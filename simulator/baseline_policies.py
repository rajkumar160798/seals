"""
Baseline retraining policies for SEALS benchmark comparison.

Implements:
- EWC (Elastic Weight Consolidation): Stabilizes by penalizing weight changes
- ER (Experience Replay): Stabilizes by replaying old data
- ADWIN: Adaptive Windowing drift detector
- DDM: Drift Detection Method
- FixedInterval: Industry standard fixed schedule
"""

import numpy as np
import torch
import copy
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class BaselinePolicy:
    """Base class for all baseline policies."""
    
    def __init__(self, name: str, seed: int = 42):
        """Initialize policy."""
        self.name = name
        self.seed = seed
        self.step = 0
        self.retrain_steps = []
        self.retrain_costs = []
    
    def decide_retrain(
        self,
        accuracy: float,
        drift_signal: float,
        model: Optional[object] = None,
        **kwargs
    ) -> bool:
        """
        Decide whether to retrain.
        
        Args:
            accuracy: Current model accuracy
            drift_signal: Drift magnitude
            model: Reference to model (for EWC)
            **kwargs: Additional context
            
        Returns:
            True if should retrain
        """
        raise NotImplementedError
    
    def on_retrain(self, X_train, y_train, model):
        """Hook called after retraining."""
        self.retrain_steps.append(self.step)
        self.retrain_costs.append(10.0)  # Default cost
    
    def log_statistics(self) -> Dict:
        """Get policy statistics."""
        return {
            'name': self.name,
            'total_retrains': len(self.retrain_steps),
            'avg_retrain_interval': (self.step / max(len(self.retrain_steps), 1)),
            'total_cost': sum(self.retrain_costs),
        }


class FixedIntervalPolicy(BaselinePolicy):
    """
    Fixed interval retraining (industry standard).
    
    Retrains every N steps regardless of drift.
    """
    
    def __init__(self, interval: int = 50, seed: int = 42):
        """Initialize."""
        super().__init__(f"FixedInterval-{interval}", seed=seed)
        self.interval = interval
    
    def decide_retrain(self, accuracy: float, drift_signal: float, model=None, **kwargs) -> bool:
        """Retrain every N steps."""
        should_retrain = (self.step % self.interval) == 0 and self.step > 0
        self.step += 1
        return should_retrain


class ADWINPolicy(BaselinePolicy):
    """
    ADWIN-based drift detection.
    
    Retrain when ADWIN detects significant drift in error rate.
    """
    
    def __init__(self, delta: float = 0.002, seed: int = 42):
        """
        Initialize ADWIN policy.
        
        Args:
            delta: Significance level for drift
        """
        super().__init__(f"ADWIN-{delta}", seed=seed)
        self.delta = delta
        self.error_history = []
        self.window_size = 100
    
    def decide_retrain(self, accuracy: float, drift_signal: float, model=None, **kwargs) -> bool:
        """Retrain when ADWIN detects error rate change."""
        error = 1.0 - accuracy
        self.error_history.append(error)
        
        # Simplified ADWIN: detect change in mean error over recent window
        if len(self.error_history) < 2 * self.window_size:
            self.step += 1
            return False
        
        recent_errors = self.error_history[-self.window_size:]
        old_errors = self.error_history[-2*self.window_size:-self.window_size]
        
        recent_mean = np.mean(recent_errors)
        old_mean = np.mean(old_errors)
        
        # Statistical test for significant change
        recent_std = np.std(recent_errors) / np.sqrt(len(recent_errors))
        old_std = np.std(old_errors) / np.sqrt(len(old_errors))
        
        threshold = 2 * np.sqrt(recent_std**2 + old_std**2)
        should_retrain = abs(recent_mean - old_mean) > threshold
        
        self.step += 1
        return should_retrain


class DDMPolicy(BaselinePolicy):
    """
    Drift Detection Method.
    
    Monitors error rate and retrains when drift is detected.
    """
    
    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0, seed: int = 42):
        """
        Initialize DDM policy.
        
        Args:
            warning_level: Warning threshold (std from baseline)
            drift_level: Drift threshold
        """
        super().__init__("DDM", seed=seed)
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.error_history = []
        self.baseline_error = None
        self.baseline_std = None
    
    def decide_retrain(self, accuracy: float, drift_signal: float, model=None, **kwargs) -> bool:
        """Retrain when DDM detects drift."""
        error = 1.0 - accuracy
        self.error_history.append(error)
        
        # Initialize baseline
        if len(self.error_history) < 50:
            self.step += 1
            return False
        
        if self.baseline_error is None:
            self.baseline_error = np.mean(self.error_history[:50])
            self.baseline_std = np.std(self.error_history[:50])
        
        # Compute warning and drift
        current_error = np.mean(self.error_history[-10:])
        std_error = np.sqrt(current_error * (1 - current_error) / len(self.error_history))
        
        warning_threshold = self.baseline_error + self.warning_level * std_error
        drift_threshold = self.baseline_error + self.drift_level * std_error
        
        should_retrain = current_error > drift_threshold
        
        self.step += 1
        return should_retrain


class EWCPolicy(BaselinePolicy):
    """
    Elastic Weight Consolidation.
    
    Prevents catastrophic forgetting by penalizing changes to important weights.
    Retrains based on drift but applies EWC loss.
    """
    
    def __init__(self, ewc_lambda: float = 0.5, seed: int = 42):
        """
        Initialize EWC policy.
        
        Args:
            ewc_lambda: EWC regularization strength
        """
        super().__init__(f"EWC-{ewc_lambda}", seed=seed)
        self.ewc_lambda = ewc_lambda
        self.important_params = {}  # Fisher information
        self.previous_task_params = {}
        self.drift_threshold = 0.1
    
    def decide_retrain(self, accuracy: float, drift_signal: float, model=None, **kwargs) -> bool:
        """
        Retrain when drift detected, apply EWC loss.
        
        Args:
            accuracy: Current accuracy
            drift_signal: Drift magnitude
            model: DeepModel instance
        """
        should_retrain = drift_signal > self.drift_threshold and self.step > 50
        
        if should_retrain and model is not None:
            # Store old parameters before retraining
            self.previous_task_params = copy.deepcopy(model._get_param_dict())
        
        self.step += 1
        return should_retrain
    
    def apply_ewc_loss(self, model) -> float:
        """
        Compute EWC regularization loss.
        
        Args:
            model: DeepModel instance
            
        Returns:
            EWC loss value
        """
        if not self.important_params or not self.previous_task_params:
            return 0.0
        
        ewc_loss = 0.0
        for name, param in model.model.named_parameters():
            if name in self.important_params:
                fisher = self.important_params[name]
                previous_param = self.previous_task_params.get(name, param.data)
                
                # EWC: penalize deviation from previous task weighted by importance
                ewc_loss += (fisher * (param - previous_param) ** 2).sum()
        
        return self.ewc_lambda * ewc_loss


class ExperienceReplayPolicy(BaselinePolicy):
    """
    Experience Replay.
    
    Prevents catastrophic forgetting by replaying samples from previous tasks.
    """
    
    def __init__(self, replay_ratio: float = 0.5, seed: int = 42):
        """
        Initialize ER policy.
        
        Args:
            replay_ratio: Fraction of batch that should be replay samples
        """
        super().__init__(f"ER-{replay_ratio}", seed=seed)
        self.replay_ratio = replay_ratio
        self.drift_threshold = 0.1
    
    def decide_retrain(self, accuracy: float, drift_signal: float, model=None, **kwargs) -> bool:
        """Retrain when drift detected."""
        should_retrain = drift_signal > self.drift_threshold and self.step > 50
        self.step += 1
        return should_retrain
    
    def augment_batch_with_replay(
        self,
        X_new: torch.Tensor,
        y_new: torch.Tensor,
        model
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mix new data with replay buffer samples.
        
        Args:
            X_new: New training data
            y_new: New labels
            model: DeepModel with buffer
            
        Returns:
            (augmented_X, augmented_y)
        """
        if len(model.buffer['X']) == 0:
            return X_new, y_new
        
        # Determine replay size
        n_replay = int(len(X_new) * self.replay_ratio / (1 - self.replay_ratio))
        n_replay = min(n_replay, len(model.buffer['X']))
        
        if n_replay == 0:
            return X_new, y_new
        
        # Sample from buffer
        replay_indices = np.random.choice(len(model.buffer['X']), size=n_replay, replace=False)
        X_replay = torch.stack([model.buffer['X'][i] for i in replay_indices])
        y_replay = torch.stack([model.buffer['y'][i] for i in replay_indices])
        
        # Concatenate
        X_augmented = torch.cat([X_new, X_replay], dim=0)
        y_augmented = torch.cat([y_new, y_replay], dim=0)
        
        return X_augmented, y_augmented


class SEALSPolicy(BaselinePolicy):
    """
    SEALS (our method): Regret-minimizing adaptive retraining.
    
    Retrains based on drift + SPI + risk considerations.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.1, seed: int = 42):
        """
        Initialize SEALS policy.
        
        Args:
            alpha: Accuracy weight
            beta: Cost weight
            gamma: Risk weight
        """
        super().__init__(f"SEALS-{alpha:.1f}_{beta:.1f}_{gamma:.1f}", seed=seed)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.drift_threshold = 0.1
        self.spi_threshold = 0.3
    
    def decide_retrain(self, accuracy: float, drift_signal: float, model=None, **kwargs) -> bool:
        """
        Retrain using regret minimization.
        
        Args:
            accuracy: Current accuracy
            drift_signal: Drift magnitude
            model: Model reference
            **kwargs: May include 'spi' and 'risk'
        """
        spi = kwargs.get('spi', 0.5)
        risk = kwargs.get('risk', 1.0 - accuracy)
        
        # Retrain if drift is significant AND SPI indicates we can benefit
        should_retrain = (
            drift_signal > self.drift_threshold and
            spi > self.spi_threshold and
            self.step > 50
        )
        
        self.step += 1
        return should_retrain


class ComparableBaselines:
    """
    Factory for creating comparable baselines.
    
    Ensures all policies are evaluated under same conditions.
    """


class AutoSEALSPolicy(SEALSPolicy):
    """
    Auto-SEALS: SEALS with learned weights.
    
    Instead of fixed α, β, γ, this policy learns them based on
    observed regret patterns in the deployment environment.
    """
    
    def __init__(
        self,
        initial_alpha: float = 1.0,
        initial_beta: float = 0.1,
        initial_gamma: float = 0.1,
        learning_rate: float = 0.05,
        seed: int = 42
    ):
        """
        Initialize Auto-SEALS with learning capability.
        
        Args:
            initial_alpha: Starting accuracy weight
            initial_beta: Starting cost weight
            initial_gamma: Starting risk weight
            learning_rate: Gradient step size for adaptation
            seed: Random seed
        """
        super().__init__(alpha=initial_alpha, beta=initial_beta, gamma=initial_gamma, seed=seed)
        self.name = "Auto-SEALS"
        
        # Learning state
        self.accuracy_history = []
        self.cost_history = []
        self.risk_history = []
        self.regret_history = []
        self.learning_rate = learning_rate
        self.warmup_steps = 30
    
    def update_weights_from_feedback(
        self,
        accuracy: float,
        cost: float,
        risk: float,
        cumulative_regret: float
    ):
        """
        Update α, β, γ based on observed metrics.
        
        Args:
            accuracy: Current model accuracy
            cost: Cost of last decision
            risk: Risk of last decision
            cumulative_regret: Total regret so far
        """
        self.accuracy_history.append(accuracy)
        self.cost_history.append(cost)
        self.risk_history.append(risk)
        self.regret_history.append(cumulative_regret)
        
        # Only learn after warmup
        if len(self.accuracy_history) < self.warmup_steps:
            return
        
        # Analyze recent performance
        window = min(20, len(self.accuracy_history))
        recent_acc = np.mean(self.accuracy_history[-window:])
        recent_cost = np.mean(self.cost_history[-window:])
        recent_risk = np.mean(self.risk_history[-window:])
        
        # Compute "pain points" - areas where performance is weak
        target_acc = 0.95
        target_cost = 10.0
        target_risk = 0.1
        
        pain_acc = max(0, target_acc - recent_acc)
        pain_cost = max(0, recent_cost - 5.0) / max(target_cost, 1.0)
        pain_risk = max(0, recent_risk - target_risk) / max(target_risk, 1.0)
        
        # Normalize pains
        total_pain = pain_acc + pain_cost + pain_risk + 1e-6
        
        # Adjust weights: increase weight on painful dimensions
        delta_alpha = (pain_acc / total_pain) * self.learning_rate
        delta_beta = (pain_cost / total_pain) * self.learning_rate
        delta_gamma = (pain_risk / total_pain) * self.learning_rate
        
        self.alpha += delta_alpha
        self.beta += delta_beta
        self.gamma += delta_gamma
        
        # Keep weights normalized and bounded
        current_sum = self.alpha + self.beta + self.gamma
        target_sum = 1.2
        scale = target_sum / current_sum
        
        self.alpha = np.clip(self.alpha * scale, 0.01, 5.0)
        self.beta = np.clip(self.beta * scale, 0.01, 5.0)
        self.gamma = np.clip(self.gamma * scale, 0.01, 5.0)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning progress statistics."""
        return {
            'steps_observed': len(self.accuracy_history),
            'current_alpha': self.alpha,
            'current_beta': self.beta,
            'current_gamma': self.gamma,
            'avg_accuracy': np.mean(self.accuracy_history[-20:]) if self.accuracy_history else 0.0,
            'avg_cost': np.mean(self.cost_history[-20:]) if self.cost_history else 0.0,
            'avg_risk': np.mean(self.risk_history[-20:]) if self.risk_history else 0.0,
        }
    
    @staticmethod
    def create_all() -> Dict[str, BaselinePolicy]:
        """Create all baseline policies."""
        return {
            'FixedInterval-50': FixedIntervalPolicy(interval=50),
            'FixedInterval-100': FixedIntervalPolicy(interval=100),
            'ADWIN': ADWINPolicy(),
            'DDM': DDMPolicy(),
            'EWC': EWCPolicy(ewc_lambda=0.5),
            'ER': ExperienceReplayPolicy(replay_ratio=0.5),
            'SEALS': SEALSPolicy(alpha=1.0, beta=0.1, gamma=0.1),
            'Auto-SEALS': AutoSEALSPolicy(),
        }
