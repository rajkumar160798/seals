"""
Retraining Policy Module

Defines and manages adaptive retraining decisions.

Implements three retraining regimes:
1. Over-plastic: Retrains too frequently (unstable)
2. Over-stable: Retrains too rarely (stale)
3. Balanced: Optimal adaptive retraining

Core mechanism uses drift signal and stability-plasticity index (SPI).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from enum import Enum


class RetrainingRegime(Enum):
    """Retraining policies."""
    OVER_PLASTIC = "over_plastic"
    OVER_STABLE = "over_stable"
    BALANCED = "balanced"
    CUSTOM = "custom"


@dataclass
class RetrainingConfig:
    """Configuration for retraining policy."""
    regime: RetrainingRegime
    drift_threshold: float = 0.3  # Threshold for drift signal
    spi_threshold: float = 0.5  # Threshold for stability-plasticity index
    max_risk: float = 0.5  # Maximum acceptable risk
    cost_budget: float = 1000.0  # Computational budget
    min_interval: int = 1  # Minimum steps between retraining
    max_interval: int = 100  # Maximum steps without retraining
    seed: int = 42


class RetrainingPolicy:
    """
    Adaptive retraining decision engine.
    
    Makes decisions based on:
    - Drift signal D_t
    - Stability-plasticity index SPI_t
    - Computational cost constraints
    - Risk bounds
    """
    
    def __init__(self, config: RetrainingConfig):
        """
        Args:
            config: RetrainingConfig instance
        """
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.last_retrain_step = 0
        self.current_step = 0
        self.retrain_history = []
        self.cost_spent = 0.0
        
    def decide_retrain(
        self,
        drift_signal: np.ndarray,
        spi: float,
        risk: float,
        cost: float
    ) -> bool:
        """
        Make retraining decision based on current state.
        
        Args:
            drift_signal: Multi-dimensional drift vector [D_KS, D_ADWIN, D_Error, D_SHAP]
            spi: Stability-plasticity index
            risk: Current risk level
            cost: Cost of retraining
            
        Returns:
            Whether to retrain
        """
        self.current_step += 1
        
        # Check minimum interval constraint
        steps_since_last = self.current_step - self.last_retrain_step
        if steps_since_last < self.config.min_interval:
            return False
        
        # Check cost budget
        if self.cost_spent + cost > self.config.cost_budget:
            return False
        
        # Check risk bounds
        if risk > self.config.max_risk:
            return True  # Force retrain if risk too high
        
        # Apply regime-specific logic
        if self.config.regime == RetrainingRegime.OVER_PLASTIC:
            return self._retrain_over_plastic(drift_signal, spi)
        elif self.config.regime == RetrainingRegime.OVER_STABLE:
            return self._retrain_over_stable(drift_signal, spi)
        elif self.config.regime == RetrainingRegime.BALANCED:
            return self._retrain_balanced(drift_signal, spi)
        else:
            return False
    
    def _retrain_over_plastic(self, drift_signal: np.ndarray, spi: float) -> bool:
        """
        Over-plastic regime: retrain frequently.
        
        High plasticity, low stability → oscillation.
        """
        # Always retrain if maximum interval exceeded
        steps_since = self.current_step - self.last_retrain_step
        if steps_since >= 5:  # Very short interval
            return True
        
        # Retrain on any detected drift
        return np.mean(drift_signal) > 0.1  # Aggressive threshold
    
    def _retrain_over_stable(self, drift_signal: np.ndarray, spi: float) -> bool:
        """
        Over-stable regime: retrain rarely.
        
        Low plasticity, high stability → stagnation.
        """
        # Retrain only if maximum interval exceeded
        steps_since = self.current_step - self.last_retrain_step
        if steps_since >= self.config.max_interval:
            return True
        
        # Retrain only on severe drift
        return np.mean(drift_signal) > 0.7  # Very conservative threshold
    
    def _retrain_balanced(self, drift_signal: np.ndarray, spi: float) -> bool:
        """
        Balanced regime: adaptive retraining.
        
        Retrains when drift detected AND SPI confirms plasticity opportunity.
        """
        drift_magnitude = np.mean(drift_signal)
        steps_since = self.current_step - self.last_retrain_step
        
        # Condition 1: Significant drift detected
        drift_detected = drift_magnitude > self.config.drift_threshold
        
        # Condition 2: Sufficient plasticity (SPI indicates learning opportunity)
        plasticity_good = spi > self.config.spi_threshold
        
        # Condition 3: Respect minimum interval
        interval_ok = steps_since >= self.config.min_interval
        
        # Condition 4: Force retrain if too long without update
        force_retrain = steps_since >= self.config.max_interval
        
        decision = (drift_detected and plasticity_good and interval_ok) or force_retrain
        
        return decision
    
    def log_retrain(self, cost: float, reason: str = ""):
        """
        Log a retraining event.
        
        Args:
            cost: Cost of the retraining
            reason: Reason for retraining
        """
        self.last_retrain_step = self.current_step
        self.cost_spent += cost
        self.retrain_history.append({
            "step": self.current_step,
            "cost": cost,
            "reason": reason
        })
    
    def get_regime_stats(self) -> Dict:
        """
        Get statistics about retraining behavior.
        
        Returns:
            Dictionary with retraining statistics
        """
        if not self.retrain_history:
            return {
                "total_retrains": 0,
                "avg_interval": self.current_step,
                "total_cost": 0.0,
                "cost_per_retrain": 0.0,
                "retraining_frequency": 0.0
            }
        
        retrains = len(self.retrain_history)
        intervals = []
        for i in range(1, len(self.retrain_history)):
            interval = (self.retrain_history[i]["step"] - 
                       self.retrain_history[i-1]["step"])
            intervals.append(interval)
        
        avg_interval = np.mean(intervals) if intervals else self.current_step
        total_cost = sum(r["cost"] for r in self.retrain_history)
        cost_per_retrain = total_cost / retrains if retrains > 0 else 0.0
        frequency = retrains / max(self.current_step, 1)
        
        return {
            "total_retrains": retrains,
            "avg_interval": avg_interval,
            "total_cost": total_cost,
            "cost_per_retrain": cost_per_retrain,
            "retraining_frequency": frequency
        }
    
    def get_next_retrain_window(self) -> Tuple[int, int]:
        """
        Get recommended window for next retraining.
        
        Returns:
            (earliest_step, latest_step) for next retrain
        """
        earliest = self.current_step + self.config.min_interval
        latest = self.current_step + self.config.max_interval
        return earliest, latest
    
    def reset(self):
        """Reset policy state."""
        self.last_retrain_step = 0
        self.current_step = 0
        self.retrain_history = []
        self.cost_spent = 0.0


class AdaptiveRetrainingScheduler:
    """
    Manages multiple retraining policies and selects best one.
    """
    
    def __init__(self, n_features: int = 10, seed: int = 42):
        """
        Args:
            n_features: Number of features for SHAP-based decisions
            seed: Random seed
        """
        self.n_features = n_features
        self.rng = np.random.RandomState(seed)
        
        # Create three policies
        self.policies = {
            "plastic": RetrainingPolicy(
                RetrainingConfig(regime=RetrainingRegime.OVER_PLASTIC)
            ),
            "stable": RetrainingPolicy(
                RetrainingConfig(regime=RetrainingRegime.OVER_STABLE)
            ),
            "balanced": RetrainingPolicy(
                RetrainingConfig(regime=RetrainingRegime.BALANCED)
            ),
        }
        
        self.current_policy = "balanced"
        
    def select_policy(self, performance_history: np.ndarray) -> str:
        """
        Select best policy based on recent performance.
        
        Args:
            performance_history: Recent accuracy scores
            
        Returns:
            Selected policy name
        """
        if len(performance_history) < 10:
            return "balanced"
        
        recent_variance = np.var(performance_history[-10:])
        recent_mean = np.mean(performance_history[-10:])
        
        # If high variance, reduce plasticity
        if recent_variance > 0.05:
            self.current_policy = "stable"
        # If improving, increase plasticity
        elif recent_mean > np.mean(performance_history[-20:-10]):
            self.current_policy = "plastic"
        # Otherwise, stay balanced
        else:
            self.current_policy = "balanced"
        
        return self.current_policy
    
    def decide_retrain(
        self,
        drift_signal: np.ndarray,
        spi: float,
        risk: float,
        cost: float
    ) -> bool:
        """Delegate to current policy."""
        policy = self.policies[self.current_policy]
        return policy.decide_retrain(drift_signal, spi, risk, cost)
    
    def log_retrain(self, cost: float, reason: str = ""):
        """Log retrain in current policy."""
        self.policies[self.current_policy].log_retrain(cost, reason)
    
    def get_stats(self) -> Dict:
        """Get stats for all policies."""
        return {
            name: policy.get_regime_stats()
            for name, policy in self.policies.items()
        }
