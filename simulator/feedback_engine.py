"""
Feedback Engine Module

Manages three feedback regimes for ML systems:
1. Passive feedback (error-based, no human)
2. Human-in-the-loop feedback (selective relabeling)
3. Policy feedback (business rule interventions)

Models feedback as a probabilistic signal dependent on trust and cost.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum


class FeedbackRegime(Enum):
    """Types of feedback mechanisms."""
    PASSIVE = "passive"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"
    POLICY = "policy"


@dataclass
class FeedbackConfig:
    """Configuration for feedback system."""
    regime: FeedbackRegime
    cost_per_label: float  # Monetary or computational cost
    human_trust: float  # 0 to 1, confidence in human feedback
    labeling_accuracy: float  # 0 to 1, accuracy of labels
    budget: float  # Total feedback budget
    seed: int = 42


class FeedbackEngine:
    """
    Orchestrates feedback signal generation and management.
    
    Models feedback probabilistically based on trust and cost.
    """
    
    def __init__(self, config: FeedbackConfig):
        """
        Args:
            config: FeedbackConfig instance
        """
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.total_cost_spent = 0.0
        self.feedback_history = []
        
    def passive_feedback(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainty: np.ndarray = None
    ) -> Tuple[np.ndarray, List[int], float]:
        """
        Passive feedback: obtain labels for misclassified samples.
        
        Args:
            y_true: True labels (oracle)
            y_pred: Predicted labels
            uncertainty: Prediction uncertainty (optional)
            
        Returns:
            (corrected_labels, labeled_indices, cost_incurred)
        """
        # Identify errors
        errors = (y_true != y_pred)
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            return y_pred.copy(), [], 0.0
            
        # Cost of labeling all errors
        cost = len(error_indices) * self.config.cost_per_label
        
        # Budget check
        if self.total_cost_spent + cost > self.config.budget:
            # Label only a subset within budget
            affordable = int((self.config.budget - self.total_cost_spent) / self.config.cost_per_label)
            error_indices = error_indices[:affordable]
            cost = affordable * self.config.cost_per_label
        
        # Apply corrections with labeling error
        corrected = y_pred.copy()
        labeled_indices = list(error_indices)
        
        for idx in labeled_indices:
            # Labeling error
            if self.rng.rand() < self.config.labeling_accuracy:
                corrected[idx] = y_true[idx]
            # If labeling fails, keep prediction
                
        self.total_cost_spent += cost
        self.feedback_history.append({
            "regime": "passive",
            "labeled_count": len(labeled_indices),
            "cost": cost
        })
        
        return corrected, labeled_indices, cost
    
    def human_in_the_loop_feedback(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainty: np.ndarray,
        selection_strategy: str = "uncertainty"
    ) -> Tuple[np.ndarray, List[int], float]:
        """
        Human-in-the-loop: selectively query labels based on uncertainty.
        
        Args:
            y_true: True labels (oracle)
            y_pred: Predicted labels
            uncertainty: Prediction uncertainty (0 to 1, higher = more uncertain)
            selection_strategy: "uncertainty" or "margin"
            
        Returns:
            (corrected_labels, labeled_indices, cost_incurred)
        """
        # Normalize uncertainty
        uncertainty = np.clip(uncertainty, 0, 1)
        
        if selection_strategy == "uncertainty":
            # Query most uncertain samples
            ranking = np.argsort(-uncertainty)
        elif selection_strategy == "margin":
            # Query near-decision-boundary samples
            ranking = np.argsort(np.abs(uncertainty - 0.5))
        else:
            raise ValueError(f"Unknown strategy: {selection_strategy}")
        
        # Determine how many we can afford
        n_affordable = int(self.config.budget - self.total_cost_spent / self.config.cost_per_label)
        n_to_label = min(n_affordable, len(ranking))
        
        selected_indices = ranking[:n_to_label]
        labeled_indices = list(selected_indices)
        
        # Apply corrections with labeling error
        corrected = y_pred.copy()
        for idx in labeled_indices:
            if self.rng.rand() < self.config.labeling_accuracy:
                corrected[idx] = y_true[idx]
        
        cost = n_to_label * self.config.cost_per_label
        self.total_cost_spent += cost
        
        # Model trust as confidence boost
        trust_boost = self.config.human_trust
        
        self.feedback_history.append({
            "regime": "human_in_the_loop",
            "labeled_count": n_to_label,
            "cost": cost,
            "strategy": selection_strategy,
            "trust_boost": trust_boost
        })
        
        return corrected, labeled_indices, cost
    
    def policy_feedback(
        self,
        y_pred: np.ndarray,
        risk_flags: np.ndarray = None
    ) -> Tuple[np.ndarray, List[int], float]:
        """
        Policy feedback: override predictions based on safety/business rules.
        
        Args:
            y_pred: Predicted labels
            risk_flags: Boolean array indicating high-risk predictions
            
        Returns:
            (corrected_labels, overridden_indices, cost_incurred)
        """
        if risk_flags is None:
            risk_flags = np.zeros(len(y_pred), dtype=bool)
        
        # Identify high-risk predictions
        risk_indices = np.where(risk_flags)[0]
        
        # Policy cost: less than human labeling but non-zero
        cost = len(risk_indices) * self.config.cost_per_label * 0.2
        
        if self.total_cost_spent + cost > self.config.budget:
            # Limit to budget
            affordable = int((self.config.budget - self.total_cost_spent) / (self.config.cost_per_label * 0.2))
            risk_indices = risk_indices[:affordable]
            cost = affordable * self.config.cost_per_label * 0.2
        
        corrected = y_pred.copy()
        overridden_indices = list(risk_indices)
        
        # Override high-risk predictions with conservative choice (0)
        for idx in overridden_indices:
            corrected[idx] = 0  # Conservative override
        
        self.total_cost_spent += cost
        
        self.feedback_history.append({
            "regime": "policy",
            "overridden_count": len(overridden_indices),
            "cost": cost
        })
        
        return corrected, overridden_indices, cost
    
    def probabilistic_feedback(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainty: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute feedback probability based on trust and cost.
        
        F_t ~ P(F_t | trust, cost) = exp(β₁·trust - β₂·cost) / Z
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            uncertainty: Prediction uncertainty
            
        Returns:
            (corrected_labels, feedback_probability, cost_incurred)
        """
        # Compute raw probability
        β1 = 0.5  # Trust coefficient
        β2 = 0.01  # Cost coefficient
        
        raw_prob = np.exp(
            β1 * self.config.human_trust - β2 * self.config.cost_per_label
        )
        feedback_prob = raw_prob / (1 + raw_prob)  # Normalize to [0, 1]
        
        # Sample feedback occurrence
        will_feedback = self.rng.rand() < feedback_prob
        
        if will_feedback:
            # Human-in-the-loop with uncertainty sampling
            corrected, labeled_indices, cost = self.human_in_the_loop_feedback(
                y_true, y_pred, uncertainty
            )
            return corrected, feedback_prob, cost
        else:
            return y_pred.copy(), feedback_prob, 0.0
    
    def get_feedback_signal(self) -> np.ndarray:
        """
        Get summary of feedback activity.
        
        Returns:
            Array [total_samples_labeled, total_cost, avg_accuracy_improvement]
        """
        if not self.feedback_history:
            return np.array([0, 0, 0])
        
        total_labeled = sum(
            entry.get("labeled_count", 0) + entry.get("overridden_count", 0)
            for entry in self.feedback_history
        )
        total_cost = self.total_cost_spent
        
        return np.array([total_labeled, total_cost])
    
    def remaining_budget(self) -> float:
        """Get remaining feedback budget."""
        return self.config.budget - self.total_cost_spent
    
    def reset(self):
        """Reset feedback engine state."""
        self.total_cost_spent = 0.0
        self.feedback_history = []
