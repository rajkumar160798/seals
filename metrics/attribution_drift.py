"""
Attribution Drift Module

Detects and measures changes in feature importance over time.

Concept: While accuracy may remain stable, the reasoning changes.
This is important for:
- Explainability systems
- Regulatory compliance (model interpretability)
- Debugging model behavior
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class AttributionSnapshot:
    """Single point-in-time attribution measurement."""
    step: int
    shap_values: np.ndarray  # Mean absolute SHAP per feature
    feature_ranking: np.ndarray  # Ranking of features by importance
    entropy: float  # Entropy of importance distribution
    dominant_features: List[int]  # Top-k important features


class AttributionDriftDetector:
    """
    Detects and quantifies changes in feature importance.
    
    Even if accuracy is stable, explanations (SHAP values) may drift.
    This indicates concept drift that isn't yet visible in performance.
    """
    
    def __init__(self, n_features: int, top_k: int = 5, window_size: int = 10):
        """
        Args:
            n_features: Number of features
            top_k: Number of top features to track
            window_size: Historical window for trend analysis
        """
        self.n_features = n_features
        self.top_k = top_k
        self.window_size = window_size
        
        # History
        self.shap_history = []
        self.ranking_history = []
        self.entropy_history = []
        self.snapshots = []
        
    def update(self, shap_values: np.ndarray, step: int) -> float:
        """
        Update with new SHAP values.
        
        Args:
            shap_values: Mean absolute SHAP values (n_features,)
            step: Current step
            
        Returns:
            Attribution drift magnitude
        """
        # Normalize SHAP values
        shap_normalized = np.abs(shap_values)
        shap_normalized = shap_normalized / (np.sum(shap_normalized) + 1e-10)
        
        self.shap_history.append(shap_normalized)
        
        # Compute ranking
        ranking = np.argsort(-shap_normalized)
        self.ranking_history.append(ranking)
        
        # Compute entropy of importance distribution
        entropy = -np.sum(shap_normalized * np.log(shap_normalized + 1e-10))
        self.entropy_history.append(entropy)
        
        # Get dominant features
        dominant = ranking[:self.top_k].tolist()
        
        # Create snapshot
        snapshot = AttributionSnapshot(
            step=step,
            shap_values=shap_normalized,
            feature_ranking=ranking,
            entropy=entropy,
            dominant_features=dominant
        )
        self.snapshots.append(snapshot)
        
        # Compute drift
        drift = self._compute_drift()
        
        # Keep history bounded
        if len(self.shap_history) > self.window_size:
            self.shap_history.pop(0)
            self.ranking_history.pop(0)
            self.entropy_history.pop(0)
        
        return drift
    
    def _compute_drift(self) -> float:
        """
        Compute attribution drift as change in top-k features.
        
        Returns:
            Drift magnitude (0 to 1)
        """
        if len(self.ranking_history) < 2:
            return 0.0
        
        # Compare last two snapshots
        prev_ranking = self.ranking_history[-2]
        curr_ranking = self.ranking_history[-1]
        
        # Get top-k features
        prev_top_k = set(prev_ranking[:self.top_k])
        curr_top_k = set(curr_ranking[:self.top_k])
        
        # Jaccard distance
        intersection = len(prev_top_k & curr_top_k)
        union = len(prev_top_k | curr_top_k)
        
        jaccard_dist = 1 - (intersection / union) if union > 0 else 0.0
        
        # Also consider magnitude of SHAP changes
        magnitude_change = np.linalg.norm(
            self.shap_history[-1] - self.shap_history[-2]
        )
        
        # Combined drift
        drift = 0.5 * jaccard_dist + 0.5 * magnitude_change
        return float(np.clip(drift, 0, 1))
    
    def get_drift_signal(self) -> np.ndarray:
        """
        Get attribution drift component of full drift signal.
        
        Returns:
            D_SHAP value from drift signal vector
        """
        if not self.snapshots:
            return 0.0
        
        recent_drifts = []
        for i in range(max(0, len(self.snapshots) - self.window_size), len(self.snapshots)):
            drift = self._compute_drift_at_index(i)
            recent_drifts.append(drift)
        
        return float(np.mean(recent_drifts)) if recent_drifts else 0.0
    
    def _compute_drift_at_index(self, idx: int) -> float:
        """Compute drift at specific index."""
        if idx < 1:
            return 0.0
        
        prev_ranking = self.ranking_history[idx - 1]
        curr_ranking = self.ranking_history[idx]
        
        prev_top_k = set(prev_ranking[:self.top_k])
        curr_top_k = set(curr_ranking[:self.top_k])
        
        intersection = len(prev_top_k & curr_top_k)
        union = len(prev_top_k | curr_top_k)
        
        jaccard_dist = 1 - (intersection / union) if union > 0 else 0.0
        magnitude_change = np.linalg.norm(
            self.shap_history[idx] - self.shap_history[idx - 1]
        )
        
        return 0.5 * jaccard_dist + 0.5 * magnitude_change
    
    def get_feature_importance_trajectory(self, feature_idx: int) -> List[float]:
        """
        Get importance trajectory for a single feature.
        
        Args:
            feature_idx: Index of feature
            
        Returns:
            List of importance values over time
        """
        trajectory = []
        for snapshot in self.snapshots:
            trajectory.append(float(snapshot.shap_values[feature_idx]))
        return trajectory
    
    def get_top_k_stability(self) -> float:
        """
        Measure stability of top-k features.
        
        High stability (1) = same top features throughout.
        Low stability (0) = constantly changing features.
        
        Returns:
            Stability score (0 to 1)
        """
        if not self.snapshots:
            return 0.0
        
        # Get all top-k sets
        top_k_sets = [set(s.dominant_features) for s in self.snapshots]
        
        if len(top_k_sets) < 2:
            return 1.0
        
        # Compute intersection of all top-k sets
        intersection = set(top_k_sets[0])
        for s in top_k_sets[1:]:
            intersection &= s
        
        stability = len(intersection) / self.top_k if self.top_k > 0 else 0.0
        return float(stability)
    
    def get_entropy_trend(self) -> float:
        """
        Get trend in entropy (order vs disorder of importances).
        
        Increasing entropy = spreading importance across features
        Decreasing entropy = concentrating importance on few features
        
        Returns:
            Entropy change rate
        """
        if len(self.entropy_history) < 2:
            return 0.0
        
        recent_entropies = self.entropy_history[-10:]
        if len(recent_entropies) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(recent_entropies))
        slope = np.polyfit(x, recent_entropies, 1)[0]
        return float(slope)


class DiscrepancyAnalyzer:
    """
    Analyzes discrepancy between performance and explanation stability.
    
    Key insight: Accuracy may stay high while explanations drift.
    This indicates the model is right for wrong reasons.
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.accuracy_history = []
        self.attribution_drift_history = []
        self.discrepancy_history = []
        
    def record(self, accuracy: float, attribution_drift: float):
        """
        Record accuracy and attribution drift at same step.
        
        Args:
            accuracy: Model accuracy (0 to 1)
            attribution_drift: Magnitude of attribution drift (0 to 1)
        """
        self.accuracy_history.append(accuracy)
        self.attribution_drift_history.append(attribution_drift)
        
        # Discrepancy: high accuracy despite attribution drift
        # Indicates model is unstable/brittle
        discrepancy = attribution_drift * (1 - accuracy)  # Max when high drift + low accuracy
        self.discrepancy_history.append(discrepancy)
    
    def get_discrepancy_episodes(self, threshold: float = 0.3) -> List[Tuple[int, int]]:
        """
        Identify episodes where high discrepancy occurs.
        
        Args:
            threshold: Discrepancy threshold
            
        Returns:
            List of (start_idx, end_idx) tuples for high-discrepancy episodes
        """
        episodes = []
        in_episode = False
        episode_start = 0
        
        for i, disc in enumerate(self.discrepancy_history):
            if disc > threshold and not in_episode:
                episode_start = i
                in_episode = True
            elif disc <= threshold and in_episode:
                episodes.append((episode_start, i))
                in_episode = False
        
        if in_episode:
            episodes.append((episode_start, len(self.discrepancy_history)))
        
        return episodes
    
    def get_discrepancy_score(self) -> float:
        """
        Get average discrepancy over all time.
        
        High score = explanations drift despite stable accuracy.
        
        Returns:
            Mean discrepancy (0 to 1)
        """
        if not self.discrepancy_history:
            return 0.0
        return float(np.mean(self.discrepancy_history))
    
    def get_analysis(self) -> Dict:
        """Get comprehensive analysis of accuracy-explanation discrepancy."""
        if not self.accuracy_history:
            return {}
        
        acc_array = np.array(self.accuracy_history)
        drift_array = np.array(self.attribution_drift_history)
        disc_array = np.array(self.discrepancy_history)
        
        # Correlation between accuracy and attribution drift
        if len(acc_array) > 1:
            correlation = float(np.corrcoef(acc_array, drift_array)[0, 1])
        else:
            correlation = 0.0
        
        return {
            "mean_accuracy": float(np.mean(acc_array)),
            "mean_attribution_drift": float(np.mean(drift_array)),
            "mean_discrepancy": float(np.mean(disc_array)),
            "accuracy_std": float(np.std(acc_array)),
            "attribution_drift_std": float(np.std(drift_array)),
            "correlation": correlation,
            "discrepancy_episodes": self.get_discrepancy_episodes()
        }
    
    def reset(self):
        """Reset analyzer."""
        self.accuracy_history = []
        self.attribution_drift_history = []
        self.discrepancy_history = []
