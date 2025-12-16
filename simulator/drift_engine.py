"""
Drift Engine Module

Simulates controlled drift in ML systems through multiple mechanisms:
- Covariate drift (input distribution shift)
- Label drift (output distribution shift)
- Concept drift (decision boundary shift)
- Feedback-induced drift (caused by system interventions)

Each drift type is parameterized to enable controlled experiments.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import warnings


class DriftType(Enum):
    """Enumeration of drift mechanisms."""
    COVARIATE = "covariate"
    LABEL = "label"
    CONCEPT = "concept"
    FEEDBACK_INDUCED = "feedback_induced"


@dataclass
class DriftConfig:
    """Configuration for drift generation."""
    drift_type: DriftType
    magnitude: float  # Intensity of drift (0 to 1)
    onset_time: int  # When drift starts (step number)
    duration: int  # Duration of drift (steps)
    gradual: bool = True  # Whether drift is gradual or abrupt
    time_varying_intensity: float = 0.0  # λ for time-varying drift: Drift_t = Drift_0 * (1 + λ*t)
    seed: int = 42


class KSDriftDetector:
    """
    Kolmogorov-Smirnov test for covariate drift.
    
    Detects changes in input feature distributions.
    """
    
    def __init__(self, window_size: int = 100, alpha: float = 0.05):
        """
        Args:
            window_size: Size of reference window for comparison
            alpha: Significance level for KS test
        """
        self.window_size = window_size
        self.alpha = alpha
        self.reference_window = None
        
    def update(self, X: np.ndarray) -> float:
        """
        Compute KS statistic for new data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Maximum KS statistic across features
        """
        if len(X) < self.window_size:
            warnings.warn("Not enough samples for KS test")
            return 0.0
            
        if self.reference_window is None:
            self.reference_window = X[-self.window_size:]
            return 0.0
            
        current_window = X[-self.window_size:]
        
        # Compute KS statistic per feature
        max_ks = 0.0
        for j in range(X.shape[1]):
            ks_stat = self._ks_statistic(
                self.reference_window[:, j],
                current_window[:, j]
            )
            max_ks = max(max_ks, ks_stat)
            
        return max_ks
    
    @staticmethod
    def _ks_statistic(x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute KS statistic between two samples."""
        n1, n2 = len(x1), len(x2)
        x1_sorted = np.sort(x1)
        x2_sorted = np.sort(x2)
        
        d_plus = 0.0
        d_minus = 0.0
        
        for i, x in enumerate(x1_sorted):
            # CDF values
            cdf1 = (i + 1) / n1
            cdf2 = np.sum(x2_sorted <= x) / n2
            d_plus = max(d_plus, cdf1 - cdf2)
            d_minus = max(d_minus, cdf2 - cdf1)
            
        return max(d_plus, d_minus)


class ADWINDriftDetector:
    """
    Adaptive Windowing (ADWIN) for concept drift.
    
    Detects changes in error rate or performance metric.
    """
    
    def __init__(self, max_buckets: int = 10, delta: float = 0.002):
        """
        Args:
            max_buckets: Maximum number of buckets for windowing
            delta: Significance level for drift detection
        """
        self.max_buckets = max_buckets
        self.delta = delta
        self.window = []
        self.bucket_boundaries = []
        
    def update(self, error: float) -> Tuple[bool, float]:
        """
        Check for drift in error metric.
        
        Args:
            error: Current error value (0 to 1)
            
        Returns:
            (drift_detected: bool, drift_magnitude: float)
        """
        self.window.append(error)
        
        if len(self.window) < 10:
            return False, 0.0
            
        # Check for significant change in recent vs. older data
        split_point = len(self.window) // 2
        older_mean = np.mean(self.window[:split_point])
        recent_mean = np.mean(self.window[split_point:])
        
        # ADWIN-style significance test
        m = len(self.window)
        variance = np.var(self.window) + 1e-10
        threshold = np.sqrt(np.log(2 / self.delta) / (2 * m)) * np.sqrt(variance)
        
        drift_mag = abs(recent_mean - older_mean)
        is_drift = drift_mag > threshold
        
        # Clip older data if drift detected (adaptive window)
        if is_drift and len(self.window) > self.max_buckets:
            self.window = self.window[split_point:]
            
        return is_drift, drift_mag


class SHAPDriftDetector:
    """
    Attribution drift detector based on SHAP values.
    
    Detects changes in feature importance rankings.
    """
    
    def __init__(self, n_features: int, window_size: int = 100):
        """
        Args:
            n_features: Number of features
            window_size: Size of historical window
        """
        self.n_features = n_features
        self.window_size = window_size
        self.shap_history = []
        
    def update(self, shap_values: np.ndarray) -> float:
        """
        Compute attribution drift as L2 distance in importance space.
        
        Args:
            shap_values: Mean absolute SHAP values (n_features,)
            
        Returns:
            Attribution drift magnitude
        """
        # Normalize to ranking
        normalized_shap = np.argsort(np.argsort(-shap_values))
        self.shap_history.append(normalized_shap)
        
        if len(self.shap_history) < 2:
            return 0.0
            
        # Compute drift as change in rankings
        prev_ranking = self.shap_history[-2]
        curr_ranking = self.shap_history[-1]
        
        drift = np.linalg.norm(curr_ranking - prev_ranking) / np.sqrt(self.n_features)
        
        # Keep window bounded
        if len(self.shap_history) > self.window_size:
            self.shap_history.pop(0)
            
        return drift


class DriftEngine:
    """
    Main drift engine orchestrating multiple drift mechanisms.
    
    Generates realistic drift scenarios for ML system simulation.
    """
    
    def __init__(self, n_features: int = 10, n_samples_per_step: int = 100, seed: int = 42):
        """
        Args:
            n_features: Number of features in the system
            n_samples_per_step: Samples generated per time step
            seed: Random seed for reproducibility
        """
        self.n_features = n_features
        self.n_samples_per_step = n_samples_per_step
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Detectors
        self.ks_detector = KSDriftDetector()
        self.adwin_detector = ADWINDriftDetector()
        self.shap_detector = SHAPDriftDetector(n_features)
        
        # State
        self.current_step = 0
        self.base_distribution_params = self._init_base_distribution()
        self.drift_config = None  # Will store DriftConfig when active
        
    def _init_base_distribution(self) -> Dict:
        """Initialize base data distribution parameters."""
        return {
            "mean": self.rng.randn(self.n_features),
            "cov": np.eye(self.n_features) * 0.5 + 0.5 * np.ones((self.n_features, self.n_features)) * 0.1,
            "label_bias": 0.5,
        }
        
    def _compute_time_varying_magnitude(self, base_magnitude: float, time_varying_intensity: float = 0.0) -> float:
        """
        Compute magnitude with time-varying intensity.
        
        Formula: Drift_t = Drift_0 * (1 + λ*t)
        
        Args:
            base_magnitude: Base drift magnitude (Drift_0)
            time_varying_intensity: λ coefficient for linear time scaling
            
        Returns:
            Adjusted magnitude accounting for time
        """
        if time_varying_intensity <= 0:
            return base_magnitude
        
        # Drift_t = Drift_0 * (1 + λ*t)
        adjusted = base_magnitude * (1.0 + time_varying_intensity * self.current_step)
        return np.clip(adjusted, 0.0, 1.0)  # Keep bounded
    
    def generate_covariate_drift(
        self,
        magnitude: float = 0.5,
        duration: int = 50,
        gradual: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate covariate drift (input distribution shift).
        
        Args:
            magnitude: Intensity of drift (0-1)
            duration: Number of steps to apply drift
            gradual: Whether drift is gradual or abrupt
            
        Returns:
            (X: features, y: labels)
        """
        # Create shifted mean
        drift_direction = self.rng.randn(self.n_features)
        drift_direction /= np.linalg.norm(drift_direction)
        
        progress = self.current_step % duration / duration if duration > 0 else 1.0
        shift_amount = magnitude * progress if gradual else magnitude
        
        shifted_mean = self.base_distribution_params["mean"] + shift_amount * drift_direction
        
        # Generate data
        X = self.rng.multivariate_normal(
            shifted_mean,
            self.base_distribution_params["cov"],
            self.n_samples_per_step
        )
        
        # Generate labels (influenced by shift)
        linear_predictor = X @ self.rng.randn(self.n_features)
        y = (linear_predictor + self.rng.randn(self.n_samples_per_step) > 0).astype(int)
        
        self.current_step += 1
        return X, y
    
    def generate_label_drift(
        self,
        magnitude: float = 0.5,
        duration: int = 50,
        gradual: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate label drift (output distribution shift).
        
        Args:
            magnitude: Intensity of drift (0-1)
            duration: Number of steps to apply drift
            gradual: Whether drift is gradual or abrupt
            
        Returns:
            (X: features, y: labels with shifted distribution)
        """
        # Generate base features (no covariate drift)
        X = self.rng.multivariate_normal(
            self.base_distribution_params["mean"],
            self.base_distribution_params["cov"],
            self.n_samples_per_step
        )
        
        # Generate base labels
        linear_predictor = X @ self.rng.randn(self.n_features)
        base_labels = (linear_predictor + self.rng.randn(self.n_samples_per_step) > 0).astype(int)
        
        # Flip labels with increasing probability
        progress = self.current_step % duration / duration if duration > 0 else 1.0
        flip_rate = magnitude * progress if gradual else magnitude
        
        flip_mask = self.rng.rand(self.n_samples_per_step) < flip_rate
        y = base_labels.copy()
        y[flip_mask] = 1 - y[flip_mask]
        
        self.current_step += 1
        return X, y
    
    def generate_concept_drift(
        self,
        magnitude: float = 0.5,
        duration: int = 50,
        gradual: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate concept drift (decision boundary shift).
        
        Args:
            magnitude: Intensity of drift (0-1)
            duration: Number of steps to apply drift
            gradual: Whether drift is gradual or abrupt
            
        Returns:
            (X: features, y: labels with shifted decision boundary)
        """
        # Generate features
        X = self.rng.multivariate_normal(
            self.base_distribution_params["mean"],
            self.base_distribution_params["cov"],
            self.n_samples_per_step
        )
        
        # Shift decision boundary
        progress = self.current_step % duration / duration if duration > 0 else 1.0
        shift_amount = magnitude * progress if gradual else magnitude
        
        decision_boundary_shift = shift_amount * self.rng.randn(self.n_features)
        threshold_shift = shift_amount * 2 - 1  # Shift from -1 to 1
        
        linear_predictor = X @ (self.rng.randn(self.n_features) + decision_boundary_shift)
        y = (linear_predictor + self.rng.randn(self.n_samples_per_step) > threshold_shift).astype(int)
        
        self.current_step += 1
        return X, y
    
    def compute_drift_vector(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
        shap_values: np.ndarray
    ) -> np.ndarray:
        """
        Compute multi-dimensional drift signal.
        
        Args:
            X: Feature matrix
            y: True labels
            y_pred: Predicted labels
            shap_values: SHAP attribution values (mean absolute per feature)
            
        Returns:
            Drift vector [D_KS, D_ADWIN, D_Error, D_SHAP]
        """
        # Covariate drift
        d_ks = self.ks_detector.update(X)
        
        # Concept drift
        errors = (y != y_pred).astype(float)
        error_rate = np.mean(errors)
        d_adwin, d_error = self.adwin_detector.update(error_rate)
        d_error = float(d_error)  # Already computed above
        
        # Attribution drift
        d_shap = self.shap_detector.update(shap_values)
        
        return np.array([d_ks, d_error, d_error, d_shap])
    
    def reset(self):
        """Reset drift engine state."""
        self.current_step = 0
        self.ks_detector = KSDriftDetector()
        self.adwin_detector = ADWINDriftDetector()
        self.shap_detector = SHAPDriftDetector(self.n_features)
