"""
Stability-Plasticity Index (SPI) Metric

Quantifies the trade-off between stability and plasticity.

TWO METRICS:

1. Raw SPI (legacy, clipped to ±100):
   SPI = ΔAccuracy / (‖ΔModel‖ + ε) * exp(-λ * Risk)

2. NORMALIZED SPI (nSPI - preferred, bounded in [-1, 1]):
   nSPI = tanh(ΔAccuracy / (‖ΔModel‖ + ε))
   
   Properties:
   - Bounded in [-1, 1], no arbitrary clipping
   - Comparable across datasets
   - Smooth saturation instead of hard bounds
   - Better interpretability for control theory
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class SPISnapshot:
    """Single SPI measurement."""
    step: int
    spi: float
    delta_accuracy: float
    parameter_change: float
    regime: str = "unknown"


class SPICalculator:
    """
    Computes Stability-Plasticity Index over time.
    
    Tracks parameter changes and accuracy improvements.
    
    IMPROVED DEFINITION:
    
    SPI_t = (ΔAcc_t / (Δθ_t + ε)) * exp(-λ * Risk_t)
    
    This bounded version accounts for risk, preventing numerical explosion.
    """
    
    def __init__(self, window_size: int = 5, epsilon: float = 1e-6, lambda_risk: float = 1.0):
        """
        Args:
            window_size: Number of steps to look back
            epsilon: Regularization to avoid division by zero
            lambda_risk: Risk penalty coefficient
        """
        self.window_size = window_size
        self.epsilon = epsilon
        self.lambda_risk = lambda_risk
        
        # History
        self.accuracy_history = []
        self.parameter_norms = []
        self.risk_history = []
        self.steps = []
        self.spi_history = []
        
    def compute_parameter_change(self, params_old: np.ndarray, params_new: np.ndarray) -> float:
        """
        Compute L2 distance between parameter sets.
        
        Args:
            params_old: Old parameters (flattened)
            params_new: New parameters (flattened)
            
        Returns:
            L2 norm of parameter change
        """
        params_old = np.asarray(params_old).flatten()
        params_new = np.asarray(params_new).flatten()
        
        return np.linalg.norm(params_new - params_old)
    
    def update(
        self,
        step: int,
        accuracy: float,
        parameter_change: float,
        risk: float = 0.0
    ) -> float:
        """
        Update SPI with new measurement.
        
        Args:
            step: Current time step
            accuracy: Current accuracy (0 to 1)
            parameter_change: L2 norm of parameter changes
            risk: Current risk level (0 to 1, default 0)
            
        Returns:
            Current SPI value
        """
        self.steps.append(step)
        self.accuracy_history.append(accuracy)
        self.parameter_norms.append(parameter_change)
        self.risk_history.append(risk)
        
        # Compute SPI with window
        if len(self.accuracy_history) < 2:
            spi = 0.0
        else:
            window_start = max(0, len(self.accuracy_history) - self.window_size)
            delta_accuracy = (
                self.accuracy_history[-1] - self.accuracy_history[window_start]
            )
            total_change = sum(self.parameter_norms[window_start:])
            mean_risk = np.mean(self.risk_history[window_start:])
            
            # Normalized SPI with risk penalty
            raw_spi = delta_accuracy / (total_change + self.epsilon)
            
            # Apply risk penalty: high risk lowers SPI
            spi = raw_spi * np.exp(-self.lambda_risk * mean_risk)
            
            # Clip to reasonable range for stability
            spi = np.clip(spi, -100, 100)
        
        self.spi_history.append(spi)
        return spi
    
    def compute_normalized_spi(self) -> float:
        """
        Compute Normalized SPI (nSPI) - bounded version using tanh.
        
        nSPI_t = tanh(ΔAccuracy_t / (‖ΔModel‖_t + ε))
        
        Properties:
        - Bounded in [-1, 1]
        - No arbitrary clipping, smooth saturation
        - Comparable across datasets
        - Better for control theory interpretation
        
        Returns:
            nSPI value in [-1, 1]
        """
        if len(self.accuracy_history) < 2:
            nspi = 0.0
        else:
            window_start = max(0, len(self.accuracy_history) - self.window_size)
            delta_accuracy = (
                self.accuracy_history[-1] - self.accuracy_history[window_start]
            )
            total_change = sum(self.parameter_norms[window_start:])
            
            # Normalized SPI using tanh (bounded, smooth)
            raw_ratio = delta_accuracy / (total_change + self.epsilon)
            nspi = np.tanh(raw_ratio)
        
        return nspi
    
    def get_regime(self, spi: float) -> str:
        """
        Classify SPI into regime.
        
        Args:
            spi: SPI value
            
        Returns:
            Regime name
        """
        if spi > 0.5:
            return "balanced"
        elif spi > 0.0:
            return "slightly_plastic"
        elif spi > -0.5:
            return "over_plastic"
        else:
            return "stagnant"
    
    def get_spi_in_optimal_band(self, min_spi: float = 0.3, max_spi: float = 1.2) -> float:
        """
        Fraction of time SPI is in optimal band.
        
        Args:
            min_spi: Minimum optimal SPI
            max_spi: Maximum optimal SPI
            
        Returns:
            Fraction in optimal band (0 to 1)
        """
        if not self.spi_history:
            return 0.0
        
        in_band = sum(1 for spi in self.spi_history if min_spi <= spi <= max_spi)
        return in_band / len(self.spi_history)
    
    def get_current_spi(self) -> float:
        """Get most recent SPI value."""
        return self.spi_history[-1] if self.spi_history else 0.0
    
    def get_nspi_statistics(self) -> Dict[str, float]:
        """
        Compute comprehensive nSPI statistics.
        
        Returns:
            Dict with:
            - mean_nspi: Mean nSPI
            - std_nspi: Standard deviation
            - max_nspi: Maximum value
            - min_nspi: Minimum value
            - fraction_optimal: Fraction in [-0.7, 1.0] band
        """
        if len(self.accuracy_history) < 2:
            return {
                'mean_nspi': 0.0,
                'std_nspi': 0.0,
                'max_nspi': 0.0,
                'min_nspi': 0.0,
                'fraction_optimal': 0.0,
            }
        
        # Compute nSPI history
        nspi_values = []
        for t in range(1, len(self.accuracy_history)):
            # Recompute for each time point
            delta_accuracy = self.accuracy_history[t] - self.accuracy_history[0]
            total_change = sum(self.parameter_norms[:t])
            raw_ratio = delta_accuracy / (total_change + self.epsilon)
            nspi_values.append(np.tanh(raw_ratio))
        
        nspi_array = np.array(nspi_values)
        
        # Count optimal band [-0.7, 1.0]
        in_optimal = np.sum((nspi_array >= -0.7) & (nspi_array <= 1.0))
        
        return {
            'mean_nspi': float(np.mean(nspi_array)),
            'std_nspi': float(np.std(nspi_array)),
            'max_nspi': float(np.max(nspi_array)),
            'min_nspi': float(np.min(nspi_array)),
            'fraction_optimal': float(in_optimal / len(nspi_array)),
        }
    
    def get_spi_trend(self, window: int = 10) -> float:
        """
        Compute trend in SPI (increasing or decreasing).
        
        Args:
            window: Number of recent steps to analyze
            
        Returns:
            Slope of SPI trend
        """
        if len(self.spi_history) < window:
            return 0.0
        
        recent_spi = np.array(self.spi_history[-window:])
        x = np.arange(len(recent_spi))
        
        # Linear regression slope
        slope = np.polyfit(x, recent_spi, 1)[0]
        return slope
    
    def get_spi_statistics(self) -> Dict[str, float]:
        """Get summary statistics of SPI."""
        if not self.spi_history:
            return {}
        
        spi_array = np.array(self.spi_history)
        return {
            "mean": float(np.mean(spi_array)),
            "std": float(np.std(spi_array)),
            "min": float(np.min(spi_array)),
            "max": float(np.max(spi_array)),
            "median": float(np.median(spi_array)),
            "current": float(spi_array[-1]),
            "fraction_in_optimal_band": self.get_spi_in_optimal_band()
        }
    
    def reset(self):
        """Reset calculator state."""
        self.accuracy_history = []
        self.parameter_norms = []
        self.risk_history = []
        self.steps = []
        self.spi_history = []


class RegimeTracker:
    """
    Tracks system behavior across different regimes.
    
    Identifies transitions and characteristics of each regime.
    """
    
    def __init__(self):
        """Initialize regime tracker."""
        self.regime_boundaries = []
        self.current_regime = "unknown"
        self.regime_durations = {}
        self.regime_performance = {}
        
    def update_regime(self, spi: float, accuracy: float):
        """
        Update current regime based on SPI.
        
        Args:
            spi: Stability-plasticity index
            accuracy: Current accuracy
        """
        # Determine regime
        if spi > 0.5:
            new_regime = "balanced"
        elif spi > 0.0:
            new_regime = "slightly_plastic"
        elif spi > -0.5:
            new_regime = "over_plastic"
        else:
            new_regime = "stagnant"
        
        # Track transitions
        if new_regime != self.current_regime:
            self.regime_boundaries.append({
                "regime": new_regime,
                "from_regime": self.current_regime
            })
            self.current_regime = new_regime
        
        # Track performance per regime
        if new_regime not in self.regime_performance:
            self.regime_performance[new_regime] = []
        self.regime_performance[new_regime].append(accuracy)
    
    def get_regime_analysis(self) -> Dict:
        """
        Analyze regime behavior.
        
        Returns:
            Dictionary with regime statistics
        """
        analysis = {}
        for regime, accuracies in self.regime_performance.items():
            if accuracies:
                analysis[regime] = {
                    "mean_accuracy": float(np.mean(accuracies)),
                    "std_accuracy": float(np.std(accuracies)),
                    "duration": len(accuracies),
                    "max_accuracy": float(np.max(accuracies)),
                    "min_accuracy": float(np.min(accuracies))
                }
        return analysis
    
    def reset(self):
        """Reset tracker."""
        self.regime_boundaries = []
        self.current_regime = "unknown"
        self.regime_durations = {}
        self.regime_performance = {}


class RegretCalculator:
    """
    Computes cumulative regret for retraining policies.
    
    Two-stage regret formula:
    - Early stage (t ≤ T): Regret_t = α·(Acc_max - Acc_t) + β·Cost_t + γ·Risk_t
    - Late stage (t > T): Add late-error amplification: λ·(Acc_max - Acc_t)
    
    Late-stage amplification makes late errors expensive, favoring:
    - Balanced policies that adapt smoothly
    - Against over-plastic (oscillation late) and over-stable (stagnation late)
    
    Balanced policies minimize regret, not just maximize accuracy.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.1,
        late_stage_threshold: int = 100,
        late_stage_penalty: float = 0.5
    ):
        """
        Args:
            alpha: Weight for accuracy gap (early stage)
            beta: Weight for cost
            gamma: Weight for risk
            late_stage_threshold: Time T after which to apply error amplification
            late_stage_penalty: λ coefficient for late-stage error (multiplies accuracy gap)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.late_stage_threshold = late_stage_threshold
        self.late_stage_penalty = late_stage_penalty
        
        self.accuracy_history = []
        self.cost_history = []
        self.risk_history = []
        self.regret_history = []
        self.cumulative_regret = 0.0
        
    def update(
        self,
        accuracy: float,
        cost: float,
        risk: float,
        max_accuracy: float = 1.0,
        step: int = None
    ) -> float:
        """
        Update regret calculation with late-stage error amplification.
        
        Formula:
            Early (t ≤ T): Regret_t = α·(Acc* - Acc_t) + β·Cost_t + γ·Risk_t
            Late (t > T):  Regret_t += λ·(Acc* - Acc_t)  [late errors are expensive]
        
        Args:
            accuracy: Current accuracy (0 to 1)
            cost: Current cost (normalized 0 to 1)
            risk: Current risk (0 to 1)
            max_accuracy: Best possible accuracy (default 1.0)
            step: Current time step (for late-stage detection)
            
        Returns:
            Cumulative regret
        """
        self.accuracy_history.append(accuracy)
        self.cost_history.append(cost)
        self.risk_history.append(risk)
        
        # Compute current step
        if step is None:
            step = len(self.accuracy_history)
        
        # Base regret (early stage)
        regret_t = (
            self.alpha * (max_accuracy - accuracy) +
            self.beta * cost +
            self.gamma * risk
        )
        
        # Late-stage error amplification: after threshold T, late errors are expensive
        # This makes balanced policies shine (they adapt; extreme policies collapse)
        if step > self.late_stage_threshold:
            late_stage_error = self.late_stage_penalty * (max_accuracy - accuracy)
            regret_t += late_stage_error
        
        self.regret_history.append(regret_t)
        self.cumulative_regret += regret_t
        
        return self.cumulative_regret
    
    def get_regret_statistics(self) -> Dict[str, float]:
        """Get regret statistics."""
        if not self.regret_history:
            return {}
        
        regret_array = np.array(self.regret_history)
        return {
            "cumulative_regret": float(self.cumulative_regret),
            "mean_regret": float(np.mean(regret_array)),
            "std_regret": float(np.std(regret_array)),
            "min_regret": float(np.min(regret_array)),
            "max_regret": float(np.max(regret_array)),
        }
    
    def reset(self):
        """Reset calculator."""
        self.accuracy_history = []
        self.cost_history = []
        self.risk_history = []
        self.regret_history = []
        self.cumulative_regret = 0.0


class AdaptivityMetric:
    """
    Measures system adaptivity: ability to respond to drift.
    
    Adaptivity = (ΔAccuracy after drift) / (Severity of drift)
    """
    
    def __init__(self):
        """Initialize adaptivity metric."""
        self.drift_events = []
        self.post_drift_improvements = []
        
    def record_drift_event(self, drift_magnitude: float, accuracy_before: float):
        """
        Record drift event.
        
        Args:
            drift_magnitude: Magnitude of drift signal
            accuracy_before: Accuracy before drift response
        """
        self.drift_events.append({
            "magnitude": drift_magnitude,
            "accuracy_before": accuracy_before,
            "accuracy_after": None,
            "improvement": None
        })
    
    def record_post_drift_accuracy(self, accuracy_after: float):
        """
        Record accuracy after drift response (e.g., after retraining).
        
        Args:
            accuracy_after: Accuracy after response
        """
        if self.drift_events:
            event = self.drift_events[-1]
            event["accuracy_after"] = accuracy_after
            event["improvement"] = accuracy_after - event["accuracy_before"]
            
            if event["magnitude"] > 0:
                adaptivity = event["improvement"] / event["magnitude"]
                self.post_drift_improvements.append(adaptivity)
    
    def get_adaptivity(self) -> float:
        """
        Get average adaptivity metric.
        
        Returns:
            Mean adaptivity (higher = better)
        """
        if not self.post_drift_improvements:
            return 0.0
        return float(np.mean(self.post_drift_improvements))
    
    def get_adaptivity_per_event(self) -> List[float]:
        """Get adaptivity for each drift event."""
        adaptivities = []
        for event in self.drift_events:
            if event["magnitude"] > 0 and event["improvement"] is not None:
                adaptivity = event["improvement"] / event["magnitude"]
                adaptivities.append(adaptivity)
        return adaptivities
