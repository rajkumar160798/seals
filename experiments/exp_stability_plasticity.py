"""
Experiment 1: Stability vs Plasticity

This experiment demonstrates the fundamental trade-off:
- Over-plastic: Retrains too frequently → oscillation, high variance
- Over-stable: Retrains too rarely → collapse, stale decisions
- Balanced: Optimal retraining → smooth evolution

Results:
- Accuracy over time for each regime
- Stability-Plasticity Index (SPI) over time
- Parameter change magnitude
- System regime transitions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Import simulator modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.drift_engine import DriftEngine, DriftConfig, DriftType
from simulator.retraining_policy import (
    RetrainingPolicy,
    RetrainingConfig,
    RetrainingRegime
)
from metrics.spi import SPICalculator, RegimeTracker, RegretCalculator
from metrics.attribution_drift import DiscrepancyAnalyzer


class SimpleModel:
    """Minimal ML model for simulation."""
    
    def __init__(self, n_features: int = 10):
        """Initialize model with random parameters."""
        self.n_features = n_features
        self.params = np.random.randn(n_features)
        self.training_loss = None
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        scores = X @ self.params
        return (scores > 0).astype(int)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_shap_approximation(self, X: np.ndarray) -> np.ndarray:
        """Approximate SHAP values as absolute weights."""
        return np.abs(self.params)
    
    def retrain(self, X: np.ndarray, y: np.ndarray, epochs: int = 5):
        """Simple gradient-based retraining."""
        learning_rate = 0.01
        old_params = self.params.copy()
        
        for _ in range(epochs):
            # Compute loss and gradient
            scores = X @ self.params
            predictions = (scores > 0).astype(float)
            errors = predictions - y
            
            # Gradient step
            gradient = X.T @ errors / len(X)
            self.params -= learning_rate * gradient
        
        # Record change
        param_change = np.linalg.norm(self.params - old_params)
        return param_change


class StabilityPlasticityExperiment:
    """
    Main experiment class.
    
    Runs three retraining regimes and compares performance.
    """
    
    def __init__(self, n_steps: int = 200, n_features: int = 10, seed: int = 42):
        """
        Args:
            n_steps: Number of time steps
            n_features: Number of features
            seed: Random seed
        """
        self.n_steps = n_steps
        self.n_features = n_features
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Results storage
        self.results = {}
        
    def run_single_regime(
        self,
        regime: RetrainingRegime,
        regime_name: str
    ) -> Dict:
        """
        Run experiment for single retraining regime.
        
        Args:
            regime: Retraining regime to test
            regime_name: Name for logging
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"Running regime: {regime_name}")
        print(f"{'='*60}")
        
        # Initialize components
        drift_engine = DriftEngine(n_features=self.n_features, seed=self.seed)
        policy = RetrainingPolicy(RetrainingConfig(regime=regime, seed=self.seed))
        model = SimpleModel(n_features=self.n_features)
        
        spi_calc = SPICalculator()
        regime_tracker = RegimeTracker()
        regret_calc = RegretCalculator(alpha=1.0, beta=0.1, gamma=0.1)
        discrepancy = DiscrepancyAnalyzer()
        
        # History
        accuracy_history = []
        spi_history = []
        param_change_history = []
        retrain_steps = []
        drift_signals = []
        risk_history = []
        regret_history = []
        
        # Main loop
        for step in range(self.n_steps):
            # 1. Generate data with time-varying drift intensity
            # Drift_t = Drift_0 * (1 + λ*t)  where λ increases over time
            if step < 50:
                # No drift (warm-up phase)
                X, y = drift_engine.generate_covariate_drift(magnitude=0.0)
                current_drift = 0.0
            elif step < 100:
                # Gradual covariate drift with TIME-VARYING intensity
                # Start at 0.3, increase by 0.5% per step: λ = 0.005
                time_varying_intensity = 0.005  # Strong time scaling
                adjusted_mag = 0.3 * (1.0 + time_varying_intensity * (step - 50))
                X, y = drift_engine.generate_covariate_drift(magnitude=adjusted_mag, gradual=True)
                current_drift = adjusted_mag
            elif step < 150:
                # Concept drift with TIME-VARYING intensity
                # Start at 0.4, increase by 0.5% per step
                time_varying_intensity = 0.005
                adjusted_mag = 0.4 * (1.0 + time_varying_intensity * (step - 100))
                X, y = drift_engine.generate_concept_drift(magnitude=adjusted_mag, gradual=True)
                current_drift = adjusted_mag
            else:
                # Label drift with TIME-VARYING intensity (MOST SEVERE)
                # Start at 0.5, increase by 1.0% per step (more aggressive)
                time_varying_intensity = 0.010  # Even stronger
                adjusted_mag = 0.5 * (1.0 + time_varying_intensity * (step - 150))
                X, y = drift_engine.generate_label_drift(magnitude=adjusted_mag, gradual=True)
                current_drift = adjusted_mag
            
            # 2. Evaluate current model
            accuracy = model.accuracy(X, y)
            y_pred = model.predict(X)
            shap_approx = model.get_shap_approximation(X)
            
            accuracy_history.append(accuracy)
            
            # 3. Compute drift signal
            drift_vector = drift_engine.compute_drift_vector(
                X, y, y_pred, shap_approx
            )
            drift_signals.append(np.mean(drift_vector))
            
            # 4. Compute SPI
            old_params = model.params.copy()
            spi = spi_calc.update(step, accuracy, 0.0)  # Update with zero change first
            spi_history.append(spi)
            regime_tracker.update_regime(spi, accuracy)
            
            # 5. Compute risk (simplified: inverse accuracy + drift)
            risk = (1 - accuracy) * (1 + np.mean(drift_vector))
            risk_history.append(risk)
            
            # 6. Decide retraining
            should_retrain = policy.decide_retrain(
                drift_vector,
                spi,
                risk,
                cost=10.0
            )
            
            # 7. Retrain if needed
            param_change = 0.0
            if should_retrain:
                param_change = model.retrain(X, y, epochs=5)
                policy.log_retrain(10.0, f"drift={np.mean(drift_vector):.3f}")
                retrain_steps.append(step)
                
                # Update SPI with actual parameter change
                spi = spi_calc.update(step, accuracy, param_change)
                spi_history[-1] = spi
            
            param_change_history.append(param_change)
            
            # 8. Record discrepancy (new metric)
            shap_drift = drift_vector[3]  # Attribution drift
            discrepancy.record(accuracy, shap_drift)
            
            # 9. Compute regret and record
            cost = 10.0 if should_retrain else 0.0
            max_accuracy = 0.95  # Reference optimal accuracy
            regret_calc.update(accuracy, cost, risk, max_accuracy)
            regret_history.append(regret_calc.cumulative_regret)
            
            # Progress
            if step % 50 == 0:
                print(f"Step {step:3d} | Acc: {accuracy:.3f} | "
                      f"SPI: {spi:5.2f} | Drift: {np.mean(drift_vector):.3f} | "
                      f"Retrain: {'YES' if should_retrain else 'NO':3s}")
        
        # Compile results
        results = {
            "accuracy": np.array(accuracy_history),
            "spi": np.array(spi_history),
            "param_change": np.array(param_change_history),
            "retrain_steps": np.array(retrain_steps),
            "drift_signals": np.array(drift_signals),
            "risk": np.array(risk_history),
            "cumulative_regret": np.array(regret_history),
            "regime_stats": regime_tracker.get_regime_analysis(),
            "spi_stats": spi_calc.get_spi_statistics(),
            "regret_stats": regret_calc.get_regret_statistics(),
            "discrepancy": discrepancy.get_analysis(),
            "policy_stats": policy.get_regime_stats(),
        }
        
        return results
    
    def run_all_regimes(self) -> Dict:
        """
        Run experiment for all three regimes.
        
        Returns:
            Dictionary with results for each regime
        """
        regimes = [
            (RetrainingRegime.OVER_PLASTIC, "Over-Plastic (Oscillating)"),
            (RetrainingRegime.OVER_STABLE, "Over-Stable (Stagnant)"),
            (RetrainingRegime.BALANCED, "Balanced (Optimal)"),
        ]
        
        for regime, name in regimes:
            self.results[name] = self.run_single_regime(regime, name)
        
        return self.results
    
    def plot_results(self, output_dir: Path = None):
        """
        Create comprehensive visualization with main figure + supporting plots.
        
        Main figure: Accuracy, SPI, Cumulative Regret
        Supporting: Retraining frequency, drift, risk details
        
        Args:
            output_dir: Directory to save plots
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "paper" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        regime_names = list(self.results.keys())
        colors = ['red', 'orange', 'green']
        
        # ============================================================================
        # MAIN FIGURE: Three key findings (Accuracy, SPI, Cumulative Regret)
        # ============================================================================
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("SEALS: Stability-Plasticity Trade-Off Analysis", fontsize=14, fontweight='bold')
        
        # Main Plot 1: Accuracy over time
        ax = axes[0]
        for (name, color) in zip(regime_names, colors):
            ax.plot(self.results[name]["accuracy"], label=name, color=color, linewidth=2.5)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.axvline(x=100, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        ax.set_xlabel("Time Step", fontweight='bold')
        ax.set_ylabel("Accuracy", fontweight='bold')
        ax.set_title("(A) Accuracy Under Escalating Drift")
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.3, 1.0])
        
        # Main Plot 2: SPI over time (NEW - IMPROVED BOUNDED DEFINITION)
        ax = axes[1]
        for (name, color) in zip(regime_names, colors):
            spi_vals = self.results[name]["spi"]
            ax.plot(spi_vals, label=name, color=color, linewidth=2.5)
        ax.axhspan(0.3, 1.2, alpha=0.1, color='green', label='Optimal band')
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_xlabel("Time Step", fontweight='bold')
        ax.set_ylabel("SPI (Bounded: [-100, 100])", fontweight='bold')
        ax.set_title("(B) Stability-Plasticity Index")
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 3])
        
        # Main Plot 3: Cumulative Regret (NEW - CORE CONTRIBUTION)
        ax = axes[2]
        for (name, color) in zip(regime_names, colors):
            regret = self.results[name]["cumulative_regret"]
            ax.plot(regret, label=name, color=color, linewidth=2.5)
        ax.set_xlabel("Time Step", fontweight='bold')
        ax.set_ylabel("Cumulative Regret", fontweight='bold')
        ax.set_title("(C) Regret Minimization")
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = output_dir / "fig_main_stability_plasticity.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Main figure saved to {fig_path}")
        plt.close()
        
        # ============================================================================
        # SUPPLEMENTARY FIGURE 1: Detailed dynamics (3x3 grid)
        # ============================================================================
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle("Supplementary: Detailed System Dynamics", fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy over time
        ax = axes[0, 0]
        for (name, color) in zip(regime_names, colors):
            ax.plot(self.results[name]["accuracy"], label=name, color=color, linewidth=2)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel("Accuracy", fontweight='bold')
        ax.set_title("Accuracy Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: SPI over time
        ax = axes[0, 1]
        for (name, color) in zip(regime_names, colors):
            ax.plot(self.results[name]["spi"], label=name, color=color, linewidth=2)
        ax.axhspan(0.3, 1.2, alpha=0.1, color='green')
        ax.set_ylabel("SPI", fontweight='bold')
        ax.set_title("Stability-Plasticity Index")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Parameter change magnitude
        ax = axes[0, 2]
        for (name, color) in zip(regime_names, colors):
            ax.plot(self.results[name]["param_change"], label=name, color=color, linewidth=1.5, alpha=0.8)
        ax.set_ylabel("Parameter Change", fontweight='bold')
        ax.set_title("Model Plasticity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Retraining frequency
        ax = axes[1, 0]
        for (name, color) in zip(regime_names, colors):
            retrain_steps = self.results[name]["retrain_steps"]
            ax.scatter(retrain_steps, np.ones_like(retrain_steps) * regime_names.index(name),
                      color=color, s=50, alpha=0.6, label=name)
        ax.set_yticks(range(len(regime_names)))
        ax.set_yticklabels(regime_names)
        ax.set_ylabel("Regime")
        ax.set_title("Retraining Events")
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 5: Drift signals
        ax = axes[1, 1]
        for (name, color) in zip(regime_names, colors):
            ax.plot(self.results[name]["drift_signals"], label=name, color=color, linewidth=2)
        ax.set_ylabel("Drift Magnitude", fontweight='bold')
        ax.set_title("Drift Signal")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Risk over time
        ax = axes[1, 2]
        for (name, color) in zip(regime_names, colors):
            ax.plot(self.results[name]["risk"], label=name, color=color, linewidth=2)
        ax.set_ylabel("Risk", fontweight='bold')
        ax.set_title("System Risk")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 7: Summary statistics
        ax = axes[2, 0]
        ax.axis('off')
        summary_text = "Performance Summary\n\n"
        for name, color in zip(regime_names, colors):
            stats = self.results[name]["spi_stats"]
            regret_stats = self.results[name]["regret_stats"]
            summary_text += f"{name}:\n"
            summary_text += f"  Acc: {self.results[name]['accuracy'].mean():.3f}\n"
            summary_text += f"  SPI: {stats.get('mean', 0):.3f}\n"
            summary_text += f"  Regret: {regret_stats['cumulative_regret']:.1f}\n"
            summary_text += f"  Retrains: {self.results[name]['policy_stats']['total_retrains']}\n\n"
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 8: SPI distribution
        ax = axes[2, 1]
        spi_data = [self.results[name]["spi"] for name in regime_names]
        bp = ax.boxplot(spi_data, labels=regime_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("SPI Distribution", fontweight='bold')
        ax.set_title("SPI Variability")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 9: Cumulative Regret (expanded for detail)
        ax = axes[2, 2]
        final_regrets = []
        for name in regime_names:
            regret = self.results[name]["cumulative_regret"][-1]
            final_regrets.append(regret)
        bars = ax.bar(regime_names, final_regrets, color=colors, alpha=0.6, edgecolor='black', linewidth=2)
        ax.set_ylabel("Final Cumulative Regret", fontweight='bold')
        ax.set_title("Regret Comparison (Lower is Better)")
        ax.set_xticklabels(regime_names, rotation=15, ha='right')
        
        # Add value labels and highlight winner
        for i, (bar, regret) in enumerate(zip(bars, final_regrets)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            if i == 2:  # Balanced (should be best)
                bar.set_edgecolor('gold')
                bar.set_linewidth(3)
        
        plt.tight_layout()
        fig_path = output_dir / "exp_stability_plasticity_supplementary.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Supplementary figure saved to {fig_path}")
        plt.close()
        
        # ============================================================================
        # PRINT SUMMARY TABLE
        # ============================================================================
        print("\n" + "="*80)
        print("RESULTS SUMMARY TABLE")
        print("="*80)
        print(f"{'Regime':<20} {'Mean Acc':<12} {'Mean SPI':<12} {'Final Regret':<15} {'Retrains':<10}")
        print("-"*80)
        for name in regime_names:
            mean_acc = self.results[name]["accuracy"].mean()
            mean_spi = self.results[name]["spi_stats"].get('mean', 0)
            final_regret = self.results[name]["cumulative_regret"][-1]
            n_retrains = self.results[name]["policy_stats"]["total_retrains"]
            print(f"{name:<20} {mean_acc:<12.4f} {mean_spi:<12.4f} {final_regret:<15.1f} {n_retrains:<10}")
        print("="*80)



def main():
    """Run full experiment."""
    print("\n" + "="*80)
    print("SEALS Experiment 1: Stability vs Plasticity Trade-Off")
    print("="*80)
    
    experiment = StabilityPlasticityExperiment(
        n_steps=200,
        n_features=10,
        seed=42
    )
    
    # Run all regimes
    results = experiment.run_all_regimes()
    
    # Generate plots
    experiment.plot_results()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for regime_name, regime_results in results.items():
        print(f"\n{regime_name}:")
        print(f"  Mean Accuracy: {regime_results['accuracy'].mean():.4f}")
        print(f"  Std Accuracy: {regime_results['accuracy'].std():.4f}")
        print(f"  Mean SPI: {regime_results['spi_stats']['mean']:.4f}")
        print(f"  Total Retrains: {regime_results['policy_stats']['total_retrains']}")
        print(f"  Avg Retrain Interval: {regime_results['policy_stats']['avg_interval']:.1f}")
        print(f"  Discrepancy Score: {regime_results['discrepancy']['mean_discrepancy']:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
