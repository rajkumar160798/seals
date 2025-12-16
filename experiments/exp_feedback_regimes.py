"""
Experiment 2: Feedback Regimes

Compares three feedback mechanisms:
1. Passive feedback: Error-based, no human (cheapest)
2. Human-in-the-loop: Selective labeling based on uncertainty (expensive)
3. Policy feedback: Business rule interventions (moderate cost)

Demonstrates non-linear returns to feedback investment.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.drift_engine import DriftEngine, DriftConfig, DriftType
from simulator.feedback_engine import FeedbackEngine, FeedbackConfig, FeedbackRegime
from simulator.retraining_policy import RetrainingPolicy, RetrainingConfig, RetrainingRegime
from metrics.spi import SPICalculator


class SimpleModel:
    """Minimal ML model for simulation."""
    
    def __init__(self, n_features: int = 10):
        """Initialize model with random parameters."""
        self.n_features = n_features
        self.params = np.random.randn(n_features)
        self.training_samples = []
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and compute uncertainty.
        
        Returns:
            (predictions, uncertainty)
        """
        scores = X @ self.params
        predictions = (scores > 0).astype(int)
        
        # Uncertainty: distance from decision boundary
        uncertainty = np.abs(scores) / (np.linalg.norm(self.params) + 1e-10)
        uncertainty = 1.0 / (1.0 + uncertainty)  # Sigmoid-like
        
        return predictions, uncertainty
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        y_pred, _ = self.predict(X)
        return np.mean(y_pred == y)
    
    def retrain(self, X: np.ndarray, y: np.ndarray, epochs: int = 3):
        """Simple retraining."""
        old_params = self.params.copy()
        
        for _ in range(epochs):
            scores = X @ self.params
            predictions = (scores > 0).astype(float)
            errors = predictions - y
            gradient = X.T @ errors / max(len(X), 1)
            self.params -= 0.01 * gradient
        
        param_change = np.linalg.norm(self.params - old_params)
        return param_change


class FeedbackRegimeExperiment:
    """
    Experiment comparing feedback regimes.
    """
    
    def __init__(self, n_steps: int = 200, n_features: int = 10, seed: int = 42):
        """
        Args:
            n_steps: Number of simulation steps
            n_features: Number of features
            seed: Random seed
        """
        self.n_steps = n_steps
        self.n_features = n_features
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.results = {}
    
    def run_feedback_regime(
        self,
        feedback_regime: FeedbackRegime,
        regime_name: str,
        budget: float = 500.0
    ) -> Dict:
        """
        Run experiment with specific feedback regime.
        
        Args:
            feedback_regime: Type of feedback
            regime_name: Name for logging
            budget: Total feedback budget
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"Running feedback regime: {regime_name}")
        print(f"Budget: ${budget}")
        print(f"{'='*60}")
        
        # Initialize components
        drift_engine = DriftEngine(n_features=self.n_features, seed=self.seed)
        feedback_engine = FeedbackEngine(
            FeedbackConfig(
                regime=feedback_regime,
                cost_per_label=10.0,
                human_trust=0.9,
                labeling_accuracy=0.95,
                budget=budget,
                seed=self.seed
            )
        )
        retrain_policy = RetrainingPolicy(
            RetrainingConfig(regime=RetrainingRegime.BALANCED, seed=self.seed)
        )
        model = SimpleModel(n_features=self.n_features)
        spi_calc = SPICalculator()
        
        # History
        accuracy_history = []
        feedback_cost_history = []
        total_feedback_history = []
        spi_history = []
        retrain_history = []
        
        # Main loop
        for step in range(self.n_steps):
            # Generate data with drift
            if step < 100:
                X, y = drift_engine.generate_covariate_drift(magnitude=0.2, gradual=True)
            else:
                X, y = drift_engine.generate_concept_drift(magnitude=0.4, gradual=True)
            
            # 1. Evaluate model
            y_pred, uncertainty = model.predict(X)
            accuracy = np.mean(y_pred == y)
            accuracy_history.append(accuracy)
            
            # 2. Apply feedback based on regime
            if feedback_regime == FeedbackRegime.PASSIVE:
                y_corrected, labeled_idx, cost = feedback_engine.passive_feedback(
                    y, y_pred, uncertainty
                )
            elif feedback_regime == FeedbackRegime.HUMAN_IN_THE_LOOP:
                y_corrected, labeled_idx, cost = feedback_engine.human_in_the_loop_feedback(
                    y, y_pred, uncertainty, selection_strategy="uncertainty"
                )
            else:  # POLICY
                risk_flags = accuracy < 0.5  # Mark high-error predictions
                y_corrected, labeled_idx, cost = feedback_engine.policy_feedback(
                    y_pred, risk_flags
                )
            
            feedback_cost_history.append(cost)
            total_fed = np.sum(feedback_engine.get_feedback_signal())
            total_feedback_history.append(total_fed)
            
            # 3. Decide retraining
            drift_signal = np.array([0.3, 0.3, 1 - accuracy, 0.2])
            spi = spi_calc.update(step, accuracy, 0.0)
            
            should_retrain = (
                np.mean(drift_signal) > 0.3 and
                feedback_engine.remaining_budget() > 20.0
            )
            
            # 4. Retrain if needed
            param_change = 0.0
            if should_retrain and len(labeled_idx) > 0:
                # Use corrected labels for retraining
                X_labeled = X[labeled_idx]
                y_labeled = y_corrected[labeled_idx]
                param_change = model.retrain(X_labeled, y_labeled, epochs=5)
                retrain_policy.log_retrain(10.0, "feedback-driven")
                retrain_history.append(step)
            
            spi = spi_calc.update(step, accuracy, param_change)
            spi_history.append(spi)
            
            # Progress
            if step % 40 == 0:
                remaining = feedback_engine.remaining_budget()
                print(f"Step {step:3d} | Acc: {accuracy:.3f} | "
                      f"Feedback Cost: ${cost:6.1f} | "
                      f"Budget Remaining: ${remaining:7.1f}")
        
        results = {
            "accuracy": np.array(accuracy_history),
            "feedback_cost": np.array(feedback_cost_history),
            "total_feedback": np.array(total_feedback_history),
            "spi": np.array(spi_history),
            "retrain_steps": np.array(retrain_history),
            "feedback_stats": {
                "total_cost": feedback_engine.total_cost_spent,
                "budget": budget,
                "cost_efficiency": feedback_engine.total_cost_spent / (np.mean(accuracy_history) + 1e-10)
            }
        }
        
        return results
    
    def run_all_regimes(self, budget: float = 500.0) -> Dict:
        """
        Run all feedback regimes.
        
        Args:
            budget: Total feedback budget for each regime
            
        Returns:
            Dictionary with results for each regime
        """
        regimes = [
            (FeedbackRegime.PASSIVE, "Passive (Error-Based)"),
            (FeedbackRegime.HUMAN_IN_THE_LOOP, "Human-in-the-Loop"),
            (FeedbackRegime.POLICY, "Policy (Business Rules)"),
        ]
        
        for feedback_regime, regime_name in regimes:
            self.results[regime_name] = self.run_feedback_regime(
                feedback_regime, regime_name, budget=budget
            )
        
        return self.results
    
    def plot_results(self, output_dir: Path = None):
        """
        Create visualization comparing feedback regimes.
        
        Args:
            output_dir: Output directory for plots
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "paper" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Feedback Regimes Comparison", fontsize=16, fontweight='bold')
        
        regime_names = list(self.results.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Plot 1: Accuracy comparison
        ax = axes[0, 0]
        for name, color in zip(regime_names, colors):
            acc = self.results[name]["accuracy"]
            ax.plot(acc, label=name, color=color, linewidth=2)
        ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel("Accuracy", fontweight='bold')
        ax.set_title("Model Accuracy Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative feedback cost
        ax = axes[0, 1]
        for name, color in zip(regime_names, colors):
            costs = self.results[name]["feedback_cost"]
            cumsum = np.cumsum(costs)
            ax.plot(cumsum, label=name, color=color, linewidth=2)
        ax.set_ylabel("Cumulative Cost ($)", fontweight='bold')
        ax.set_title("Feedback Cost Accumulation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: SPI over time
        ax = axes[0, 2]
        for name, color in zip(regime_names, colors):
            spi = self.results[name]["spi"]
            ax.plot(spi, label=name, color=color, linewidth=2)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
        ax.set_ylabel("SPI", fontweight='bold')
        ax.set_title("Stability-Plasticity Index")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Accuracy vs Cost (Pareto frontier)
        ax = axes[1, 0]
        for name, color in zip(regime_names, colors):
            acc = self.results[name]["accuracy"]
            costs = self.results[name]["feedback_cost"]
            total_cost = np.cumsum(costs)
            
            # Plot trajectory
            ax.plot(total_cost, acc, 'o-', label=name, color=color, linewidth=2, markersize=4)
        
        ax.set_xlabel("Total Feedback Cost ($)", fontweight='bold')
        ax.set_ylabel("Final Accuracy", fontweight='bold')
        ax.set_title("Accuracy-Cost Trade-Off (Pareto)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Cost efficiency
        ax = axes[1, 1]
        efficiencies = [
            self.results[name]["feedback_stats"]["cost_efficiency"]
            for name in regime_names
        ]
        bars = ax.bar(regime_names, efficiencies, color=colors, alpha=0.6, edgecolor='black')
        ax.set_ylabel("Cost per Accuracy Point", fontweight='bold')
        ax.set_title("Feedback Efficiency (Lower is Better)")
        ax.set_xticklabels(regime_names, rotation=15, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 6: Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = "**Feedback Summary**\n\n"
        for name, color in zip(regime_names, colors):
            stats = self.results[name]["feedback_stats"]
            final_acc = self.results[name]["accuracy"][-1]
            summary_text += f"{name}:\n"
            summary_text += f"  Final Accuracy: {final_acc:.3f}\n"
            summary_text += f"  Total Cost: ${stats['total_cost']:.1f}\n"
            summary_text += f"  Cost/Accuracy: {stats['cost_efficiency']:.2f}\n\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save
        fig_path = output_dir / "exp_feedback_regimes.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to {fig_path}")
        
        pdf_path = fig_path.with_suffix('.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"✓ PDF saved to {pdf_path}")
        
        plt.close()


def main():
    """Run full experiment."""
    print("\n" + "="*80)
    print("SEALS Experiment 2: Feedback Regimes")
    print("="*80)
    
    experiment = FeedbackRegimeExperiment(
        n_steps=200,
        n_features=10,
        seed=42
    )
    
    # Run all regimes
    results = experiment.run_all_regimes(budget=500.0)
    
    # Generate plots
    experiment.plot_results()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for regime_name, regime_results in results.items():
        acc = regime_results["accuracy"]
        stats = regime_results["feedback_stats"]
        
        print(f"\n{regime_name}:")
        print(f"  Initial Accuracy: {acc[0]:.4f}")
        print(f"  Final Accuracy: {acc[-1]:.4f}")
        print(f"  Mean Accuracy: {np.mean(acc):.4f}")
        print(f"  Total Feedback Cost: ${stats['total_cost']:.2f}")
        print(f"  Cost Efficiency: {stats['cost_efficiency']:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
