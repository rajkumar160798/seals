"""
Experiment 3: Real Dataset Evaluation

Tests the SEALS framework on real industrial datasets:
- CMAPSS: Engine predictive maintenance
- AI4I: Industrial predictive maintenance

Demonstrates how drift detection and adaptive retraining perform on real data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.data_loader import get_cmapss_data, get_ai4i_data
from simulator.drift_engine import DriftEngine
from simulator.retraining_policy import RetrainingPolicy, RetrainingConfig, RetrainingRegime
from metrics.spi import SPICalculator
from metrics.attribution_drift import DiscrepancyAnalyzer


class RealDataExperiment:
    """
    Run SEALS on real datasets with time-series splits.
    """
    
    def __init__(self, test_size: float = 0.2, seed: int = 42):
        """
        Args:
            test_size: Proportion for test/holdout
            seed: Random seed
        """
        self.test_size = test_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.results = {}
    
    def run_cmapss_experiment(self, dataset: str = "FD001", retrain_interval: int = 10) -> Dict:
        """
        Run experiment on CMAPSS dataset.
        
        Args:
            dataset: CMAPSS variant (FD001-FD004)
            retrain_interval: Steps between retraining
            
        Returns:
            Results dictionary
        """
        print(f"\n{'='*60}")
        print(f"CMAPSS {dataset} Experiment")
        print(f"{'='*60}")
        
        try:
            # Load data
            data = get_cmapss_data(dataset=dataset, normalize=True)
            X_train = data["X_train"]
            X_test = data["X_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]
            
            print(f"✓ Loaded {dataset}")
            print(f"  Train shape: {X_train.shape}")
            print(f"  Test shape: {X_test.shape}")
            print(f"  Classes: {len(np.unique(y_train))}")
            
            # Convert regression (RUL) to classification (will fail soon)
            # Use median as threshold for better balance
            median_rul = np.median(y_train)
            y_train_binary = (y_train <= median_rul).astype(int)  # 1 if RUL <= median
            y_test_binary = (y_test <= median_rul).astype(int)
            
            # Time-based split for temporal validation
            n_train = len(X_train)
            n_test = len(X_test)
            window_size = max(50, n_train // 20)  # ~5% windows
            
            # Initialize components
            model = LogisticRegression(max_iter=1000, random_state=self.seed)
            spi_calc = SPICalculator()
            discrepancy = DiscrepancyAnalyzer()
            
            # History
            accuracy_history = []
            auc_history = []
            spi_history = []
            retrain_steps = []
            
            # Train on initial window
            X_init = X_train[:window_size]
            y_init = y_train_binary[:window_size]
            model.fit(X_init, y_init)
            
            # Evaluate on test set in windows
            n_windows = max(1, n_test // window_size)
            
            for window_idx in range(n_windows):
                start_idx = window_idx * window_size
                end_idx = min((window_idx + 1) * window_size, n_test)
                
                X_window = X_test[start_idx:end_idx]
                y_window = y_test_binary[start_idx:end_idx]
                
                # Evaluate
                y_pred = model.predict(X_window)
                y_proba = model.predict_proba(X_window)[:, 1]
                
                acc = accuracy_score(y_window, y_pred)
                
                # Only compute AUC if we have both classes
                try:
                    if len(np.unique(y_window)) > 1:
                        auc = roc_auc_score(y_window, y_proba)
                    else:
                        auc = 0.5
                except:
                    auc = 0.5
                
                accuracy_history.append(acc)
                auc_history.append(auc)
                
                # Compute SPI (approximate parameter change)
                old_coef = model.coef_.copy()
                
                # Decide to retrain
                if window_idx % retrain_interval == 0 and window_idx > 0:
                    # Retrain on window
                    model.fit(X_window, y_window)
                    retrain_steps.append(window_idx)
                    param_change = np.linalg.norm(model.coef_ - old_coef)
                else:
                    param_change = 0.0
                
                # Update SPI
                spi = spi_calc.update(window_idx, acc, param_change)
                spi_history.append(spi)
                
                # Record discrepancy (approximate)
                drift_signal = np.std(X_window) if len(X_window) > 1 else 0.0
                discrepancy.record(acc, drift_signal)
                
                print(f"Window {window_idx:2d} | Acc: {acc:.3f} | AUC: {auc:.3f} | SPI: {spi:.2f}")
            
            return {
                "dataset": f"CMAPSS-{dataset}",
                "accuracy": np.array(accuracy_history),
                "auc": np.array(auc_history),
                "spi": np.array(spi_history),
                "retrain_steps": np.array(retrain_steps),
                "spi_stats": spi_calc.get_spi_statistics(),
                "discrepancy": discrepancy.get_analysis(),
                "mean_accuracy": float(np.mean(accuracy_history)),
                "mean_auc": float(np.mean(auc_history)),
            }
        
        except Exception as e:
            print(f"✗ Error loading CMAPSS: {e}")
            return None
    
    def run_ai4i_experiment(self, retrain_interval: int = 10) -> Dict:
        """
        Run experiment on AI4I dataset.
        
        Args:
            retrain_interval: Steps between retraining
            
        Returns:
            Results dictionary
        """
        print(f"\n{'='*60}")
        print(f"AI4I 2020 Experiment")
        print(f"{'='*60}")
        
        try:
            # Load data
            data = get_ai4i_data(normalize=True, test_size=self.test_size)
            X_train = data["X_train"]
            X_test = data["X_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]
            feature_names = data["feature_names"]
            
            print(f"✓ Loaded AI4I 2020")
            print(f"  Train shape: {X_train.shape}")
            print(f"  Test shape: {X_test.shape}")
            print(f"  Features: {len(feature_names)}")
            print(f"  Classes: {len(np.unique(y_train))}")
            print(f"  Class balance: {np.sum(y_train)}/{len(y_train)}")
            
            # Time-based evaluation (sequential chunks)
            n_test = len(X_test)
            window_size = max(50, n_test // 20)
            
            # Initialize components
            model = RandomForestClassifier(n_estimators=100, random_state=self.seed)
            spi_calc = SPICalculator()
            discrepancy = DiscrepancyAnalyzer()
            
            # Train on training set
            model.fit(X_train, y_train)
            
            # History
            accuracy_history = []
            precision_history = []
            recall_history = []
            auc_history = []
            spi_history = []
            retrain_steps = []
            
            # Evaluate on test set in windows
            n_windows = max(1, n_test // window_size)
            
            for window_idx in range(n_windows):
                start_idx = window_idx * window_size
                end_idx = min((window_idx + 1) * window_size, n_test)
                
                X_window = X_test[start_idx:end_idx]
                y_window = y_test[start_idx:end_idx]
                
                # Evaluate
                y_pred = model.predict(X_window)
                y_proba = model.predict_proba(X_window)[:, 1] if len(np.unique(y_pred)) > 1 else y_pred
                
                acc = accuracy_score(y_window, y_pred)
                precision = precision_score(y_window, y_pred, zero_division=0)
                recall = recall_score(y_window, y_pred, zero_division=0)
                
                try:
                    auc = roc_auc_score(y_window, y_proba)
                except:
                    auc = 0.5
                
                accuracy_history.append(acc)
                precision_history.append(precision)
                recall_history.append(recall)
                auc_history.append(auc)
                
                # Approximate parameter change
                param_change = 0.0
                
                # Decide to retrain
                if window_idx % retrain_interval == 0 and window_idx > 0:
                    model.fit(X_window, y_window)
                    retrain_steps.append(window_idx)
                    param_change = 0.01 * (1 - acc)  # Approximate
                
                # Update SPI
                spi = spi_calc.update(window_idx, acc, param_change)
                spi_history.append(spi)
                
                # Record discrepancy
                drift_signal = np.std(X_window) if len(X_window) > 1 else 0.0
                discrepancy.record(acc, drift_signal)
                
                print(f"Window {window_idx:2d} | Acc: {acc:.3f} | Prec: {precision:.3f} | "
                      f"Rec: {recall:.3f} | AUC: {auc:.3f}")
            
            return {
                "dataset": "AI4I-2020",
                "accuracy": np.array(accuracy_history),
                "precision": np.array(precision_history),
                "recall": np.array(recall_history),
                "auc": np.array(auc_history),
                "spi": np.array(spi_history),
                "retrain_steps": np.array(retrain_steps),
                "spi_stats": spi_calc.get_spi_statistics(),
                "discrepancy": discrepancy.get_analysis(),
                "mean_accuracy": float(np.mean(accuracy_history)),
                "mean_precision": float(np.mean(precision_history)),
                "mean_recall": float(np.mean(recall_history)),
                "mean_auc": float(np.mean(auc_history)),
            }
        
        except Exception as e:
            print(f"✗ Error loading AI4I: {e}")
            return None
    
    def plot_results(self, output_dir: Path = None):
        """Create visualization of results."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "paper" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            print("No results to plot")
            return
        
        n_datasets = len(self.results)
        fig, axes = plt.subplots(n_datasets, 3, figsize=(15, 5 * n_datasets))
        
        if n_datasets == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, (dataset_name, result) in enumerate(self.results.items()):
            if result is None:
                continue
            
            # Accuracy
            ax = axes[row_idx, 0]
            ax.plot(result["accuracy"], 'o-', linewidth=2, markersize=4, color='blue')
            for retrain_step in result["retrain_steps"]:
                ax.axvline(x=retrain_step, color='red', linestyle='--', alpha=0.5)
            ax.set_ylabel("Accuracy", fontweight='bold')
            ax.set_title(f"{dataset_name} - Accuracy")
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            # SPI
            ax = axes[row_idx, 1]
            ax.plot(result["spi"], 'o-', linewidth=2, markersize=4, color='green')
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
            ax.set_ylabel("SPI", fontweight='bold')
            ax.set_title(f"{dataset_name} - Stability-Plasticity")
            ax.grid(True, alpha=0.3)
            
            # Other metrics
            ax = axes[row_idx, 2]
            metrics = {}
            if "auc" in result:
                metrics["AUC"] = np.mean(result["auc"])
            if "precision" in result:
                metrics["Precision"] = np.mean(result["precision"])
            if "recall" in result:
                metrics["Recall"] = np.mean(result["recall"])
            metrics["Mean Acc"] = result["mean_accuracy"]
            
            ax.barh(list(metrics.keys()), list(metrics.values()), color='steelblue', alpha=0.7)
            ax.set_xlim([0, 1])
            ax.set_title(f"{dataset_name} - Metrics Summary")
            for i, v in enumerate(metrics.values()):
                ax.text(v, i, f' {v:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        fig_path = output_dir / "exp_real_datasets.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to {fig_path}")
        
        plt.close()


def main():
    """Run experiments on real datasets."""
    print("\n" + "="*80)
    print("SEALS Experiment 3: Real Dataset Evaluation")
    print("="*80)
    
    experiment = RealDataExperiment(test_size=0.2, seed=42)
    
    # Run CMAPSS
    cmapss_result = experiment.run_cmapss_experiment(dataset="FD001", retrain_interval=5)
    if cmapss_result:
        experiment.results["CMAPSS-FD001"] = cmapss_result
    
    # Run AI4I
    ai4i_result = experiment.run_ai4i_experiment(retrain_interval=5)
    if ai4i_result:
        experiment.results["AI4I-2020"] = ai4i_result
    
    # Plot
    experiment.plot_results()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for dataset_name, result in experiment.results.items():
        if result:
            print(f"\n{dataset_name}:")
            print(f"  Mean Accuracy: {result['mean_accuracy']:.4f}")
            if "mean_auc" in result:
                print(f"  Mean AUC: {result['mean_auc']:.4f}")
            if "mean_precision" in result:
                print(f"  Mean Precision: {result['mean_precision']:.4f}")
                print(f"  Mean Recall: {result['mean_recall']:.4f}")
            print(f"  Total Retrains: {len(result['retrain_steps'])}")
            spi_mean = result["spi_stats"].get("mean", 0)
            print(f"  Mean SPI: {spi_mean:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
