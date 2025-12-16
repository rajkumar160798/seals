#!/usr/bin/env python
"""
SEALS - Master Runner Script

Runs all experiments and generates comprehensive report.
"""

import sys
from pathlib import Path
import traceback

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def run_experiment(name: str, module_path: str):
    """Run single experiment."""
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"{'='*80}\n")
    
    try:
        # Dynamically import and run
        if "stability_plasticity" in module_path:
            from experiments.exp_stability_plasticity import main
        elif "feedback" in module_path:
            from experiments.exp_feedback_regimes import main
        elif "real_datasets" in module_path:
            from experiments.exp_real_datasets import main
        else:
            print(f"Unknown experiment: {module_path}")
            return False
        
        main()
        return True
    
    except Exception as e:
        print(f"\n✗ Error running {name}:")
        print(f"  {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all experiments."""
    print("\n" + "="*80)
    print("SEALS - Self-Evolving AI Lifecycle Simulator")
    print("Reference Implementation for ML System Dynamics")
    print("="*80)
    
    experiments = [
        ("Experiment 1: Stability vs Plasticity", "exp_stability_plasticity"),
        ("Experiment 2: Feedback Regimes", "exp_feedback_regimes"),
        ("Experiment 3: Real Dataset Evaluation", "exp_real_datasets"),
    ]
    
    results = {}
    
    for exp_name, exp_module in experiments:
        success = run_experiment(exp_name, exp_module)
        results[exp_name] = "✓ PASSED" if success else "✗ FAILED"
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for exp_name, status in results.items():
        print(f"{status}: {exp_name}")
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("Results saved to: paper/figures/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
