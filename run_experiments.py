#!/usr/bin/env python
"""
SEALS Quick Start Script

Run this to execute all SEALS experiments and see results.
"""

import subprocess
import sys
from pathlib import Path
import time


def run_cmd(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*80}")
    print(f"▶ {description}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=Path(__file__).parent,
            capture_output=False
        )
        if result.returncode == 0:
            print(f"\n✓ {description} completed successfully")
            return True
        else:
            print(f"\n✗ {description} failed with code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n✗ Error running {description}: {e}")
        return False


def main():
    """Run all experiments."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("SEALS - Self-Evolving AI Lifecycle Simulator")
    print("="*80)
    print("\nThis script will run 3 experiments:")
    print("  1. Stability vs Plasticity (synthetic data)")
    print("  2. Feedback Regimes (synthetic data)")
    print("  3. Real Dataset Evaluation (CMAPSS + AI4I)")
    print("\nResults will be saved to: paper/figures/")
    
    results = {}
    
    # Experiment 1
    results["Exp 1: Stability vs Plasticity"] = run_cmd(
        "python experiments/exp_stability_plasticity.py",
        "Experiment 1: Stability vs Plasticity"
    )
    
    # Experiment 2
    results["Exp 2: Feedback Regimes"] = run_cmd(
        "python experiments/exp_feedback_regimes.py",
        "Experiment 2: Feedback Regimes"
    )
    
    # Experiment 3
    results["Exp 3: Real Datasets"] = run_cmd(
        "python experiments/exp_real_datasets.py",
        "Experiment 3: Real Dataset Evaluation"
    )
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for exp_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {exp_name}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")
    
    print("\n" + "="*80)
    print("Results saved to:")
    print("  - paper/figures/exp_stability_plasticity.png")
    print("  - paper/figures/exp_feedback_regimes.png")
    print("  - paper/figures/exp_real_datasets.png")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
