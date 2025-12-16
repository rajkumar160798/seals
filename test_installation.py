#!/usr/bin/env python
"""
SEALS Quick Test

Run this to verify installation and data loading.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules import correctly."""
    print("\n" + "="*80)
    print("Testing imports...")
    print("="*80 + "\n")
    
    try:
        from simulator import (
            DriftEngine,
            FeedbackEngine,
            RetrainingPolicy,
            get_cmapss_data,
            get_ai4i_data
        )
        print("✓ Simulator modules imported successfully")
        
        from metrics import SPICalculator, AttributionDriftDetector
        print("✓ Metrics modules imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_data_loading():
    """Test that datasets can be loaded."""
    print("\n" + "="*80)
    print("Testing data loading...")
    print("="*80 + "\n")
    
    try:
        from simulator import get_cmapss_data, get_ai4i_data
        import numpy as np
        
        # Test CMAPSS
        print("Loading CMAPSS-FD001...")
        data = get_cmapss_data(dataset="FD001", normalize=True)
        assert "X_train" in data and "X_test" in data
        assert isinstance(data["X_train"], np.ndarray)
        print(f"  ✓ Train: {data['X_train'].shape}, Test: {data['X_test'].shape}")
        
        # Test AI4I
        print("Loading AI4I-2020...")
        data = get_ai4i_data(normalize=True)
        assert "X_train" in data and "X_test" in data
        assert isinstance(data["X_train"], np.ndarray)
        print(f"  ✓ Train: {data['X_train'].shape}, Test: {data['X_test'].shape}")
        print(f"  ✓ Features: {len(data['feature_names'])}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_core_components():
    """Test core SEALS components."""
    print("\n" + "="*80)
    print("Testing core components...")
    print("="*80 + "\n")
    
    try:
        from simulator import (
            DriftEngine,
            FeedbackEngine,
            RetrainingPolicy,
            FeedbackConfig,
            RetrainingConfig,
            FeedbackRegime,
            RetrainingRegime
        )
        from metrics import SPICalculator
        import numpy as np
        
        # Test DriftEngine
        print("Creating DriftEngine...")
        drift_engine = DriftEngine(n_features=10)
        X, y = drift_engine.generate_covariate_drift(magnitude=0.3)
        assert X.shape[0] > 0 and y.shape[0] > 0
        print(f"  ✓ Generated drift data: X shape {X.shape}, y shape {y.shape}")
        
        # Test FeedbackEngine
        print("Creating FeedbackEngine...")
        config = FeedbackConfig(
            regime=FeedbackRegime.PASSIVE,
            cost_per_label=10.0,
            human_trust=0.9,
            labeling_accuracy=0.95,
            budget=500.0
        )
        feedback = FeedbackEngine(config)
        print(f"  ✓ Feedback engine created")
        
        # Test RetrainingPolicy
        print("Creating RetrainingPolicy...")
        policy_config = RetrainingConfig(
            regime=RetrainingRegime.BALANCED
        )
        policy = RetrainingPolicy(policy_config)
        print(f"  ✓ Retraining policy created")
        
        # Test SPICalculator
        print("Creating SPICalculator...")
        spi = SPICalculator()
        spi_value = spi.update(0, 0.8, 0.1)
        print(f"  ✓ SPI calculator created, SPI value: {spi_value:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SEALS Installation Test")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Core Components", test_core_components),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    
    if all_passed:
        print("\n✓ All tests passed! You can now run:")
        print("  python experiments/exp_stability_plasticity.py")
        print("  python experiments/exp_feedback_regimes.py")
        print("  python experiments/exp_real_datasets.py")
        print("  python run_experiments.py  # Run all at once")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
    
    print()


if __name__ == "__main__":
    main()
