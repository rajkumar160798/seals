"""
Data Loading Module

Utilities for loading CMAPSS and AI4I datasets.

Datasets:
- CMAPSS: Engine Remaining Useful Life (RUL) prediction (time series)
- AI4I 2020: Predictive maintenance data (tabular)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import warnings


class DataLoader:
    """Base class for data loading."""
    
    def __init__(self, data_dir: Path = None):
        """
        Args:
            data_dir: Path to data directory
        """
        if data_dir is None:
            # Find data directory relative to this file
            data_dir = Path(__file__).parent.parent / "data"
        
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")


class CMAPSSDataLoader(DataLoader):
    """
    Load CMAPSS (Commercial Modular Aero-Propulsion System Simulation) data.
    
    NASA dataset for predictive maintenance with engine degradation.
    Multiple datasets with increasing difficulty:
    - FD001: Single operating condition
    - FD002: Multiple operating conditions
    - FD003: Single operating condition with introduced faults
    - FD004: Multiple operating conditions with introduced faults
    """
    
    def __init__(self, data_dir: Path = None, dataset: str = "FD001"):
        """
        Args:
            data_dir: Path to data directory
            dataset: Which CMAPSS dataset to use (FD001, FD002, FD003, FD004)
        """
        super().__init__(data_dir)
        self.dataset = dataset
        self.cmapss_dir = self.data_dir / "CMAPSSData"
        
        if not self.cmapss_dir.exists():
            raise FileNotFoundError(f"CMAPSS directory not found: {self.cmapss_dir}")
    
    def load_train_test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load train and test data.
        
        Returns:
            (X_train, y_train, X_test, y_test)
        """
        # Load files
        train_file = self.cmapss_dir / f"train_{self.dataset}.txt"
        test_file = self.cmapss_dir / f"test_{self.dataset}.txt"
        rul_file = self.cmapss_dir / f"RUL_{self.dataset}.txt"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        if not rul_file.exists():
            raise FileNotFoundError(f"RUL file not found: {rul_file}")
        
        # Load data (space-separated, no headers)
        train_data = np.loadtxt(train_file)
        test_data = np.loadtxt(test_file)
        rul_values = np.loadtxt(rul_file)
        
        # Format: unit, cycle, 26 sensor readings
        # Columns: [unit_id, time_in_cycles, op_setting_1, op_setting_2, op_setting_3, sensor_1, ..., sensor_21]
        
        X_train = train_data[:, 2:]  # Drop unit_id and cycle, keep sensors + settings
        X_test = test_data[:, 2:]
        
        # Create RUL labels for train data
        # RUL decreases as cycle increases
        unit_ids_train = train_data[:, 0].astype(int)
        cycles_train = train_data[:, 1]
        
        y_train = np.zeros(len(X_train))
        for unit_id in np.unique(unit_ids_train):
            unit_mask = unit_ids_train == unit_id
            unit_cycles = cycles_train[unit_mask]
            max_cycle = np.max(unit_cycles)
            y_train[unit_mask] = max_cycle - unit_cycles
        
        # For test data, RUL comes from file
        unit_ids_test = test_data[:, 0].astype(int)
        cycles_test = test_data[:, 1]
        
        y_test = np.zeros(len(X_test))
        for i, unit_id in enumerate(np.unique(unit_ids_test)):
            unit_mask = unit_ids_test == unit_id
            unit_cycles = cycles_test[unit_mask]
            max_cycle = np.max(unit_cycles)
            y_test[unit_mask] = max_cycle - unit_cycles + rul_values[i]
        
        return X_train, y_train, X_test, y_test
    
    def load_sequences(self, max_samples: int = None) -> List[Tuple[np.ndarray, int]]:
        """
        Load CMAPSS as sequences (time series per engine unit).
        
        Args:
            max_samples: Maximum number of engines to load
            
        Returns:
            List of (sequence, RUL) tuples
        """
        X_train, y_train, X_test, y_test = self.load_train_test()
        train_data = np.loadtxt(self.cmapss_dir / f"train_{self.dataset}.txt")
        
        unit_ids = train_data[:, 0].astype(int)
        unique_units = np.unique(unit_ids)
        
        if max_samples:
            unique_units = unique_units[:max_samples]
        
        sequences = []
        for unit_id in unique_units:
            unit_mask = unit_ids == unit_id
            seq = X_train[unit_mask]
            rul = int(y_train[unit_mask][-1])  # RUL at end of life
            sequences.append((seq, rul))
        
        return sequences
    
    def normalize(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize data using train set statistics.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            (X_train_norm, X_test_norm)
        """
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8
        
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        
        return X_train_norm, X_test_norm


class AI4IDataLoader(DataLoader):
    """
    Load AI4I 2020 Predictive Maintenance Dataset.
    
    UCI Machine Learning Repository dataset for predictive maintenance
    on industrial equipment.
    """
    
    def __init__(self, data_dir: Path = None):
        """
        Args:
            data_dir: Path to data directory
        """
        super().__init__(data_dir)
        self.csv_file = self.data_dir / "ai4i2020.csv"
        
        if not self.csv_file.exists():
            raise FileNotFoundError(f"AI4I CSV not found: {self.csv_file}")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load AI4I data.
        
        Returns:
            (X, y) where y is machine failure indicator (binary)
        """
        df = pd.read_csv(self.csv_file)
        
        # Drop non-numeric columns and target
        target_col = 'Machine failure'
        
        # Identify feature columns (numeric, not target)
        feature_cols = [col for col in df.columns 
                       if col != target_col and df[col].dtype in [np.float64, np.int64]]
        
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(int) if target_col in df.columns else np.zeros(len(X), dtype=int)
        
        return X, y
    
    def load_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load AI4I with train/test split.
        
        Args:
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        X, y = self.load_data()
        
        n = len(X)
        n_test = int(n * test_size)
        
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n)
        
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    
    def get_feature_names(self) -> List[str]:
        """Get feature column names."""
        df = pd.read_csv(self.csv_file)
        target_col = 'Machine failure'
        return [col for col in df.columns 
               if col != target_col and df[col].dtype in [np.float64, np.int64]]
    
    def normalize(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize data using train set statistics.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            (X_train_norm, X_test_norm)
        """
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8
        
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        
        return X_train_norm, X_test_norm


def get_cmapss_data(dataset: str = "FD001", normalize: bool = True) -> Dict:
    """
    Convenience function to load CMAPSS data.
    
    Args:
        dataset: Dataset variant (FD001-FD004)
        normalize: Whether to normalize features
        
    Returns:
        Dictionary with X_train, X_test, y_train, y_test
    """
    loader = CMAPSSDataLoader(dataset=dataset)
    X_train, y_train, X_test, y_test = loader.load_train_test()
    
    if normalize:
        X_train, X_test = loader.normalize(X_train, X_test)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "dataset_name": f"CMAPSS-{dataset}"
    }


def get_ai4i_data(normalize: bool = True, test_size: float = 0.2) -> Dict:
    """
    Convenience function to load AI4I data.
    
    Args:
        normalize: Whether to normalize features
        test_size: Proportion for test set
        
    Returns:
        Dictionary with X_train, X_test, y_train, y_test, feature_names
    """
    loader = AI4IDataLoader()
    X_train, X_test, y_train, y_test = loader.load_train_test_split(test_size=test_size)
    
    if normalize:
        X_train, X_test = loader.normalize(X_train, X_test)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": loader.get_feature_names(),
        "dataset_name": "AI4I-2020"
    }
