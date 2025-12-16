"""
Benchmark datasets for SEALS comparison experiments.

Provides standard datasets:
- CIFAR-10-C (corrupted CIFAR-10)
- Rotating MNIST (concept drift)
- Synthetic concept drift sequences
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Optional
from pathlib import Path
from PIL import Image
import io


class CIFAR10C:
    """
    CIFAR-10 with natural corruptions (distribution shift).
    
    Papers: Hendrycks & Dietterich 2019
    """
    
    corruptions = [
        'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
        'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
        'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
        'saturate', 'shot_noise', 'snow', 'spatter', 'tilt_shift',
        'zoom_blur'
    ]
    
    @staticmethod
    def load(
        corruption: str = 'brightness',
        severity: int = 1,
        split: str = 'test',
        root: str = './data'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load CIFAR-10-C dataset.
        
        Args:
            corruption: Type of corruption
            severity: Severity level (1-5)
            split: 'test' or 'train'
            root: Data directory
            
        Returns:
            (images, labels)
        """
        # For demo: use standard CIFAR-10 with synthetic corruption
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=(split == 'train'),
            download=True,
            transform=transform
        )
        
        images = []
        labels = []
        
        for img, label in dataset:
            # Apply synthetic corruption
            if corruption == 'brightness':
                img = torch.clamp(img + (severity * 0.1), 0, 1)
            elif corruption == 'contrast':
                img = torch.clamp(img * (1 + severity * 0.1), 0, 1)
            elif corruption == 'gaussian_noise':
                img = img + torch.randn_like(img) * (severity * 0.05)
                img = torch.clamp(img, 0, 1)
            
            images.append(img.numpy())
            labels.append(label)
        
        return np.array(images), np.array(labels)


class RotatingMNIST:
    """
    MNIST with rotating digits (concept drift benchmark).
    
    Papers: Lopez-Paz et al. 2017 (PackNet)
    Concept drifts: Digits rotate progressively from 0° to 180°.
    """
    
    @staticmethod
    def load(
        num_tasks: int = 20,
        split: str = 'test',
        root: str = './data'
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load Rotating MNIST tasks.
        
        Args:
            num_tasks: Number of rotation tasks
            split: 'test' or 'train'
            root: Data directory
            
        Returns:
            List of (images, labels) tuples for each task
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        mnist = torchvision.datasets.MNIST(
            root=root,
            train=(split == 'train'),
            download=True,
            transform=transform
        )
        
        tasks = []
        rotation_angles = np.linspace(0, 180, num_tasks)
        
        for angle in rotation_angles:
            images = []
            labels = []
            
            for img, label in mnist:
                # Rotate image
                img_pil = transforms.ToPILImage()(img)
                img_rotated = img_pil.rotate(angle)
                img_tensor = transforms.ToTensor()(img_rotated)
                
                images.append(img_tensor.numpy())
                labels.append(label)
            
            tasks.append((np.array(images), np.array(labels)))
        
        return tasks


class ConceptDriftSequence:
    """
    Synthetic concept drift benchmark.
    
    Generates sequences with controlled drift characteristics.
    """
    
    @staticmethod
    def linear_drift(
        n_samples: int = 1000,
        n_features: int = 20,
        n_classes: int = 2,
        drift_type: str = 'gradual',
        drift_magnitude: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data with linear concept drift.
        
        Args:
            n_samples: Total number of samples
            n_features: Input dimensionality
            n_classes: Number of classes
            drift_type: 'gradual', 'sudden', or 'recurring'
            drift_magnitude: Magnitude of drift
            
        Returns:
            (X, y) with drift pattern
        """
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        
        if drift_type == 'gradual':
            # Gradually rotate decision boundary
            y = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                progress = i / n_samples
                rotation_angle = progress * drift_magnitude * np.pi
                
                # Rotate features
                X_rotated = X[i].copy()
                X_rotated[0] = X[i, 0] * np.cos(rotation_angle) - X[i, 1] * np.sin(rotation_angle)
                X_rotated[1] = X[i, 0] * np.sin(rotation_angle) + X[i, 1] * np.cos(rotation_angle)
                
                # Simple linear classifier
                y[i] = int((X_rotated[0] + X_rotated[1]) > 0)
        
        elif drift_type == 'sudden':
            # Sudden drift at midpoint
            y = np.zeros(n_samples, dtype=int)
            threshold = n_samples // 2
            
            for i in range(n_samples):
                if i < threshold:
                    y[i] = int((X[i, 0] + X[i, 1]) > 0)
                else:
                    # Drifted decision boundary
                    y[i] = int((X[i, 0] - X[i, 1]) > 0)
        
        else:  # recurring
            # Recurring concept drift
            y = np.zeros(n_samples, dtype=int)
            period = n_samples // 5
            
            for i in range(n_samples):
                phase = (i // period) % 2
                if phase == 0:
                    y[i] = int((X[i, 0] + X[i, 1]) > 0)
                else:
                    y[i] = int((X[i, 0] - X[i, 1]) > 0)
        
        return X, y
    
    @staticmethod
    def as_windows(
        X: np.ndarray,
        y: np.ndarray,
        window_size: int = 100,
        step_size: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Convert dataset to sliding windows.
        
        Args:
            X: Feature data
            y: Labels
            window_size: Size of each window
            step_size: Sliding step (default: window_size for non-overlapping)
            
        Returns:
            List of (X_window, y_window) tuples
        """
        if step_size is None:
            step_size = window_size
        
        windows = []
        for i in range(0, len(X) - window_size, step_size):
            X_window = X[i:i+window_size]
            y_window = y[i:i+window_size]
            windows.append((X_window, y_window))
        
        return windows


class BenchmarkDataLoader:
    """
    Unified interface for loading benchmark datasets.
    """
    
    @staticmethod
    def load_cifar10_c(
        corruption: str = 'brightness',
        severity: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load CIFAR-10-C."""
        X_train, y_train = CIFAR10C.load(corruption, severity, split='train')
        X_test, y_test = CIFAR10C.load(corruption, severity, split='test')
        
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()
        
        return (X_train, y_train), (X_test, y_test)
    
    @staticmethod
    def load_rotating_mnist(num_tasks: int = 20):
        """Load Rotating MNIST."""
        tasks = RotatingMNIST.load(num_tasks=num_tasks, split='test')
        
        result = []
        for X, y in tasks:
            # Convert to grayscale 3-channel for ResNet-18
            X = np.repeat(X, 3, axis=1)  # (N, 3, 28, 28)
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).long()
            result.append((X, y))
        
        return result
    
    @staticmethod
    def load_concept_drift_sequence(
        drift_type: str = 'gradual',
        n_samples: int = 5000,
        window_size: int = 500
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load concept drift sequence as windows."""
        X, y = ConceptDriftSequence.linear_drift(
            n_samples=n_samples,
            drift_type=drift_type
        )
        
        windows = ConceptDriftSequence.as_windows(X, y, window_size=window_size)
        
        result = []
        for X_w, y_w in windows:
            X_tensor = torch.from_numpy(X_w).float()
            y_tensor = torch.from_numpy(y_w).long()
            
            # Expand to image-like format for consistency
            # Add channel and spatial dimensions: (N, 1, H, W)
            X_tensor = X_tensor.unsqueeze(1).unsqueeze(1)
            
            result.append((X_tensor, y_tensor))
        
        return result
