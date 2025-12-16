"""
Deep Learning Model for SEALS: ResNet-18 with catastrophic forgetting detection.

This module provides a PyTorch-based ResNet-18 implementation for studying
stability-plasticity trade-offs in deep neural networks under concept drift.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import copy


class ResNet18(nn.Module):
    """ResNet-18 for CIFAR-10 scale images."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        """Initialize ResNet-18."""
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a residual layer."""
        layers = []
        
        # Downsample if needed
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        # First block
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    """Basic residual block."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        """Forward pass."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class DeepModel:
    """
    Deep learning model wrapper for SEALS framework.
    
    Provides:
    - Training and evaluation
    - Parameter change tracking
    - Catastrophic forgetting detection
    - Fisher information matrix (for EWC)
    - Experience replay buffer
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.01,
        seed: int = 42
    ):
        """
        Initialize deep model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB)
            device: Torch device (cuda/cpu)
            learning_rate: Optimizer learning rate
            seed: Random seed
        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.num_classes = num_classes
        self.model = ResNet18(num_classes=num_classes, input_channels=input_channels)
        self.model.to(self.device)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        # Track parameter changes
        self.initial_params = copy.deepcopy(self._get_params())
        self.previous_params = copy.deepcopy(self._get_params())
        
        # Catastrophic forgetting tracking
        self.task_accuracies = []  # Accuracy on each task
        self.forgetting_events = []  # Steps where forgetting detected
        
        # Experience replay buffer
        self.buffer = {'X': [], 'y': []}
        self.buffer_size = 500
        
        # Fisher information (for EWC)
        self.fisher_information = None
        
    def _get_params(self) -> np.ndarray:
        """Get flattened parameters."""
        params = []
        for p in self.model.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def _get_param_dict(self) -> Dict[str, torch.Tensor]:
        """Get parameter dict for EWC."""
        return {name: p.clone().detach() for name, p in self.model.named_parameters()}
    
    def get_parameter_change(self) -> float:
        """
        Compute L2 norm of parameter change since last evaluation.
        
        Returns:
            Frobenius norm of weight change
        """
        current_params = self._get_params()
        prev_params = self.previous_params
        change = np.linalg.norm(current_params - prev_params)
        return change
    
    def get_cumulative_parameter_change(self) -> float:
        """
        Compute total parameter change from initialization.
        
        Returns:
            L2 distance from initial parameters
        """
        current_params = self._get_params()
        initial_params = self.initial_params
        change = np.linalg.norm(current_params - initial_params)
        return change
    
    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 1) -> float:
        """
        Train model for one or more epochs.
        
        Args:
            X_train: Training data (N, C, H, W)
            y_train: Training labels (N,)
            epochs: Number of epochs
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_size = 32
            n_batches = len(X_train) // batch_size
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_X = X_train[start_idx:end_idx].to(self.device)
                batch_y = y_train[start_idx:end_idx].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / n_batches
        
        # Update parameter tracking
        self.previous_params = copy.deepcopy(self._get_params())
        
        return total_loss / epochs
    
    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> Tuple[float, float]:
        """
        Evaluate model accuracy.
        
        Args:
            X_test: Test data (N, C, H, W)
            y_test: Test labels (N,)
            
        Returns:
            (accuracy, loss)
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            batch_size = 32
            n_batches = len(X_test) // batch_size
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_X = X_test[start_idx:end_idx].to(self.device)
                batch_y = y_test[start_idx:end_idx].to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                total_loss += loss.item()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / max(n_batches, 1)
        
        return accuracy, avg_loss
    
    def compute_fisher_information(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """
        Compute Fisher information matrix for EWC.
        
        Args:
            X_train: Training data
            y_train: Training labels
        """
        self.model.eval()
        fisher = {}
        
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param.data)
        
        batch_size = 32
        n_batches = len(X_train) // batch_size
        
        for i in range(min(n_batches, 10)):  # Use subset for efficiency
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_train[start_idx:end_idx].to(self.device)
            batch_y = y_train[start_idx:end_idx].to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += (param.grad ** 2) / max(min(n_batches, 10), 1)
        
        self.fisher_information = fisher
    
    def add_to_buffer(self, X: torch.Tensor, y: torch.Tensor):
        """
        Add samples to experience replay buffer.
        
        Args:
            X: Data samples
            y: Labels
        """
        self.buffer['X'].extend([x for x in X])
        self.buffer['y'].extend([label for label in y])
        
        # Keep buffer at max size
        if len(self.buffer['X']) > self.buffer_size:
            excess = len(self.buffer['X']) - self.buffer_size
            self.buffer['X'] = self.buffer['X'][excess:]
            self.buffer['y'] = self.buffer['y'][excess:]
    
    def detect_catastrophic_forgetting(self, current_acc: float, threshold: float = 0.1) -> bool:
        """
        Detect if catastrophic forgetting occurred.
        
        Args:
            current_acc: Current accuracy
            threshold: Forgetting threshold (fraction of performance loss)
            
        Returns:
            True if forgetting detected
        """
        if len(self.task_accuracies) == 0:
            self.task_accuracies.append(current_acc)
            return False
        
        previous_acc = self.task_accuracies[-1]
        forgetting = (previous_acc - current_acc) > threshold
        
        if forgetting:
            self.forgetting_events.append(len(self.task_accuracies))
        
        self.task_accuracies.append(current_acc)
        return forgetting
    
    def state_dict(self) -> Dict:
        """Get model state."""
        return self.model.state_dict()
    
    def load_state_dict(self, state: Dict):
        """Load model state."""
        self.model.load_state_dict(state)
    
    def get_statistics(self) -> Dict:
        """Get model statistics."""
        return {
            'num_tasks_seen': len(self.task_accuracies),
            'forgetting_events': len(self.forgetting_events),
            'cumulative_param_change': self.get_cumulative_parameter_change(),
            'buffer_size': len(self.buffer['X']),
        }
