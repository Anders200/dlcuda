"""
Model architecture definitions for MNIST classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Baseline CNN model for MNIST."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNNWithBatchNorm(nn.Module):
    """CNN with Batch Normalization."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class DeepCNN(nn.Module):
    """Deeper CNN with more layers and batch normalization."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Conv block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv block 1 + pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Conv block 2 + pooling
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Conv block 3 + pooling
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Flattening
        x = torch.flatten(x, 1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class SimpleCNNWithLeakyReLU(nn.Module):
    """CNN with LeakyReLU activation."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.fc2(x)
        return x


class SimpleCNNWithELU(nn.Module):
    """CNN with ELU activation."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.elu(self.conv1(x), alpha=1.0)
        x = F.elu(self.conv2(x), alpha=1.0)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.elu(self.fc1(x), alpha=1.0)
        x = self.fc2(x)
        return x


def get_model(architecture: str, in_channels: int = 1, num_classes: int = 10, device: str = "cuda") -> nn.Module:
    """
    Get model by name.
    
    Args:
        architecture: Model architecture name
        in_channels: Number of input channels
        num_classes: Number of output classes
        device: Device to move model to
    
    Returns:
        Model instance
    """
    models = {
        "SimpleCNN": SimpleCNN,
        "SimpleCNNWithBatchNorm": SimpleCNNWithBatchNorm,
        "DeepCNN": DeepCNN,
        "SimpleCNNWithLeakyReLU": SimpleCNNWithLeakyReLU,
        "SimpleCNNWithELU": SimpleCNNWithELU,
    }
    
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(models.keys())}")
    
    model = models[architecture](in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> dict:
    """Get model information."""
    return {
        "total_params": count_parameters(model),
        "model_class": model.__class__.__name__,
        "device": next(model.parameters()).device
    }
