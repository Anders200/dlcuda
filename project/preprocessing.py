"""
Data preprocessing and loading utilities for MNIST dataset.
"""

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist_transforms(augmentation: bool = False) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get MNIST data transforms.
    
    Args:
        augmentation: If True, apply augmentation to training data
    
    Returns:
        Tuple of (train_transform, test_transform)
    """
    
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return train_transform, test_transform


def load_mnist_data(
    data_dir: str = "./data",
    train_size: int = 10000,
    test_size: int = 2000,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    augmentation: bool = False,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset and return DataLoaders.
    
    Args:
        data_dir: Directory to download/store MNIST data
        train_size: Number of training samples to use
        test_size: Number of test samples to use
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for DataLoader
        pin_memory: Pin memory for faster GPU transfer
        augmentation: Apply data augmentation to training set
        shuffle_train: Shuffle training data
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Get transforms
    train_transform, test_transform = get_mnist_transforms(augmentation=augmentation)
    
    # Download/load datasets
    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    full_test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create subsets
    train_subset = Subset(full_train_dataset, range(min(train_size, len(full_train_dataset))))
    test_subset = Subset(full_test_dataset, range(min(test_size, len(full_test_dataset))))
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def get_dataset_info(data_dir: str = "./data") -> dict:
    """Get information about MNIST dataset."""
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True)
    
    return {
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "class_names": [str(i) for i in range(10)]
    }
