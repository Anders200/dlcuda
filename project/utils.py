"""
Utility functions for training, evaluation, and logging.
"""

import os
import random
import json
from typing import Dict, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Set CUBLAS environment variable for deterministic behavior with CUDA >= 10.2
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "cuda") -> str:
    if device_name == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def print_device_info(device: str):
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    
    if device == "cuda":
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        import platform
        print(f"CPU: {platform.processor()}")
    
    print(f"{'='*60}\n")


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str
) -> Dict:
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": np.array(all_preds),
        "targets": np.array(all_targets)
    }


def create_experiment_dir(base_dir: str = "./runs") -> str:
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def create_tensorboard_dir(base_dir: str = "./runs", exp_name: str = None) -> str:
    os.makedirs(base_dir, exist_ok=True)
    
    if exp_name:
        log_dir = os.path.join(base_dir, exp_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def save_config(config: Dict, save_dir: str, filename: str = "config.json"):
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, filename)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def plot_training_curves(
    history: Dict,
    save_path: str = None,
    title: str = "Training Curves"
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history["train_loss"], label="Train Loss", marker='o')
    axes[0].plot(history["val_loss"], label="Val Loss", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(history["val_accuracy"], label="Val Accuracy", marker='o')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_comparison(
    histories: Dict[str, Dict],
    metric: str = "val_accuracy",
    save_path: str = None
):
    plt.figure(figsize=(10, 6))
    
    for name, history in histories.items():
        plt.plot(history[metric], label=name, marker='o')
    
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Comparison of {metric}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def create_results_table(
    experiments: List[Dict],
    save_path: str = None
) -> str:
    if not experiments:
        return "No experiments to display"
    
    # Get column names from first experiment
    columns = list(experiments[0].keys())
    
    # Calculate column widths
    widths = {col: max(len(str(col)), max(len(str(exp.get(col, ""))) for exp in experiments)) for col in columns}
    
    # Create header
    header = " | ".join(f"{col:<{widths[col]}}" for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    
    # Create rows
    rows = []
    for exp in experiments:
        row = " | ".join(f"{str(exp.get(col, '')):<{widths[col]}}" for col in columns)
        rows.append(row)
    
    # Combine
    table = header + "\n" + separator + "\n" + "\n".join(rows)
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(table)
        print(f"Results table saved to {save_path}")
    
    return table


class AverageMeter:
    """Track metric running average."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"
