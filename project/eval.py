"""
Evaluate a trained model without training.
"""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from model import get_model, get_model_info
from preprocessing import load_mnist_data
from utils import get_device, print_device_info, evaluate_model


def get_architecture_from_checkpoint(checkpoint_path: str) -> str:
    """
    Try to get architecture from experiment config.json.
    Falls back to SimpleCNN if not found.
    """
    # Try to find config.json in parent directories
    current_path = os.path.dirname(os.path.abspath(checkpoint_path))
    
    for _ in range(3):  # Check up to 3 levels up
        config_path = os.path.join(current_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    architecture = config.get('model', {}).get('architecture', 'SimpleCNN')
                    return architecture
            except:
                pass
        current_path = os.path.dirname(current_path)
    
    return 'SimpleCNN'  # Default fallback


def evaluate_checkpoint(checkpoint_path: str, device: str = "cuda", batch_size: int = 64, architecture: str = None):
    """
    Evaluate a model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to use
        batch_size: Batch size for evaluation
        architecture: Model architecture (auto-detected if not provided)
    """
    device = get_device(device)
    print_device_info(device)
    
    # Auto-detect architecture if not provided
    if architecture is None:
        architecture = get_architecture_from_checkpoint(checkpoint_path)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model info from checkpoint metadata
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best Val Accuracy: {checkpoint.get('best_val_acc', 'unknown'):.2f}%\n")
    
    # Load data
    print("Loading MNIST test set...")
    _, test_loader = load_mnist_data(
        batch_size=batch_size,
        train_size=10000,
        test_size=2000,
        augmentation=False
    )
    print(f"  Test batches: {len(test_loader)}\n")
    
    # Recreate model with detected architecture
    print(f"Creating {architecture} model...")
    model = get_model(architecture=architecture, device=device)
    model_info = get_model_info(model)
    print(f"  Architecture: {model_info['model_class']}")
    print(f"  Total parameters: {model_info['total_params']:,}\n")
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded from checkpoint\n")
    
    # Evaluate
    print("Evaluating on test set...")
    metrics = evaluate_model(model, test_loader, device)
    
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Test Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Correct: {metrics['correct']}/{metrics['total']}")
    print(f"{'='*70}\n")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model without training")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--architecture', type=str, default=None,
                        help='Model architecture (auto-detected from config if not provided)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    if not args.checkpoint:
        print("Error: --checkpoint is required")
        print("Usage: python eval.py --checkpoint ./runs/exp_name/checkpoints/best_model.pt")
        exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        exit(1)
    
    evaluate_checkpoint(args.checkpoint, device=args.device, batch_size=args.batch_size, architecture=args.architecture)

