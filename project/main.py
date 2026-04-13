

import os
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import json

import torch
from torch.utils.tensorboard import SummaryWriter

from preprocessing import load_mnist_data, get_dataset_info
from model import get_model, get_model_info
from train import train_model
from utils import (
    set_seed,
    get_device,
    print_device_info,
    evaluate_model,
    create_tensorboard_dir,
    save_config,
    plot_training_curves,
    create_results_table
)


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(exp_name: str, log_dir: str = './runs') -> tuple:
    """
    Find the latest checkpoint for an experiment.
    
    Returns:
        (checkpoint_path, epoch) or (None, 0) if no checkpoint found
    """
    checkpoint_dir = os.path.join(log_dir, exp_name, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # Try to load latest.pt first
    latest_checkpoint = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        epoch = checkpoint.get('epoch', 0)
        return latest_checkpoint, epoch
    
    return None, 0


def create_experiment_config(base_config: dict, args: argparse.Namespace = None) -> dict:
    """Create experiment configuration from base config and CLI arguments."""
    config = base_config.copy()
    
    # Override with CLI arguments if provided
    if args:
        if args.batch_size:
            config['data']['batch_size'] = args.batch_size
        if args.learning_rate:
            config['training']['learning_rate'] = args.learning_rate
        if args.epochs:
            config['training']['epochs'] = args.epochs
        if args.model:
            config['model']['architecture'] = args.model
        if args.optimizer:
            config['training']['optimizer'] = args.optimizer
        if args.weight_decay:
            config['training']['weight_decay'] = args.weight_decay
        if args.seed:
            config['seed'] = args.seed
    
    return config


def main():
    """Main training pipeline."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Deep Learning Pipeline for MNIST Classification")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--model', type=str, default=None,
                        help='Model architecture')
    parser.add_argument('--optimizer', type=str, default=None,
                        help='Optimizer (Adam or SGD)')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable TensorBoard logging')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save training plots')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Load base configuration
    if os.path.exists(args.config):
        base_config = load_yaml_config(args.config)
    else:
        print(f"Warning: Config file {args.config} not found. Using defaults.")
        base_config = {
            'training': {'epochs': 10, 'batch_size': 64, 'learning_rate': 0.001},
            'data': {'dataset': 'MNIST', 'data_dir': './data'},
            'model': {'architecture': 'SimpleCNN'},
            'device': 'cuda',
            'seed': 42,
            'deterministic': True,
            'logging': {'log_dir': './runs', 'save_interval': 5}
        }
    
    # Create experiment configuration
    config = create_experiment_config(base_config, args)
    
    # Create experiment directory
    if args.exp_name:
        exp_name = args.exp_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"exp_{timestamp}"
    
    tensorboard_dir = create_tensorboard_dir(config['logging']['log_dir'], exp_name)
    checkpoint_dir = os.path.join(tensorboard_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, tensorboard_dir)
    
    print(f"\n{'='*70}")
    print(f"DEEP LEARNING PIPELINE - EXPERIMENT: {exp_name}")
    print(f"{'='*70}\n")
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    deterministic = config.get('deterministic', True)
    set_seed(seed, deterministic=deterministic)
    print(f"Random seed set to: {seed}, Deterministic: {deterministic}\n")
    
    # Get device
    device_name = config.get('device', 'cuda')
    device = get_device(device_name)
    print_device_info(device)
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(
        data_dir=config['data']['data_dir'],
        train_size=config['data']['train_size'],
        test_size=config['data']['test_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        augmentation=False
    )
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(test_loader)}\n")
    
    # Create model
    print("Creating model...")
    model = get_model(
        architecture=config['model']['architecture'],
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        device=device
    )
    model_info = get_model_info(model)
    print(f"  Architecture: {model_info['model_class']}")
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Device: {model_info['device']}\n")
    
    # Check for checkpoint if resuming
    resume_epoch = 0
    if args.resume:
        checkpoint_path, epoch = find_latest_checkpoint(exp_name, config['logging']['log_dir'])
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            resume_epoch = checkpoint.get('epoch', 0)
            print(f"  Resuming from epoch {resume_epoch}")
            print(f"  Best validation accuracy: {checkpoint.get('best_val_acc', 0):.2f}%\n")
        else:
            print(f"Warning: No checkpoint found for experiment '{exp_name}'")
            print(f"  Starting fresh training\n")
            args.resume = False
    
    # TensorBoard writer
    writer = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"TensorBoard logs: {tensorboard_dir}")
        print(f"  Run: tensorboard --logdir={tensorboard_dir}\n")
    
    # Training
    print(f"{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        optimizer_name=config['training']['optimizer'],
        weight_decay=config['training']['weight_decay'],
        device=device,
        writer=writer,
        scheduler_name=config['training'].get('scheduler'),
        checkpoint_dir=checkpoint_dir,
        start_epoch=resume_epoch
    )
    
    # Validation on test set
    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}\n")
    
    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"  Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  Correct: {test_metrics['correct']}/{test_metrics['total']}\n")
    
    # Save results
    results = {
        'config': config,
        'model_info': {
            'architecture': model_info['model_class'],
            'total_parameters': model_info['total_params']
        },
        'results': {
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'final_val_accuracy': float(history['val_accuracy'][-1]),
            'test_accuracy': float(test_metrics['accuracy']),
            'best_val_accuracy': max(history['val_accuracy'])
        }
    }
    
    results_path = os.path.join(tensorboard_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot and save training curves if requested
    if args.save_plots:
        plot_path = os.path.join(tensorboard_dir, 'training_curves.png')
        plot_training_curves(history, save_path=plot_path, title=f"Training: {exp_name}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"Model: {model_info['model_class']} ({model_info['total_params']:,} params)")
    print(f"Best Val Accuracy: {max(history['val_accuracy']):.2f}%")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"Results saved to: {tensorboard_dir}")
    print(f"{'='*70}\n")
    
    if writer:
        writer.close()
    
    return results


if __name__ == "__main__":
    main()
