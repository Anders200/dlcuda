"""
Architecture exploration and comparison.
Basic Direction #3: Compare different model architectures.
"""

import os
import json
from typing import List, Dict

import torch
from torch.utils.tensorboard import SummaryWriter

from preprocessing import load_mnist_data
from model import get_model, get_model_info
from train import train_model
from utils import (
    set_seed,
    get_device,
    evaluate_model,
    create_tensorboard_dir,
    create_results_table,
    plot_comparison
)


def compare_architectures(
    config: Dict,
    architectures: List[str]
) -> List[Dict]:
    """
    Compare different model architectures.
    
    Args:
        config: Base configuration
        architectures: List of architecture names to compare
    
    Returns:
        List of results for each architecture
    """
    
    results = []
    device = get_device(config.get('device', 'cuda'))
    
    # Load data once
    train_loader, test_loader = load_mnist_data(
        data_dir=config['data']['data_dir'],
        train_size=config['data']['train_size'],
        test_size=config['data']['test_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    for arch_idx, architecture in enumerate(architectures):
        # Set seed for each architecture
        set_seed(config.get('seed', 42) + arch_idx, deterministic=config.get('deterministic', True))
        
        print(f"\n{'='*70}")
        print(f"Architecture {arch_idx + 1}/{len(architectures)}: {architecture}")
        print(f"{'='*70}")
        
        # Create model
        model = get_model(
            architecture=architecture,
            in_channels=config['model']['in_channels'],
            num_classes=config['model']['num_classes'],
            device=device
        )
        
        model_info = get_model_info(model)
        print(f"Model: {model_info['model_class']}")
        print(f"Parameters: {model_info['total_params']:,}")
        
        # Create TensorBoard writer
        exp_name = f"arch_{architecture}"
        log_dir = create_tensorboard_dir(config['logging']['log_dir'], exp_name)
        writer = SummaryWriter(log_dir=log_dir)
        
        # Train
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
            scheduler_name=config['training'].get('scheduler')
        )
        
        # Evaluate
        test_metrics = evaluate_model(model, test_loader, device)
        
        # Close writer
        writer.close()
        
        # Store results
        result = {
            'architecture': architecture,
            'parameters': model_info['total_params'],
            'final_train_loss': round(float(history['train_loss'][-1]), 4),
            'final_val_loss': round(float(history['val_loss'][-1]), 4),
            'best_val_accuracy': round(max(history['val_accuracy']), 2),
            'test_accuracy': round(float(test_metrics['accuracy']), 2),
            'convergence_epochs': min(5, len(history['val_loss'])),  # approximate
            'log_dir': log_dir,
            'history': {
                'train_loss': [float(x) for x in history['train_loss']],
                'val_loss': [float(x) for x in history['val_loss']],
                'val_accuracy': [float(x) for x in history['val_accuracy']]
            }
        }
        
        results.append(result)
    
    return results


def main():
    """Main architecture comparison."""
    
    import argparse
    parser = argparse.ArgumentParser(description="Architecture Exploration")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Load base config
    if os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            base_config = yaml.safe_load(f)
    else:
        print(f"Error: Config file {args.config} not found.")
        return
    
    # Architectures to compare
    architectures = [
        'SimpleCNN',
        'SimpleCNNWithBatchNorm',
        'DeepCNN',
        'SimpleCNNWithLeakyReLU',
        'SimpleCNNWithELU'
    ]
    
    # Run comparison
    results = compare_architectures(base_config, architectures)
    
    # Create results directory
    if args.exp_name:
        results_dir = os.path.join(base_config['logging']['log_dir'], f"architectures_{args.exp_name}")
    else:
        results_dir = os.path.join(base_config['logging']['log_dir'], "architectures")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    results_json_path = os.path.join(results_dir, 'architecture_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print results table
    print(f"\n{'='*70}")
    print("ARCHITECTURE COMPARISON RESULTS")
    print(f"{'='*70}\n")
    
    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
    
    # Create table data
    table_data = [
        {
            'architecture': r['architecture'],
            'parameters': f"{r['parameters']:,}",
            'train_loss': f"{r['final_train_loss']:.4f}",
            'val_loss': f"{r['final_val_loss']:.4f}",
            'val_accuracy': f"{r['best_val_accuracy']:.2f}%",
            'test_accuracy': f"{r['test_accuracy']:.2f}%"
        }
        for r in results_sorted
    ]
    
    table_path = os.path.join(results_dir, 'architecture_comparison.txt')
    table = create_results_table(table_data, save_path=table_path)
    print(table)
    
    print(f"\nResults saved to: {results_dir}")
    
    return results_sorted


if __name__ == "__main__":
    main()
