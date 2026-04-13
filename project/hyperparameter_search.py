"""
Hyperparameter optimization with grid search and random search.
Deep Direction #2: Run 10-20 experiments with hyperparameter tuning.
"""

import os
import json
import random
from typing import List, Dict
from datetime import datetime
from itertools import product

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
    save_config,
    create_results_table
)


def run_hyperparameter_grid_search(
    config: Dict,
    search_space: Dict,
    num_trials: int = None
) -> List[Dict]:
    """
    Run grid search over hyperparameter space.
    
    Args:
        config: Base configuration
        search_space: Dictionary with parameter lists
        num_trials: Max number of trials (for limiting grid size)
    
    Returns:
        List of results from all trials
    """
    
    # Create parameter combinations
    keys = search_space.keys()
    values = search_space.values()
    combinations = list(product(*values))
    
    if num_trials and len(combinations) > num_trials:
        print(f"Grid has {len(combinations)} combinations. Using first {num_trials}.")
        combinations = combinations[:num_trials]
    
    results = []
    base_seed = config.get('seed', 42)
    
    for trial_idx, combo in enumerate(combinations):
        trial_config = config.copy()
        
        # Update config with current hyperparameters
        for param_name, param_value in zip(keys, combo):
            if param_name in trial_config['training']:
                trial_config['training'][param_name] = param_value
            elif param_name in trial_config['data']:
                trial_config['data'][param_name] = param_value
        
        # Run trial
        print(f"\n{'='*70}")
        print(f"Trial {trial_idx + 1}/{len(combinations)}")
        print(f"{'='*70}")
        
        trial_result = run_single_trial(
            trial_config,
            trial_idx=trial_idx,
            base_seed=base_seed
        )
        results.append(trial_result)
    
    return results


def run_hyperparameter_random_search(
    config: Dict,
    search_space: Dict,
    num_trials: int = 10
) -> List[Dict]:
    """
    Run random search over hyperparameter space.
    
    Args:
        config: Base configuration
        search_space: Dictionary with parameter ranges/lists
        num_trials: Number of random trials
    
    Returns:
        List of results from all trials
    """
    
    results = []
    base_seed = config.get('seed', 42)
    
    for trial_idx in range(num_trials):
        trial_config = config.copy()
        
        # Randomly sample hyperparameters
        sampled_params = {}
        for param_name, param_range in search_space.items():
            if isinstance(param_range, list):
                sampled_params[param_name] = random.choice(param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                # Assume log scale for learning rate
                if "learning_rate" in param_name:
                    log_min, log_max = param_range
                    sampled_params[param_name] = 10 ** random.uniform(log_min, log_max)
                else:
                    min_val, max_val = param_range
                    sampled_params[param_name] = random.uniform(min_val, max_val)
        
        # Update config with sampled hyperparameters
        for param_name, param_value in sampled_params.items():
            if param_name in trial_config['training']:
                trial_config['training'][param_name] = param_value
            elif param_name in trial_config['data']:
                trial_config['data'][param_name] = param_value
        
        # Run trial
        print(f"\n{'='*70}")
        print(f"Trial {trial_idx + 1}/{num_trials}")
        print(f"{'='*70}")
        
        trial_result = run_single_trial(
            trial_config,
            trial_idx=trial_idx,
            base_seed=base_seed + trial_idx
        )
        results.append(trial_result)
    
    return results


def run_single_trial(config: Dict, trial_idx: int = 0, base_seed: int = 42) -> Dict:
    """
    Run a single training trial.
    
    Args:
        config: Configuration for this trial
        trial_idx: Trial index
        base_seed: Base seed for reproducibility
    
    Returns:
        Dictionary with trial results
    """
    
    # Set seed
    set_seed(base_seed, deterministic=config.get('deterministic', True))
    
    # Device
    device = get_device(config.get('device', 'cuda'))
    
    # Load data
    train_loader, test_loader = load_mnist_data(
        data_dir=config['data']['data_dir'],
        train_size=config['data']['train_size'],
        test_size=config['data']['test_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Create model
    model = get_model(
        architecture=config['model']['architecture'],
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        device=device
    )
    
    # Create TensorBoard writer
    exp_name = f"trial_{trial_idx:03d}_lr_{config['training']['learning_rate']:.0e}_bs_{config['training']['batch_size']}"
    log_dir = create_tensorboard_dir(config['logging']['log_dir'], exp_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Print trial info
    model_info = get_model_info(model)
    print(f"Learning Rate: {config['training']['learning_rate']:.6f}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Optimizer: {config['training']['optimizer']}")
    print(f"Weight Decay: {config['training']['weight_decay']}")
    print(f"Model: {model_info['model_class']} ({model_info['total_params']:,} params)")
    
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
    
    # Close TensorBoard
    writer.close()
    
    # Prepare results
    trial_result = {
        'trial_idx': trial_idx,
        'learning_rate': config['training']['learning_rate'],
        'batch_size': config['training']['batch_size'],
        'optimizer': config['training']['optimizer'],
        'weight_decay': config['training']['weight_decay'],
        'epochs': config['training']['epochs'],
        'model': model_info['model_class'],
        'parameters': model_info['total_params'],
        'final_train_loss': round(float(history['train_loss'][-1]), 4),
        'final_val_loss': round(float(history['val_loss'][-1]), 4),
        'best_val_accuracy': round(max(history['val_accuracy']), 2),
        'test_accuracy': round(float(test_metrics['accuracy']), 2),
        'log_dir': log_dir
    }
    
    return trial_result


def main():
    """Main hyperparameter search."""
    
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--method', type=str, default='grid',
                        choices=['grid', 'random'],
                        help='Search method: grid or random')
    parser.add_argument('--num-trials', type=int, default=12,
                        help='Number of trials for random search')
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
    
    # Define hyperparameter search space
    if args.method == 'grid':
        # Grid search: combinations of learning rates and batch sizes
        search_space = {
            'learning_rate': [0.1, 0.01, 0.001, 0.0001],
            'batch_size': [32, 64, 128],
            'optimizer': ['Adam', 'SGD'],
            'weight_decay': [0.0, 0.0001]
        }
        results = run_hyperparameter_grid_search(
            base_config,
            search_space,
            num_trials=args.num_trials
        )
    else:
        # Random search
        search_space = {
            'learning_rate': (-4, -1),  # 10^-4 to 10^-1 in log scale
            'batch_size': [16, 32, 64, 128, 256],
            'optimizer': ['Adam', 'SGD'],
            'weight_decay': [0.0, 0.00001, 0.0001, 0.001]
        }
        results = run_hyperparameter_random_search(
            base_config,
            search_space,
            num_trials=args.num_trials
        )
    
    # Create search directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.exp_name:
        search_dir = os.path.join(base_config['logging']['log_dir'], f"search_{args.exp_name}")
    else:
        search_dir = os.path.join(base_config['logging']['log_dir'], f"search_{timestamp}")
    
    os.makedirs(search_dir, exist_ok=True)
    
    # Save results
    results_json_path = os.path.join(search_dir, 'results.json')
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create and print results table
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SEARCH RESULTS ({args.method.upper()})")
    print(f"{'='*70}\n")
    
    # Sort by test accuracy
    results_sorted = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
    
    # Display table
    table_path = os.path.join(search_dir, 'results_table.txt')
    table = create_results_table(results_sorted, save_path=table_path)
    print(table)
    
    # Find best configuration
    best_result = results_sorted[0]
    print(f"\n{'='*70}")
    print("BEST CONFIGURATION")
    print(f"{'='*70}")
    print(f"Trial: {best_result['trial_idx']}")
    print(f"Learning Rate: {best_result['learning_rate']:.6f}")
    print(f"Batch Size: {best_result['batch_size']}")
    print(f"Optimizer: {best_result['optimizer']}")
    print(f"Weight Decay: {best_result['weight_decay']}")
    print(f"Test Accuracy: {best_result['test_accuracy']:.2f}%")
    print(f"Best Val Accuracy: {best_result['best_val_accuracy']:.2f}%")
    print(f"Log Directory: {best_result['log_dir']}")
    print(f"{'='*70}\n")
    
    print(f"Results saved to: {search_dir}")
    
    return results_sorted


if __name__ == "__main__":
    main()
