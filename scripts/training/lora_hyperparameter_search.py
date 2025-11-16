#!/usr/bin/env python3
"""
LoRA Hyperparameter Search
===========================

Automated hyperparameter optimization for LoRA training.
Uses random search or grid search to find optimal configurations.

Author: Claude Code
Date: 2025-11-14
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List
import random
import itertools
from datetime import datetime

def generate_config_variations(
    base_config: Path,
    search_space: Dict,
    n_trials: int,
    method: str = "random"
) -> List[Dict]:
    """Generate hyperparameter configurations to try"""
    
    if method == "grid":
        # Grid search: try all combinations
        keys = list(search_space.keys())
        values = [search_space[k] for k in keys]
        combinations = list(itertools.product(*values))
        
        configs = []
        for combo in combinations[:n_trials]:
            config = dict(zip(keys, combo))
            configs.append(config)
    
    elif method == "random":
        # Random search: sample randomly
        configs = []
        for _ in range(n_trials):
            config = {
                param: random.choice(values)
                for param, values in search_space.items()
            }
            configs.append(config)
    
    return configs


def create_trial_config(base_config: Path, params: Dict, trial_id: int, output_dir: Path) -> Path:
    """Create a training config file for this trial"""
    import toml
    
    # Load base config
    with open(base_config, 'r') as f:
        config = toml.load(f)
    
    # Update with trial parameters
    if 'network_dim' in params:
        config['model']['network_dim'] = params['network_dim']

    if 'network_alpha' in params:
        config['model']['network_alpha'] = params['network_alpha']

    if 'network_dropout' in params:
        config['model']['network_dropout'] = params['network_dropout']

    if 'learning_rate' in params:
        config['training']['learning_rate'] = params['learning_rate']
        config['training']['unet_lr'] = params['learning_rate']

    if 'lr_scheduler' in params:
        config['training']['lr_scheduler'] = params['lr_scheduler']

    if 'min_snr_gamma' in params:
        config['training']['min_snr_gamma'] = params['min_snr_gamma']

    if 'max_train_epochs' in params:
        config['training']['max_train_epochs'] = params['max_train_epochs']
    
    # Update output paths for this trial
    trial_name = f"trial_{trial_id:03d}"
    config['model']['output_name'] = trial_name
    config['model']['output_dir'] = str(output_dir / trial_name)
    
    # Save trial config
    trial_config_path = output_dir / f"{trial_name}_config.toml"
    with open(trial_config_path, 'w') as f:
        toml.dump(config, f)
    
    return trial_config_path


def run_trial(config_path: Path, trial_id: int, log_dir: Path) -> Dict:
    """Run a single training trial"""
    
    log_file = log_dir / f"trial_{trial_id:03d}.log"
    
    print(f"\n{'='*80}")
    print(f"Starting Trial {trial_id}")
    print(f"Config: {config_path}")
    print(f"Log: {log_file}")
    print(f"{'='*80}\n")
    
    cmd = [
        "conda", "run", "-n", "ai_env",
        "python", "sd-scripts/train_network.py",
        "--config_file", str(config_path)
    ]
    
    start_time = datetime.now()
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "trial_id": trial_id,
            "config": str(config_path),
            "status": "success",
            "duration_seconds": duration,
            "log_file": str(log_file)
        }
    
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "trial_id": trial_id,
            "config": str(config_path),
            "status": "failed",
            "error": str(e),
            "duration_seconds": duration,
            "log_file": str(log_file)
        }


def main():
    parser = argparse.ArgumentParser(description="LoRA Hyperparameter Search")
    parser.add_argument('--base-config', type=Path, required=True,
                        help='Base training configuration file')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for trials')
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of trials to run')
    parser.add_argument('--method', choices=['random', 'grid'], default='random',
                        help='Search method')
    parser.add_argument('--search-space', type=Path, default=None,
                        help='JSON file with search space (optional)')
    
    args = parser.parse_args()
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Define search space
    if args.search_space and args.search_space.exists():
        with open(args.search_space, 'r') as f:
            search_space = json.load(f)
    else:
        # Default search space for 3D character LoRA
        # Based on successful Trial 3.6 parameters and hyperparameter optimization guide
        search_space = {
            "network_dim": [32, 64, 128],
            "network_alpha": [16, 32, 64],
            "network_dropout": [0, 0.05, 0.1],
            "learning_rate": [5e-5, 6e-5, 8e-5, 1e-4, 1.5e-4],  # Added 6e-5 from Trial 3.6
            "lr_scheduler": ["cosine_with_restarts", "cosine", "constant"],
            "min_snr_gamma": [0, 5, 10],
            "max_train_epochs": [12, 16, 20, 24]  # Flexible epochs based on dataset size
        }
    
    # Generate trial configurations
    print(f"Generating {args.n_trials} trial configurations...")
    configs = generate_config_variations(
        args.base_config,
        search_space,
        args.n_trials,
        args.method
    )
    
    # Save search space and configurations
    metadata = {
        "search_method": args.method,
        "n_trials": args.n_trials,
        "search_space": search_space,
        "base_config": str(args.base_config),
        "start_time": datetime.now().isoformat()
    }
    
    with open(args.output_dir / "search_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Run trials
    results = []
    for i, params in enumerate(configs):
        trial_id = i + 1
        
        # Create trial config
        trial_config = create_trial_config(
            args.base_config,
            params,
            trial_id,
            args.output_dir
        )
        
        # Run trial
        result = run_trial(trial_config, trial_id, log_dir)
        result['parameters'] = params
        results.append(result)
        
        # Save incremental results
        with open(args.output_dir / "trial_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Final summary
    print(f"\n{'='*80}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Total trials: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"\nResults saved to: {args.output_dir / 'trial_results.json'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
