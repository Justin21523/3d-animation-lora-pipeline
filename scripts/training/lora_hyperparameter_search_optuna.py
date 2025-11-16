#!/usr/bin/env python3
"""
LoRA Hyperparameter Search with Optuna TPE
===========================================

Automated hyperparameter optimization using Tree-structured Parzen Estimator (TPE).
Based on HYPERPARAMETER_OPTIMIZATION_GUIDE.md V2.1 specification.

This implementation uses Optuna's TPE sampler for intelligent, adaptive hyperparameter search.

Author: Claude Code
Date: 2025-01-14
Version: 2.0 (Optuna TPE)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import optuna
from optuna.samplers import TPESampler

# Check if toml is available
try:
    import toml
except ImportError:
    print("ERROR: toml package not installed. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "toml"], check=True)
    import toml


class LoRAHyperparameterOptimizer:
    """
    Optuna-based hyperparameter optimizer for LoRA training.

    Uses TPE (Tree-structured Parzen Estimator) to intelligently search
    the hyperparameter space and find optimal configurations.
    """

    def __init__(
        self,
        base_config: Path,
        output_dir: Path,
        n_trials: int = 20,
        search_strategy: str = "aggressive"
    ):
        self.base_config = base_config
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.search_strategy = search_strategy
        self.trial_counter = 0

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # Load base config
        with open(self.base_config, 'r') as f:
            self.base_config_data = toml.load(f)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """
        Use Optuna TPE to suggest hyperparameters.

        This follows the V2.1 specification from HYPERPARAMETER_OPTIMIZATION_GUIDE.md
        with alpha_ratio approach and safety constraints.
        """

        # Network architecture - using alpha_ratio (0.25-0.9)
        network_dim = trial.suggest_categorical("network_dim", [64, 128, 256])
        network_alpha_ratio = trial.suggest_float("network_alpha_ratio", 0.25, 0.9, step=0.05)
        network_alpha = int(network_dim * network_alpha_ratio)
        network_dropout = trial.suggest_categorical("network_dropout", [0, 0.05, 0.1])

        # Learning rates (log-uniform for better exploration)
        learning_rate = trial.suggest_float("learning_rate", 6e-5, 1.2e-4, log=True)
        text_encoder_lr = trial.suggest_float("text_encoder_lr", 3e-5, 8e-5, log=True)

        # Scheduler
        lr_scheduler = trial.suggest_categorical(
            "lr_scheduler",
            ["cosine_with_restarts", "cosine", "constant"]
        )

        # Training duration
        max_train_epochs = trial.suggest_categorical("max_train_epochs", [12, 16, 20, 24])

        # Regularization
        min_snr_gamma = trial.suggest_categorical("min_snr_gamma", [0, 5, 10])

        # Optimizer (removed Lion as per V2.1)
        optimizer_type = trial.suggest_categorical("optimizer_type", ["AdamW", "AdamW8bit"])

        # Batch settings
        gradient_accumulation_steps = trial.suggest_categorical(
            "gradient_accumulation_steps",
            [1, 2, 4]
        )

        # Warmup
        lr_warmup_steps = trial.suggest_categorical("lr_warmup_steps", [50, 100, 150, 200])

        params = {
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "network_alpha_ratio": network_alpha_ratio,
            "network_dropout": network_dropout,
            "learning_rate": learning_rate,
            "text_encoder_lr": text_encoder_lr,
            "lr_scheduler": lr_scheduler,
            "max_train_epochs": max_train_epochs,
            "min_snr_gamma": min_snr_gamma,
            "optimizer_type": optimizer_type,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lr_warmup_steps": lr_warmup_steps
        }

        return params

    def check_safety_constraints(self, params: Dict) -> Tuple[bool, str]:
        """
        Check if parameter combination is safe to train.

        Implements safety constraints from V2.1 specification to avoid
        unstable or ineffective configurations.

        Returns:
            (is_valid, rejection_reason)
        """
        lr = params["learning_rate"]
        text_lr = params["text_encoder_lr"]
        dim = params["network_dim"]
        alpha = params["network_alpha"]
        alpha_ratio = params["network_alpha_ratio"]
        optimizer = params["optimizer_type"]
        epochs = params["max_train_epochs"]
        grad_accum = params["gradient_accumulation_steps"]
        warmup = params["lr_warmup_steps"]

        # Constraint 1: High LR + 8bit optimizer
        if lr > 0.00012 and optimizer == "AdamW8bit":
            return False, "High LR (>0.00012) + AdamW8bit = unstable"

        # Constraint 2: Exact Alpha = Dim (avoid float comparison issues)
        if abs(alpha - dim) < 1:
            return False, "Alpha ≈ Dim causes overfitting + instability"

        # Constraint 3: High Dim + Few Epochs
        if dim >= 256 and epochs < 16:
            return False, "High dim (≥256) needs ≥16 epochs for convergence"

        # Constraint 4: Very high LR + low warmup
        if lr > 0.0001 and warmup < 100:
            return False, "High LR (>0.0001) needs ≥100 warmup steps"

        # Constraint 5: Gradient Accumulation 1 + High LR
        if grad_accum == 1 and lr > 0.00011:
            return False, "High LR needs gradient accumulation ≥2"

        # Constraint 6: High alpha ratio (>= 0.75) requires strong stability
        if alpha_ratio >= 0.75:
            required_checks = [
                ("epochs >= 16", epochs >= 16),
                ("grad_accum >= 2", grad_accum >= 2),
                ("optimizer = AdamW", optimizer == "AdamW"),
                ("warmup >= 150", warmup >= 150),
            ]

            failed_checks = [name for name, passed in required_checks if not passed]
            if failed_checks:
                return False, f"High alpha (≥0.75) requires: {', '.join(failed_checks)}"

        # Constraint 7: Very high alpha (>= 0.85) + high LR is dangerous
        if alpha_ratio >= 0.85 and lr > 0.0001:
            return False, "Very high alpha (≥0.85) + LR >0.0001 = explosion risk"

        return True, ""

    def create_trial_config(self, params: Dict, trial_id: int) -> Path:
        """Create a training config file for this trial"""

        # Copy base config
        config = self.base_config_data.copy()

        # Update with trial parameters
        config['model']['network_dim'] = params['network_dim']
        config['model']['network_alpha'] = params['network_alpha']
        config['model']['network_dropout'] = params['network_dropout']

        config['training']['learning_rate'] = params['learning_rate']
        config['training']['unet_lr'] = params['learning_rate']
        config['training']['text_encoder_lr'] = params['text_encoder_lr']
        config['training']['lr_scheduler'] = params['lr_scheduler']
        config['training']['max_train_epochs'] = params['max_train_epochs']
        config['training']['min_snr_gamma'] = params['min_snr_gamma']
        config['training']['optimizer_type'] = params['optimizer_type']
        config['training']['gradient_accumulation_steps'] = params['gradient_accumulation_steps']
        config['training']['lr_warmup_steps'] = params['lr_warmup_steps']

        # Update output paths for this trial
        trial_name = f"trial_{trial_id:03d}"
        config['model']['output_name'] = trial_name
        config['model']['output_dir'] = str(self.output_dir / trial_name)

        # Save trial config
        trial_config_path = self.output_dir / f"{trial_name}_config.toml"
        with open(trial_config_path, 'w') as f:
            toml.dump(config, f)

        return trial_config_path

    def run_training(self, config_path: Path, trial_id: int) -> Dict:
        """Run a single training trial"""

        log_file = self.log_dir / f"trial_{trial_id:03d}.log"

        print(f"\n{'='*80}")
        print(f"Starting Trial {trial_id}")
        print(f"Config: {config_path}")
        print(f"Log: {log_file}")
        print(f"{'='*80}\n")

        cmd = [
            "conda", "run", "-n", "kohya_ss",
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

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        This is called by Optuna for each trial. It suggests parameters,
        checks constraints, runs training, and returns a score.
        """

        self.trial_counter += 1

        # Suggest hyperparameters using TPE
        params = self.suggest_hyperparameters(trial)

        # Check safety constraints
        is_valid, rejection_reason = self.check_safety_constraints(params)
        if not is_valid:
            print(f"\n⚠️  Trial {self.trial_counter} REJECTED: {rejection_reason}")
            print(f"Parameters: {json.dumps(params, indent=2)}\n")
            # Raise exception to tell Optuna this trial is pruned
            raise optuna.exceptions.TrialPruned(rejection_reason)

        # Create trial config
        trial_config = self.create_trial_config(params, self.trial_counter)

        # Run training
        result = self.run_training(trial_config, self.trial_counter)
        result['parameters'] = params

        # Save incremental results
        results_file = self.output_dir / "trial_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = []

        all_results.append(result)

        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        # For now, return a placeholder score
        # In real implementation, this would evaluate the trained model
        # Higher is better (we want to maximize)
        if result['status'] == 'success':
            # Placeholder: use negative duration as proxy (faster = better)
            # Real implementation should use CLIP score or other quality metrics
            score = -result['duration_seconds'] / 3600.0  # Convert to hours
            return score
        else:
            # Failed trials get worst score
            return float('-inf')

    def run_optimization(self):
        """Run the Optuna hyperparameter optimization"""

        print(f"\n{'='*80}")
        print("LoRA Hyperparameter Optimization with Optuna TPE")
        print(f"{'='*80}")
        print(f"Base config: {self.base_config}")
        print(f"Output dir: {self.output_dir}")
        print(f"Trials: {self.n_trials}")
        print(f"Strategy: {self.search_strategy}")
        print(f"{'='*80}\n")

        # Save optimization metadata
        metadata = {
            "optimization_method": "Optuna TPE",
            "base_config": str(self.base_config),
            "n_trials": self.n_trials,
            "search_strategy": self.search_strategy,
            "start_time": datetime.now().isoformat(),
            "optuna_version": optuna.__version__
        }

        with open(self.output_dir / "optimization_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create Optuna study with TPE sampler
        study = optuna.create_study(
            study_name="lora_hyperparameter_search",
            direction="maximize",  # We want to maximize score
            sampler=TPESampler(
                n_startup_trials=5,  # First 5 trials are random exploration
                multivariate=True,   # Consider parameter interactions
                seed=42              # Reproducibility
            ),
            storage=f"sqlite:///{self.output_dir / 'optuna_study.db'}",
            load_if_exists=True
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        # Save best parameters
        best_params = study.best_params
        best_params['alpha_ratio'] = best_params.get('network_alpha_ratio', 0.5)
        best_params['network_alpha'] = int(
            best_params['network_dim'] * best_params['alpha_ratio']
        )

        with open(self.output_dir / "best_parameters.json", 'w') as f:
            json.dump(best_params, f, indent=2)

        # Generate summary report
        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total trials: {len(study.trials)}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best score: {study.best_value:.4f}")
        print(f"\nBest parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA Hyperparameter Optimization with Optuna TPE"
    )
    parser.add_argument(
        '--base-config',
        type=Path,
        required=True,
        help='Base training configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for trials'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of trials to run (default: 20)'
    )
    parser.add_argument(
        '--search-strategy',
        choices=['conservative', 'aggressive'],
        default='aggressive',
        help='Search strategy (default: aggressive)'
    )

    args = parser.parse_args()

    # Create optimizer and run
    optimizer = LoRAHyperparameterOptimizer(
        base_config=args.base_config,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        search_strategy=args.search_strategy
    )

    optimizer.run_optimization()


if __name__ == "__main__":
    main()
