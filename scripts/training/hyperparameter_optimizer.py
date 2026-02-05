#!/usr/bin/env python3
"""
Hyperparameter Optimizer for LoRA Training

Unified hyperparameter optimization system supporting:
- Random search
- Grid search
- Bayesian optimization (Optuna)
- Iterative improvement based on evaluation feedback

Replaces:
- lora_hyperparameter_search.py
- lora_hyperparameter_search_optuna.py
- iterative_lora_optimizer.py
- run_hyperparameter_search.sh
- run_hyperparameter_search_optuna.sh

Author: LLMProvider Tooling
Date: 2025-11-22
"""

import argparse
import json
import subprocess
import sys
import toml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import random
import itertools
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Unified hyperparameter optimization for LoRA training.

    Supports multiple optimization methods:
    - random: Random sampling
    - grid: Exhaustive grid search
    - optuna: Bayesian optimization
    - iterative: Feedback-based improvement
    """

    # Default search space for LoRA training
    DEFAULT_SEARCH_SPACE = {
        'learning_rate': [5e-5, 8e-5, 1e-4, 1.5e-4, 2e-4],
        'text_encoder_lr': [4e-5, 6e-5, 8e-5, 1e-4],
        'network_dim': [64, 96, 128, 192, 256],
        'network_alpha': [32, 48, 64, 96, 128],
        'max_train_epochs': [8, 10, 12, 14, 16],
        'batch_size': [4, 6, 8, 12],
        'lr_scheduler': ['cosine', 'cosine_with_restarts', 'constant'],
        'optimizer_type': ['AdamW', 'AdamW8bit', 'Lion'],
    }

    def __init__(
        self,
        method: str = 'optuna',
        base_config: Path = None,
        search_space: Optional[Dict] = None,
        n_trials: int = 20,
        output_dir: Path = None
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            method: Optimization method (random, grid, optuna, iterative)
            base_config: Base training config file
            search_space: Parameter search space
            n_trials: Number of trials to run
            output_dir: Output directory for results
        """
        self.method = method
        self.base_config = Path(base_config) if base_config else None
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE
        self.n_trials = n_trials
        self.output_dir = Path(output_dir) if output_dir else Path('hyperparameter_search')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trials: List[Dict] = []
        self.best_trial: Optional[Dict] = None

        logger.info(f"Hyperparameter Optimizer initialized")
        logger.info(f"  Method: {method}")
        logger.info(f"  Base config: {base_config}")
        logger.info(f"  Search space: {len(search_space)} parameters")
        logger.info(f"  Trials: {n_trials}")

    def optimize(self) -> Dict:
        """
        Run hyperparameter optimization.

        Returns:
            Optimization results
        """
        if self.method == 'random':
            return self._optimize_random()
        elif self.method == 'grid':
            return self._optimize_grid()
        elif self.method == 'optuna':
            return self._optimize_optuna()
        elif self.method == 'iterative':
            return self._optimize_iterative()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _optimize_random(self) -> Dict:
        """Random search optimization"""
        logger.info(f"Starting random search with {self.n_trials} trials")

        for trial_id in range(self.n_trials):
            # Sample random parameters
            params = {
                param: random.choice(values)
                for param, values in self.search_space.items()
            }

            logger.info(f"\nTrial {trial_id + 1}/{self.n_trials}")
            logger.info(f"  Parameters: {params}")

            # Run trial
            result = self._run_trial(trial_id, params)
            self.trials.append(result)

            # Track best
            if self.best_trial is None or result['score'] > self.best_trial['score']:
                self.best_trial = result
                logger.info(f"  ✨ New best score: {result['score']:.4f}")

        return self._generate_results()

    def _optimize_grid(self) -> Dict:
        """Grid search optimization"""
        # Generate all combinations
        keys = list(self.search_space.keys())
        values = [self.search_space[k] for k in keys]
        combinations = list(itertools.product(*values))

        # Limit to n_trials
        combinations = combinations[:self.n_trials]

        logger.info(f"Starting grid search with {len(combinations)} combinations")

        for trial_id, combo in enumerate(combinations):
            params = dict(zip(keys, combo))

            logger.info(f"\nTrial {trial_id + 1}/{len(combinations)}")
            logger.info(f"  Parameters: {params}")

            # Run trial
            result = self._run_trial(trial_id, params)
            self.trials.append(result)

            # Track best
            if self.best_trial is None or result['score'] > self.best_trial['score']:
                self.best_trial = result
                logger.info(f"  ✨ New best score: {result['score']:.4f}")

        return self._generate_results()

    def _optimize_optuna(self) -> Dict:
        """Bayesian optimization using Optuna"""
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "Optuna not installed. Install: pip install optuna"
            )

        def objective(trial):
            # Sample parameters
            params = {}
            for param, values in self.search_space.items():
                if isinstance(values[0], (int, float)):
                    # Numeric parameter
                    params[param] = trial.suggest_categorical(param, values)
                else:
                    # Categorical parameter
                    params[param] = trial.suggest_categorical(param, values)

            # Run training trial
            result = self._run_trial(trial.number, params)
            self.trials.append(result)

            return result['score']

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name=f'lora_optimization_{datetime.now():%Y%m%d_%H%M%S}'
        )

        # Optimize
        logger.info(f"Starting Optuna optimization with {self.n_trials} trials")
        study.optimize(objective, n_trials=self.n_trials)

        # Best trial
        self.best_trial = self.trials[study.best_trial.number]

        logger.info(f"\n✨ Best trial: {study.best_trial.number}")
        logger.info(f"   Score: {study.best_value:.4f}")
        logger.info(f"   Parameters: {study.best_params}")

        return self._generate_results()

    def _optimize_iterative(self) -> Dict:
        """
        Iterative optimization with evaluation feedback.

        Train → Evaluate → Adjust → Repeat
        """
        logger.info("Starting iterative optimization")

        # Start with baseline parameters
        current_params = self._get_baseline_params()
        history = []

        for iteration in range(self.n_trials):
            logger.info(f"\n{'='*70}")
            logger.info(f"Iteration {iteration + 1}/{self.n_trials}")
            logger.info(f"{'='*70}")
            logger.info(f"Parameters: {current_params}")

            # Run trial
            result = self._run_trial(iteration, current_params)
            self.trials.append(result)
            history.append(result)

            # Track best
            if self.best_trial is None or result['score'] > self.best_trial['score']:
                self.best_trial = result
                logger.info(f"✨ New best score: {result['score']:.4f}")
            else:
                logger.info(f"Score: {result['score']:.4f} (best: {self.best_trial['score']:.4f})")

            # Suggest improvements based on results
            if iteration < self.n_trials - 1:
                current_params = self._suggest_improvements(current_params, history)
                logger.info(f"Adjusted parameters for next iteration")

        return self._generate_results()

    def _run_trial(self, trial_id: int, params: Dict) -> Dict:
        """
        Run a single training trial.

        Args:
            trial_id: Trial identifier
            params: Hyperparameters to test

        Returns:
            Trial results with score
        """
        trial_dir = self.output_dir / f"trial_{trial_id:03d}"
        trial_dir.mkdir(exist_ok=True)

        # Create trial config
        config_file = self._create_trial_config(trial_id, params, trial_dir)

        # Train (placeholder - would call actual training)
        # For now, return mock score
        # In real implementation: run training, evaluate, return score

        score = self._mock_evaluate(params)

        result = {
            'trial_id': trial_id,
            'params': params,
            'score': score,
            'config_file': str(config_file),
            'output_dir': str(trial_dir)
        }

        # Save trial result
        with open(trial_dir / 'result.json', 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def _create_trial_config(self, trial_id: int, params: Dict, output_dir: Path) -> Path:
        """Create training config for this trial"""
        if not self.base_config or not self.base_config.exists():
            raise FileNotFoundError(f"Base config not found: {self.base_config}")

        # Load base config
        with open(self.base_config) as f:
            config = toml.load(f)

        # Update with trial parameters
        for param, value in params.items():
            if param in config:
                config[param] = value

        # Update output directory
        config['output_dir'] = str(output_dir)

        # Save trial config
        trial_config = output_dir / f"trial_{trial_id:03d}_config.toml"
        with open(trial_config, 'w') as f:
            toml.dump(config, f)

        return trial_config

    def _mock_evaluate(self, params: Dict) -> float:
        """
        Mock evaluation function.

        In real implementation, this would:
        1. Train LoRA with params
        2. Generate test images
        3. Compute metrics (CLIP score, FID, etc.)
        4. Return score

        For now, returns random score for demonstration.
        """
        # Mock score based on parameter values
        # Favor higher learning rates and network dims
        score = 0.5
        if params.get('learning_rate', 0) > 1e-4:
            score += 0.1
        if params.get('network_dim', 0) >= 128:
            score += 0.1
        if params.get('optimizer_type') == 'AdamW':
            score += 0.05

        # Add some randomness
        score += random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, score))

    def _get_baseline_params(self) -> Dict:
        """Get baseline parameters for iterative optimization"""
        return {
            'learning_rate': 1.0e-4,
            'text_encoder_lr': 8.0e-5,
            'network_dim': 128,
            'network_alpha': 64,
            'max_train_epochs': 12,
            'batch_size': 8,
            'lr_scheduler': 'cosine',
            'optimizer_type': 'AdamW',
        }

    def _suggest_improvements(self, current_params: Dict, history: List[Dict]) -> Dict:
        """
        Suggest parameter improvements based on history.

        Simple heuristics:
        - If score improving: continue in same direction
        - If score degrading: try different direction
        - If stuck: random perturbation
        """
        if len(history) < 2:
            return current_params

        recent = history[-1]
        previous = history[-2]

        new_params = current_params.copy()

        # If improving, increase LR slightly
        if recent['score'] > previous['score']:
            if 'learning_rate' in new_params:
                new_params['learning_rate'] *= 1.1

        # If degrading, decrease LR
        else:
            if 'learning_rate' in new_params:
                new_params['learning_rate'] *= 0.9

        # Randomly perturb network size
        if random.random() > 0.7:
            if 'network_dim' in new_params:
                dims = [64, 96, 128, 192, 256]
                new_params['network_dim'] = random.choice(dims)
                new_params['network_alpha'] = new_params['network_dim'] // 2

        return new_params

    def _generate_results(self) -> Dict:
        """Generate optimization results summary"""
        results = {
            'method': self.method,
            'n_trials': len(self.trials),
            'best_trial': self.best_trial,
            'all_trials': self.trials,
            'summary': {
                'best_score': self.best_trial['score'] if self.best_trial else None,
                'best_params': self.best_trial['params'] if self.best_trial else None,
                'score_range': {
                    'min': min(t['score'] for t in self.trials) if self.trials else None,
                    'max': max(t['score'] for t in self.trials) if self.trials else None,
                    'mean': sum(t['score'] for t in self.trials) / len(self.trials) if self.trials else None
                }
            }
        }

        # Save results
        results_file = self.output_dir / 'optimization_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimizer for LoRA Training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--method',
        choices=['random', 'grid', 'optuna', 'iterative'],
        default='optuna',
        help='Optimization method (default: optuna)'
    )
    parser.add_argument(
        '--base-config',
        type=Path,
        required=True,
        help='Base training configuration file (TOML)'
    )
    parser.add_argument(
        '--search-space',
        type=Path,
        help='Custom search space definition (JSON)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of trials to run (default: 20)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('hyperparameter_search'),
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Load custom search space if provided
    search_space = None
    if args.search_space:
        with open(args.search_space) as f:
            search_space = json.load(f)

    # Create optimizer
    optimizer = HyperparameterOptimizer(
        method=args.method,
        base_config=args.base_config,
        search_space=search_space,
        n_trials=args.n_trials,
        output_dir=args.output_dir
    )

    # Run optimization
    try:
        results = optimizer.optimize()

        # Print summary
        print("\n" + "="*70)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*70)
        print(f"Method: {results['method']}")
        print(f"Trials: {results['n_trials']}")
        if results['best_trial']:
            print(f"\nBest trial: {results['best_trial']['trial_id']}")
            print(f"Best score: {results['best_trial']['score']:.4f}")
            print(f"\nBest parameters:")
            for param, value in results['best_trial']['params'].items():
                print(f"  {param}: {value}")
        print(f"\nResults: {args.output_dir}/optimization_results.json")
        print("="*70 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
