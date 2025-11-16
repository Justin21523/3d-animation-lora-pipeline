#!/usr/bin/env python3
"""
Iterative LoRA Training Optimizer

Automatically trains, evaluates, and improves LoRA models through multiple iterations.
Uses evaluation feedback to adjust hyperparameters for next training round.

Training Strategy:
- Round 1: Baseline training with default parameters
- Round 2+: Adjust parameters based on evaluation metrics
- Alternate between Luca and Alberto for GPU efficiency
- Generate sample images each iteration for progress monitoring
"""

import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import shutil
import sys

# Add path for prompt loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.utils.prompt_loader import load_character_prompts


class HyperparameterOptimizer:
    """Optimize hyperparameters based on evaluation results"""

    def __init__(self, character: str):
        self.character = character
        self.history = []
        self.best_score = 0.0
        self.consecutive_degradations = 0

        # Baseline parameters (Iteration V6 - Enhanced dataset + detailed captions)
        # Key improvements:
        #   - 372 diverse images (vs 191) with inpaint-only (preserves Pixar low-contrast)
        #   - Enhanced captions with detailed expression + action descriptions
        #   - Higher LR to leverage better data quality
        #   - Larger network capacity (128) for detailed caption encoding
        # Strategy: Leverage improved data quality for better results
        self.current_params = {
            'learning_rate': 1.0e-4,      # Raised from 8.0e-5 (leverage better data)
            'text_encoder_lr': 8.0e-5,    # Maintained 80% ratio (identity preservation)
            'network_dim': 128,           # Raised from 96 (more capacity for detailed captions)
            'network_alpha': 64,          # Proportional to dim (dim/2)
            'max_train_epochs': 12,       # Raised from 10 (more data, can train longer)
            'batch_size': 8,              # Keep at 8 (proven stable)
            'lr_scheduler': 'cosine',     # Smooth convergence
            'optimizer_type': 'AdamW',    # Proven optimizer (NO xformers)
        }

    def suggest_improvements(self, evaluation_result: Dict) -> Dict:
        """Suggest parameter improvements based on SOTA evaluation with early stopping"""

        new_params = self.current_params.copy()
        reasons = []

        # Extract SOTA metrics
        prompt_alignment = evaluation_result.get('internvl_score', evaluation_result.get('clip_score', 0))
        consistency = evaluation_result.get('character_consistency', 0)
        aesthetics = evaluation_result.get('aesthetic_score', evaluation_result.get('image_quality', 0))
        quality = evaluation_result.get('image_quality', 0)
        diversity = evaluation_result.get('diversity', 0)
        composite = evaluation_result.get('composite_score', 0)

        # ===== EARLY STOPPING: Check for quality degradation =====
        if len(self.history) >= 2:
            prev_score = self.history[-1].get('composite_score', 0)

            # Update best score
            if composite > self.best_score:
                self.best_score = composite
                self.consecutive_degradations = 0
            elif composite < prev_score:
                self.consecutive_degradations += 1

            # WARNING and RECOVERY if quality degrading for 2 consecutive iterations
            if self.consecutive_degradations >= 2:
                print("\n" + "="*70)
                print("⚠️  QUALITY DEGRADATION DETECTED - AUTO-RECOVERY MODE")
                print("="*70)
                print(f"Quality degrading for {self.consecutive_degradations} consecutive iterations")
                print(f"Best score: {self.best_score:.4f}")
                print(f"Current score: {composite:.4f}")
                print("\n🔧 Applying recovery strategy:")
                print("   1. Reverting to best iteration parameters")
                print("   2. Reducing learning rate by 20%")
                print("   3. Reducing epochs by 2")
                print("   4. Continuing training with adjusted settings")
                print("="*70 + "\n")

                # Revert to best iteration parameters
                best_iteration = max(self.history, key=lambda x: x.get('composite_score', 0))
                new_params = best_iteration['params'].copy()

                # Apply conservative adjustments
                new_params['learning_rate'] *= 0.8      # Reduce 20%
                new_params['text_encoder_lr'] *= 1.05   # Slightly boost relative ratio
                new_params['max_train_epochs'] = max(8, new_params['max_train_epochs'] - 2)

                reasons.append("Quality degradation recovery: reverted to best params + reduced LR")
                self.consecutive_degradations = 0  # Reset counter
                self.current_params = new_params

                # Continue training (DO NOT STOP)
                return {
                    'params': new_params,
                    'reasons': reasons,
                    'improvement_expected': True,
                    'recovery_mode': True
                }

        # Update best score on first iteration
        if len(self.history) == 0:
            self.best_score = composite

        # Strategy 1: Low prompt alignment → Increase epochs or LR
        if prompt_alignment < 0.28:
            if self.current_params['max_train_epochs'] < 20:
                new_params['max_train_epochs'] += 3
                reasons.append("Increasing epochs due to low prompt alignment score")
            else:
                new_params['learning_rate'] *= 1.2
                new_params['text_encoder_lr'] *= 1.2
                reasons.append("Increasing learning rate due to low prompt alignment")

        # Strategy 2: Low consistency → FOCUS ON FACIAL FEATURE STABILITY
        # Key insight: Facial inconsistency usually comes from training instability, NOT lack of capacity
        if consistency < 0.75:
            # DO NOT increase network_dim (already at 96 which is plenty)
            # Instead, reduce learning rate to stabilize feature learning
            new_params['learning_rate'] *= 0.85    # Reduce UNet LR by 15%
            new_params['text_encoder_lr'] *= 1.1   # Boost Text Encoder by 10% (relative increase)
            reasons.append("⚠️ Low consistency detected - reducing LR to stabilize facial features")

            # Also slightly reduce epochs if too high
            if new_params['max_train_epochs'] > 10:
                new_params['max_train_epochs'] -= 1
                reasons.append("Reducing epochs to prevent overfitting (facial consistency)")

        # Strategy 3: Low diversity → Reduce overfitting
        if diversity < 0.15:
            if self.current_params['max_train_epochs'] > 10:
                new_params['max_train_epochs'] -= 3
                reasons.append("Reducing epochs to prevent overfitting")

            new_params['learning_rate'] *= 0.8
            new_params['text_encoder_lr'] *= 0.8
            reasons.append("Reducing learning rate to improve generalization")

        # Strategy 4: Check improvement trend
        if len(self.history) >= 2:
            prev_score = self.history[-1].get('composite_score', 0)

            if composite < prev_score * 0.95:  # Performance dropped significantly
                # Revert to previous best params
                best_iteration = max(self.history, key=lambda x: x.get('composite_score', 0))
                new_params = best_iteration['params'].copy()
                reasons.append("Reverting to previous best parameters due to performance drop")

                # Small random perturbation to escape local minimum
                new_params['learning_rate'] *= 0.9
                reasons.append("Applying small learning rate adjustment")

            elif composite > prev_score * 1.02:  # Good improvement
                # Continue in same direction but more conservatively
                if self.current_params['learning_rate'] < new_params['learning_rate']:
                    new_params['learning_rate'] *= 1.1
                    reasons.append("Continuing learning rate increase (good trend)")

        # Strategy 5: Adaptive learning rate schedule
        if len(self.history) >= 3:
            recent_scores = [h.get('composite_score', 0) for h in self.history[-3:]]

            if max(recent_scores) - min(recent_scores) < 0.01:  # Plateaued
                new_params['lr_scheduler'] = 'cosine'
                reasons.append("Switching to cosine scheduler (plateau detected)")

        # ===== TEXT ENCODER RELATIVE WEIGHT PROTECTION =====
        # Ensure text_encoder_lr is at least 60% of unet_lr for identity preservation
        te_ratio = new_params['text_encoder_lr'] / new_params['learning_rate']
        if te_ratio < 0.6:
            new_params['text_encoder_lr'] = new_params['learning_rate'] * 0.6
            reasons.append("⚠️ Adjusted Text Encoder LR to maintain 60% ratio (identity preservation)")

        # Clamp parameters to reasonable ranges (Updated for iteration_v6)
        new_params['learning_rate'] = max(5e-5, min(1.5e-4, new_params['learning_rate']))      # Allow higher LR for better data
        new_params['text_encoder_lr'] = max(3e-5, min(1.2e-4, new_params['text_encoder_lr']))  # Proportionally higher
        new_params['network_dim'] = max(64, min(128, new_params['network_dim']))               # Allow up to 128 for detailed captions
        new_params['max_train_epochs'] = max(8, min(15, new_params['max_train_epochs']))       # Allow up to 15 epochs

        # Record history
        self.history.append({
            'params': self.current_params.copy(),
            'composite_score': composite,
            'metrics': evaluation_result
        })

        self.current_params = new_params

        return {
            'params': new_params,
            'reasons': reasons,
            'improvement_expected': len(reasons) > 0
        }


class IterativeTrainingOrchestrator:
    """Orchestrate iterative training and evaluation"""

    def __init__(
        self,
        characters: List[str],
        base_dataset_dir: Path,
        base_model_path: str,
        output_base_dir: Path,
        sd_scripts_dir: Path,
        max_iterations: int = 5,
        time_limit_hours: float = 14.0,
        pretrained_lora_dir: Path = None  # Optional: Directory containing pretrained LoRA models
    ):
        self.characters = characters
        self.base_dataset_dir = base_dataset_dir
        self.base_model_path = base_model_path
        self.output_base_dir = output_base_dir
        self.sd_scripts_dir = sd_scripts_dir
        self.max_iterations = max_iterations
        self.time_limit_seconds = time_limit_hours * 3600
        self.pretrained_lora_dir = pretrained_lora_dir  # Store pretrained LoRA directory

        self.optimizers = {char: HyperparameterOptimizer(char) for char in characters}
        self.start_time = time.time()

        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Log if continuing from pretrained models
        if self.pretrained_lora_dir and self.pretrained_lora_dir.exists():
            print(f"\n{'='*70}")
            print(f"📦 CONTINUATION TRAINING MODE")
            print(f"{'='*70}")
            print(f"Will load pretrained LoRA weights from: {self.pretrained_lora_dir}")
            for char in self.characters:
                pretrained_path = self.pretrained_lora_dir / f"{char}_BEST.safetensors"
                if pretrained_path.exists():
                    print(f"  ✓ Found: {char}_BEST.safetensors ({pretrained_path.stat().st_size / 1024 / 1024:.1f}MB)")
                else:
                    print(f"  ⚠️  Missing: {char}_BEST.safetensors")
            print(f"{'='*70}\n")

    def time_remaining(self) -> float:
        """Get remaining time in seconds"""
        elapsed = time.time() - self.start_time
        return max(0, self.time_limit_seconds - elapsed)

    def estimate_training_time(self, params: Dict, num_images: int) -> float:
        """Estimate training time in seconds"""

        # Rough estimate: 2 seconds per image per epoch
        images_per_epoch = num_images / params['batch_size']
        total_iterations = images_per_epoch * params['max_train_epochs']
        estimated_seconds = total_iterations * 2

        return estimated_seconds

    def generate_config(
        self,
        character: str,
        params: Dict,
        iteration: int,
        config_output_path: Path
    ) -> Path:
        """Generate training config TOML"""

        image_dir = self.base_dataset_dir / f"{character}_db"
        image_dir_images = self.base_dataset_dir / character / 'images'
        output_dir = self.output_base_dir / f"iteration_{iteration}" / character
        logging_dir = self.output_base_dir / "logs" / f"iteration_{iteration}" / character

        output_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)

        # Store training params for command line (not in config file)
        self.training_params = {
            'learning_rate': params['learning_rate'],
            'unet_lr': params['learning_rate'],
            'text_encoder_lr': params['text_encoder_lr'],
            'lr_scheduler': params['lr_scheduler'],
            'lr_warmup_steps': 200,
            'optimizer_type': params['optimizer_type'],
            'network_dim': params['network_dim'],
            'network_alpha': params['network_alpha'],
            'max_train_epochs': params['max_train_epochs'],
            'save_every_n_epochs': max(1, params['max_train_epochs'] // 5),
            'output_dir': str(output_dir),
            'output_name': f"{character}_iter{iteration}_v1",
            'logging_dir': str(logging_dir),
            'log_prefix': f"{character}_iter{iteration}",
            'seed': 42 + iteration,
        }

        # Dataset config file (use --dataset_config flag, NOT --config_file)
        # This file contains ONLY dataset configuration
        config_content = f"""# Dataset configuration for {character} - Iteration {iteration}

[general]
shuffle_caption = true
keep_tokens = 3

[[datasets]]
resolution = 512
batch_size = {params['batch_size']}
enable_bucket = true
min_bucket_reso = 384
max_bucket_reso = 768
bucket_reso_steps = 64
bucket_no_upscale = false

  [[datasets.subsets]]
  image_dir = "{image_dir_images}"
  class_tokens = "{character.replace('_', ' ')}"
  num_repeats = 1
  caption_extension = ".txt"
"""

        config_path = config_output_path / f"{character}_iter{iteration}.toml"
        with open(config_path, 'w') as f:
            f.write(config_content)

        return config_path

    def train_lora(self, config_path: Path, character: str, iteration: int) -> bool:
        """Execute training"""

        print(f"\n{'='*70}")
        print(f"TRAINING: {character.upper()} - Iteration {iteration}")
        print(f"{'='*70}")
        print(f"Config: {config_path}")
        print(f"Time remaining: {self.time_remaining() / 3600:.1f} hours")
        print(f"{'='*70}\n")

        train_script = self.sd_scripts_dir / 'train_network.py'

        # Build command with all training params from self.training_params
        # CRITICAL: Use --dataset_config (NOT --config_file) for dataset configuration!
        cmd = [
            'conda', 'run', '-n', 'ai_env',
            'python', str(train_script),
            '--dataset_config', str(config_path),  # Use --dataset_config for dataset config!
            '--pretrained_model_name_or_path', self.base_model_path,
            '--output_dir', self.training_params['output_dir'],
            '--output_name', self.training_params['output_name'],
            '--logging_dir', self.training_params['logging_dir'],
            '--log_prefix', self.training_params['log_prefix'],
            '--learning_rate', str(self.training_params['learning_rate']),
            '--unet_lr', str(self.training_params['unet_lr']),
            '--text_encoder_lr', str(self.training_params['text_encoder_lr']),
            '--lr_scheduler', self.training_params['lr_scheduler'],
            '--lr_warmup_steps', str(self.training_params['lr_warmup_steps']),
            '--optimizer_type', self.training_params['optimizer_type'],
            '--network_module', 'networks.lora',
            '--network_dim', str(self.training_params['network_dim']),
            '--network_alpha', str(self.training_params['network_alpha']),
            '--max_train_epochs', str(self.training_params['max_train_epochs']),
            '--save_every_n_epochs', str(self.training_params['save_every_n_epochs']),
            '--mixed_precision', 'fp16',
            '--save_model_as', 'safetensors',
            '--cache_latents',
            '--cache_latents_to_disk',
            '--gradient_checkpointing',
            '--gradient_accumulation_steps', '3',  # Increased from 2 (effective batch: 8*3=24, was 10*2=20)
            '--max_data_loader_n_workers', '8',
            '--persistent_data_loader_workers',
            '--seed', str(self.training_params['seed']),
            '--save_precision', 'fp16',
            '--clip_skip', '2',
            '--prior_loss_weight', '1.0',
        ]

        # Add pretrained LoRA weights if available (CONTINUATION TRAINING)
        if self.pretrained_lora_dir:
            pretrained_lora_path = self.pretrained_lora_dir / f"{character}_BEST.safetensors"
            if pretrained_lora_path.exists():
                cmd.extend(['--network_weights', str(pretrained_lora_path)])
                print(f"\n🔄 CONTINUATION TRAINING: Loading pretrained weights from:")
                print(f"   {pretrained_lora_path}")
                print(f"   Size: {pretrained_lora_path.stat().st_size / 1024 / 1024:.1f}MB")
                print(f"   This will continue training from iteration 3 best model\n")

        print("Training command:")
        print(" ".join(cmd[:5]) + " \\")
        print("  " + " \\\n  ".join(cmd[5:]))
        print()

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Add timeout to prevent hanging (4 hours per training)
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=14400  # 4 hours timeout
                )
                print("✓ Training completed successfully")

                # Clear GPU cache after training
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("✓ GPU cache cleared")
                except:
                    pass

                return True

            except subprocess.TimeoutExpired:
                print(f"✗ Training timeout after 4 hours (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    print("  Retrying with reduced batch size...")
                    # Reduce batch size for retry
                    self.training_params['batch_size'] = max(4, self.training_params.get('batch_size', 10) - 2)
                    continue
                return False

            except subprocess.CalledProcessError as e:
                print(f"✗ Training failed (attempt {attempt + 1}/{max_retries}): {e}")

                # Check for OOM error
                if e.stderr and ('out of memory' in e.stderr.lower() or 'oom' in e.stderr.lower()):
                    print("  Detected OOM error - GPU memory insufficient")
                    if attempt < max_retries - 1:
                        print("  Retrying with reduced parameters...")
                        # Reduce parameters for OOM
                        self.training_params['batch_size'] = max(4, self.training_params.get('batch_size', 10) - 3)
                        continue

                # Log error details
                error_log = self.output_base_dir / "logs" / f"error_{character}_iter{iteration}.log"
                error_log.parent.mkdir(parents=True, exist_ok=True)
                with open(error_log, 'w') as f:
                    f.write(f"Command: {' '.join(cmd)}\n\n")
                    f.write(f"Return code: {e.returncode}\n\n")
                    f.write(f"STDOUT:\n{e.stdout}\n\n")
                    f.write(f"STDERR:\n{e.stderr}\n")
                print(f"  Error log saved to: {error_log}")

                if attempt < max_retries - 1:
                    print("  Retrying...")
                    continue

                return False

            except Exception as e:
                print(f"✗ Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print("  Retrying...")
                    continue
                return False

        return False

    def evaluate_lora(
        self,
        lora_dir: Path,
        character: str,
        iteration: int
    ) -> Dict:
        """Evaluate trained LoRA using SOTA models"""

        print(f"\n{'='*70}")
        print(f"EVALUATING: {character.upper()} - Iteration {iteration}")
        print(f"{'='*70}\n")

        eval_output_dir = self.output_base_dir / "evaluations" / f"iteration_{iteration}" / character
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Use SOTA evaluator
        eval_script = Path(__file__).parent.parent / 'evaluation' / 'sota_lora_evaluator.py'

        cmd = [
            'conda', 'run', '-n', 'ai_env',
            'python', str(eval_script),
            '--lora-dir', str(lora_dir),
            '--character', character,
            '--base-model', self.base_model_path,
            '--output-dir', str(eval_output_dir)
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            # Load evaluation report (SOTA report name)
            report_path = eval_output_dir / 'sota_evaluation_report.json'
            with open(report_path, 'r') as f:
                report = json.load(f)

            print("✓ SOTA Evaluation completed")
            print(f"  Models used: {', '.join(report.get('evaluation_models', {}).values())}")
            print(f"  Best Checkpoint: {report['best_checkpoint']}")
            print(f"  Composite Score: {report['best_score']:.4f}")

            return report['rankings'][0]  # Return best checkpoint metrics

        except Exception as e:
            print(f"✗ Evaluation failed: {e}")
            return {}

    def run_iteration(self, character: str, iteration: int) -> Dict:
        """Run single training iteration"""

        print(f"\n{'#'*70}")
        print(f"# ITERATION {iteration} - {character.upper()}")
        print(f"{'#'*70}\n")

        # Check if model already exists (skip training if so)
        lora_output_dir = self.output_base_dir / f"iteration_{iteration}" / character
        model_files = list(lora_output_dir.glob("*.safetensors")) if lora_output_dir.exists() else []
        skip_training = len(model_files) > 0

        if skip_training:
            print(f"✓ Found existing model: {model_files[0].name}")
            print(f"  Skipping training, will evaluate existing model\n")
            # Load params from config or use defaults
            params = self.optimizers[character].current_params
            improvement_info = {'reasons': ['Using existing model']}
        else:
            # Get hyperparameters
            if iteration == 1 or len(self.optimizers[character].history) == 0:
                # Baseline (first iteration or no previous successful training)
                params = self.optimizers[character].current_params
                improvement_info = {'reasons': ['Baseline training']}
            else:
                # Optimize based on previous results
                prev_eval = self.optimizers[character].history[-1]['metrics']
                improvement_info = self.optimizers[character].suggest_improvements(prev_eval)
                params = improvement_info['params']

            print("Hyperparameters:")
            for key, value in params.items():
                print(f"  {key}: {value}")

            print("\nImprovement strategy:")
            for reason in improvement_info.get('reasons', []):
                print(f"  • {reason}")
            print()

            # Check time budget
            image_dir = self.base_dataset_dir / f"{character}_db"
            num_images = len(list(image_dir.glob('*.png')))
            estimated_time = self.estimate_training_time(params, num_images)

            if estimated_time > self.time_remaining():
                print(f"⚠️  Insufficient time remaining ({self.time_remaining() / 3600:.1f}h < {estimated_time / 3600:.1f}h)")
                return None

            # Generate config
            config_dir = self.output_base_dir / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)

            config_path = self.generate_config(character, params, iteration, config_dir)

            # Train
            success = self.train_lora(config_path, character, iteration)

            if not success:
                return None

        # Evaluate (either new model or existing model)
        eval_result = self.evaluate_lora(lora_output_dir, character, iteration)

        if not eval_result:
            return None

        # Save iteration result
        iteration_result = {
            'iteration': iteration,
            'character': character,
            'params': params,
            'metrics': eval_result,
            'timestamp': datetime.now().isoformat(),
            'improvement_strategy': improvement_info.get('reasons', [])
        }

        result_path = self.output_base_dir / f"iteration_{iteration}_{character}_result.json"
        with open(result_path, 'w') as f:
            json.dump(iteration_result, f, indent=2)

        return iteration_result

    def is_training_complete(self, character: str, iteration: int) -> bool:
        """Check if training for this character/iteration already completed"""
        # Check for final model file
        output_dir = self.output_base_dir / f"iteration_{iteration}" / character

        # Look for any .safetensors files (final model)
        model_files = list(output_dir.glob("*.safetensors"))

        if model_files:
            print(f"  ✓ Found existing model: {model_files[0].name}")

            # Also check for result JSON
            result_path = self.output_base_dir / f"iteration_{iteration}_{character}_result.json"
            if result_path.exists():
                print(f"  ✓ Found existing result file")
                return True
            else:
                print(f"  ⚠️  Model exists but result file missing - will re-evaluate")
                return False

        return False

    def load_existing_result(self, character: str, iteration: int) -> Dict:
        """Load existing training result if available"""
        result_path = self.output_base_dir / f"iteration_{iteration}_{character}_result.json"

        if result_path.exists():
            with open(result_path, 'r') as f:
                result = json.load(f)
            print(f"  ✓ Loaded existing result (score: {result['metrics'].get('composite_score', 0):.4f})")
            return result

        return None

    def save_checkpoint(self, all_results: List[Dict], iteration: int, character_completed: Dict = None):
        """Save checkpoint for recovery with character-level tracking"""
        checkpoint = {
            'last_completed_iteration': iteration,
            'results': all_results,
            'optimizers': {char: opt.history for char, opt in self.optimizers.items()},
            'timestamp': datetime.now().isoformat(),
            'character_completion': character_completed or {}
        }
        checkpoint_path = self.output_base_dir / 'checkpoint.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self) -> Tuple[List[Dict], int, Dict]:
        """Load checkpoint if exists, returns (results, next_iteration, completion_status)"""
        checkpoint_path = self.output_base_dir / 'checkpoint.json'

        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            # Restore optimizer histories
            for char, history in checkpoint['optimizers'].items():
                if char in self.optimizers:
                    self.optimizers[char].history = history

            # CRITICAL FIX: Return next iteration (last_completed + 1)
            last_completed = checkpoint.get('last_completed_iteration', 0)
            next_iteration = last_completed + 1 if last_completed > 0 else 1

            character_completion = checkpoint.get('character_completion', {})

            return checkpoint['results'], next_iteration, character_completion

        return [], 1, {}

    def run_optimization(self):
        """Run full optimization process with error handling and checkpointing"""

        print(f"\n{'='*70}")
        print(f"ITERATIVE LORA OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Characters:      {', '.join(self.characters)}")
        print(f"Max Iterations:  {self.max_iterations}")
        print(f"Time Limit:      {self.time_limit_seconds / 3600:.1f} hours")
        print(f"{'='*70}\n")

        # Try to resume from checkpoint
        all_results, start_iteration, character_completion = self.load_checkpoint()
        if start_iteration > 1:
            print(f"✓ Resuming from checkpoint")
            print(f"  Last completed iteration: {start_iteration - 1}")
            print(f"  Starting from iteration: {start_iteration}")
            print(f"  Loaded {len(all_results)} previous results\n")
        else:
            all_results = []
            character_completion = {}

        try:
            for iteration in range(start_iteration, self.max_iterations + 1):
                if self.time_remaining() < 1800:  # Less than 30 minutes
                    print(f"\n⚠️  Time budget exhausted. Stopping optimization.")
                    break

                # Progress header for this iteration round
                elapsed_hours = (time.time() - self.start_time) / 3600
                remaining_hours = self.time_remaining() / 3600
                print(f"\n{'▓'*70}")
                print(f"▓ ITERATION ROUND {iteration}")
                print(f"▓ Time: {elapsed_hours:.1f}h elapsed / {remaining_hours:.1f}h remaining")
                print(f"▓ Completed: {len(all_results)} iterations total")
                print(f"{'▓'*70}\n")

                # Initialize completion tracking for this iteration
                iteration_key = f"iter_{iteration}"
                if iteration_key not in character_completion:
                    character_completion[iteration_key] = {}

                # Alternate between characters
                for character in self.characters:
                    if self.time_remaining() < 900:  # Less than 15 minutes
                        print(f"\n⚠️  Insufficient time remaining. Stopping.")
                        break

                    # Check if this character/iteration already completed
                    if character_completion.get(iteration_key, {}).get(character, False):
                        print(f"\n✓ {character.upper()} iteration {iteration} already completed (skipping)")
                        # Load existing result and add to all_results if not already there
                        existing_result = self.load_existing_result(character, iteration)
                        if existing_result and existing_result not in all_results:
                            all_results.append(existing_result)
                        continue

                    # Also check for existing model files
                    if self.is_training_complete(character, iteration):
                        print(f"\n✓ {character.upper()} iteration {iteration} model exists (loading result)")
                        existing_result = self.load_existing_result(character, iteration)

                        if existing_result:
                            all_results.append(existing_result)
                            character_completion[iteration_key][character] = True
                            self.save_checkpoint(all_results, iteration, character_completion)
                            continue
                        else:
                            print(f"  ⚠️  Model exists but no result file - will re-evaluate")

                    try:
                        result = self.run_iteration(character, iteration)

                        if result:
                            all_results.append(result)
                            character_completion[iteration_key][character] = True

                            # Show immediate result summary
                            score = result['metrics'].get('composite_score', 0)
                            char_results = [r for r in all_results if r['character'] == character]
                            if len(char_results) > 1:
                                prev_score = char_results[-2]['metrics'].get('composite_score', 0)
                                improvement = score - prev_score
                                trend = "📈" if improvement > 0 else "📉"
                                print(f"\n{trend} {character.upper()} Score: {score:.4f} ({improvement:+.4f})")
                            else:
                                print(f"\n✓ {character.upper()} Baseline Score: {score:.4f}")

                            # Save checkpoint after each successful character iteration
                            self.save_checkpoint(all_results, iteration, character_completion)
                        else:
                            print(f"\n⚠️  Iteration failed for {character}. Continuing with next character.")

                    except KeyboardInterrupt:
                        print(f"\n⚠️  Training interrupted by user.")
                        self.save_checkpoint(all_results, iteration, character_completion)
                        raise

                    except Exception as e:
                        print(f"\n✗ Unexpected error during {character} iteration {iteration}: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"  Continuing with next character...")

                    # Small break between trainings for GPU cooling
                    time.sleep(10)

        except KeyboardInterrupt:
            print(f"\n{'='*70}")
            print(f"OPTIMIZATION INTERRUPTED")
            print(f"{'='*70}\n")
            print(f"Checkpoint saved. You can resume later.")

        finally:
            # Always generate report, even if interrupted
            self.generate_final_report(all_results)

    def generate_final_report(self, all_results: List[Dict]):
        """Generate comprehensive final report"""

        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}\n")

        report = {
            'total_iterations': len(all_results),
            'time_elapsed_hours': (time.time() - self.start_time) / 3600,
            'characters': {},
            'best_models': {}
        }

        # Analyze per character
        for character in self.characters:
            char_results = [r for r in all_results if r['character'] == character]

            if not char_results:
                continue

            best = max(char_results, key=lambda x: x['metrics'].get('composite_score', 0))

            report['characters'][character] = {
                'iterations_completed': len(char_results),
                'best_iteration': best['iteration'],
                'best_score': best['metrics'].get('composite_score', 0),
                'improvement_over_baseline': (
                    best['metrics'].get('composite_score', 0) -
                    char_results[0]['metrics'].get('composite_score', 0)
                ),
                'final_params': best['params']
            }

            report['best_models'][character] = best['metrics'].get('checkpoint', 'unknown')

            print(f"{character.upper()}:")
            print(f"  Iterations:       {len(char_results)}")
            print(f"  Best Iteration:   #{best['iteration']}")
            print(f"  Best Score:       {best['metrics'].get('composite_score', 0):.4f}")
            print(f"  Improvement:      {report['characters'][character]['improvement_over_baseline']:+.4f}")
            print(f"  Best Checkpoint:  {report['best_models'][character]}")
            print()

        # Save report
        report_path = self.output_base_dir / 'optimization_final_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Full report saved: {report_path}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Iterative LoRA training optimizer")
    parser.add_argument('--characters', nargs='+', required=True, help='Characters to train')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Curated dataset directory')
    parser.add_argument('--base-model', type=str, required=True, help='Base model path')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--sd-scripts', type=str, required=True, help='Path to sd-scripts')
    parser.add_argument('--max-iterations', type=int, default=5, help='Max iterations per character')
    parser.add_argument('--time-limit', type=float, default=14.0, help='Time limit in hours')

    args = parser.parse_args()

    orchestrator = IterativeTrainingOrchestrator(
        characters=args.characters,
        base_dataset_dir=Path(args.dataset_dir),
        base_model_path=args.base_model,
        output_base_dir=Path(args.output_dir),
        sd_scripts_dir=Path(args.sd_scripts),
        max_iterations=args.max_iterations,
        time_limit_hours=args.time_limit
    )

    orchestrator.run_optimization()


if __name__ == '__main__':
    main()
