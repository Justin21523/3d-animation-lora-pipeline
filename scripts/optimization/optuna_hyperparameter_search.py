#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Optimization for LoRA Training
Uses TPE (Tree-structured Parzen Estimator) and multi-objective optimization
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import optuna
from optuna.trial import Trial
from optuna.visualization import plot_optimization_history, plot_param_importances
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.enhanced_metrics import EnhancedMetrics


class LoRAOptimizer:
    """
    Hyperparameter optimizer for LoRA training

    Uses Optuna to find optimal parameters by:
    1. Suggesting parameters from search space
    2. Training LoRA with those parameters
    3. Evaluating trained checkpoint
    4. Optimizing multi-objective metrics
    """

    def __init__(
        self,
        dataset_config: str,
        base_model: str,
        output_dir: str,
        study_name: str = "lora_optimization",
        storage: str = None,
        n_trials: int = 30,
        n_jobs: int = 1,
        device: str = "cuda",
    ):
        self.dataset_config = Path(dataset_config).resolve()  # Convert to absolute path
        self.base_model = base_model
        self.output_dir = Path(output_dir).resolve()  # Convert to absolute path
        self.study_name = study_name
        self.storage = storage or f"sqlite:///{self.output_dir}/optuna_study.db"
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.device = device

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize enhanced metrics evaluator
        self.evaluator = EnhancedMetrics(device=device)

        # Trial counter
        self.trial_counter = 0

    def suggest_hyperparameters(self, trial: Trial) -> Dict[str, any]:
        """
        Define hyperparameter search space (V2.1)

        Based on Trial 1-5 analysis:
        - learning_rate: 6e-5 to 1.2e-4 (narrowed from Trial 3/4 success range)
        - text_encoder_lr: 3.5e-5 to 8e-5 (refined ratio)
        - network_dim: 64, 128, 256 (focus on proven dimensions)
        - network_alpha_ratio: 0.25 to 0.9 (EXPANDED to explore Trial 3/4 success range)
        - optimizer: AdamW, Lion (removed 8bit for high ratios)
        - lr_scheduler: cosine_with_restarts, polynomial (focus on best performers)
        - gradient_accumulation_steps: 2, 3, 4 (minimum 2 for stability)
        - max_epochs: 12, 14, 16 (minimum 12 for proper convergence)
        """
        # Network architecture with RATIO-BASED alpha
        network_dim = trial.suggest_categorical("network_dim", [64, 128, 256])
        network_alpha_ratio = trial.suggest_float("network_alpha_ratio", 0.25, 0.9, step=0.05)
        network_alpha = int(network_dim * network_alpha_ratio)

        params = {
            # Learning rates (NARROWED based on Trial 3/4 analysis)
            "learning_rate": trial.suggest_float("learning_rate", 6e-5, 1.2e-4, log=True),
            "text_encoder_lr": trial.suggest_float("text_encoder_lr", 3.5e-5, 8e-5, log=True),

            # Network architecture (ratio-based)
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "network_alpha_ratio": network_alpha_ratio,  # Store for analysis

            # Optimizer (FIXED choices for Optuna compatibility)
            # Will enforce constraints post-suggestion
            "optimizer_type": trial.suggest_categorical(
                "optimizer_type",
                ["AdamW", "AdamW8bit", "Lion"]
            ),

            # Learning rate scheduler (focus on best performers)
            "lr_scheduler": trial.suggest_categorical(
                "lr_scheduler",
                ["cosine_with_restarts", "polynomial"]
            ),

            # Training dynamics (MINIMUM 2 for stability)
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps",
                [2, 3, 4]
            ),
            "max_train_epochs": trial.suggest_categorical("max_train_epochs", [12, 14, 16]),

            # Scheduler-specific parameters
            "lr_warmup_steps": trial.suggest_int("lr_warmup_steps", 100, 200, step=50),
        }

        # Add scheduler-specific parameters
        if params["lr_scheduler"] == "cosine_with_restarts":
            params["lr_scheduler_num_cycles"] = trial.suggest_int("lr_scheduler_num_cycles", 2, 3)

        # STRATIFIED SAFETY CONSTRAINTS for high alpha ratios
        if network_alpha_ratio >= 0.75:
            # High alpha ratio: enforce stability mechanisms
            params["max_grad_norm"] = 0.8
            params["min_snr_gamma"] = 5.0
            if params["gradient_accumulation_steps"] < 3:
                params["gradient_accumulation_steps"] = 3  # Minimum 3 for high ratios
            if params["optimizer_type"] == "AdamW8bit":
                params["optimizer_type"] = "AdamW"  # Force full precision for high ratios

        return params

    def train_lora(self, trial: Trial, params: Dict[str, any]) -> Path:
        """
        Train LoRA with given hyperparameters

        Returns:
            Path to trained checkpoint
        """
        self.trial_counter += 1
        trial_dir = self.output_dir / f"trial_{self.trial_counter:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save trial parameters
        with open(trial_dir / "params.json", 'w') as f:
            json.dump(params, f, indent=2)

        print(f"\n{'='*60}")
        print(f"🔬 TRIAL {self.trial_counter}/{self.n_trials}")
        print(f"{'='*60}")
        print(f"Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")

        # Build training command
        cmd = [
            "conda", "run", "-n", "kohya_ss",
            "python", "/mnt/c/AI_LLM_projects/kohya_ss/sd-scripts/train_network.py",
            "--dataset_config", str(self.dataset_config),
            "--pretrained_model_name_or_path", self.base_model,
            "--output_dir", str(trial_dir),
            "--output_name", f"lora_trial_{self.trial_counter}",
            "--network_module", "networks.lora",
            "--network_dim", str(params["network_dim"]),
            "--network_alpha", str(params["network_alpha"]),
            "--learning_rate", str(params["learning_rate"]),
            "--text_encoder_lr", str(params["text_encoder_lr"]),
            "--max_train_epochs", str(params["max_train_epochs"]),
            "--save_every_n_epochs", "2",  # Save checkpoint every 2 epochs for intermediate evaluation
            "--save_model_as", "safetensors",
            "--save_precision", "fp16",
            "--mixed_precision", "fp16",
            "--gradient_checkpointing",
            "--gradient_accumulation_steps", str(params["gradient_accumulation_steps"]),
            "--optimizer_type", params["optimizer_type"],
            "--lr_scheduler", params["lr_scheduler"],
            "--lr_warmup_steps", str(params["lr_warmup_steps"]),
            "--logging_dir", str(trial_dir / "logs"),
            "--seed", "42",
            "--clip_skip", "2",
            "--cache_latents",
            "--cache_latents_to_disk",
            "--max_data_loader_n_workers", "8",
        ]

        # Add scheduler-specific parameters
        if params["lr_scheduler"] == "cosine_with_restarts":
            cmd.extend([
                "--lr_scheduler_num_cycles",
                str(params.get("lr_scheduler_num_cycles", 3))
            ])

        # Add safety parameters for high alpha ratios (V2.1)
        if "max_grad_norm" in params:
            cmd.extend(["--max_grad_norm", str(params["max_grad_norm"])])
        if "min_snr_gamma" in params:
            cmd.extend(["--min_snr_gamma", str(params["min_snr_gamma"])])

        # Run training with process group management
        log_file = trial_dir / "training.log"
        print(f"📝 Training log: {log_file}")

        process = None
        try:
            with open(log_file, 'w') as f:
                # Start process in new session to create process group
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd="/mnt/c/AI_LLM_projects/kohya_ss/sd-scripts",
                    start_new_session=True  # Critical: creates new process group
                )

                # Wait for process to complete
                returncode = process.wait()

                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, cmd)

            print(f"✅ Training completed successfully")

            # Find checkpoint
            checkpoint_pattern = f"lora_trial_{self.trial_counter}.safetensors"
            checkpoint_path = trial_dir / checkpoint_pattern

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            return checkpoint_path

        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed with return code {e.returncode}")
            print(f"   Check log: {log_file}")
            # Kill entire process group
            if process and process.pid:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=5)
                except:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except:
                        pass
            raise optuna.TrialPruned(f"Training failed: {e}")

        except Exception as e:
            print(f"❌ Error during training: {e}")
            # Kill entire process group on any error
            if process and process.pid:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=5)
                except:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except:
                        pass
            raise optuna.TrialPruned(f"Training error: {e}")

    def evaluate_checkpoint(self, checkpoint_path: Path, trial: Trial) -> Dict[str, float]:
        """
        Evaluate trained checkpoint using enhanced SOTA metrics

        Uses comprehensive metrics:
        - LPIPS diversity (perceptual similarity, avoids mode collapse)
        - CLIP text-image consistency (ensures proper generation)
        - Basic quality metrics (brightness, contrast, saturation)
        - Pixar style adherence score

        Returns:
            Dictionary of metric values for optimization
        """
        print(f"\n📊 Evaluating checkpoint: {checkpoint_path.name}")

        eval_dir = checkpoint_path.parent / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        # Generate test images using evaluation script
        cmd = [
            "/home/b0979/.conda/envs/kohya_ss/bin/python",
            "/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/evaluation/evaluate_single_checkpoint.py",
            "--checkpoint", str(checkpoint_path),
            "--output-dir", str(eval_dir),
            "--num-samples", "8",
            "--device", self.device,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed: {e}")
            raise optuna.TrialPruned(f"Evaluation failed: {e}")

        # Load basic metrics from evaluation
        metrics_file = eval_dir / "metrics.json"
        if not metrics_file.exists():
            raise optuna.TrialPruned(f"Metrics file not found: {metrics_file}")

        with open(metrics_file, 'r') as f:
            basic_metrics = json.load(f)

        # Extract basic quality metrics
        quality = basic_metrics["metrics"]

        # Calculate standard deviation from arrays (for consistency metrics)
        import numpy as np
        std_brightness = float(np.std(quality["avg_brightness"]))
        std_contrast = float(np.std(quality["avg_contrast"]))

        results = {
            "mean_brightness": quality["mean_brightness"],
            "std_brightness": std_brightness,
            "mean_contrast": quality["mean_contrast"],
            "std_contrast": std_contrast,
        }

        # === SOTA METRICS: LPIPS and CLIP ===
        # Collect generated image paths
        image_paths = sorted(eval_dir.glob("sample_*.png"))
        if len(image_paths) < 2:
            print("⚠️  Not enough images for LPIPS/CLIP calculation")
            lpips_diversity = {"mean_lpips": 0.0}
            clip_consistency = {"mean_clip_score": 0.0}
        else:
            # Calculate LPIPS diversity (higher = more diverse, avoiding mode collapse)
            print("   Calculating LPIPS diversity...")
            lpips_diversity = self.evaluator.calculate_lpips_diversity(
                [str(p) for p in image_paths]
            )

            # Calculate CLIP text-image consistency
            # Use simple test prompts that match evaluation script
            print("   Calculating CLIP consistency...")
            test_prompts = [
                "a 3d animated character, pixar style, luca paguro, young boy",
                "a 3d animated character, pixar style, luca paguro, happy expression",
                "a 3d animated character, pixar style, luca paguro, looking at camera",
                "a 3d animated character, pixar style, luca paguro, three-quarter view",
                "a 3d animated character, pixar style, luca paguro, concerned expression",
                "a 3d animated character, pixar style, luca paguro, neutral stance",
                "a 3d animated character, pixar style, luca paguro, outdoor lighting",
                "a 3d animated character, pixar style, luca paguro, cinematic quality",
            ]
            clip_consistency = self.evaluator.calculate_clip_consistency(
                [str(p) for p in image_paths],
                test_prompts[:len(image_paths)]
            )

        # Store SOTA metrics
        results["lpips_diversity"] = lpips_diversity["mean_lpips"]
        results["clip_consistency"] = clip_consistency["mean_clip_score"]

        # === COMPREHENSIVE SCORING ===
        # 1. Pixar Style Score (brightness + contrast adherence)
        brightness_error = abs(results["mean_brightness"] - 0.50)
        contrast_error = abs(results["mean_contrast"] - 0.20)
        brightness_consistency = results["std_brightness"]
        contrast_consistency = results["std_contrast"]

        pixar_score = (brightness_error + 0.5 * brightness_consistency) + \
                     (contrast_error + 0.5 * contrast_consistency)

        # 2. LPIPS Diversity Score
        # Target: mean_lpips around 0.3-0.5 (diverse but not chaotic)
        # Too low (<0.2) = mode collapse, too high (>0.6) = inconsistent character
        lpips_target = 0.40
        lpips_error = abs(results["lpips_diversity"] - lpips_target)

        # 3. CLIP Consistency Score
        # Target: high consistency (>0.30 is good for CLIP ViT-B/32)
        # Convert to error: lower clip_score = worse, so invert it
        clip_error = max(0.0, 0.35 - results["clip_consistency"])  # Penalize if below 0.35

        # === FINAL COMBINED SCORE ===
        # Weight distribution (total = 1.0):
        # - Pixar style: 0.40 (most important for your use case)
        # - LPIPS diversity: 0.30 (prevents mode collapse, ensures variety)
        # - CLIP consistency: 0.30 (ensures proper generation of features)
        results["pixar_score"] = pixar_score
        results["lpips_score"] = lpips_error
        results["clip_score"] = clip_error

        results["combined_score"] = \
            0.40 * pixar_score + \
            0.30 * lpips_error + \
            0.30 * clip_error

        # Detailed logging
        print(f"\n✅ Comprehensive Evaluation Results:")
        print(f"   📸 Basic Quality:")
        print(f"      Brightness: {results['mean_brightness']:.3f} ± {results['std_brightness']:.3f}")
        print(f"      Contrast: {results['mean_contrast']:.3f} ± {results['std_contrast']:.3f}")
        print(f"\n   🎨 SOTA Metrics:")
        print(f"      LPIPS Diversity: {results['lpips_diversity']:.4f} (target: ~0.40)")
        print(f"      CLIP Consistency: {results['clip_consistency']:.4f} (target: >0.35)")
        print(f"\n   📊 Component Scores:")
        print(f"      Pixar Score: {results['pixar_score']:.4f} (weight: 0.40)")
        print(f"      LPIPS Score: {results['lpips_score']:.4f} (weight: 0.30)")
        print(f"      CLIP Score: {results['clip_score']:.4f} (weight: 0.30)")
        print(f"\n   🎯 Combined Score: {results['combined_score']:.4f} (lower = better)")

        # Save all metrics to trial attributes
        trial.set_user_attr("mean_brightness", results["mean_brightness"])
        trial.set_user_attr("std_brightness", results["std_brightness"])
        trial.set_user_attr("mean_contrast", results["mean_contrast"])
        trial.set_user_attr("std_contrast", results["std_contrast"])
        trial.set_user_attr("lpips_diversity", results["lpips_diversity"])
        trial.set_user_attr("clip_consistency", results["clip_consistency"])
        trial.set_user_attr("pixar_score", results["pixar_score"])
        trial.set_user_attr("lpips_score", results["lpips_score"])
        trial.set_user_attr("clip_score", results["clip_score"])

        return results

    def objective(self, trial: Trial) -> float:
        """
        Optuna objective function

        Returns:
            Score to minimize (combined brightness + contrast error)
        """
        # Suggest hyperparameters
        params = self.suggest_hyperparameters(trial)

        # Train LoRA
        checkpoint_path = self.train_lora(trial, params)

        # Evaluate checkpoint
        metrics = self.evaluate_checkpoint(checkpoint_path, trial)

        # Return combined score (lower is better)
        return metrics["combined_score"]

    def run_optimization(self):
        """
        Run Optuna hyperparameter optimization
        """
        print(f"\n{'='*60}")
        print(f"🚀 STARTING HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Study name: {self.study_name}")
        print(f"Storage: {self.storage}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Parallel jobs: {self.n_jobs}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        # Create Optuna study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction="minimize",  # Minimize combined error
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Run optimization
        try:
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\n⚠️  Optimization interrupted by user")

        # Print results
        self.print_results(study)

        # Save results
        self.save_results(study)

    def print_results(self, study: optuna.Study):
        """Print optimization results"""
        print(f"\n{'='*60}")
        print(f"✅ OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Number of pruned trials: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
        print(f"Number of complete trials: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")

        if len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])) > 0:
            print(f"\n{'='*60}")
            print(f"🏆 BEST TRIAL")
            print(f"{'='*60}")
            best_trial = study.best_trial

            print(f"Trial number: {best_trial.number}")
            print(f"Combined score: {best_trial.value:.4f}")
            print(f"\nBest hyperparameters:")
            for key, value in best_trial.params.items():
                print(f"  {key}: {value}")

            print(f"\nBest trial metrics:")
            print(f"  Brightness: {best_trial.user_attrs.get('mean_brightness', 'N/A'):.3f} ± "
                  f"{best_trial.user_attrs.get('std_brightness', 'N/A'):.3f}")
            print(f"  Contrast: {best_trial.user_attrs.get('mean_contrast', 'N/A'):.3f} ± "
                  f"{best_trial.user_attrs.get('std_contrast', 'N/A'):.3f}")
            print(f"{'='*60}\n")

    def save_results(self, study: optuna.Study):
        """Save optimization results"""
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)

        # Save best parameters
        if len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])) > 0:
            best_params_file = results_dir / "best_parameters.json"
            with open(best_params_file, 'w') as f:
                json.dump({
                    "trial_number": study.best_trial.number,
                    "combined_score": study.best_trial.value,
                    "parameters": study.best_trial.params,
                    "metrics": {
                        "mean_brightness": study.best_trial.user_attrs.get("mean_brightness"),
                        "std_brightness": study.best_trial.user_attrs.get("std_brightness"),
                        "mean_contrast": study.best_trial.user_attrs.get("mean_contrast"),
                        "std_contrast": study.best_trial.user_attrs.get("std_contrast"),
                    }
                }, f, indent=2)
            print(f"✅ Best parameters saved to: {best_params_file}")

        # Save all trials data
        trials_file = results_dir / "all_trials.json"
        trials_data = []
        for trial in study.trials:
            trials_data.append({
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            })

        with open(trials_file, 'w') as f:
            json.dump(trials_data, f, indent=2)
        print(f"✅ All trials data saved to: {trials_file}")

        # Save optimization history plot (if matplotlib available)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            fig = plot_optimization_history(study)
            fig.write_image(str(results_dir / "optimization_history.png"))
            print(f"✅ Optimization history plot saved")

            # Parameter importances
            if len(study.trials) >= 2:
                fig = plot_param_importances(study)
                fig.write_image(str(results_dir / "param_importances.png"))
                print(f"✅ Parameter importance plot saved")

        except ImportError:
            print("⚠️  matplotlib/plotly not available, skipping plots")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for LoRA training using Optuna"
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        help="Path to dataset configuration TOML file"
    )
    parser.add_argument(
        "--base-model",
        default="/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors",
        help="Path to base Stable Diffusion model"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for optimization results"
    )
    parser.add_argument(
        "--study-name",
        default="lora_optimization",
        help="Name of Optuna study"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (use 1 for GPU)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for evaluation"
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = LoRAOptimizer(
        dataset_config=args.dataset_config,
        base_model=args.base_model,
        output_dir=args.output_dir,
        study_name=args.study_name,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        device=args.device,
    )

    # Run optimization
    optimizer.run_optimization()


if __name__ == "__main__":
    main()
