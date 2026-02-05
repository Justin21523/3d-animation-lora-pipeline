#!/usr/bin/env python3
"""
Training Config Generator for Kohya_ss SDXL LoRA Training

Automatically generates optimized TOML configuration files for Kohya_ss training
based on dataset characteristics and user preferences.

Part of Module 6: Training Pipeline Integration

Features:
- Auto-calculate optimal training steps based on dataset size
- Smart epoch and save interval selection
- Template-based TOML generation with full customization
- Dataset analysis for repeat count optimization
- Validation of all paths and parameters

Author: LLMProvider Tooling
Date: 2025-11-30
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import toml


@dataclass
class TrainingConfig:
    """
    Complete training configuration for Kohya_ss SDXL LoRA training

    All fields map directly to Kohya_ss TOML config format
    """

    # Model configuration
    pretrained_model_name_or_path: str
    network_module: str = "networks.lora"
    network_dim: int = 64
    network_alpha: int = 32
    network_dropout: float = 0.0

    # Paths
    train_data_dir: str = ""
    output_dir: str = ""
    output_name: str = ""
    logging_dir: str = ""
    log_prefix: str = ""

    # Training parameters
    optimizer_type: str = "AdamW8bit"
    mixed_precision: str = "bf16"
    full_bf16: bool = True
    gradient_checkpointing: bool = True

    train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 0.0001
    text_encoder_lr: float = 0.00006
    unet_lr: float = 0.0001

    # Learning rate scheduler
    lr_scheduler: str = "cosine_with_restarts"
    lr_scheduler_num_cycles: int = 3
    lr_warmup_steps: int = 100

    # Training quality
    min_snr_gamma: float = 5.0
    noise_offset: float = 0.05
    adaptive_noise_scale: float = 0.0
    multires_noise_iterations: int = 0
    multires_noise_discount: float = 0.3

    # Epochs and saving
    max_train_epochs: int = 4
    save_every_n_epochs: int = 2
    save_last_n_epochs: int = 2
    save_model_as: str = "safetensors"

    # Latent caching
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    vae_batch_size: int = 2

    # Resolution and bucketing
    resolution: str = "1024,1024"
    enable_bucket: bool = True
    min_bucket_reso: int = 768
    max_bucket_reso: int = 1280
    bucket_reso_steps: int = 128
    bucket_no_upscale: bool = False

    # Advanced options
    prior_loss_weight: float = 1.0
    clip_skip: int = 2
    max_token_length: int = 225
    shuffle_caption: bool = False
    keep_tokens: int = 0

    # Memory optimization
    xformers: bool = True
    lowram: bool = False
    max_data_loader_n_workers: int = 0
    persistent_data_loader_workers: bool = False

    # Logging and debugging
    log_with: str = "tensorboard"
    logging_level: str = "INFO"
    wandb_project: Optional[str] = None

    # Optional metadata
    comment: str = ""
    character_name: str = ""
    dataset_size: int = 0
    repeat_count: int = 12

    def to_toml_dict(self) -> Dict[str, Any]:
        """
        Convert to Kohya_ss TOML format

        Returns:
            Dict organized by TOML sections
        """
        return {
            "model": {
                "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
                "v2": False,
                "v_parameterization": False,
                "sdxl": True,
                "network_module": self.network_module,
                "network_dim": self.network_dim,
                "network_alpha": self.network_alpha,
                "network_dropout": self.network_dropout,
                "network_args": [],
            },
            "paths": {
                "train_data_dir": self.train_data_dir,
                "output_dir": self.output_dir,
                "output_name": self.output_name,
                "logging_dir": self.logging_dir,
                "log_prefix": self.log_prefix,
            },
            "training": {
                "optimizer_type": self.optimizer_type,
                "mixed_precision": self.mixed_precision,
                "full_bf16": self.full_bf16,
                "gradient_checkpointing": self.gradient_checkpointing,
                "train_batch_size": self.train_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "learning_rate": self.learning_rate,
                "text_encoder_lr": self.text_encoder_lr,
                "unet_lr": self.unet_lr,
                "lr_scheduler": self.lr_scheduler,
                "lr_scheduler_num_cycles": self.lr_scheduler_num_cycles,
                "lr_warmup_steps": self.lr_warmup_steps,
                "min_snr_gamma": self.min_snr_gamma,
                "noise_offset": self.noise_offset,
                "adaptive_noise_scale": self.adaptive_noise_scale,
                "multires_noise_iterations": self.multires_noise_iterations,
                "multires_noise_discount": self.multires_noise_discount,
                "max_train_epochs": self.max_train_epochs,
                "save_every_n_epochs": self.save_every_n_epochs,
                "save_last_n_epochs": self.save_last_n_epochs,
                "save_model_as": self.save_model_as,
                "cache_latents": self.cache_latents,
                "cache_latents_to_disk": self.cache_latents_to_disk,
                "vae_batch_size": self.vae_batch_size,
            },
            "resolution": {
                "resolution": self.resolution,
                "enable_bucket": self.enable_bucket,
                "min_bucket_reso": self.min_bucket_reso,
                "max_bucket_reso": self.max_bucket_reso,
                "bucket_reso_steps": self.bucket_reso_steps,
                "bucket_no_upscale": self.bucket_no_upscale,
            },
            "advanced": {
                "prior_loss_weight": self.prior_loss_weight,
                "clip_skip": self.clip_skip,
                "max_token_length": self.max_token_length,
                "shuffle_caption": self.shuffle_caption,
                "keep_tokens": self.keep_tokens,
                "xformers": self.xformers,
                "lowram": self.lowram,
                "max_data_loader_n_workers": self.max_data_loader_n_workers,
                "persistent_data_loader_workers": self.persistent_data_loader_workers,
            },
            "logging": {
                "log_with": self.log_with,
                "logging_dir": self.logging_dir,
                "log_prefix": self.log_prefix,
            }
        }


class DatasetAnalyzer:
    """
    Analyze dataset characteristics for optimal training config
    """

    @staticmethod
    def analyze_dataset(dataset_dir: Path) -> Dict[str, Any]:
        """
        Analyze dataset directory to extract metadata

        Args:
            dataset_dir: Path to Kohya-format dataset directory
                         (expects {repeat}_{concept}/ subdirectories)

        Returns:
            Dict with dataset analysis:
            - image_count: Total number of images
            - repeat_count: Extracted from directory name
            - concept_name: Extracted from directory name
            - has_captions: Whether .txt files exist
            - steps_per_epoch: Calculated training steps per epoch
        """
        analysis = {
            "image_count": 0,
            "repeat_count": 12,
            "concept_name": "concept",
            "has_captions": False,
            "steps_per_epoch": 0,
            "concept_dirs": []
        }

        if not dataset_dir.exists():
            logging.warning(f"Dataset directory not found: {dataset_dir}")
            return analysis

        # Find concept directories (format: {repeat}_{concept}/)
        concept_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

        for concept_dir in concept_dirs:
            # Parse directory name
            dir_name = concept_dir.name
            if '_' in dir_name:
                try:
                    repeat_str, concept = dir_name.split('_', 1)
                    repeat_count = int(repeat_str)
                    analysis["repeat_count"] = repeat_count
                    analysis["concept_name"] = concept
                except ValueError:
                    logging.warning(f"Could not parse concept dir: {dir_name}")

            # Count images
            images = list(concept_dir.glob("*.png")) + list(concept_dir.glob("*.jpg"))
            analysis["image_count"] += len(images)

            # Check for captions
            captions = list(concept_dir.glob("*.txt"))
            if captions:
                analysis["has_captions"] = True

            analysis["concept_dirs"].append(str(concept_dir))

        # Calculate steps per epoch
        # steps_per_epoch = (num_images * repeat_count) / batch_size
        # We use batch_size = 1 as default
        if analysis["image_count"] > 0:
            analysis["steps_per_epoch"] = analysis["image_count"] * analysis["repeat_count"]

        return analysis


class TrainingConfigGenerator:
    """
    Main class for generating Kohya_ss training configurations
    """

    # Type-specific default parameters
    TYPE_SPECIFIC_PARAMS = {
        'pose': {
            'network_dim': 112,
            'network_alpha': 56,
            'conv_dim': 56,
            'conv_alpha': 28,
            'max_train_epochs': 20,
            'learning_rate': 1.2e-4,
            'unet_lr': 1.2e-4,
            'text_encoder_lr': 6e-5,
            'lr_scheduler_num_cycles': 3,
            'description': 'Pose LoRA - body structure and limb positioning'
        },
        'action': {
            'network_dim': 128,
            'network_alpha': 64,
            'conv_dim': 64,
            'conv_alpha': 32,
            'max_train_epochs': 25,
            'learning_rate': 1.0e-4,
            'unet_lr': 1.0e-4,
            'text_encoder_lr': 5e-5,
            'lr_scheduler_num_cycles': 4,
            'description': 'Action LoRA - dynamic motion and movement'
        },
        'expression': {
            'network_dim': 96,
            'network_alpha': 48,
            'conv_dim': 48,
            'conv_alpha': 24,
            'max_train_epochs': 25,
            'learning_rate': 1.0e-4,
            'unet_lr': 1.0e-4,
            'text_encoder_lr': 5e-5,
            'lr_scheduler_num_cycles': 3,
            'description': 'Expression LoRA - facial emotions and features'
        },
        'identity': {
            'network_dim': 96,
            'network_alpha': 48,
            'conv_dim': 48,
            'conv_alpha': 24,
            'max_train_epochs': 20,
            'learning_rate': 1.0e-4,
            'unet_lr': 1.0e-4,
            'text_encoder_lr': 5e-5,
            'lr_scheduler_num_cycles': 3,
            'description': 'Identity LoRA - character appearance and identity'
        }
    }

    def __init__(self, lora_type: Optional[str] = None):
        """
        Initialize training config generator.

        Args:
            lora_type: Optional LoRA type (pose/action/expression/identity)
                      If provided, uses type-specific defaults
        """
        self.analyzer = DatasetAnalyzer()
        self.lora_type = lora_type

        if lora_type and lora_type not in self.TYPE_SPECIFIC_PARAMS:
            logging.warning(f"Unknown LoRA type: {lora_type}. Using generic defaults.")
            self.lora_type = None

    def generate_config(
        self,
        base_model_path: str,
        dataset_dir: Path,
        output_lora_dir: Path,
        character_name: str,
        **overrides
    ) -> TrainingConfig:
        """
        Generate complete training configuration

        Args:
            base_model_path: Path to SDXL base model
            dataset_dir: Path to training dataset (Kohya format)
            output_lora_dir: Where to save LoRA checkpoints
            character_name: Character/concept name
            **overrides: Optional parameter overrides

        Returns:
            Complete TrainingConfig object
        """
        # Analyze dataset
        dataset_info = self.analyzer.analyze_dataset(dataset_dir)

        logging.info(f"Dataset analysis:")
        logging.info(f"  Images: {dataset_info['image_count']}")
        logging.info(f"  Repeat count: {dataset_info['repeat_count']}")
        logging.info(f"  Steps per epoch: {dataset_info['steps_per_epoch']}")

        # Create output directories
        output_lora_dir = Path(output_lora_dir)
        output_lora_dir.mkdir(parents=True, exist_ok=True)

        logging_dir = output_lora_dir / "logs"
        logging_dir.mkdir(parents=True, exist_ok=True)

        # Determine optimal training parameters based on dataset size
        optimal_params = self._calculate_optimal_params(dataset_info)

        # Get type-specific defaults if lora_type is set
        type_defaults = {}
        if self.lora_type:
            type_defaults = self.TYPE_SPECIFIC_PARAMS[self.lora_type].copy()
            logging.info(f"Using {self.lora_type} LoRA defaults: {type_defaults['description']}")

        # Create base config
        config = TrainingConfig(
            # Model
            pretrained_model_name_or_path=base_model_path,

            # Paths
            train_data_dir=str(dataset_dir),
            output_dir=str(output_lora_dir),
            output_name=f"{character_name}_lora_sdxl",
            logging_dir=str(logging_dir),
            log_prefix=f"{character_name}_sdxl",

            # Training parameters (use calculated optimal values or type-specific)
            max_train_epochs=type_defaults.get('max_train_epochs', optimal_params["max_epochs"]),
            save_every_n_epochs=optimal_params["save_interval"],
            lr_warmup_steps=optimal_params["warmup_steps"],

            # Network parameters (type-specific if available)
            network_dim=type_defaults.get('network_dim', 64),
            network_alpha=type_defaults.get('network_alpha', 32),
            learning_rate=type_defaults.get('learning_rate', 0.0001),
            unet_lr=type_defaults.get('unet_lr', 0.0001),
            text_encoder_lr=type_defaults.get('text_encoder_lr', 0.00006),
            lr_scheduler_num_cycles=type_defaults.get('lr_scheduler_num_cycles', 3),

            # Metadata
            comment=f"SDXL LoRA training for {character_name}" + (f" - {type_defaults.get('description', '')}" if self.lora_type else ""),
            character_name=character_name,
            dataset_size=dataset_info["image_count"],
            repeat_count=dataset_info["repeat_count"],
        )

        # Apply overrides (user-specified values override type defaults)
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logging.debug(f"Override: {key} = {value}")

        return config

    def _calculate_optimal_params(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimal training parameters based on dataset characteristics

        Args:
            dataset_info: Dataset analysis dict

        Returns:
            Dict with optimal parameters:
            - max_epochs: Recommended maximum training epochs
            - save_interval: How often to save checkpoints
            - warmup_steps: Learning rate warmup steps
        """
        image_count = dataset_info["image_count"]
        steps_per_epoch = dataset_info["steps_per_epoch"]

        # Epoch calculation based on dataset size
        if image_count < 50:
            max_epochs = 6  # Small dataset needs more epochs
            save_interval = 2
        elif image_count < 100:
            max_epochs = 5
            save_interval = 2
        elif image_count < 200:
            max_epochs = 4
            save_interval = 2
        else:
            max_epochs = 3  # Large dataset needs fewer epochs
            save_interval = 1

        # Warmup steps: ~5% of first epoch
        warmup_steps = max(50, int(steps_per_epoch * 0.05))

        return {
            "max_epochs": max_epochs,
            "save_interval": save_interval,
            "warmup_steps": warmup_steps,
        }

    def save_config(
        self,
        config: TrainingConfig,
        output_path: Path
    ) -> Path:
        """
        Save config to TOML file

        Args:
            config: TrainingConfig object
            output_path: Where to save TOML file

        Returns:
            Path to saved config file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to TOML format
        toml_dict = config.to_toml_dict()

        # Add header comment
        header = f"""# SDXL Character LoRA Training Config - AUTO-GENERATED
# Character: {config.character_name}
# Dataset: {config.dataset_size} images × {config.repeat_count} repeats = {config.dataset_size * config.repeat_count} steps/epoch
# Target: {config.max_train_epochs} epochs ({config.dataset_size * config.repeat_count * config.max_train_epochs} total steps)
# Generated: {Path(__file__).name}

"""

        # Write TOML file
        with open(output_path, 'w') as f:
            f.write(header)
            toml.dump(toml_dict, f)

        logging.info(f"Saved training config to: {output_path}")

        return output_path

    def validate_config(self, config: TrainingConfig) -> bool:
        """
        Validate configuration before training

        Args:
            config: TrainingConfig to validate

        Returns:
            True if valid, raises ValueError if invalid
        """
        errors = []

        # Check paths exist
        if not Path(config.pretrained_model_name_or_path).exists():
            errors.append(f"Base model not found: {config.pretrained_model_name_or_path}")

        if not Path(config.train_data_dir).exists():
            errors.append(f"Training data dir not found: {config.train_data_dir}")

        # Check parameters are sensible
        if config.network_dim < 1 or config.network_dim > 256:
            errors.append(f"network_dim should be 1-256, got {config.network_dim}")

        if config.learning_rate <= 0 or config.learning_rate > 0.01:
            errors.append(f"learning_rate seems wrong: {config.learning_rate}")

        if config.max_train_epochs < 1:
            errors.append(f"max_train_epochs must be >= 1, got {config.max_train_epochs}")

        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Config validation failed:\n{error_msg}")

        logging.info("✅ Config validation passed")
        return True


def main():
    """CLI interface for training config generation"""
    parser = argparse.ArgumentParser(
        description="Generate Kohya_ss SDXL LoRA training configuration"
    )

    # Required arguments
    parser.add_argument("--base-model", type=str, required=True,
                       help="Path to SDXL base model (.safetensors)")
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="Path to training dataset (Kohya format)")
    parser.add_argument("--output-lora-dir", type=str, required=True,
                       help="Output directory for LoRA checkpoints")
    parser.add_argument("--character-name", type=str, required=True,
                       help="Character/concept name")

    # LoRA type (IMPORTANT for type-specific defaults)
    parser.add_argument("--lora-type", type=str,
                       choices=['pose', 'action', 'expression', 'identity'],
                       help="LoRA type (uses optimized defaults for each type)")

    # Optional config file output
    parser.add_argument("--config-output", type=str,
                       help="Where to save generated TOML config (default: auto)")

    # Training parameter overrides
    parser.add_argument("--network-dim", type=int, default=64,
                       help="LoRA network dimension (default: 64)")
    parser.add_argument("--network-alpha", type=int, default=32,
                       help="LoRA network alpha (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                       help="Learning rate (default: 0.0001)")
    parser.add_argument("--max-train-epochs", type=int,
                       help="Maximum training epochs (default: auto)")
    parser.add_argument("--train-batch-size", type=int, default=1,
                       help="Training batch size (default: 1)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2,
                       help="Gradient accumulation steps (default: 2)")

    # Advanced options
    parser.add_argument("--optimizer", type=str, default="AdamW8bit",
                       help="Optimizer type (default: AdamW8bit)")
    parser.add_argument("--lr-scheduler", type=str, default="cosine_with_restarts",
                       help="LR scheduler (default: cosine_with_restarts)")
    parser.add_argument("--min-snr-gamma", type=float, default=5.0,
                       help="Min SNR gamma (default: 5.0)")
    parser.add_argument("--noise-offset", type=float, default=0.05,
                       help="Noise offset (default: 0.05)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create generator with lora_type
    generator = TrainingConfigGenerator(lora_type=args.lora_type)

    # Build overrides dict (only override if user explicitly specified)
    overrides = {}

    # Only override if user specified different from type default
    if args.network_dim != 64:  # 64 is the CLI default, not type default
        overrides["network_dim"] = args.network_dim
    if args.network_alpha != 32:
        overrides["network_alpha"] = args.network_alpha
    if args.learning_rate != 0.0001:
        overrides["learning_rate"] = args.learning_rate
    if args.train_batch_size != 1:
        overrides["train_batch_size"] = args.train_batch_size
    if args.gradient_accumulation_steps != 2:
        overrides["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.optimizer != "AdamW8bit":
        overrides["optimizer_type"] = args.optimizer
    if args.lr_scheduler != "cosine_with_restarts":
        overrides["lr_scheduler"] = args.lr_scheduler
    if args.min_snr_gamma != 5.0:
        overrides["min_snr_gamma"] = args.min_snr_gamma
    if args.noise_offset != 0.05:
        overrides["noise_offset"] = args.noise_offset

    if args.max_train_epochs:
        overrides["max_train_epochs"] = args.max_train_epochs

    # Generate config
    try:
        config = generator.generate_config(
            base_model_path=args.base_model,
            dataset_dir=Path(args.dataset_dir),
            output_lora_dir=Path(args.output_lora_dir),
            character_name=args.character_name,
            **overrides
        )

        # Validate
        generator.validate_config(config)

        # Determine output path
        if args.config_output:
            config_path = Path(args.config_output)
        else:
            config_path = Path(args.output_lora_dir) / f"{args.character_name}_training_config.toml"

        # Save
        saved_path = generator.save_config(config, config_path)

        print(f"\n{'='*80}")
        print(f"✅ Training config generated successfully!")
        print(f"{'='*80}")
        print(f"Config file: {saved_path}")
        print(f"Character: {config.character_name}")
        print(f"Dataset: {config.dataset_size} images")
        print(f"Training epochs: {config.max_train_epochs}")
        print(f"Steps per epoch: {config.dataset_size * config.repeat_count}")
        print(f"Total steps: {config.dataset_size * config.repeat_count * config.max_train_epochs}")
        print(f"{'='*80}\n")

    except Exception as e:
        logging.error(f"Failed to generate config: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
