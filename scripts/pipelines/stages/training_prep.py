"""
Stage 6: Training Prep

Organizes dataset into Kohya_ss training format:
/path/to/training_data/{repeats}_{class_name}/
  - image001.jpg
  - image001.txt
  - ...
  - metadata.json
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import shutil
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.pipelines.stages.base_stage import BaseStage


class TrainingPrepStage(BaseStage):
    """Stage 6: Prepare dataset for LoRA training"""

    def validate_config(self) -> bool:
        """Validate configuration"""
        required_keys = ['input_dir', 'output_dir', 'character_name', 'class_name']

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        input_dir = Path(self.config['input_dir'])
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")

        return True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute training prep stage"""

        self.logger.info("="*60)
        self.logger.info("STAGE 6: Training Preparation")
        self.logger.info("="*60)

        # Setup paths
        input_dir = Path(self.config['input_dir'])
        output_base = Path(self.config['output_dir'])

        # Build Kohya_ss format directory name: {repeats}_{class_name}
        repeats = self.config.get('repeats', 10)
        class_name = self.config['class_name']
        training_dir = output_base / f"{repeats}_{class_name}"
        training_dir.mkdir(parents=True, exist_ok=True)

        # Collect input images and captions
        image_files = sorted(
            list(input_dir.glob('*.jpg')) +
            list(input_dir.glob('*.png')) +
            list(input_dir.glob('*.jpeg'))
        )

        self.logger.info(f"Preparing {len(image_files)} images for training...")

        # Copy images and captions to training directory
        copied_images = []
        copied_captions = []
        missing_captions = []

        for img_path in tqdm(image_files, desc="Organizing training data"):
            # Copy image
            img_output = training_dir / img_path.name
            shutil.copy2(img_path, img_output)
            copied_images.append(img_output)

            # Copy caption if exists
            caption_path = img_path.with_suffix('.txt')
            if caption_path.exists():
                caption_output = training_dir / caption_path.name
                shutil.copy2(caption_path, caption_output)
                copied_captions.append(caption_output)
            else:
                missing_captions.append(img_path.name)
                self.logger.warning(f"Missing caption for {img_path.name}")

        # Generate dataset metadata
        metadata = {
            'character_name': self.config['character_name'],
            'class_name': class_name,
            'repeats': repeats,
            'num_images': len(copied_images),
            'num_captions': len(copied_captions),
            'missing_captions': len(missing_captions),
            'dataset_format': 'kohya_ss',
            'directory_structure': f"{repeats}_{class_name}/",
            'training_dir': str(training_dir),
            'source_dir': str(input_dir),
            'provenance': {
                'pipeline_version': '1.0',
                'stages_completed': [
                    'face_prefilter',
                    'quality_filter',
                    'augmentation',
                    'diversity_selection',
                    'captioning',
                    'training_prep'
                ]
            },
            'recommended_training_params': {
                'network_dim': self.config.get('network_dim', 32),
                'network_alpha': self.config.get('network_alpha', 16),
                'learning_rate': self.config.get('learning_rate', 1e-4),
                'lr_scheduler': 'cosine_with_restarts',
                'train_batch_size': 1,
                'max_train_epochs': self.config.get('max_epochs', 10),
                'save_every_n_epochs': 1,
                'mixed_precision': 'fp16',
                'optimizer_type': 'AdamW8bit',
                'clip_skip': 2,
                'enable_bucket': True,
                'min_bucket_reso': 256,
                'max_bucket_reso': 1024,
                'bucket_reso_steps': 64
            },
            'augmentation_settings': {
                'flip_aug': False,  # CRITICAL: NO horizontal flip for 3D
                'color_aug': False,  # CRITICAL: NO color jitter for 3D
                'random_crop': False  # Already done in augmentation stage
            },
            '3d_specific_notes': [
                'Dataset prepared for Pixar-style 3D character',
                'NO horizontal flip (preserves asymmetric features)',
                'NO color augmentation (preserves PBR materials)',
                'Captions emphasize smooth shading and lighting',
                'Target: 200-500 training images for 3D characters'
            ]
        }

        # Save metadata in training directory
        metadata_file = training_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save missing captions list if any
        if missing_captions:
            missing_file = training_dir / 'missing_captions.txt'
            with open(missing_file, 'w') as f:
                f.write('\n'.join(missing_captions))

        # Generate training command template
        command_template = self._generate_training_command(metadata)
        command_file = training_dir / 'train_command.sh'
        with open(command_file, 'w') as f:
            f.write(command_template)
        command_file.chmod(0o755)

        self.logger.info(f"Metadata saved to {metadata_file}")
        self.logger.info(f"Training command saved to {command_file}")

        # Print summary
        self.logger.info("\nTraining Preparation Summary:")
        self.logger.info(f"  Character: {self.config['character_name']}")
        self.logger.info(f"  Class name: {class_name}")
        self.logger.info(f"  Repeats: {repeats}")
        self.logger.info(f"  Training directory: {training_dir}")
        self.logger.info(f"  Images: {len(copied_images)}")
        self.logger.info(f"  Captions: {len(copied_captions)}")
        if missing_captions:
            self.logger.warning(f"  Missing captions: {len(missing_captions)}")
        self.logger.info(f"\n  Format: Kohya_ss compatible")
        self.logger.info(f"  Directory pattern: {repeats}_{class_name}/")
        self.logger.info(f"\n  Ready for training!")
        self.logger.info(f"  Run: bash {command_file}")

        return {
            'training_dir': training_dir,
            'output_dir': output_base,
            'metadata': metadata,
            'num_images': len(copied_images),
            'num_captions': len(copied_captions),
            'command_file': command_file
        }

    def _generate_training_command(self, metadata: Dict[str, Any]) -> str:
        """
        Generate training command script

        Args:
            metadata: Training metadata

        Returns:
            Shell script content
        """
        training_dir = metadata['training_dir']
        character_name = metadata['character_name']
        params = metadata['recommended_training_params']

        # Get project root and model paths
        project_root = Path(__file__).resolve().parents[3]
        model_base = "/mnt/data/ai_data/models/stable-diffusion"
        output_base = f"/mnt/data/ai_data/models/lora/{character_name}"

        command = f"""#!/bin/bash
# Training command for {character_name}
# Generated by Luca Dataset Preparation Pipeline

# Activate environment
conda activate ai_env

# Navigate to sd-scripts directory
cd "{project_root}/sd-scripts" || exit 1

# Set training parameters
TRAIN_DIR="{training_dir}"
OUTPUT_DIR="{output_base}"
MODEL_PATH="{model_base}/sd-v1-5-fp16.safetensors"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
accelerate launch --num_cpu_threads_per_process=8 train_network.py \\
  --pretrained_model_name_or_path="$MODEL_PATH" \\
  --train_data_dir="$TRAIN_DIR" \\
  --output_dir="$OUTPUT_DIR" \\
  --output_name="{character_name}_lora" \\
  --network_module=networks.lora \\
  --network_dim={params['network_dim']} \\
  --network_alpha={params['network_alpha']} \\
  --learning_rate={params['learning_rate']} \\
  --lr_scheduler={params['lr_scheduler']} \\
  --train_batch_size={params['train_batch_size']} \\
  --max_train_epochs={params['max_train_epochs']} \\
  --save_every_n_epochs={params['save_every_n_epochs']} \\
  --mixed_precision={params['mixed_precision']} \\
  --optimizer_type={params['optimizer_type']} \\
  --clip_skip={params['clip_skip']} \\
  --enable_bucket \\
  --min_bucket_reso={params['min_bucket_reso']} \\
  --max_bucket_reso={params['max_bucket_reso']} \\
  --bucket_reso_steps={params['bucket_reso_steps']} \\
  --save_model_as=safetensors \\
  --save_precision=fp16 \\
  --seed=42 \\
  --cache_latents \\
  --prior_loss_weight=1.0 \\
  --logging_dir="$OUTPUT_DIR/logs" \\
  --log_prefix="{character_name}_" \\
  --xformers

# CRITICAL 3D-specific settings:
# - NO --flip_aug (preserves asymmetric features)
# - NO --color_aug (preserves PBR materials)
# - enable_bucket=True (handles varied aspect ratios)
# - clip_skip=2 (standard for SD 1.5)

echo "Training complete! Models saved to: $OUTPUT_DIR"
"""

        return command
