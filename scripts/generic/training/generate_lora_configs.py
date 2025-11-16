#!/usr/bin/env python3
"""
Generate LoRA Training Configs for Each Character

Automatically generates TOML configuration files for each character
based on the curated dataset structure.
"""

import argparse
import json
from pathlib import Path
from typing import Dict


LORA_CONFIG_TEMPLATE = """# LoRA Training Config for {character_name}
# Generated for Pixar Luca (2021) - 3D Animation Character
# Compatible with kohya_ss sd-scripts

# Model parameters
pretrained_model_name_or_path = "{base_model}"
v2 = false
v_parameterization = false

# Training output
output_dir = "{output_dir}"
output_name = "{output_name}"
save_model_as = "safetensors"
save_precision = "fp16"

# Training parameters
max_train_epochs = {epochs}
save_every_n_epochs = {save_every}

learning_rate = {lr}
unet_lr = {lr}
text_encoder_lr = {te_lr}
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = {warmup_steps}

optimizer_type = "AdamW8bit"
optimizer_args = ["weight_decay=0.01"]

# Network settings (LoRA)
network_module = "networks.lora"
network_dim = {network_dim}
network_alpha = {network_alpha}

# Training precision
mixed_precision = "fp16"
full_fp16 = false

# Logging
logging_dir = "{logging_dir}"
log_prefix = "{log_prefix}"

# Performance optimization
gradient_checkpointing = true
gradient_accumulation_steps = 1
max_data_loader_n_workers = 4
persistent_data_loader_workers = true

# Memory optimization
xformers = true
cache_latents = true
cache_latents_to_disk = true

# Misc
seed = 42
clip_skip = 2  # Optimized for 3D content

# Dataset configuration
[[datasets]]
resolution = [512, 512]
batch_size = {batch_size}
enable_bucket = true
min_bucket_reso = 384
max_bucket_reso = 768
bucket_reso_steps = 64
bucket_no_upscale = false

  [[datasets.subsets]]
  image_dir = "{image_dir}"
  num_repeats = 1
  shuffle_caption = true
  keep_tokens = 3  # Keep "a 3d animated character" prefix
  caption_extension = ".txt"
  color_aug = false  # Disable for 3D - preserve PBR materials
  flip_aug = false   # Disable for 3D - preserve asymmetric features

# Training notes:
# - {num_images} images in dataset
# - Training at {lr} learning rate for {epochs} epochs
# - Checkpoints saved every {save_every} epochs
# - 3D-optimized: no color/flip augmentation, clip_skip=2
"""


def count_images(image_dir: Path) -> int:
    """Count images in directory"""
    if not image_dir.exists():
        return 0
    return len(list(image_dir.glob('*.[pj][np]g')))


def estimate_training_params(num_images: int) -> Dict:
    """Estimate optimal training parameters based on dataset size"""

    # Determine epochs and learning rate based on dataset size
    if num_images < 100:
        epochs = 20
        lr = 1.5e-4
        warmup_steps = 100
    elif num_images < 300:
        epochs = 15
        lr = 1e-4
        warmup_steps = 200
    elif num_images < 600:
        epochs = 12
        lr = 8e-5
        warmup_steps = 300
    else:
        epochs = 10
        lr = 5e-5
        warmup_steps = 400

    # Text encoder LR is typically half of main LR
    te_lr = lr / 2

    # Batch size based on expected VRAM (assume 16GB)
    if num_images < 200:
        batch_size = 4
    else:
        batch_size = 3

    # Network dimensions
    network_dim = 32  # Standard for character LoRA
    network_alpha = 16  # Half of dim

    return {
        'epochs': epochs,
        'lr': lr,
        'te_lr': te_lr,
        'warmup_steps': warmup_steps,
        'batch_size': batch_size,
        'network_dim': network_dim,
        'network_alpha': network_alpha,
        'save_every': max(3, epochs // 5)  # Save 5 checkpoints
    }


def format_character_name(char_id: str) -> str:
    """Format character ID to readable name"""
    return char_id.replace('_', ' ').title()


def generate_config(
    character_dir: Path,
    output_config_dir: Path,
    base_model: str,
    lora_output_dir: Path,
    logging_dir: Path
) -> Dict:
    """Generate LoRA config for a single character"""

    char_id = character_dir.name
    char_name = format_character_name(char_id)

    image_dir = character_dir / 'images'
    num_images = count_images(image_dir)

    if num_images == 0:
        return None

    # Estimate training parameters
    params = estimate_training_params(num_images)

    # Prepare paths
    char_output_dir = lora_output_dir / char_id
    output_name = f"luca_{char_id}_v1"
    log_prefix = f"luca_{char_id}"

    # Generate config content
    config_content = LORA_CONFIG_TEMPLATE.format(
        character_name=char_name,
        base_model=base_model,
        output_dir=str(char_output_dir),
        output_name=output_name,
        epochs=params['epochs'],
        save_every=params['save_every'],
        lr=params['lr'],
        te_lr=params['te_lr'],
        warmup_steps=params['warmup_steps'],
        network_dim=params['network_dim'],
        network_alpha=params['network_alpha'],
        logging_dir=str(logging_dir / char_id),
        log_prefix=log_prefix,
        batch_size=params['batch_size'],
        image_dir=str(image_dir),
        num_images=num_images
    )

    # Write config file
    config_path = output_config_dir / f"{char_id}.toml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    return {
        'character': char_id,
        'character_name': char_name,
        'config_path': str(config_path),
        'num_images': num_images,
        'epochs': params['epochs'],
        'learning_rate': params['lr'],
        'batch_size': params['batch_size']
    }


def main():
    parser = argparse.ArgumentParser(description="Generate LoRA training configs for each character")
    parser.add_argument(
        '--dataset-dir',
        type=str,
        required=True,
        help='Curated dataset directory (output from curator)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for config files'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='/mnt/c/AI_LLM_projects/ai_warehouse/models/base/stable-diffusion-v1-5',
        help='Base model path or HuggingFace ID'
    )
    parser.add_argument(
        '--lora-output-dir',
        type=str,
        default='/mnt/data/ai_data/models/lora/luca',
        help='Base directory for LoRA outputs'
    )
    parser.add_argument(
        '--logging-dir',
        type=str,
        default='/mnt/data/ai_data/lora_evaluation/logs/luca',
        help='Logging directory'
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    lora_output_dir = Path(args.lora_output_dir)
    logging_dir = Path(args.logging_dir)

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"GENERATING LORA TRAINING CONFIGS")
    print(f"{'='*70}")
    print(f"Dataset:    {dataset_dir}")
    print(f"Output:     {output_dir}")
    print(f"Base Model: {args.base_model}")
    print(f"{'='*70}\n")

    # Generate configs for each character
    all_configs = []

    for char_dir in sorted(dataset_dir.iterdir()):
        if not char_dir.is_dir():
            continue

        print(f"Processing {char_dir.name}...", end=' ')

        config_info = generate_config(
            char_dir,
            output_dir,
            args.base_model,
            lora_output_dir,
            logging_dir
        )

        if config_info:
            all_configs.append(config_info)
            print(f"✓ ({config_info['num_images']} images, {config_info['epochs']} epochs)")
        else:
            print("✗ (no images found)")

    # Save manifest
    manifest = {
        'project': 'Luca (2021) - Pixar 3D Characters',
        'base_model': args.base_model,
        'generated_at': str(Path(__file__).parent),
        'total_characters': len(all_configs),
        'configs': all_configs
    }

    manifest_path = output_dir / 'training_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"CONFIG GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total configs generated: {len(all_configs)}")
    print(f"Manifest saved: {manifest_path}")
    print(f"\nReady to start training!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
