#!/usr/bin/env python3
"""
Pure Iterative LoRA Training (No Evaluation)

Focuses on training only with optimized parameters for facial consistency.
Evaluation is skipped - user can manually test models later.
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime

# Training parameters optimized for facial consistency
TRAINING_PARAMS = {
    'learning_rate': 8.0e-5,      # Lower for stability
    'text_encoder_lr': 6.0e-5,    # Higher relative ratio
    'network_dim': 96,
    'network_alpha': 48,
    'max_train_epochs': 10,        # Reduced from 12
    'batch_size': 8,
    'lr_scheduler': 'cosine',
    'optimizer_type': 'AdamW',
}

# Paths
DATASET_BASE = Path('/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset')
OUTPUT_BASE = Path('/mnt/data/ai_data/models/lora/luca/iterative_overnight_v6')
BASE_MODEL = '/mnt/c/AI_LLM_projects/ai_warehouse/tool-caches/huggingface/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
SD_SCRIPTS = Path('/mnt/c/AI_LLM_projects/ai_warehouse/sd-scripts')

CHARACTERS = ['luca_human', 'alberto_human']
MAX_ITERATIONS = 15  # Fixed number
TIME_LIMIT_HOURS = 14.0

def create_toml_config(character: str, iteration: int, output_dir: Path) -> Path:
    """Create TOML config for training"""

    config_dir = output_dir / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = config_dir / f'{character}_iter{iteration}.toml'

    dataset_dir = DATASET_BASE / character
    images_dir = dataset_dir / 'images'

    # Count images
    num_images = len(list(images_dir.glob('*.png'))) + len(list(images_dir.glob('*.jpg')))

    # Calculate steps
    steps_per_epoch = num_images // TRAINING_PARAMS['batch_size']
    total_steps = steps_per_epoch * TRAINING_PARAMS['max_train_epochs']

    config_content = f"""
# LoRA Training Config - {character} - Iteration {iteration}
# Optimized for facial consistency

[general]
pretrained_model_name_or_path = "{BASE_MODEL}"
output_dir = "{output_dir / character}"
output_name = "{character}_iter{iteration}_v1"
save_model_as = "safetensors"
mixed_precision = "fp16"
save_precision = "fp16"

[dataset]
resolution = 512
batch_size = {TRAINING_PARAMS['batch_size']}
max_train_epochs = {TRAINING_PARAMS['max_train_epochs']}
save_every_n_epochs = 4

[[datasets]]
resolution = 512
batch_size = {TRAINING_PARAMS['batch_size']}

  [[datasets.subsets]]
  image_dir = "{images_dir}"
  num_repeats = 1
  shuffle_caption = true
  keep_tokens = 3
  caption_extension = ".txt"

[network]
network_module = "networks.lora"
network_dim = {TRAINING_PARAMS['network_dim']}
network_alpha = {TRAINING_PARAMS['network_alpha']}

[optimizer]
optimizer_type = "{TRAINING_PARAMS['optimizer_type']}"
learning_rate = {TRAINING_PARAMS['learning_rate']}
unet_lr = {TRAINING_PARAMS['learning_rate']}
text_encoder_lr = {TRAINING_PARAMS['text_encoder_lr']}
lr_scheduler = "{TRAINING_PARAMS['lr_scheduler']}"
lr_warmup_steps = 50

[training]
gradient_checkpointing = true
gradient_accumulation_steps = 2
max_data_loader_n_workers = 8
cache_latents = true
cache_latents_to_disk = true
seed = 42
clip_skip = 2

[logging]
logging_dir = "{output_dir / 'logs'}"
log_prefix = "{character}_iter{iteration}"
"""

    config_file.write_text(config_content)
    print(f"  ✓ Config created: {config_file}")
    return config_file

def train_character(character: str, iteration: int, output_base: Path) -> bool:
    """Train one character for one iteration"""

    print("\n" + "#"*70)
    print(f"# ITERATION {iteration} - {character.upper()}")
    print("#"*70 + "\n")

    # Create config
    config_file = create_toml_config(character, iteration, output_base)

    # Training command
    cmd = [
        'conda', 'run', '-n', 'ai_env',
        'python', str(SD_SCRIPTS / 'train_network.py'),
        '--config_file', str(config_file)
    ]

    print(f"Starting training...")
    print(f"  Character: {character}")
    print(f"  Iteration: {iteration}")
    print(f"  Epochs: {TRAINING_PARAMS['max_train_epochs']}")
    print(f"  Learning Rate: {TRAINING_PARAMS['learning_rate']}")
    print(f"  Text Encoder LR: {TRAINING_PARAMS['text_encoder_lr']}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"\n✓ Training completed in {elapsed/60:.1f} minutes\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed: {e}\n")
        return False

def main():
    print("\n" + "="*70)
    print("PURE ITERATIVE LORA TRAINING")
    print("="*70)
    print(f"Characters: {', '.join(CHARACTERS)}")
    print(f"Output: {OUTPUT_BASE}")
    print(f"Max Iterations: {MAX_ITERATIONS}")
    print(f"Time Limit: {TIME_LIMIT_HOURS} hours")
    print("="*70 + "\n")

    # Create output directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Verify datasets
    for character in CHARACTERS:
        dataset_dir = DATASET_BASE / character / 'images'
        if not dataset_dir.exists():
            print(f"✗ Dataset not found: {dataset_dir}")
            return

        num_images = len(list(dataset_dir.glob('*.png'))) + len(list(dataset_dir.glob('*.jpg')))
        caption_dir = DATASET_BASE / character / 'captions'
        num_captions = len(list(caption_dir.glob('*.txt'))) if caption_dir.exists() else 0

        print(f"✓ {character}: {num_images} images, {num_captions} captions")

    print("\n" + "="*70)
    print(f"STARTING TRAINING - {TIME_LIMIT_HOURS} hours")
    print("="*70 + "\n")

    start_time = time.time()
    iteration = 1

    while iteration <= MAX_ITERATIONS:
        # Check time limit
        elapsed_hours = (time.time() - start_time) / 3600
        remaining_hours = TIME_LIMIT_HOURS - elapsed_hours

        if remaining_hours <= 0:
            print(f"\n⏰ Time limit reached ({TIME_LIMIT_HOURS}h)")
            break

        print("\n" + "▓"*70)
        print(f"▓ ITERATION ROUND {iteration}")
        print(f"▓ Time: {elapsed_hours:.1f}h elapsed / {remaining_hours:.1f}h remaining")
        print("▓"*70 + "\n")

        # Train both characters
        success_count = 0
        for character in CHARACTERS:
            if train_character(character, iteration, OUTPUT_BASE):
                success_count += 1

        if success_count == 0:
            print("✗ All trainings failed, stopping")
            break

        print(f"\n✓ Iteration {iteration} completed ({success_count}/{len(CHARACTERS)} successful)\n")
        iteration += 1

    total_time = (time.time() - start_time) / 3600
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Total time: {total_time:.2f} hours")
    print(f"Iterations completed: {iteration - 1}")
    print(f"Output directory: {OUTPUT_BASE}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
