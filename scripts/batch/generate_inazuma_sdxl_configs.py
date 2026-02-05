#!/usr/bin/env python3
"""
Generate SDXL LoRA training configs for all Inazuma Eleven characters.
Adapted from 3D character training config to 2D anime optimization.
"""

import os
from pathlib import Path
from typing import Dict, List

# Character metadata
CHARACTERS = [
    {
        "id": "endou_mamoru",
        "dataset_name": "1_endou_mamoru",
        "canonical_name": "Endou Mamoru",
        "jp_name": "円堂守",
        "english_name": "Mark Evans",
        "trigger": "inazuma_endou_mamoru",
        "position": "goalkeeper",
        "element": "mountain",
        "key_features": "orange_headband, spiky_brown_hair, big_round_brown_eyes"
    },
    {
        "id": "gouenji_shuuya",
        "dataset_name": "1_gouenji_shuuya",
        "canonical_name": "Gouenji Shuuya",
        "jp_name": "豪炎寺修也",
        "english_name": "Axel Blaze",
        "trigger": "inazuma_gouenji_shuuya",
        "position": "forward",
        "element": "fire",
        "key_features": "cool_expression, white_headband, silver_hair"
    },
    {
        "id": "fudou_akio",
        "dataset_name": "1_fudou_akio",
        "canonical_name": "Fudou Akio",
        "jp_name": "不動明王",
        "english_name": "Axel Blaze",  # Placeholder, adjust if known
        "trigger": "inazuma_fudou_akio",
        "position": "midfielder",
        "element": "mountain",
        "key_features": "mohawk_hair, intimidating_expression, sharp_eyes"
    },
    {
        "id": "matsukaze_tenma",
        "dataset_name": "1_matsukaze_tenma",
        "canonical_name": "Matsukaze Tenma",
        "jp_name": "松風天馬",
        "english_name": "Arion Sherwind",
        "trigger": "inazuma_matsukaze_tenma",
        "position": "midfielder",
        "element": "wind",
        "key_features": "brown_spiky_hair, energetic_expression, goggles"
    },
    {
        "id": "inamori_asuto",
        "dataset_name": "1_inamori_asuto",
        "canonical_name": "Inamori Asuto",
        "jp_name": "稲森明日人",
        "english_name": "Asuto Inamori",
        "trigger": "inazuma_inamori_asuto",
        "position": "forward",
        "element": "wind",
        "key_features": "orange_hair, cheerful_expression, bright_eyes"
    },
    {
        "id": "nosaka_yuuma",
        "dataset_name": "1_nosaka_yuuma",
        "canonical_name": "Nosaka Yuuma",
        "jp_name": "野坂悠馬",
        "english_name": "Yuuma Nosaka",
        "trigger": "inazuma_nosaka_yuuma",
        "position": "midfielder",
        "element": "fire",
        "key_features": "purple_hair, strategic_expression, glasses"
    },
    {
        "id": "utsunomiya_toramaru",
        "dataset_name": "1_utsunomiya_toramaru",
        "canonical_name": "Utsunomiya Toramaru",
        "jp_name": "宇都宮虎丸",
        "english_name": "Toramaru Utsunomiya",
        "trigger": "inazuma_utsunomiya_toramaru",
        "position": "forward",
        "element": "fire",
        "key_features": "spiky_orange_hair, feral_expression, energetic"
    }
]

# Base paths
BASE_DATASET_PATH = "/mnt/data/datasets/general/inazuma-eleven/lora_data/training_data_sdxl"
BASE_OUTPUT_PATH = "/mnt/data/ai_data/models/lora_sdxl/inazuma-eleven"
BASE_MODEL_PATH = "/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
VAE_PATH = "/mnt/c/ai_models/stable-diffusion/vae/sdxl_vae.safetensors"
CONFIG_OUTPUT_DIR = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl")

# Training hyperparameters (2D anime optimized)
TRAINING_CONFIG = {
    # NOTE: Values are chosen to avoid NaN loss in Kohya SDXL training.
    "learning_rate": 5e-5,
    "unet_lr": 5e-5,
    "text_encoder_lr": 1e-5,
    "lr_scheduler": "cosine_with_restarts",
    "lr_scheduler_num_cycles": 2,
    "lr_warmup_steps": 100,
    "network_dim": 32,
    "network_alpha": 16,
    "optimizer_type": "AdamW",
    "optimizer_args": ["weight_decay=0.01", "betas=0.9,0.999"],
    "train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "max_train_epochs": 8,
    "dataset_repeats": 10,  # 200 images × 10 = 2000 steps/epoch
    "caption_dropout_rate": 0.1,
    "caption_dropout_every_n_epochs": 1,
    "keep_tokens": 2,  # Keep "inazuma_eleven, inazuma_{character}"
    "persistent_data_loader_workers": True,
    "max_data_loader_n_workers": 4,
    "vae_batch_size": 2,
    "mixed_precision": "bf16",
    "full_bf16": True,
    "cache_latents": True,
    "cache_latents_to_disk": False,  # RAM cache for speed
    "gradient_checkpointing": True,
    "no_half_vae": True,
    "noise_offset": 0.05,
    "adaptive_noise_scale": 0.00357,
    "max_grad_norm": 0.5,
    "save_every_n_epochs": 2,
    "save_last_n_epochs_state": 3,
    "save_model_as": "safetensors",
    "save_precision": "bf16",
    "clip_skip": 2,
    "max_token_length": 225,
    "seed": 42,
    "resolution": "1024,1024",
    "enable_bucket": True,
    "bucket_reso_steps": 64,
    "bucket_no_upscale": True,
    "min_bucket_reso": 768,
    "max_bucket_reso": 1280
}


def generate_toml_config(character: Dict) -> str:
    """Generate TOML configuration for a character."""

    char_id = character["id"]
    trigger = character["trigger"]
    canonical_name = character["canonical_name"]

    # Paths
    train_data_dir = f"{BASE_DATASET_PATH}/{char_id}_identity"
    output_dir = f"{BASE_OUTPUT_PATH}/{char_id}_identity"
    output_name = f"inazuma_{char_id}_lora_sdxl"
    logging_dir = f"{output_dir}/logs"

    toml_content = f'''# Inazuma Eleven SDXL LoRA Training Config
# Character: {canonical_name} ({character["jp_name"]})
# Generated: 2025-12-19
# Adapted from 3D config to 2D anime optimization

[general]
# Output configuration
output_name = "{output_name}"
output_dir = "{output_dir}"
pretrained_model_name_or_path = "{BASE_MODEL_PATH}"
vae = "{VAE_PATH}"

# Model architecture
v2 = false
v_parameterization = false
sdxl = true

[dataset]
# Dataset paths
train_data_dir = "{train_data_dir}"

# Caption format (this dataset uses .txt caption files)
caption_extension = ".txt"
shuffle_caption = true

# Resolution and bucketing (SDXL optimized)
resolution = "{TRAINING_CONFIG["resolution"]}"
enable_bucket = {str(TRAINING_CONFIG["enable_bucket"]).lower()}
bucket_reso_steps = {TRAINING_CONFIG["bucket_reso_steps"]}
bucket_no_upscale = {str(TRAINING_CONFIG["bucket_no_upscale"]).lower()}
min_bucket_reso = {TRAINING_CONFIG["min_bucket_reso"]}
max_bucket_reso = {TRAINING_CONFIG["max_bucket_reso"]}

# Dataset repeats (200 images × 10 = 2000 steps/epoch)
dataset_repeats = {TRAINING_CONFIG["dataset_repeats"]}

[training]
# Learning rates (2D anime optimized - slightly lower than 3D)
learning_rate = {TRAINING_CONFIG["learning_rate"]}
unet_lr = {TRAINING_CONFIG["unet_lr"]}
text_encoder_lr = {TRAINING_CONFIG["text_encoder_lr"]}

# LR scheduler (cosine with restarts for anime features)
lr_scheduler = "{TRAINING_CONFIG["lr_scheduler"]}"
lr_scheduler_num_cycles = {TRAINING_CONFIG["lr_scheduler_num_cycles"]}
lr_warmup_steps = {TRAINING_CONFIG["lr_warmup_steps"]}

# Gradient clipping (CRITICAL for SDXL stability)
max_grad_norm = {TRAINING_CONFIG["max_grad_norm"]}

# Network architecture (identity-focused)
network_module = "networks.lora"
network_dim = {TRAINING_CONFIG["network_dim"]}
network_alpha = {TRAINING_CONFIG["network_alpha"]}

# Optimizer (memory-efficient)
optimizer_type = "{TRAINING_CONFIG["optimizer_type"]}"
optimizer_args = {TRAINING_CONFIG["optimizer_args"]}

# Batch configuration (memory-safe)
train_batch_size = {TRAINING_CONFIG["train_batch_size"]}
gradient_accumulation_steps = {TRAINING_CONFIG["gradient_accumulation_steps"]}
# Effective batch size = 1 × 2 = 2

# Training duration
max_train_epochs = {TRAINING_CONFIG["max_train_epochs"]}
# Total steps ≈ 200 images × 10 repeats × epochs / 2

# Regularization (2D anime specific)
caption_dropout_rate = {TRAINING_CONFIG["caption_dropout_rate"]}
caption_dropout_every_n_epochs = {TRAINING_CONFIG["caption_dropout_every_n_epochs"]}
keep_tokens = {TRAINING_CONFIG["keep_tokens"]}  # Keep "inazuma_eleven, {trigger}"

# Data loading
persistent_data_loader_workers = {str(TRAINING_CONFIG["persistent_data_loader_workers"]).lower()}
max_data_loader_n_workers = {TRAINING_CONFIG["max_data_loader_n_workers"]}

# VAE configuration
vae_batch_size = {TRAINING_CONFIG["vae_batch_size"]}

# Memory optimization
mixed_precision = "{TRAINING_CONFIG["mixed_precision"]}"
full_bf16 = {str(TRAINING_CONFIG["full_bf16"]).lower()}
cache_latents = {str(TRAINING_CONFIG["cache_latents"]).lower()}
cache_latents_to_disk = {str(TRAINING_CONFIG["cache_latents_to_disk"]).lower()}  # RAM cache for speed (64GB available)
gradient_checkpointing = {str(TRAINING_CONFIG["gradient_checkpointing"]).lower()}
# xformers disabled - not available in the default environment

# Stability knobs (avoid NaN)
no_half_vae = {str(TRAINING_CONFIG["no_half_vae"]).lower()}
noise_offset = {TRAINING_CONFIG["noise_offset"]}
adaptive_noise_scale = {TRAINING_CONFIG["adaptive_noise_scale"]}

# Checkpointing
save_every_n_epochs = {TRAINING_CONFIG["save_every_n_epochs"]}
save_last_n_epochs_state = {TRAINING_CONFIG["save_last_n_epochs_state"]}
save_model_as = "{TRAINING_CONFIG["save_model_as"]}"
save_precision = "{TRAINING_CONFIG["save_precision"]}"

# Logging
logging_dir = "{logging_dir}"
log_with = "tensorboard"
log_prefix = "inazuma_{char_id}"

# Quality settings
clip_skip = {TRAINING_CONFIG["clip_skip"]}

# Reproducibility
seed = {TRAINING_CONFIG["seed"]}
max_token_length = {TRAINING_CONFIG["max_token_length"]}

# Character-specific metadata (for reference)
# Trigger word: {trigger}
# Position: {character["position"]}
# Element: {character["element"]}
# Key features: {character["key_features"]}
# Timeline support: original, go, ares, orion (via caption tags)
'''

    return toml_content


def main():
    """Generate all configuration files."""

    # Ensure output directory exists
    CONFIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating SDXL LoRA configs for {len(CHARACTERS)} Inazuma Eleven characters...")
    print(f"Output directory: {CONFIG_OUTPUT_DIR}")
    print()

    generated_files = []

    for char in CHARACTERS:
        char_id = char["id"]
        config_filename = f"inazuma_{char_id}_sdxl.toml"
        config_path = CONFIG_OUTPUT_DIR / config_filename

        # Generate TOML content
        toml_content = generate_toml_config(char)

        # Write to file
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(toml_content)

        generated_files.append(config_filename)
        print(f"✓ Generated: {config_filename}")
        print(f"  Character: {char['canonical_name']} ({char['jp_name']})")
        print(f"  Train data: {BASE_DATASET_PATH}/{char_id}_identity")
        print(f"  Output: {BASE_OUTPUT_PATH}/{char_id}_identity")
        print()

    print("=" * 60)
    print(f"Successfully generated {len(generated_files)} configuration files")
    print()
    print("Configuration Summary:")
    print(f"  Base Model: {BASE_MODEL_PATH}")
    print(f"  VAE: {VAE_PATH}")
    print(f"  Learning Rate: {TRAINING_CONFIG['learning_rate']} (unet), {TRAINING_CONFIG['text_encoder_lr']} (text encoder)")
    print(f"  Network Dim/Alpha: {TRAINING_CONFIG['network_dim']}/{TRAINING_CONFIG['network_alpha']}")
    print(f"  Batch Size: {TRAINING_CONFIG['train_batch_size']} × {TRAINING_CONFIG['gradient_accumulation_steps']} (effective=2)")
    print(f"  Epochs: {TRAINING_CONFIG['max_train_epochs']}")
    print(f"  Dataset Repeats: {TRAINING_CONFIG['dataset_repeats']}")
    print(f"  Steps/Epoch: ~2000 (200 images × 10 repeats / 2 batch)")
    print(f"  Total Steps: ~16,000 per character")
    print(f"  Estimated Time: ~5-7 hours per character")
    print()
    print("Next Steps:")
    print("  1. Review generated configs in:")
    print(f"     {CONFIG_OUTPUT_DIR}")
    print("  2. Run batch training script:")
    print("     bash scripts/batch/train_inazuma_sdxl_loras.sh")
    print()


if __name__ == "__main__":
    main()
