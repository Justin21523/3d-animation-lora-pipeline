#!/usr/bin/env python3
"""
Generate SDXL Character LoRA Training Configs
Based on successful Alberto training results (Epoch 4 best, no dropout, conservative training)
"""

import argparse
from pathlib import Path
import yaml


# ALBERTO'S SUCCESSFUL CONFIGURATION (Epoch 4 = best)
PROVEN_CONFIG = {
    "network_dropout": 0.0,  # NO dropout - proven better than 0.1
    "learning_rate": 0.0001,
    "text_encoder_lr": 0.00006,
    "unet_lr": 0.0001,
    "lr_scheduler_num_cycles": 1,  # Single cycle, no warm restarts
    "max_train_epochs": 5,  # Conservative - stop before overfitting
    "save_every_n_epochs": 1,  # Save ALL epochs to find optimal
    "save_last_n_epochs": 5,  # Keep all 5
}


def generate_character_config(
    character_name: str,
    display_name: str,
    movie: str,
    image_count: int,
    repeats: int,
    output_path: Path,
):
    """Generate SDXL training config based on proven Alberto settings"""

    # Calculate steps per epoch
    steps_per_epoch = image_count * repeats
    total_steps = steps_per_epoch * PROVEN_CONFIG["max_train_epochs"]

    # Generate config
    config = f"""# SDXL Character Identity LoRA Training Config
# Character: {display_name} from {movie.title()} (Pixar)
# Dataset: {image_count} images × {repeats} repeats = {steps_per_epoch} steps/epoch
# Target: {PROVEN_CONFIG['max_train_epochs']} epochs ({total_steps} total steps)
#
# CONFIGURATION BASED ON ALBERTO'S SUCCESSFUL TRAINING:
# - NO dropout (0.0) - dropout 0.1 caused training instability
# - Conservative LR (0.0001)
# - Single LR cycle (no warm restarts)
# - Max 5 epochs (Alberto's epoch 4 was best, stop before overfitting)
# - Save EVERY epoch to identify optimal checkpoint

[model]
pretrained_model_name_or_path = "/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
v2 = false
v_parameterization = false
sdxl = true

network_module = "networks.lora"
network_dim = 64
network_alpha = 32
network_dropout = {PROVEN_CONFIG['network_dropout']}
network_args = []

[paths]
train_data_dir = "/mnt/data/ai_data/datasets/3d-anime/{movie}/lora_data/training_data/{character_name}_identity"
output_dir = "/mnt/c/ai_models/lora_sdxl/{movie}/{character_name}_identity"
output_name = "{character_name}_lora_sdxl"
logging_dir = "/mnt/c/ai_models/lora_sdxl/{movie}/{character_name}_identity/logs"
log_prefix = "{character_name}_sdxl"

[training]
optimizer_type = "AdamW8bit"
mixed_precision = "bf16"
full_bf16 = true
gradient_checkpointing = true

train_batch_size = 1
gradient_accumulation_steps = 2

learning_rate = {PROVEN_CONFIG['learning_rate']}
text_encoder_lr = {PROVEN_CONFIG['text_encoder_lr']}
unet_lr = {PROVEN_CONFIG['unet_lr']}

lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = {PROVEN_CONFIG['lr_scheduler_num_cycles']}
lr_warmup_steps = 100

min_snr_gamma = 5.0
noise_offset = 0.05
adaptive_noise_scale = 0.0
multires_noise_iterations = 0
multires_noise_discount = 0.3

max_train_epochs = {PROVEN_CONFIG['max_train_epochs']}
save_every_n_epochs = {PROVEN_CONFIG['save_every_n_epochs']}
save_last_n_epochs = {PROVEN_CONFIG['save_last_n_epochs']}

cache_latents = true
cache_latents_to_disk = false
vae_batch_size = 2

[resolution]
resolution = "1024,1024"
enable_bucket = true
min_bucket_reso = 768
max_bucket_reso = 1280
bucket_reso_steps = 128
bucket_no_upscale = false

[augmentation]
color_aug = false
flip_aug = false
random_crop = false
shuffle_caption = true
keep_tokens = 1

[sample_generation]
# Disabled to maximize training speed - use manual testing instead

[performance]
persistent_data_loader_workers = true
max_data_loader_n_workers = 6
lowram = false
max_token_length = 225
seed = 42
clip_skip = 2
max_grad_norm = 1.0
"""

    # Write config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config)

    print(f"✅ Created: {output_path}")
    print(f"   - {image_count} images × {repeats} repeats = {steps_per_epoch} steps/epoch")
    print(f"   - {PROVEN_CONFIG['max_train_epochs']} epochs = {total_steps} total steps")
    print(f"   - Saves every epoch (1-5)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character-yaml", type=str, required=True, help="Path to character YAML config")
    parser.add_argument("--output-dir", type=str, default="configs/training/character_loras_sdxl", help="Output directory for TOML configs")
    args = parser.parse_args()

    # Load character config
    char_yaml = Path(args.character_yaml)
    with open(char_yaml) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)

    # Generate config for each character
    for char in config["characters"]:
        movie = char["movie"]
        char_dir = char["char_dir"]
        char_name = char["char_name"]
        display_name = char.get("display_name", char_name.title())
        image_count = char["image_count"]
        repeats = char["repeats"]

        # Generate output path
        output_path = output_dir / f"{movie}_{char_name}_sdxl.toml"

        generate_character_config(
            character_name=char_name,
            display_name=display_name,
            movie=movie,
            image_count=image_count,
            repeats=repeats,
            output_path=output_path,
        )
        print()


if __name__ == "__main__":
    main()
