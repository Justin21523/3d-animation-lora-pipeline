#!/usr/bin/env python3
"""
Generate SDXL LoRA training configurations for Inazuma Eleven characters.
"""

import json
from pathlib import Path
from typing import Dict, List


# Character configurations
CHARACTERS = {
    "Endou Mamoru": {
        "character_id": "endou_mamoru",
        "caption": "inazuma_eleven, inazuma_endou_mamoru, anime_style, teenage_boy, timeline_original, timeline_go, timeline_ares, timeline_orion, role_goalkeeper, element_mountain, orange_headband, spiky_brown_hair, big_round_brown_eyes, determined_expression, soccer_uniform, gloves, daytime, training, confident_pose, white_and_blue_jersey, muscular, energetic, focused, goalkeeper_gloves, sports_environment",
    },
    "Fudou Akio": {
        "character_id": "fudou_akio",
        "caption": "inazuma_eleven, inazuma_fudou_akio, anime_style, teenage_boy, timeline_original, timeline_go, timeline_ares, timeline_orion, role_midfielder, element_fire, brown_mohawk, sarcastic_smirk, blue_gray_eyes, confident_pose, soccer_uniform, intense_expression, sharp_features, athletic_build, rebellious_attitude",
    },
    "Gouenji Shuuya": {
        "character_id": "gouenji_shuuya",
        "caption": "inazuma_eleven, inazuma_gouenji_shuuya, anime_style, teenage_boy, timeline_original, timeline_go, timeline_ares, timeline_orion, role_forward, element_fire, spiky_blonde_hair, dark_brown_eyes, cool_expression, muscular, soccer_uniform, serious_pose, white_jacket, determination, athletic_build, confident_stance",
    },
    "Inamori Asuto": {
        "character_id": "inamori_asuto",
        "caption": "inazuma_eleven, inazuma_inamori_asuto, anime_style, teenage_boy, timeline_ares, timeline_orion, role_forward, element_fire, short_darkgray_spiky_hair, deep_green_eyes, sunny_smile, white_soccer_uniform, dynamic_pose, muscular, determined_expression, soccer_field, intense_gaze, athletic_build, energetic, passionate",
    },
    "Matsukaze Tenma": {
        "character_id": "matsukaze_tenma",
        "caption": "inazuma_eleven, inazuma_matsukaze_tenma, anime_style, teenage_boy, timeline_go, role_midfielder, element_wind, chestnut_brown_hair, wind_swirl_hair, greyblue_eyes, determined_expression, soccer_uniform, white_and_blue_jersey, dynamic_pose, running_pose, soccer_ball, wind_effect, short_hair, athletic_build, intense_gaze",
    },
    "Nosaka Yuuma": {
        "character_id": "nosaka_yuuma",
        "caption": "inazuma_eleven, inazuma_nosaka_yuuma, anime_style, teenage_boy, timeline_ares, timeline_orion, role_midfielder, element_fire, element_mountain, swept_left_blonde_hair, grey_dry_eyes, tall_slender, serious_expression, soccer_uniform, white_and_blue_jersey, determined_pose, soccer_field, intense_gaze, athletic_build, skilled_player",
    },
    "Utsunomiya Toramaru": {
        "character_id": "utsunomiya_toramaru",
        "caption": "inazuma_eleven, inazuma_utsunomiya_toramaru, anime_style, young_boy, timeline_original, timeline_go, role_forward, element_forest, spiky_blueblack_hair, innocent_expression, small_build",
    },
}


def generate_sdxl_lora_config(
    character_name: str,
    character_id: str,
    train_data_dir: str,
    output_dir: str,
    base_model: str = "/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors",
    network_dim: int = 64,
    network_alpha: int = 32,
    train_batch_size: int = 2,
    max_train_steps: int = 1000,
    learning_rate: float = 1e-4,
    save_every_n_steps: int = 200,
) -> str:
    """Generate SDXL LoRA training configuration."""

    config = f"""# SDXL LoRA Training Config for {character_name}
# Generated for Inazuma Eleven Character Training Pipeline

[settings]
use_shell = false

[model]
models_dir = "/mnt/c/ai_models/stable-diffusion/checkpoints"
pretrained_model_name_or_path = "{base_model}"
output_name = "{character_id}_sdxl_lora"
train_data_dir = "{train_data_dir}"
training_comment = "Inazuma Eleven {character_name} SDXL LoRA"
save_model_as = "safetensors"
save_precision = "bf16"

[folders]
output_dir = "{output_dir}/{character_id}"
logging_dir = "{output_dir}/{character_id}/logs"

[accelerate_launch]
mixed_precision = "bf16"
num_processes = 1
gpu_ids = "0"
main_process_port = 29500
dynamo_backend = "no"

[basic]
cache_latents = true
cache_latents_to_disk = false
caption_extension = ".txt"
enable_bucket = true
epoch = 10
learning_rate = {learning_rate}
learning_rate_te1 = {learning_rate / 10}
learning_rate_te2 = {learning_rate / 10}
lr_scheduler = "cosine"
lr_warmup = 0
max_bucket_reso = 2048
max_grad_norm = 1.0
max_resolution = "1024,1024"
max_train_steps = {max_train_steps}
min_bucket_reso = 256
optimizer = "AdamW8bit"
save_every_n_epochs = 0
save_every_n_steps = {save_every_n_steps}
seed = 42
train_batch_size = {train_batch_size}
clip_skip = 2

[advanced]
bucket_no_upscale = true
bucket_reso_steps = 64
color_aug = false
flip_aug = false
fp8_base = false
full_fp16 = false
full_bf16 = true
gradient_accumulation_steps = 1
gradient_checkpointing = true
keep_tokens = 0
loss_type = "l2"
max_token_length = 225
mem_eff_attn = false
min_snr_gamma = 0
mixed_precision = "bf16"
network_dropout = 0
network_train_unet_only = false
network_train_text_encoder_only = false
noise_offset = 0.1
persistent_data_loader_workers = false
prior_loss_weight = 1.0
randn_fix_seed = false
resolution_sqrt = false
resume = ""
sample_every_n_epochs = 0
sample_every_n_steps = 100
sample_sampler = "euler_a"
sdxl_cache_text_encoder_outputs = true
sdxl_no_half_vae = true
shuffle_caption = false
token_warmup_min = 1
token_warmup_step = 0
vae = ""
weighted_captions = false

[sdxl]
# SDXL-specific settings
sdxl_cache_text_encoder_outputs = true
sdxl_no_half_vae = true

[network]
network_module = "lycoris.kohya"
network_type = "lora"
network_dim = {network_dim}
network_alpha = {network_alpha}
network_args = "conv_dim=32:conv_alpha=16:algo=loha"
network_dropout = 0
network_train_unet_only = false
network_train_text_encoder_only = false

[sample]
sample_every_n_epochs = 1
sample_every_n_steps = 0
sample_sampler = "euler_a"
sample_prompts = "1girl, inazuma_{character_id}, anime_style, looking_at_viewer, masterpiece"
sample_negative_prompts = "bad quality, low quality"

[logging]
logging_dir = "{output_dir}/{character_id}/logs"
log_with = "tensorboard"

[[datasets]]
resolution = [1024, 1024]
batch_size = {train_batch_size}

[[datasets.subsets]]
image_dir = "{train_data_dir}"
caption_extension = ".txt"
enable_bucketing = true
"""

    return config


def main():
    """Generate all character SDXL LoRA configs."""

    output_base_dir = Path("/mnt/data/training/lora/inazuma_eleven")
    config_output_dir = output_base_dir / "configs"
    config_output_dir.mkdir(parents=True, exist_ok=True)

    train_data_base_dir = "/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented"

    print("=" * 70)
    print("SDXL LoRA Training Configuration Generator")
    print("=" * 70)
    print()

    generated_configs = []

    for character_name, char_info in CHARACTERS.items():
        character_id = char_info["character_id"]
        train_data_dir = f"{train_data_base_dir}/{character_name}"

        config_content = generate_sdxl_lora_config(
            character_name=character_name,
            character_id=character_id,
            train_data_dir=train_data_dir,
            output_dir=str(output_base_dir),
            base_model="/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors",
            network_dim=64,
            network_alpha=32,
            train_batch_size=2,
            max_train_steps=1000,
            learning_rate=1e-4,
            save_every_n_steps=200,
        )

        config_path = config_output_dir / f"{character_id}_sdxl.toml"
        config_path.write_text(config_content, encoding="utf-8")

        generated_configs.append({
            "character": character_name,
            "character_id": character_id,
            "config_path": str(config_path),
            "train_data_dir": train_data_dir,
        })

        print(f"✓ {character_name}: {config_path.name}")

    # Save config manifest
    manifest_path = config_output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(generated_configs, f, indent=2)

    print()
    print("=" * 70)
    print(f"✅ Generated {len(generated_configs)} SDXL LoRA configurations")
    print(f"📁 Config directory: {config_output_dir}")
    print(f"📋 Manifest: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
