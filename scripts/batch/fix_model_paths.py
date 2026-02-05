#!/usr/bin/env python3
"""Fix model paths in Inazuma SDXL LoRA configs."""

from pathlib import Path

CONFIG_DIR = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl")

CHARACTERS = [
    "endou_mamoru", "gouenji_shuuya", "fudou_akio",
    "matsukaze_tenma", "inamori_asuto", "nosaka_yuuma", "utsunomiya_toramaru"
]

# Correct paths
OLD_MODEL_PATH = '"/mnt/data/ai_data/models/sdxl/stable-diffusion-xl-base-1.0"'
NEW_MODEL_PATH = '"/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"'

OLD_VAE_PATH = '"/mnt/data/ai_data/models/sdxl/sdxl_vae.safetensors"'
NEW_VAE_PATH = '"/mnt/c/ai_models/stable-diffusion/vae/sdxl_vae.safetensors"'

for char_id in CHARACTERS:
    config_file = CONFIG_DIR / f"inazuma_{char_id}_sdxl.toml"

    if not config_file.exists():
        print(f"⚠ {config_file.name} not found")
        continue

    content = config_file.read_text()

    # Replace model path
    if OLD_MODEL_PATH in content:
        content = content.replace(OLD_MODEL_PATH, NEW_MODEL_PATH)

    # Replace VAE path
    if OLD_VAE_PATH in content:
        content = content.replace(OLD_VAE_PATH, NEW_VAE_PATH)

    config_file.write_text(content)
    print(f"✓ Updated: {config_file.name}")

print("\n✅ All model paths updated!")
print(f"  Base Model: {NEW_MODEL_PATH}")
print(f"  VAE: {NEW_VAE_PATH}")
