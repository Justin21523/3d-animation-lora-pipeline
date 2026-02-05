#!/usr/bin/env python3
"""Fix dataset paths in Inazuma SDXL LoRA configs to use Kohya format."""

from pathlib import Path

CONFIG_DIR = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl")

# Character IDs
CHARACTERS = [
    "endou_mamoru",
    "gouenji_shuuya",
    "fudou_akio",
    "matsukaze_tenma",
    "inamori_asuto",
    "nosaka_yuuma",
    "utsunomiya_toramaru"
]

NEW_BASE_PATH = "/mnt/data/datasets/general/inazuma-eleven/lora_data/training_data_sdxl"

for char_id in CHARACTERS:
    config_file = CONFIG_DIR / f"inazuma_{char_id}_sdxl.toml"

    if not config_file.exists():
        print(f"Warning: {config_file} not found")
        continue

    # Read current config
    content = config_file.read_text()

    # Replace train_data_dir path
    old_path = f'/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/1_{char_id}"'
    new_path = f'{NEW_BASE_PATH}/{char_id}_identity"'

    if old_path in content:
        content = content.replace(old_path, new_path)
        config_file.write_text(content)
        print(f"✓ Updated: {config_file.name}")
        print(f"  New path: {NEW_BASE_PATH}/{char_id}_identity")
    else:
        print(f"⚠ Path not found in {config_file.name}")

print("\nAll configs updated!")
