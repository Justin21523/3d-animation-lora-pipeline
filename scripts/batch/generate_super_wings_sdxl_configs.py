#!/usr/bin/env python3
"""
Generate SDXL Training Configs for All Super Wings Characters

Creates optimized TOML configs for all 9 characters based on successful
Pixar character LoRA training results (Epoch 1-2 best, network_dim=64).

Author: LLMProvider Tooling
Date: 2025-12-13
"""

import re
import json
from pathlib import Path
from typing import Dict

# Paths
TEMPLATE_PATH = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_lora_sdxl_template.toml")
TRAINING_DATA_BASE = Path("/mnt/data/datasets/general/super-wings/training_data")
OUTPUT_DIR = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/super_wings_loras")

# Characters
CHARACTERS = [
    "beard", "bello", "flip", "jerone", "jet",
    "paul", "shark", "tank", "tony"
]

def count_images(character_dir: Path) -> int:
    """Count training images for a character"""
    if not character_dir.exists():
        return 0

    # Look for Kohya format: {repeat}_{concept}/
    subdirs = [d for d in character_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return 0

    # Count images in first subdir (should be only one)
    img_dir = subdirs[0]
    return len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))

def calculate_optimal_params(num_images: int) -> Dict:
    """
    Calculate optimal training parameters based on dataset size

    Based on successful results:
    - 200-300 images: 2 epochs, LR 0.0001
    - <200 images: 3 epochs, slightly higher LR
    """
    if num_images < 200:
        return {
            "max_train_epochs": 3,
            "save_every_n_epochs": 1,
            "learning_rate": 0.00012,
            "network_dim": 64,
            "network_alpha": 32
        }
    else:
        return {
            "max_train_epochs": 2,
            "save_every_n_epochs": 1,
            "learning_rate": 0.0001,
            "network_dim": 64,
            "network_alpha": 32
        }

def generate_config(character: str, template: str) -> str:
    """Generate config for a single character"""

    # Count images
    char_dir = TRAINING_DATA_BASE / character
    num_images = count_images(char_dir)

    if num_images == 0:
        print(f"⚠️  Warning: No images found for {character} at {char_dir}")
        return None

    # Get optimal params
    params = calculate_optimal_params(num_images)

    # Replace placeholders
    config = template

    # Character-specific paths
    config = config.replace("{CHARACTER_NAME}", character)

    # Training parameters
    config = re.sub(r'max_train_epochs\s*=\s*\d+',
                   f'max_train_epochs = {params["max_train_epochs"]}',
                   config)
    config = re.sub(r'save_every_n_epochs\s*=\s*\d+',
                   f'save_every_n_epochs = {params["save_every_n_epochs"]}',
                   config)
    config = re.sub(r'learning_rate\s*=\s*[\d.e-]+',
                   f'learning_rate = {params["learning_rate"]}',
                   config)
    config = re.sub(r'network_dim\s*=\s*\d+',
                   f'network_dim = {params["network_dim"]}',
                   config)
    config = re.sub(r'network_alpha\s*=\s*\d+',
                   f'network_alpha = {params["network_alpha"]}',
                   config)

    return config, num_images, params

def main():
    print("=" * 70)
    print("Super Wings SDXL LoRA - Config Generation")
    print("=" * 70)
    print()

    # Load template
    if not TEMPLATE_PATH.exists():
        print(f"❌ Template not found: {TEMPLATE_PATH}")
        return

    with open(TEMPLATE_PATH, 'r') as f:
        template = f.read()

    print(f"✅ Loaded template: {TEMPLATE_PATH.name}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate configs
    configs_generated = []

    for i, character in enumerate(CHARACTERS, 1):
        print(f"[{i}/{len(CHARACTERS)}] Generating config for {character}...")

        result = generate_config(character, template)
        if result is None:
            print(f"  ⏭️  Skipped (no training data)")
            print()
            continue

        config, num_images, params = result

        # Save config
        output_path = OUTPUT_DIR / f"super-wings-{character}-sdxl.toml"
        with open(output_path, 'w') as f:
            f.write(config)

        configs_generated.append({
            "character": character,
            "config_path": str(output_path),
            "images": num_images,
            "epochs": params["max_train_epochs"],
            "learning_rate": params["learning_rate"]
        })

        print(f"  ✅ Config saved: {output_path.name}")
        print(f"     Images: {num_images}")
        print(f"     Epochs: {params['max_train_epochs']}")
        print(f"     LR: {params['learning_rate']}")
        print()

    # Save generation report
    report = {
        "total_configs": len(configs_generated),
        "output_dir": str(OUTPUT_DIR),
        "configs": configs_generated
    }

    report_path = OUTPUT_DIR / "config_generation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print("Config Generation Complete!")
    print("=" * 70)
    print(f"Total configs: {len(configs_generated)}/{len(CHARACTERS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Report: {report_path}")
    print()
    print("Next step:")
    print("  python scripts/batch/train_super_wings_sdxl_loras.py")

if __name__ == "__main__":
    main()
