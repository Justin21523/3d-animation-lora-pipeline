#!/usr/bin/env python3
"""
Generate Filtered Training Configs for Synthetic LoRAs

Creates configs ONLY for datasets with sufficient images:
- 3 universal (all included)
- 37 character-specific (excludes < 50 images)

Filters out datasets too small for effective training.

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import json
import re
from pathlib import Path
from typing import Dict, Tuple

MIN_IMAGES_THRESHOLD = 50  # Skip datasets with fewer images

def calculate_training_params(num_images: int, lora_type: str, is_universal: bool = False) -> Dict:
    """
    Calculate optimal training parameters based on dataset size

    Returns:
        Dict with epochs, save_interval, network_dim, etc.
    """
    # Base parameters by type
    if lora_type == "pose":
        network_dim = 128 if is_universal else 112
        base_epochs = 30 if is_universal else 20
    elif lora_type == "action":
        network_dim = 160 if is_universal else 128
        base_epochs = 35 if is_universal else 25
    else:  # expression
        network_dim = 112 if is_universal else 96
        base_epochs = 35 if is_universal else 25

    # Adjust epochs based on dataset size
    if num_images < 100:
        epochs = max(base_epochs + 10, 30)
    elif num_images < 300:
        epochs = base_epochs + 5
    elif num_images > 1000:
        epochs = base_epochs - 5
    else:
        epochs = base_epochs

    # Calculate save intervals
    save_every_n_epochs = max(2, epochs // 10)

    return {
        "max_train_epochs": epochs,
        "save_every_n_epochs": save_every_n_epochs,
        "network_dim": network_dim,
        "network_alpha": network_dim // 2,
        "learning_rate": 0.0001,
        "unet_lr": 0.0001,
        "text_encoder_lr": 0.00006
    }


def load_template(template_path: Path) -> str:
    """Load TOML template"""
    with open(template_path, 'r') as f:
        return f.read()


def generate_config(
    template: str,
    dataset_name: str,
    dataset_path: Path,
    output_path: Path,
    num_images: int,
    lora_type: str,
    is_universal: bool = False
) -> None:
    """Generate a single training config"""

    # Calculate parameters
    params = calculate_training_params(num_images, lora_type, is_universal)

    # Paths
    output_dir = f"/mnt/c/ai_models/lora_sdxl/synthetic/{dataset_name}"
    logging_dir = f"/mnt/c/ai_projects/3d-animation-lora-pipeline/logs/synthetic_training/{dataset_name}"

    # Replace placeholders
    config = template

    # Basic paths
    config = re.sub(r'pretrained_model_name_or_path\s*=\s*"[^"]*"',
                   f'pretrained_model_name_or_path = "/mnt/c/ai_models/stable-diffusion-xl-base-1.0"',
                   config)
    config = re.sub(r'train_data_dir\s*=\s*"[^"]*"',
                   f'train_data_dir = "{dataset_path}"',
                   config)
    config = re.sub(r'output_dir\s*=\s*"[^"]*"',
                   f'output_dir = "{output_dir}"',
                   config)
    config = re.sub(r'logging_dir\s*=\s*"[^"]*"',
                   f'logging_dir = "{logging_dir}"',
                   config)
    config = re.sub(r'output_name\s*=\s*"[^"]*"',
                   f'output_name = "{dataset_name}_lora_sdxl"',
                   config)
    config = re.sub(r'log_prefix\s*=\s*"[^"]*"',
                   f'log_prefix = "{dataset_name}"',
                   config)

    # Training parameters
    config = re.sub(r'max_train_epochs\s*=\s*\d+',
                   f'max_train_epochs = {params["max_train_epochs"]}',
                   config)
    config = re.sub(r'save_every_n_epochs\s*=\s*\d+',
                   f'save_every_n_epochs = {params["save_every_n_epochs"]}',
                   config)
    config = re.sub(r'network_dim\s*=\s*\d+',
                   f'network_dim = {params["network_dim"]}',
                   config)
    config = re.sub(r'network_alpha\s*=\s*\d+',
                   f'network_alpha = {params["network_alpha"]}',
                   config)
    config = re.sub(r'learning_rate\s*=\s*[\d.e-]+',
                   f'learning_rate = {params["learning_rate"]}',
                   config)
    config = re.sub(r'unet_lr\s*=\s*[\d.e-]+',
                   f'unet_lr = {params["unet_lr"]}',
                   config)
    config = re.sub(r'text_encoder_lr\s*=\s*[\d.e-]+',
                   f'text_encoder_lr = {params["text_encoder_lr"]}',
                   config)

    # RTX 5080 compatibility (disable xformers)
    config = re.sub(r'xformers\s*=\s*true', 'xformers = false', config, flags=re.IGNORECASE)
    config = re.sub(r'mem_eff_attn\s*=\s*true', 'mem_eff_attn = false', config, flags=re.IGNORECASE)

    # Save config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(config)


def main():
    print("=" * 60)
    print("Filtered Training Config Generation - Phase 4")
    print("=" * 60)
    print()
    print(f"Filtering threshold: {MIN_IMAGES_THRESHOLD} images minimum")
    print()

    # Paths
    TEMPLATES_DIR = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training")
    DATASETS_DIR = Path("/mnt/data/ai_data/synthetic_lora_data/datasets")
    OUTPUT_DIR = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/synthetic_loras_filtered")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load organization report
    report_path = DATASETS_DIR / "organization_report.json"
    with open(report_path) as f:
        report = json.load(f)

    # Load templates
    templates = {
        "pose": load_template(TEMPLATES_DIR / "pose_lora_sdxl_template.toml"),
        "action": load_template(TEMPLATES_DIR / "action_lora_sdxl_template.toml"),
        "expression": load_template(TEMPLATES_DIR / "expression_lora_sdxl_template.toml")
    }

    configs_generated = []
    configs_skipped = []

    # Priority 1: Universal LoRAs (先訓練通用)
    print("=== Priority 1: Universal LoRAs (3) ===")
    print()

    for lora_type in ["pose", "action", "expression"]:
        dataset_name = f"universal_{lora_type}"

        if dataset_name in report["universal"]:
            num_images = report["universal"][dataset_name]["images"]

            dataset_path = DATASETS_DIR / dataset_name
            output_path = OUTPUT_DIR / f"{dataset_name}_sdxl.toml"

            print(f"[{len(configs_generated) + 1}] {dataset_name}")
            print(f"  Images: {num_images}")

            generate_config(
                templates[lora_type],
                dataset_name,
                dataset_path,
                output_path,
                num_images,
                lora_type,
                is_universal=True
            )

            configs_generated.append({
                "name": dataset_name,
                "config_path": str(output_path),
                "images": num_images,
                "priority": 1
            })

            print(f"  ✅ Config: {output_path.name}")
            print()

    # Priority 2: Character-Specific LoRAs (FILTERED)
    print(f"=== Priority 2: Character-Specific LoRAs (filtered) ===")
    print()

    for dataset_name, info in sorted(report["character_specific"].items()):
        # Extract lora_type from dataset_name
        lora_type = dataset_name.split('_')[-1]
        num_images = info["images"]

        # FILTER: Skip datasets with too few images
        if num_images < MIN_IMAGES_THRESHOLD:
            print(f"[SKIP] {dataset_name}")
            print(f"  Images: {num_images} (< {MIN_IMAGES_THRESHOLD} threshold)")
            print(f"  ⚠️  Too few images for effective training")
            print()

            configs_skipped.append({
                "name": dataset_name,
                "images": num_images,
                "reason": f"Below {MIN_IMAGES_THRESHOLD} image threshold"
            })
            continue

        dataset_path = DATASETS_DIR / dataset_name
        output_path = OUTPUT_DIR / f"{dataset_name}_sdxl.toml"

        print(f"[{len(configs_generated) + 1}] {dataset_name}")
        print(f"  Images: {num_images}")

        generate_config(
            templates[lora_type],
            dataset_name,
            dataset_path,
            output_path,
            num_images,
            lora_type,
            is_universal=False
        )

        configs_generated.append({
            "name": dataset_name,
            "config_path": str(output_path),
            "images": num_images,
            "priority": 2
        })

        print(f"  ✅ Config: {output_path.name}")
        print()

    # Save generation report
    report_data = {
        "filtering": {
            "min_images_threshold": MIN_IMAGES_THRESHOLD,
            "configs_generated": len(configs_generated),
            "configs_skipped": len(configs_skipped),
            "total_original": len(report["universal"]) + len(report["character_specific"])
        },
        "configs": configs_generated,
        "skipped": configs_skipped,
        "universal_configs": 3,
        "character_configs": len(configs_generated) - 3,
        "output_dir": str(OUTPUT_DIR)
    }

    report_output = OUTPUT_DIR / "config_generation_report.json"
    with open(report_output, 'w') as f:
        json.dump(report_data, f, indent=2)

    print("=" * 60)
    print("Filtered Config Generation Complete!")
    print("=" * 60)
    print()
    print(f"Total configs generated: {len(configs_generated)}")
    print(f"  Priority 1 (Universal): 3")
    print(f"  Priority 2 (Character): {len(configs_generated) - 3}")
    print()
    print(f"⚠️  Configs skipped: {len(configs_skipped)}")
    for skipped in configs_skipped:
        print(f"    - {skipped['name']:35s} ({skipped['images']:3d} images)")
    print()
    print(f"Configs saved to: {OUTPUT_DIR}")
    print(f"Report: {report_output}")
    print()
    print("✅ Phase 4 Complete! Ready for Phase 5 (Training)")
    print()
    print("Training Order:")
    print("  1. universal_pose")
    print("  2. universal_action")
    print("  3. universal_expression")
    print(f"  4-{len(configs_generated)}. Character-specific LoRAs")
    print()
    print("Estimated time saved by filtering: 10-15 hours")


if __name__ == "__main__":
    main()
