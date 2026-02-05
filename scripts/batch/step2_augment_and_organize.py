#!/usr/bin/env python3
"""
Step 2: Augment Datasets and Organize for Training

Assumes Step 1 (caption generation) is complete.
- Augments all datasets to 200+ images
- Organizes into Kohya format
- Generates SDXL training configs

Author: LLMProvider Tooling
Date: 2025-12-13
"""

import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageOps, ImageEnhance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CHARACTERS_DIR = Path("/mnt/data/datasets/general/super-wings/lora_data/characters")
OUTPUT_BASE = Path("/mnt/data/datasets/general/super-wings/training_data_sdxl")
CONFIGS_OUTPUT = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/super_wings_loras_sdxl")
TEMPLATE_PATH = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/elio_elio_identity_sdxl.toml")

MIN_TARGET_IMAGES = 200
CHARACTERS = ["jett", "jerome", "donnie", "chase", "flip", "todd", "paul", "bello", "beard"]


def count_images(directory: Path) -> int:
    """Count images in directory."""
    return len(list(directory.glob("*.png"))) + len(list(directory.glob("*.jpg")))


def augment_dataset(character: str) -> Tuple[bool, Path]:
    """
    Augment dataset to reach 200+ images.
    Uses SDXL captions from sdxl_captions/ directory.
    """
    char_dir = CHARACTERS_DIR / character
    sdxl_captions_dir = char_dir / "sdxl_captions"
    aug_dir = char_dir / "augmented"

    if not sdxl_captions_dir.exists():
        logger.error(f"[{character}] SDXL captions not found! Run step 1 first.")
        return False, aug_dir

    aug_dir.mkdir(exist_ok=True)

    logger.info(f"[{character}] Augmenting dataset...")

    # Get original images
    original_images = list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpg"))
    num_original = len(original_images)

    logger.info(f"  Original images: {num_original}")

    # Copy originals + SDXL captions
    for img_path in original_images:
        shutil.copy2(img_path, aug_dir / img_path.name)

        sdxl_caption = sdxl_captions_dir / f"{img_path.stem}.txt"
        if sdxl_caption.exists():
            shutil.copy2(sdxl_caption, aug_dir / f"{img_path.stem}.txt")

    # Check if augmentation needed
    if num_original >= MIN_TARGET_IMAGES:
        logger.info(f"  Already {num_original} >= {MIN_TARGET_IMAGES}, no augmentation needed")
        return True, aug_dir

    needed = MIN_TARGET_IMAGES - num_original
    logger.info(f"  Need {needed} more images to reach {MIN_TARGET_IMAGES}")

    # Augmentation techniques
    augmentations = [
        ("flip", lambda img: ImageOps.mirror(img)),
        ("bright_up", lambda img: ImageEnhance.Brightness(img).enhance(1.15)),
        ("bright_down", lambda img: ImageEnhance.Brightness(img).enhance(0.85)),
        ("rotate_left", lambda img: img.rotate(5, expand=False, fillcolor=(128, 128, 128))),
        ("rotate_right", lambda img: img.rotate(-5, expand=False, fillcolor=(128, 128, 128))),
        ("contrast_up", lambda img: ImageEnhance.Contrast(img).enhance(1.1)),
    ]

    augmented = 0
    aug_per_image = (needed // num_original) + 1

    for img_path in original_images:
        if augmented >= needed:
            break

        try:
            img = Image.open(img_path).convert("RGB")
            sdxl_caption = sdxl_captions_dir / f"{img_path.stem}.txt"

            # Apply multiple augmentations
            for aug_name, aug_func in augmentations[:aug_per_image]:
                if augmented >= needed:
                    break

                aug_img = aug_func(img)
                aug_filename = f"{img_path.stem}_{aug_name}{img_path.suffix}"
                aug_img.save(aug_dir / aug_filename, quality=95)

                # Copy SDXL caption
                if sdxl_caption.exists():
                    shutil.copy2(sdxl_caption, aug_dir / f"{img_path.stem}_{aug_name}.txt")
                    augmented += 1

        except Exception as e:
            logger.warning(f"  Failed to augment {img_path.name}: {e}")

    final_count = count_images(aug_dir)
    logger.info(f"  ✓ Augmented: {augmented} images")
    logger.info(f"  Final count: {final_count} images")

    return True, aug_dir


def organize_kohya_format(character: str, source_dir: Path) -> bool:
    """Organize into Kohya training format."""
    num_images = count_images(source_dir)

    # Calculate repeats for ~1500-2000 steps/epoch
    if num_images >= 250:
        repeats = 5
    elif num_images >= 200:
        repeats = 7
    else:
        repeats = 10

    training_dir = OUTPUT_BASE / f"{character}_identity"
    kohya_dir = training_dir / f"{repeats}_{character}"

    logger.info(f"[{character}] Organizing Kohya format...")
    logger.info(f"  Images: {num_images}, Repeats: {repeats}")
    logger.info(f"  Steps/epoch: {num_images * repeats}")

    kohya_dir.mkdir(parents=True, exist_ok=True)

    # Copy all images and captions
    copied = 0
    for img_path in list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg")):
        shutil.copy2(img_path, kohya_dir / img_path.name)

        caption_path = source_dir / f"{img_path.stem}.txt"
        if caption_path.exists():
            shutil.copy2(caption_path, kohya_dir / f"{img_path.stem}.txt")
            copied += 1

    logger.info(f"  ✓ Organized: {copied} image-caption pairs")

    # Save metadata
    metadata = {
        "character": character,
        "num_images": num_images,
        "repeats": repeats,
        "steps_per_epoch": num_images * repeats,
        "kohya_dir": str(kohya_dir)
    }

    with open(training_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return True


def generate_training_config(character: str) -> bool:
    """Generate SDXL training config."""
    logger.info(f"[{character}] Generating SDXL training config...")

    training_dir = OUTPUT_BASE / f"{character}_identity"
    metadata_file = training_dir / "metadata.json"

    if not metadata_file.exists():
        logger.error(f"[{character}] Metadata not found")
        return False

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    num_images = metadata["num_images"]
    steps_per_epoch = metadata["steps_per_epoch"]

    # Training parameters
    max_epochs = 10
    save_every = 2
    learning_rate = 0.0001

    # Load template
    if not TEMPLATE_PATH.exists():
        logger.error(f"Template not found: {TEMPLATE_PATH}")
        return False

    with open(TEMPLATE_PATH, 'r') as f:
        template = f.read()

    config = template

    # Update paths
    config = re.sub(
        r'train_data_dir = ".*?"',
        f'train_data_dir = "/mnt/data/datasets/general/super-wings/training_data_sdxl/{character}_identity"',
        config
    )
    config = re.sub(
        r'output_dir = ".*?"',
        f'output_dir = "/mnt/data/training/lora/super-wings/{character}_identity"',
        config
    )
    config = re.sub(
        r'output_name = ".*?"',
        f'output_name = "super-wings-{character}-identity-sdxl"',
        config
    )
    config = re.sub(
        r'logging_dir = ".*?"',
        f'logging_dir = "/mnt/data/training/lora/super-wings/{character}_identity/logs"',
        config
    )
    config = re.sub(
        r'log_prefix = ".*?"',
        f'log_prefix = "super-wings-{character}"',
        config
    )

    # Update parameters
    config = re.sub(r'max_train_epochs = \d+', f'max_train_epochs = {max_epochs}', config)
    config = re.sub(r'save_every_n_epochs = \d+', f'save_every_n_epochs = {save_every}', config)

    # Update header
    header = f"""# SDXL Character Identity LoRA Training Config
# Character: {character} (Super Wings)
# Dataset: {num_images} images × {metadata["repeats"]} repeats = {steps_per_epoch} steps/epoch
# Target: {max_epochs} epochs ({steps_per_epoch * max_epochs} total steps)
"""
    config = re.sub(r'^#.*?\n(?=\[)', header, config, flags=re.DOTALL)

    # Save config
    CONFIGS_OUTPUT.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS_OUTPUT / f"super-wings-{character}-identity-sdxl.toml"

    with open(config_path, 'w') as f:
        f.write(config)

    logger.info(f"  ✓ Config saved: {config_path.name}")
    logger.info(f"    Epochs: {max_epochs}, Steps/epoch: {steps_per_epoch}")

    return True


def main():
    logger.info("=" * 70)
    logger.info("Step 2: Augment and Organize All Datasets")
    logger.info("=" * 70)
    logger.info(f"Characters: {', '.join(CHARACTERS)}")
    logger.info(f"Target: {MIN_TARGET_IMAGES}+ images per character")
    logger.info("")

    stats = {"success": 0, "failed": 0}

    for i, char in enumerate(CHARACTERS, 1):
        logger.info(f"\n[{i}/{len(CHARACTERS)}] {char.upper()}")
        logger.info("=" * 70)

        try:
            # Augment
            success, aug_dir = augment_dataset(char)
            if not success:
                stats["failed"] += 1
                continue

            # Organize
            if not organize_kohya_format(char, aug_dir):
                stats["failed"] += 1
                continue

            # Generate config
            if not generate_training_config(char):
                stats["failed"] += 1
                continue

            logger.info(f"[{char}] ✓✓✓ COMPLETE! ✓✓✓")
            stats["success"] += 1

        except Exception as e:
            logger.error(f"[{char}] Error: {e}", exc_info=True)
            stats["failed"] += 1

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Step 2 Complete!")
    logger.info("=" * 70)
    logger.info(f"Success: {stats['success']}/{len(CHARACTERS)}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info("")
    logger.info(f"Training data: {OUTPUT_BASE}")
    logger.info(f"Configs: {CONFIGS_OUTPUT}")
    logger.info("")
    logger.info("Ready to start training!")
    logger.info("  bash scripts/batch/train_super_wings_sdxl_loras.sh")

    return 0 if stats["failed"] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
