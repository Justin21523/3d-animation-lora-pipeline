#!/usr/bin/env python3
"""
Complete Super Wings SDXL LoRA Data Preparation with Augmentation

Pipeline:
1. Generate captions using LLMProvider API (skip existing)
2. Expand to SDXL format (225 tokens)
3. Apply simple augmentation to reach 200+ images
4. Organize into Kohya training format
5. Generate SDXL training configs (using successful template)

Author: LLMProvider Tooling
Date: 2025-12-13
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
CHARACTERS_DIR = Path("/mnt/data/datasets/general/super-wings/lora_data/characters")
OUTPUT_BASE = Path("/mnt/data/datasets/general/super-wings/training_data_sdxl")
CONFIGS_OUTPUT = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/super_wings_loras_sdxl")

# Scripts
RECAPTION_SCRIPT = SCRIPT_DIR / "generic/training/caption_engines/recaption_with_llm_provider.py"
SDXL_EXPANDER_SCRIPT = SCRIPT_DIR / "generic/training/sdxl_caption_expander.py"

# Template config (successful configuration)
TEMPLATE_PATH = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/elio_elio_identity_sdxl.toml")

# Target: 200+ images per character
MIN_TARGET_IMAGES = 200

# Character definitions
CHARACTERS = {
    "jett": "small red and white jet plane with blue eyes",
    "jerome": "large blue cargo plane with friendly eyes",
    "donnie": "orange construction transport plane",
    "chase": "blue police jet plane",
    "flip": "yellow acrobatic jet plane",
    "todd": "orange rocket plane",
    "paul": "blue police helicopter",
    "bello": "purple and pink glamorous jet plane",
    "beard": "bearded human character with orange biplane"
}


class SuperWingsDataPrep:
    """Complete data preparation with augmentation for Super Wings SDXL LoRA."""

    def __init__(self, characters: List[str] = None, skip_captions: bool = False):
        """
        Initialize preparation pipeline.

        Args:
            characters: List of character names to process (None = all)
            skip_captions: Skip caption generation for all characters
        """
        self.characters = characters or list(CHARACTERS.keys())
        self.skip_captions = skip_captions
        self.stats = {
            "processed": 0,
            "skipped": 0,
            "failed": 0
        }

        # Check API key
        if not skip_captions and not os.getenv('LLM_VENDOR_API_KEY'):
            raise ValueError(
                "LLM_VENDOR_API_KEY not found. Set it with:\n"
                "export LLM_VENDOR_API_KEY='your-api-key-here'"
            )

    def count_images(self, directory: Path) -> int:
        """Count images in directory."""
        return len(list(directory.glob("*.png"))) + len(list(directory.glob("*.jpg")))

    def count_captions(self, directory: Path) -> int:
        """Count caption files."""
        return len(list(directory.glob("*.txt")))

    def has_captions(self, character_dir: Path) -> bool:
        """Check if character already has captions."""
        num_images = self.count_images(character_dir)
        num_captions = self.count_captions(character_dir)
        return num_captions >= num_images * 0.9  # 90% threshold

    def step1_generate_captions(self, character: str) -> bool:
        """
        Step 1: Generate base captions using LLMProvider API.

        Args:
            character: Character name

        Returns:
            True if successful or skipped
        """
        char_dir = CHARACTERS_DIR / character

        if self.skip_captions or self.has_captions(char_dir):
            logger.info(f"[{character}] Captions already exist, skipping")
            return True

        logger.info(f"[{character}] Generating captions with LLMProvider API...")

        cmd = [
            "python3",
            str(RECAPTION_SCRIPT),
            "--input-dir", str(char_dir),
            "--output-dir", str(char_dir),
            "--lora-type", "character",
            "--force"
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"[{character}] ✓ Captions generated")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"[{character}] Caption generation failed: {e.stderr}")
            return False

    def step2_expand_to_sdxl(self, character: str) -> bool:
        """
        Step 2: Expand captions to SDXL format.

        Args:
            character: Character name

        Returns:
            True if successful
        """
        char_dir = CHARACTERS_DIR / character
        sdxl_dir = char_dir / "sdxl_captions"

        # Check if already done
        if sdxl_dir.exists() and self.count_captions(sdxl_dir) > 0:
            logger.info(f"[{character}] SDXL captions already exist, skipping")
            return True

        logger.info(f"[{character}] Expanding captions to SDXL format...")

        cmd = [
            "python3",
            str(SDXL_EXPANDER_SCRIPT),
            "--input-dir", str(char_dir),
            "--output-dir", str(sdxl_dir),
            "--character-name", character,
            "--style", "dreamworks"
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"[{character}] ✓ SDXL captions generated")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"[{character}] SDXL expansion failed: {e.stderr}")
            return False

    def step3_augment_dataset(self, character: str) -> Tuple[bool, Path]:
        """
        Step 3: Augment dataset to reach 200+ images.

        Multiple augmentation techniques:
        - Horizontal flip
        - Slight rotations (-5°, +5°)
        - Brightness adjustments
        - Caption is copied from original

        Args:
            character: Character name

        Returns:
            Tuple of (success, augmented_dir)
        """
        from PIL import ImageEnhance

        char_dir = CHARACTERS_DIR / character
        aug_dir = char_dir / "augmented"
        aug_dir.mkdir(exist_ok=True)

        logger.info(f"[{character}] Augmenting dataset...")

        # Copy all original images and SDXL captions
        sdxl_captions_dir = char_dir / "sdxl_captions"

        original_images = list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpg"))
        num_original = len(original_images)

        logger.info(f"  Original images: {num_original}")

        # Copy originals
        for img_path in original_images:
            shutil.copy2(img_path, aug_dir / img_path.name)

            # Copy SDXL caption
            sdxl_caption = sdxl_captions_dir / f"{img_path.stem}.txt"
            if sdxl_caption.exists():
                shutil.copy2(sdxl_caption, aug_dir / f"{img_path.stem}.txt")

        # Calculate how many augmentations needed
        if num_original >= MIN_TARGET_IMAGES:
            logger.info(f"  Already {num_original} >= {MIN_TARGET_IMAGES}, no augmentation needed")
            return True, aug_dir

        needed = MIN_TARGET_IMAGES - num_original
        logger.info(f"  Need {needed} more images to reach {MIN_TARGET_IMAGES}")

        # Augmentation techniques (in order of priority)
        augmentations = [
            ("flip", lambda img: ImageOps.mirror(img)),
            ("bright_up", lambda img: ImageEnhance.Brightness(img).enhance(1.15)),
            ("bright_down", lambda img: ImageEnhance.Brightness(img).enhance(0.85)),
            ("rotate_left", lambda img: img.rotate(5, expand=False, fillcolor=(128, 128, 128))),
            ("rotate_right", lambda img: img.rotate(-5, expand=False, fillcolor=(128, 128, 128))),
            ("contrast_up", lambda img: ImageEnhance.Contrast(img).enhance(1.1)),
        ]

        augmented = 0
        aug_per_image = (needed // num_original) + 1  # How many augmentations per original image

        for img_path in original_images:
            if augmented >= needed:
                break

            try:
                img = Image.open(img_path).convert("RGB")
                caption_path = sdxl_captions_dir / f"{img_path.stem}.txt"

                # Apply multiple augmentations to this image
                for aug_name, aug_func in augmentations[:aug_per_image]:
                    if augmented >= needed:
                        break

                    # Apply augmentation
                    aug_img = aug_func(img)

                    # Save augmented image
                    aug_filename = f"{img_path.stem}_{aug_name}{img_path.suffix}"
                    aug_img.save(aug_dir / aug_filename, quality=95)

                    # Copy caption
                    if caption_path.exists():
                        shutil.copy2(caption_path, aug_dir / f"{img_path.stem}_{aug_name}.txt")
                        augmented += 1

            except Exception as e:
                logger.warning(f"  Failed to augment {img_path.name}: {e}")

        final_count = self.count_images(aug_dir)
        logger.info(f"  ✓ Augmented: {augmented} images")
        logger.info(f"  Final count: {final_count} images")

        return True, aug_dir

    def step4_organize_kohya_format(self, character: str, source_dir: Path) -> bool:
        """
        Step 4: Organize into Kohya training format.

        Args:
            character: Character name
            source_dir: Source directory with images and captions

        Returns:
            True if successful
        """
        num_images = self.count_images(source_dir)

        # Calculate repeats for ~1500-2000 steps/epoch
        # 200-250 images: 7 repeats = 1400-1750 steps
        # 250+ images: 5-6 repeats
        if num_images >= 250:
            repeats = 5
        elif num_images >= 200:
            repeats = 7
        else:
            repeats = 10  # For smaller datasets

        # Create Kohya directory
        training_dir = OUTPUT_BASE / f"{character}_identity"
        kohya_dir = training_dir / f"{repeats}_{character}"

        logger.info(f"[{character}] Organizing Kohya format...")
        logger.info(f"  Images: {num_images}, Repeats: {repeats}")
        logger.info(f"  Steps/epoch: {num_images * repeats}")
        logger.info(f"  Output: {kohya_dir}")

        kohya_dir.mkdir(parents=True, exist_ok=True)

        # Copy all images and captions
        copied = 0
        for img_path in source_dir.glob("*.png"):
            shutil.copy2(img_path, kohya_dir / img_path.name)

            caption_path = source_dir / f"{img_path.stem}.txt"
            if caption_path.exists():
                shutil.copy2(caption_path, kohya_dir / f"{img_path.stem}.txt")
                copied += 1

        for img_path in source_dir.glob("*.jpg"):
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

    def step5_generate_training_config(self, character: str) -> bool:
        """
        Step 5: Generate SDXL training config from successful template.

        Args:
            character: Character name

        Returns:
            True if successful
        """
        logger.info(f"[{character}] Generating SDXL training config...")

        # Load metadata
        training_dir = OUTPUT_BASE / f"{character}_identity"
        metadata_file = training_dir / "metadata.json"

        if not metadata_file.exists():
            logger.error(f"[{character}] Metadata not found")
            return False

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        num_images = metadata["num_images"]
        steps_per_epoch = metadata["steps_per_epoch"]

        # Training parameters (based on successful config)
        # 200-250 images, ~1500 steps/epoch: 10 epochs optimal
        max_epochs = 10
        save_every = 2
        learning_rate = 0.0001

        # Load template
        if not TEMPLATE_PATH.exists():
            logger.error(f"Template not found: {TEMPLATE_PATH}")
            return False

        with open(TEMPLATE_PATH, 'r') as f:
            template = f.read()

        # Replace character-specific values
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

        # Update training parameters
        config = re.sub(r'max_train_epochs = \d+', f'max_train_epochs = {max_epochs}', config)
        config = re.sub(r'save_every_n_epochs = \d+', f'save_every_n_epochs = {save_every}', config)

        # Update header comment
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
        logger.info(f"    Epochs: {max_epochs}, LR: {learning_rate}")
        logger.info(f"    Steps/epoch: {steps_per_epoch}")

        return True

    def process_character(self, character: str) -> bool:
        """Process a single character through all steps."""
        logger.info("=" * 70)
        logger.info(f"Processing: {character.upper()}")
        logger.info("=" * 70)

        try:
            # Step 1: Generate captions
            if not self.step1_generate_captions(character):
                logger.error(f"[{character}] Failed at step 1")
                return False

            # Step 2: Expand to SDXL
            if not self.step2_expand_to_sdxl(character):
                logger.error(f"[{character}] Failed at step 2")
                return False

            # Step 3: Augment dataset
            success, aug_dir = self.step3_augment_dataset(character)
            if not success:
                logger.error(f"[{character}] Failed at step 3")
                return False

            # Step 4: Organize Kohya format
            if not self.step4_organize_kohya_format(character, aug_dir):
                logger.error(f"[{character}] Failed at step 4")
                return False

            # Step 5: Generate training config
            if not self.step5_generate_training_config(character):
                logger.error(f"[{character}] Failed at step 5")
                return False

            logger.info(f"[{character}] ✓✓✓ COMPLETE! ✓✓✓")
            self.stats["processed"] += 1
            return True

        except Exception as e:
            logger.error(f"[{character}] Unexpected error: {e}", exc_info=True)
            self.stats["failed"] += 1
            return False

    def run(self) -> Dict:
        """Run complete pipeline for all characters."""
        logger.info("=" * 70)
        logger.info("Super Wings SDXL LoRA - Complete Preparation with Augmentation")
        logger.info("=" * 70)
        logger.info(f"Characters: {', '.join(self.characters)}")
        logger.info(f"Target: {MIN_TARGET_IMAGES}+ images per character")
        logger.info("")

        for i, character in enumerate(self.characters, 1):
            logger.info(f"\n[{i}/{len(self.characters)}] {character.upper()}")
            self.process_character(character)

        # Final summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("Pipeline Complete!")
        logger.info("=" * 70)
        logger.info(f"Processed: {self.stats['processed']}/{len(self.characters)}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info("")
        logger.info(f"Training data: {OUTPUT_BASE}")
        logger.info(f"Configs: {CONFIGS_OUTPUT}")
        logger.info("")
        logger.info("Next step: Start training!")
        logger.info("  bash scripts/batch/train_super_wings_sdxl_loras.sh")

        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description="Complete Super Wings SDXL LoRA preparation with augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--characters',
        nargs='+',
        help='Specific characters to process (default: all)'
    )
    parser.add_argument(
        '--skip-captions',
        action='store_true',
        help='Skip caption generation (use existing captions)'
    )

    args = parser.parse_args()

    # Validate characters
    if args.characters:
        invalid = [c for c in args.characters if c not in CHARACTERS]
        if invalid:
            logger.error(f"Invalid characters: {invalid}")
            logger.error(f"Available: {list(CHARACTERS.keys())}")
            return 1

    # Run pipeline
    try:
        pipeline = SuperWingsDataPrep(
            characters=args.characters,
            skip_captions=args.skip_captions
        )
        stats = pipeline.run()

        return 0 if stats["failed"] == 0 else 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
