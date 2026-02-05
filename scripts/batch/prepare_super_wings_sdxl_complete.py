#!/usr/bin/env python3
"""
Complete Super Wings SDXL LoRA Data Preparation Pipeline

Automated end-to-end pipeline:
1. Generate captions using LLMProvider API (recaption_with_llm_provider.py)
2. Expand to SDXL format (sdxl_caption_expander.py)
3. Organize into Kohya training format
4. Generate training configs

Author: LLMProvider Tooling
Date: 2025-12-13
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
CHARACTERS_DIR = Path("/mnt/data/datasets/general/super-wings/lora_data/characters")
OUTPUT_BASE = Path("/mnt/data/datasets/general/super-wings/training_data")
CONFIGS_OUTPUT = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/super_wings_loras")

# Scripts
RECAPTION_SCRIPT = SCRIPT_DIR / "generic/training/caption_engines/recaption_with_llm_provider.py"
SDXL_EXPANDER_SCRIPT = SCRIPT_DIR / "generic/training/sdxl_caption_expander.py"

# Character definitions (Super Wings)
CHARACTERS = {
    "jett": {
        "name": "jett",
        "description": "small red and white jet plane with blue eyes",
        "style": "dreamworks"
    },
    "jerome": {
        "name": "jerome",
        "description": "large blue cargo plane with friendly eyes",
        "style": "dreamworks"
    },
    "donnie": {
        "name": "donnie",
        "description": "orange construction transport plane",
        "style": "dreamworks"
    },
    "chase": {
        "name": "chase",
        "description": "blue police jet plane",
        "style": "dreamworks"
    },
    "flip": {
        "name": "flip",
        "description": "yellow acrobatic jet plane",
        "style": "dreamworks"
    },
    "todd": {
        "name": "todd",
        "description": "orange rocket plane",
        "style": "dreamworks"
    },
    "paul": {
        "name": "paul",
        "description": "blue police helicopter",
        "style": "dreamworks"
    },
    "bello": {
        "name": "bello",
        "description": "purple and pink glamorous jet plane",
        "style": "dreamworks"
    },
    "beard": {
        "name": "beard",
        "description": "bearded human character",
        "style": "dreamworks"
    }
}


class SuperWingsDataPreparation:
    """Complete data preparation pipeline for Super Wings SDXL LoRA training."""

    def __init__(self, characters: List[str] = None, skip_existing: bool = True):
        """
        Initialize preparation pipeline.

        Args:
            characters: List of character names to process (None = all)
            skip_existing: Skip characters that already have captions
        """
        self.characters = characters or list(CHARACTERS.keys())
        self.skip_existing = skip_existing
        self.stats = {
            "processed": 0,
            "skipped": 0,
            "failed": 0
        }

        # Check API key
        if not os.getenv('LLM_VENDOR_API_KEY'):
            raise ValueError(
                "LLM_VENDOR_API_KEY not found. Set it with:\n"
                "export LLM_VENDOR_API_KEY='your-api-key-here'"
            )

    def count_images(self, character_dir: Path) -> int:
        """Count images in character directory."""
        return len(list(character_dir.glob("*.png"))) + len(list(character_dir.glob("*.jpg")))

    def count_captions(self, character_dir: Path) -> int:
        """Count existing caption files."""
        return len(list(character_dir.glob("*.txt")))

    def step1_generate_captions(self, character: str) -> bool:
        """
        Step 1: Generate base captions using LLMProvider API.

        Args:
            character: Character name

        Returns:
            True if successful
        """
        char_dir = CHARACTERS_DIR / character
        if not char_dir.exists():
            logger.warning(f"Character directory not found: {char_dir}")
            return False

        num_images = self.count_images(char_dir)
        num_captions = self.count_captions(char_dir)

        logger.info(f"[{character}] Images: {num_images}, Existing captions: {num_captions}")

        if self.skip_existing and num_captions > 0:
            logger.info(f"[{character}] Captions already exist, skipping caption generation")
            return True

        # Generate captions using recaption_with_llm_provider.py
        logger.info(f"[{character}] Generating captions with LLMProvider API...")

        cmd = [
            "python3",
            str(RECAPTION_SCRIPT),
            "--input-dir", str(char_dir),
            "--output-dir", str(char_dir),  # Output in same directory
            "--lora-type", "character",
            "--force"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"[{character}] ✓ Captions generated")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"[{character}] Caption generation failed: {e.stderr}")
            return False

    def step2_expand_to_sdxl(self, character: str) -> bool:
        """
        Step 2: Expand captions to SDXL format (225 tokens).

        Args:
            character: Character name

        Returns:
            True if successful
        """
        char_dir = CHARACTERS_DIR / character
        temp_sdxl_dir = char_dir / "sdxl_captions"

        logger.info(f"[{character}] Expanding captions to SDXL format...")

        char_config = CHARACTERS[character]

        cmd = [
            "python3",
            str(SDXL_EXPANDER_SCRIPT),
            "--input-dir", str(char_dir),
            "--output-dir", str(temp_sdxl_dir),
            "--character-name", char_config["name"],
            "--style", char_config["style"]
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"[{character}] ✓ SDXL captions generated")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"[{character}] SDXL expansion failed: {e.stderr}")
            return False

    def step3_organize_kohya_format(self, character: str) -> bool:
        """
        Step 3: Organize into Kohya training format.

        Kohya format:
        training_data/
          character_name/
            10_character_name/
              image001.png
              image001.txt
              ...

        Args:
            character: Character name

        Returns:
            True if successful
        """
        char_dir = CHARACTERS_DIR / character
        sdxl_captions_dir = char_dir / "sdxl_captions"

        num_images = self.count_images(char_dir)

        # Calculate optimal repeats (aim for 1000-1500 total steps per epoch)
        # For 200 images: ~7 repeats = 1400 steps
        # For 100 images: ~12 repeats = 1200 steps
        if num_images >= 200:
            repeats = 7
        elif num_images >= 150:
            repeats = 9
        elif num_images >= 100:
            repeats = 12
        else:
            repeats = 15

        # Create Kohya directory structure
        training_dir = OUTPUT_BASE / character
        kohya_dir = training_dir / f"{repeats}_{character}"

        logger.info(f"[{character}] Organizing into Kohya format...")
        logger.info(f"  Images: {num_images}, Repeats: {repeats}")
        logger.info(f"  Output: {kohya_dir}")

        # Create directory
        kohya_dir.mkdir(parents=True, exist_ok=True)

        # Copy images and SDXL captions
        copied = 0
        for img_path in char_dir.glob("*.png"):
            # Copy image
            dst_img = kohya_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            # Copy corresponding SDXL caption
            sdxl_caption = sdxl_captions_dir / f"{img_path.stem}.txt"
            if sdxl_caption.exists():
                dst_caption = kohya_dir / f"{img_path.stem}.txt"
                shutil.copy2(sdxl_caption, dst_caption)
                copied += 1
            else:
                logger.warning(f"  Missing SDXL caption for {img_path.name}")

        # Also copy JPG images if any
        for img_path in char_dir.glob("*.jpg"):
            dst_img = kohya_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            sdxl_caption = sdxl_captions_dir / f"{img_path.stem}.txt"
            if sdxl_caption.exists():
                dst_caption = kohya_dir / f"{img_path.stem}.txt"
                shutil.copy2(sdxl_caption, dst_caption)
                copied += 1

        logger.info(f"[{character}] ✓ Organized: {copied} image-caption pairs")

        # Save metadata
        metadata = {
            "character": character,
            "num_images": num_images,
            "repeats": repeats,
            "total_steps_per_epoch": num_images * repeats,
            "kohya_dir": str(kohya_dir)
        }

        with open(training_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return True

    def step4_generate_training_config(self, character: str) -> bool:
        """
        Step 4: Generate SDXL training config.

        Args:
            character: Character name

        Returns:
            True if successful
        """
        logger.info(f"[{character}] Generating SDXL training config...")

        # Load metadata
        training_dir = OUTPUT_BASE / character
        metadata_file = training_dir / "metadata.json"

        if not metadata_file.exists():
            logger.error(f"[{character}] Metadata not found")
            return False

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Calculate training parameters
        num_images = metadata["num_images"]

        # Optimal epochs based on dataset size
        if num_images >= 200:
            max_epochs = 2
            learning_rate = 0.0001
        elif num_images >= 150:
            max_epochs = 2
            learning_rate = 0.00011
        else:
            max_epochs = 3
            learning_rate = 0.00012

        # Load template
        template_path = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_lora_sdxl_template.toml")

        if not template_path.exists():
            logger.error(f"Template not found: {template_path}")
            return False

        with open(template_path, 'r') as f:
            template = f.read()

        # Replace placeholders
        config = template.replace("{CHARACTER_NAME}", character)

        # Update training parameters
        import re
        config = re.sub(r'max_train_epochs\s*=\s*\d+', f'max_train_epochs = {max_epochs}', config)
        config = re.sub(r'learning_rate\s*=\s*[\d.e-]+', f'learning_rate = {learning_rate}', config)

        # Save config
        CONFIGS_OUTPUT.mkdir(parents=True, exist_ok=True)
        config_path = CONFIGS_OUTPUT / f"super-wings-{character}-sdxl.toml"

        with open(config_path, 'w') as f:
            f.write(config)

        logger.info(f"[{character}] ✓ Config saved: {config_path.name}")
        logger.info(f"  Epochs: {max_epochs}, LR: {learning_rate}")

        return True

    def process_character(self, character: str) -> bool:
        """
        Process a single character through all steps.

        Args:
            character: Character name

        Returns:
            True if successful
        """
        logger.info("=" * 70)
        logger.info(f"Processing: {character}")
        logger.info("=" * 70)

        try:
            # Step 1: Generate base captions
            if not self.step1_generate_captions(character):
                logger.error(f"[{character}] Failed at step 1 (caption generation)")
                return False

            # Step 2: Expand to SDXL
            if not self.step2_expand_to_sdxl(character):
                logger.error(f"[{character}] Failed at step 2 (SDXL expansion)")
                return False

            # Step 3: Organize Kohya format
            if not self.step3_organize_kohya_format(character):
                logger.error(f"[{character}] Failed at step 3 (Kohya organization)")
                return False

            # Step 4: Generate training config
            if not self.step4_generate_training_config(character):
                logger.error(f"[{character}] Failed at step 4 (config generation)")
                return False

            logger.info(f"[{character}] ✓ Complete!")
            self.stats["processed"] += 1
            return True

        except Exception as e:
            logger.error(f"[{character}] Unexpected error: {e}", exc_info=True)
            self.stats["failed"] += 1
            return False

    def run(self) -> Dict:
        """
        Run complete pipeline for all characters.

        Returns:
            Statistics dictionary
        """
        logger.info("=" * 70)
        logger.info("Super Wings SDXL LoRA - Complete Data Preparation")
        logger.info("=" * 70)
        logger.info(f"Characters to process: {len(self.characters)}")
        logger.info(f"Characters: {', '.join(self.characters)}")
        logger.info("")

        for i, character in enumerate(self.characters, 1):
            logger.info(f"\n[{i}/{len(self.characters)}] {character.upper()}")
            self.process_character(character)

        # Final summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("Pipeline Complete!")
        logger.info("=" * 70)
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Skipped: {self.stats['skipped']}")
        logger.info("")
        logger.info(f"Training data: {OUTPUT_BASE}")
        logger.info(f"Training configs: {CONFIGS_OUTPUT}")
        logger.info("")
        logger.info("Next step:")
        logger.info("  python scripts/batch/train_super_wings_sdxl_loras.py")

        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description="Complete Super Wings SDXL LoRA data preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Process all characters
  python prepare_super_wings_sdxl_complete.py

  # Process specific characters
  python prepare_super_wings_sdxl_complete.py --characters jett jerome donnie

  # Force regenerate captions
  python prepare_super_wings_sdxl_complete.py --no-skip-existing

Requirements:
  export LLM_VENDOR_API_KEY='your-api-key-here'
        """
    )

    parser.add_argument(
        '--characters',
        nargs='+',
        help='Specific characters to process (default: all)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Regenerate captions even if they exist'
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
        pipeline = SuperWingsDataPreparation(
            characters=args.characters,
            skip_existing=not args.no_skip_existing
        )
        stats = pipeline.run()

        return 0 if stats["failed"] == 0 else 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
