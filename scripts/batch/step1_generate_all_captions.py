#!/usr/bin/env python3
"""
Step 1: Generate Base Captions for All Super Wings Characters

Only generates captions, does NOT do augmentation yet.

Author: LLMProvider Tooling
Date: 2025-12-13
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.parent
CHARACTERS_DIR = Path("/mnt/data/datasets/general/super-wings/lora_data/characters")
RECAPTION_SCRIPT = SCRIPT_DIR / "generic/training/caption_engines/recaption_with_llm_provider.py"
SDXL_EXPANDER_SCRIPT = SCRIPT_DIR / "generic/training/sdxl_caption_expander.py"

CHARACTERS = ["jett", "jerome", "donnie", "chase", "flip", "todd", "paul", "bello", "beard"]


def count_files(directory: Path, pattern: str) -> int:
    """Count files matching pattern."""
    return len(list(directory.glob(pattern)))


def has_captions(char_dir: Path) -> bool:
    """Check if character has captions."""
    num_images = count_files(char_dir, "*.png") + count_files(char_dir, "*.jpg")
    num_captions = count_files(char_dir, "*.txt")
    return num_captions >= num_images * 0.9


def has_sdxl_captions(char_dir: Path) -> bool:
    """Check if character has SDXL captions."""
    sdxl_dir = char_dir / "sdxl_captions"
    if not sdxl_dir.exists():
        return False
    return count_files(sdxl_dir, "*.txt") > 0


def generate_base_captions(character: str) -> bool:
    """Generate base captions using LLMProvider API."""
    char_dir = CHARACTERS_DIR / character

    if has_captions(char_dir):
        logger.info(f"[{character}] Base captions already exist, skipping")
        return True

    logger.info(f"[{character}] Generating base captions...")

    cmd = [
        "python3",
        str(RECAPTION_SCRIPT),
        "--input-dir", str(char_dir),
        "--output-dir", str(char_dir),
        "--lora-type", "character",
        "--character-name", character,
        "--force"
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"[{character}] ✓ Base captions generated")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[{character}] Failed: {e.stderr}")
        return False


def expand_to_sdxl(character: str) -> bool:
    """Expand captions to SDXL format."""
    char_dir = CHARACTERS_DIR / character
    sdxl_dir = char_dir / "sdxl_captions"

    if has_sdxl_captions(char_dir):
        logger.info(f"[{character}] SDXL captions already exist, skipping")
        return True

    logger.info(f"[{character}] Expanding to SDXL format...")

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
        logger.error(f"[{character}] Failed: {e.stderr}")
        return False


def main():
    if not os.getenv('LLM_VENDOR_API_KEY'):
        logger.error("LLM_VENDOR_API_KEY not found!")
        return 1

    logger.info("=" * 70)
    logger.info("Step 1: Generate All Captions (Base + SDXL)")
    logger.info("=" * 70)
    logger.info(f"Characters: {', '.join(CHARACTERS)}")
    logger.info("")

    stats = {"success": 0, "failed": 0, "skipped": 0}

    # Step 1a: Generate base captions
    logger.info("=" * 70)
    logger.info("Step 1a: Base Caption Generation")
    logger.info("=" * 70)

    for i, char in enumerate(CHARACTERS, 1):
        logger.info(f"\n[{i}/{len(CHARACTERS)}] {char.upper()}")
        if generate_base_captions(char):
            stats["success"] += 1
        else:
            stats["failed"] += 1

    # Step 1b: Expand to SDXL
    logger.info("\n" + "=" * 70)
    logger.info("Step 1b: SDXL Caption Expansion")
    logger.info("=" * 70)

    for i, char in enumerate(CHARACTERS, 1):
        logger.info(f"\n[{i}/{len(CHARACTERS)}] {char.upper()}")
        if not expand_to_sdxl(char):
            stats["failed"] += 1

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Step 1 Complete!")
    logger.info("=" * 70)
    logger.info(f"Processed: {len(CHARACTERS)} characters")
    logger.info(f"Failed: {stats['failed']}")
    logger.info("")
    logger.info("Next step:")
    logger.info("  python3 scripts/batch/step2_augment_and_organize.py")

    return 0 if stats["failed"] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
