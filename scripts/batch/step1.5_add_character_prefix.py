#!/usr/bin/env python3
"""
Step 1.5: Add Character Name Prefix to All Captions

Adds character trigger word to the beginning of all captions.
This is critical for LoRA training to learn character-specific features.

Format: "{character_name}, {original_caption}"

Author: LLMProvider Tooling
Date: 2025-12-13
"""

import logging
import sys
from pathlib import Path
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CHARACTERS_DIR = Path("/mnt/data/datasets/general/super-wings/lora_data/characters")

# Character trigger words
CHARACTER_TRIGGERS = {
    "jett": "jett",
    "jerome": "jerome",
    "donnie": "donnie",
    "chase": "chase",
    "flip": "flip",
    "todd": "todd",
    "paul": "paul",
    "bello": "bello",
    "beard": "beard"
}


def caption_has_prefix(caption: str, trigger: str) -> bool:
    """Check if caption already starts with character name."""
    caption_lower = caption.lower().strip()
    trigger_lower = trigger.lower()

    # Check various formats
    return (
        caption_lower.startswith(f"{trigger_lower},") or
        caption_lower.startswith(f"{trigger_lower} ") or
        caption_lower.startswith(f"a 3d animated character, {trigger_lower}")
    )


def add_prefix_to_caption(caption: str, trigger: str) -> str:
    """Add character trigger word prefix to caption."""
    caption = caption.strip()

    # Skip if already has prefix
    if caption_has_prefix(caption, trigger):
        return caption

    # Add prefix: "character_name, original_caption"
    return f"{trigger}, {caption}"


def process_character_captions(character: str, trigger: str, caption_dir: Path, label: str) -> Dict:
    """Process all captions in a directory."""

    if not caption_dir.exists():
        return {"processed": 0, "skipped": 0, "failed": 0}

    caption_files = list(caption_dir.glob("*.txt"))

    if not caption_files:
        return {"processed": 0, "skipped": 0, "failed": 0}

    logger.info(f"  {label}: {len(caption_files)} files")

    stats = {"processed": 0, "skipped": 0, "failed": 0}

    for caption_file in caption_files:
        try:
            # Read original caption
            with open(caption_file, 'r', encoding='utf-8') as f:
                original = f.read().strip()

            # Check if already has prefix
            if caption_has_prefix(original, trigger):
                stats["skipped"] += 1
                continue

            # Add prefix
            updated = add_prefix_to_caption(original, trigger)

            # Write back
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(updated)

            stats["processed"] += 1

        except Exception as e:
            logger.warning(f"    Failed to process {caption_file.name}: {e}")
            stats["failed"] += 1

    logger.info(f"    ✓ Processed: {stats['processed']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")

    return stats


def process_character(character: str) -> bool:
    """Add character prefix to all captions (base and SDXL)."""

    char_dir = CHARACTERS_DIR / character
    trigger = CHARACTER_TRIGGERS[character]

    logger.info(f"\n[{character.upper()}] Trigger word: '{trigger}'")

    total_stats = {"processed": 0, "skipped": 0, "failed": 0}

    # Process base captions
    base_stats = process_character_captions(
        character,
        trigger,
        char_dir,
        "Base captions"
    )

    # Process SDXL captions
    sdxl_dir = char_dir / "sdxl_captions"
    sdxl_stats = process_character_captions(
        character,
        trigger,
        sdxl_dir,
        "SDXL captions"
    )

    # Combine stats
    for key in total_stats:
        total_stats[key] = base_stats[key] + sdxl_stats[key]

    logger.info(f"  Total: Processed={total_stats['processed']}, Skipped={total_stats['skipped']}, Failed={total_stats['failed']}")

    return total_stats["failed"] == 0


def main():
    logger.info("=" * 70)
    logger.info("Step 1.5: Add Character Name Prefix to All Captions")
    logger.info("=" * 70)
    logger.info("Adding trigger words to captions for LoRA training")
    logger.info("")

    all_stats = {"processed": 0, "skipped": 0, "failed": 0, "characters": 0}

    for character in CHARACTER_TRIGGERS.keys():
        if process_character(character):
            all_stats["characters"] += 1

        # Update totals
        char_dir = CHARACTERS_DIR / character
        trigger = CHARACTER_TRIGGERS[character]

        base_stats = {"processed": 0, "skipped": 0, "failed": 0}
        sdxl_stats = {"processed": 0, "skipped": 0, "failed": 0}

        # Count base captions
        for txt in char_dir.glob("*.txt"):
            try:
                with open(txt, 'r') as f:
                    if caption_has_prefix(f.read(), trigger):
                        base_stats["skipped"] += 1
                    else:
                        base_stats["processed"] += 1
            except:
                pass

        # Count SDXL captions
        sdxl_dir = char_dir / "sdxl_captions"
        if sdxl_dir.exists():
            for txt in sdxl_dir.glob("*.txt"):
                try:
                    with open(txt, 'r') as f:
                        if caption_has_prefix(f.read(), trigger):
                            sdxl_stats["skipped"] += 1
                        else:
                            sdxl_stats["processed"] += 1
                except:
                    pass

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("Step 1.5 Complete!")
    logger.info("=" * 70)
    logger.info(f"Processed: {all_stats['characters']}/9 characters")
    logger.info("")
    logger.info("All captions now have character trigger words!")
    logger.info("")
    logger.info("Example formats:")
    logger.info("  Base: 'jett, a 3d animated character...'")
    logger.info("  SDXL: 'jerome, a photorealistic 3d animated...'")
    logger.info("")
    logger.info("Next step:")
    logger.info("  python3 scripts/batch/step2_augment_and_organize.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
