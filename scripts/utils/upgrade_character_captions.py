#!/usr/bin/env python3
"""
Upgrade character captions from simple to detailed descriptions
升級角色 captions，從簡單描述到詳細描述

Usage:
    python scripts/utils/upgrade_character_captions.py --character luca
    python scripts/utils/upgrade_character_captions.py --character alberto
"""

import argparse
from pathlib import Path
import shutil
from datetime import datetime

# Old simple descriptions to replace
OLD_DESCRIPTORS = {
    'luca': "12-year-old boy, teenage character, slim youthful build, ",
    'alberto': "14-year-old boy, teenage character, athletic lean build, ",
}

# New detailed descriptions from ChatGPT analysis
NEW_DESCRIPTORS = {
    'luca': (
        "12-year-old italian pre-teen boy, short and slim build, "
        "large round brown eyes, thick arched eyebrows, button red-tinted nose, rosy cheeks, "
        "soft oval face, short dark-brown wavy curls with front quiff, barefoot, "
        "pixar stylized skin with subtle SSS and smooth shading, "
    ),
    'alberto': (
        "14-year-old Italian teen boy, wiry athletic build, "
        "sun-tanned skin with light freckles, large bright green eyes, thick arched brows, "
        "red-tinted button nose, short tight curls with tall front quiff, "
        "yellow ribbed tank top, brown rolled shorts with rope-and-shell belt, barefoot, "
        "Pixar smooth SSS, "
    ),
}

# Dataset paths
DATASET_PATHS = {
    'luca': Path("/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset/luca_human/captions"),
    'alberto': Path("/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset/alberto_human/captions"),
}


def backup_captions(captions_dir, backup_dir):
    """Create backup of original captions"""
    if backup_dir.exists():
        # Add timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_dir.parent / f"captions_backup_upgrade_{timestamp}"

    shutil.copytree(captions_dir, backup_dir)
    print(f"✓ Backup created: {backup_dir}")
    return backup_dir


def upgrade_caption(original_text, character):
    """
    Upgrade caption by replacing old simple description with new detailed description

    Args:
        original_text: Original caption text
        character: Character name (luca or alberto)

    Returns:
        Upgraded caption text or None if no upgrade needed
    """
    old_desc = OLD_DESCRIPTORS[character]
    new_desc = NEW_DESCRIPTORS[character]

    # Check if old description exists
    if old_desc not in original_text:
        return None  # Already upgraded or different format

    # Replace old with new
    upgraded = original_text.replace(old_desc, new_desc, 1)

    return upgraded


def main():
    parser = argparse.ArgumentParser(
        description="Upgrade character captions from simple to detailed descriptions"
    )
    parser.add_argument(
        "--character",
        required=True,
        choices=list(OLD_DESCRIPTORS.keys()),
        help="Character name (luca, alberto)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing files"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation"
    )
    args = parser.parse_args()

    character = args.character
    captions_dir = DATASET_PATHS[character]
    backup_dir = captions_dir.parent / "captions_backup_upgrade"

    print("=" * 70)
    print(f"{character.upper()} CAPTION UPGRADER")
    print("=" * 70)
    print(f"Character: {character}")
    print(f"\nOLD (simple):")
    print(f"  {OLD_DESCRIPTORS[character]}")
    print(f"\nNEW (detailed):")
    print(f"  {NEW_DESCRIPTORS[character][:100]}...")
    print()

    # Check if captions directory exists
    if not captions_dir.exists():
        print(f"❌ Captions directory not found: {captions_dir}")
        return 1

    # Get all caption files
    caption_files = list(captions_dir.glob("*.txt"))
    if not caption_files:
        print(f"❌ No caption files found in {captions_dir}")
        return 1

    print(f"Found {len(caption_files)} caption files")
    print()

    # Create backup (unless --no-backup or --dry-run)
    if not args.no_backup and not args.dry_run:
        backup_path = backup_captions(captions_dir, backup_dir)
        print()

    # Process each caption
    upgraded_count = 0
    skipped_count = 0

    for caption_file in caption_files:
        # Read original
        with open(caption_file, 'r', encoding='utf-8') as f:
            original = f.read().strip()

        # Upgrade
        upgraded = upgrade_caption(original, character)

        if upgraded is None:
            skipped_count += 1
            continue

        # Show changes in dry-run mode
        if args.dry_run:
            print(f"{'='*70}")
            print(f"File: {caption_file.name}")
            print(f"{'-'*70}")
            print(f"OLD:\n{original[:150]}...")
            print(f"{'-'*70}")
            print(f"NEW:\n{upgraded[:150]}...")
            print()
        else:
            # Write upgraded caption
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(upgraded)

        upgraded_count += 1

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(caption_files)}")
    print(f"Upgraded: {upgraded_count}")
    print(f"Skipped (already upgraded or different format): {skipped_count}")

    if args.dry_run:
        print()
        print("⚠️  DRY RUN MODE - No files were modified")
        print("Remove --dry-run flag to apply changes")
    else:
        print()
        print("✓ Caption upgrade completed!")
        if not args.no_backup:
            print(f"✓ Backup saved to: {backup_path}")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
