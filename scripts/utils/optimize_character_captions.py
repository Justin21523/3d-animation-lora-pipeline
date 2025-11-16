#!/usr/bin/env python3
"""
Optimize character captions to emphasize age and body characteristics
優化角色 captions，強調年齡和身材特徵

Usage:
    # For Luca
    python scripts/utils/optimize_character_captions.py \
        --character luca \
        --captions-dir /path/to/luca_human/captions

    # For Alberto
    python scripts/utils/optimize_character_captions.py \
        --character alberto \
        --captions-dir /path/to/alberto_human/captions
"""

import argparse
from pathlib import Path
import shutil
from datetime import datetime

# Character-specific age and body descriptors
# Enhanced descriptions from ChatGPT image analysis (2025-11-11)
CHARACTER_DESCRIPTORS = {
    'luca': {
        'prefix': (
            "12-year-old italian pre-teen boy, short and slim build, "
            "large round brown eyes, thick arched eyebrows, button red-tinted nose, rosy cheeks, "
            "soft oval face, short dark-brown wavy curls with front quiff, barefoot, "
            "pixar stylized skin with subtle SSS and smooth shading, "
        ),
        'name_pattern': "Luca Paguro",
        'check_keywords': ["12-year-old", "pre-teen boy", "large round brown eyes"],
    },
    'alberto': {
        'prefix': (
            "14-year-old Italian teen boy, wiry athletic build, "
            "sun-tanned skin with light freckles, large bright green eyes, thick arched brows, "
            "red-tinted button nose, short tight curls with tall front quiff, "
            "yellow ribbed tank top, brown rolled shorts with rope-and-shell belt, barefoot, "
            "Pixar smooth SSS, "
        ),
        'name_pattern': "Alberto Scorfano",
        'check_keywords': ["14-year-old", "Italian teen boy", "large bright green eyes"],
    },
}


def backup_captions(captions_dir, backup_dir):
    """Create backup of original captions"""
    if backup_dir.exists():
        # Add timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_dir.parent / f"captions_backup_{timestamp}"

    shutil.copytree(captions_dir, backup_dir)
    print(f"✓ Backup created: {backup_dir}")
    return backup_dir


def optimize_caption(original_text, character_config):
    """
    Optimize a single caption by adding age and body descriptors

    Args:
        original_text: Original caption text
        character_config: Dictionary with prefix, name_pattern, check_keywords

    Returns:
        Optimized caption text or None if already optimized
    """
    prefix = character_config['prefix']
    name_pattern = character_config['name_pattern']
    check_keywords = character_config['check_keywords']

    # Check if already optimized
    if any(keyword in original_text for keyword in check_keywords):
        return None  # Already optimized

    # Strategy: Insert age descriptors before character name
    if name_pattern in original_text:
        optimized = original_text.replace(
            name_pattern,
            f"{prefix}{name_pattern}",
            1  # Only replace first occurrence
        )
    else:
        # Fallback: prepend to the beginning (after the generic prefix)
        optimized = original_text.replace(
            "a 3d animated character, ",
            f"a 3d animated character, {prefix}",
            1
        )

    return optimized


def main():
    parser = argparse.ArgumentParser(
        description="Optimize character captions for better age/body representation"
    )
    parser.add_argument(
        "--character",
        required=True,
        choices=list(CHARACTER_DESCRIPTORS.keys()),
        help="Character name (luca, alberto, etc.)"
    )
    parser.add_argument(
        "--captions-dir",
        type=Path,
        required=True,
        help="Path to captions directory"
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

    # Get character configuration
    character_config = CHARACTER_DESCRIPTORS[args.character]
    captions_dir = args.captions_dir
    backup_dir = captions_dir.parent / "captions_backup"

    print("=" * 70)
    print(f"{args.character.upper()} CAPTION OPTIMIZER")
    print("=" * 70)
    print(f"Character: {args.character}")
    print(f"Age/Body Prefix: {character_config['prefix']}")
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
    optimized_count = 0
    skipped_count = 0

    for caption_file in caption_files:
        # Read original
        with open(caption_file, 'r', encoding='utf-8') as f:
            original = f.read().strip()

        # Optimize
        optimized = optimize_caption(original, character_config)

        if optimized is None:
            skipped_count += 1
            continue

        # Show changes in dry-run mode
        if args.dry_run:
            print(f"{'='*70}")
            print(f"File: {caption_file.name}")
            print(f"{'-'*70}")
            print(f"Original:\n{original[:200]}...")
            print(f"{'-'*70}")
            print(f"Optimized:\n{optimized[:200]}...")
            print()
        else:
            # Write optimized caption
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(optimized)

        optimized_count += 1

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(caption_files)}")
    print(f"Optimized: {optimized_count}")
    print(f"Skipped (already optimized): {skipped_count}")

    if args.dry_run:
        print()
        print("⚠️  DRY RUN MODE - No files were modified")
        print("Remove --dry-run flag to apply changes")
    else:
        print()
        print("✓ Caption optimization completed!")
        if not args.no_backup:
            print(f"✓ Backup saved to: {backup_path}")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
