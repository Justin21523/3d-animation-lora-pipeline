#!/usr/bin/env python3
"""
Optimize Luca captions to emphasize age and body characteristics
優化 Luca 的 captions，強調年齡和身材特徵

Usage:
    python scripts/utils/optimize_luca_captions.py [--dry-run]
"""

import argparse
from pathlib import Path
import shutil
from datetime import datetime

# Luca captions directory
CAPTIONS_DIR = Path("/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset/luca_human/captions")

# Age and body descriptors to add
AGE_PREFIX = "12-year-old boy, teenage character, slim youthful build, "

# Backup directory
BACKUP_DIR = CAPTIONS_DIR.parent / "captions_backup"


def backup_captions(captions_dir, backup_dir):
    """Create backup of original captions"""
    if backup_dir.exists():
        # Add timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_dir.parent / f"captions_backup_{timestamp}"

    shutil.copytree(captions_dir, backup_dir)
    print(f"✓ Backup created: {backup_dir}")
    return backup_dir


def optimize_caption(original_text):
    """
    Optimize a single caption by adding age and body descriptors

    Args:
        original_text: Original caption text

    Returns:
        Optimized caption text
    """
    # Check if already optimized
    if "12-year-old" in original_text or "teenage character" in original_text:
        return None  # Already optimized

    # Strategy: Insert age descriptors before "Luca Paguro"
    if "Luca Paguro" in original_text:
        optimized = original_text.replace(
            "Luca Paguro",
            f"{AGE_PREFIX}Luca Paguro",
            1  # Only replace first occurrence
        )
    else:
        # Fallback: prepend to the beginning (after the generic prefix)
        optimized = original_text.replace(
            "a 3d animated character, ",
            f"a 3d animated character, {AGE_PREFIX}",
            1
        )

    return optimized


def main():
    parser = argparse.ArgumentParser(description="Optimize Luca captions for better age/body representation")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing files")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    args = parser.parse_args()

    print("=" * 70)
    print("LUCA CAPTION OPTIMIZER")
    print("=" * 70)
    print()

    # Check if captions directory exists
    if not CAPTIONS_DIR.exists():
        print(f"❌ Captions directory not found: {CAPTIONS_DIR}")
        return 1

    # Get all caption files
    caption_files = list(CAPTIONS_DIR.glob("*.txt"))
    if not caption_files:
        print(f"❌ No caption files found in {CAPTIONS_DIR}")
        return 1

    print(f"Found {len(caption_files)} caption files")
    print()

    # Create backup (unless --no-backup or --dry-run)
    if not args.no_backup and not args.dry_run:
        backup_path = backup_captions(CAPTIONS_DIR, BACKUP_DIR)
        print()

    # Process each caption
    optimized_count = 0
    skipped_count = 0

    for caption_file in caption_files:
        # Read original
        with open(caption_file, 'r', encoding='utf-8') as f:
            original = f.read().strip()

        # Optimize
        optimized = optimize_caption(original)

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
