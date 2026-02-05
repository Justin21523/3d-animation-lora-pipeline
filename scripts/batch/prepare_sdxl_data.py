#!/usr/bin/env python3
"""
Prepare SDXL training data by copying SD1.5 images to SDXL directories.

This script:
1. Reads TOML config files to find SDXL target directories
2. Maps them to corresponding SD1.5 source directories
3. Copies PNG images from SD1.5 to SDXL folders
4. Prepares data for preprocessing to 1024x1024

Usage:
    # Process specific TOML configs
    python scripts/batch/prepare_sdxl_data.py \
        configs/training/character_loras_sdxl/orion_orion_identity_sdxl.toml \
        configs/training/character_loras_sdxl/elio_bryce_identity_sdxl.toml

    # Process all TOML files in a directory
    python scripts/batch/prepare_sdxl_data.py \
        --config-dir configs/training/character_loras_sdxl

    # Process from a batch config file
    python scripts/batch/prepare_sdxl_data.py \
        --batch-config configs/training/quick_batch_6_characters.txt
"""
import os
import sys
import shutil
import glob
import argparse
from pathlib import Path
import re


def extract_train_data_dir(toml_path):
    """Extract train_data_dir from TOML file"""
    try:
        with open(toml_path, 'r') as f:
            for line in f:
                if 'train_data_dir' in line and '=' in line:
                    # Extract path from quotes
                    match = re.search(r'"([^"]+)"', line)
                    if match:
                        return match.group(1)
    except FileNotFoundError:
        print(f"  ❌ TOML file not found: {toml_path}")
        return None
    except Exception as e:
        print(f"  ❌ Error reading TOML: {e}")
        return None
    return None


def find_sd15_training_data(sdxl_path):
    """Find corresponding SD1.5 training data path"""
    # Convert SDXL path to SD1.5 path
    sd15_path = sdxl_path.replace('/training_data_sdxl/', '/training_data/')

    if not os.path.exists(sd15_path):
        return None

    # Find subdirectory with images
    subdirs = glob.glob(os.path.join(sd15_path, '*/'))
    for subdir in subdirs:
        images = glob.glob(os.path.join(subdir, '*.png'))
        if images:
            return subdir
    return None


def get_sdxl_subdir(sdxl_path):
    """Find the numbered subdirectory in SDXL path (e.g., 10_orion)"""
    if not os.path.exists(sdxl_path):
        return None

    subdirs = glob.glob(os.path.join(sdxl_path, '*/'))
    for subdir in subdirs:
        dirname = os.path.basename(subdir.rstrip('/'))
        # Match pattern like "10_orion", "12_bryce", etc.
        if re.match(r'^\d+_\w+$', dirname):
            return subdir
    return None


def copy_images(sd15_dir, sdxl_dir, force=False):
    """
    Copy PNG images from SD1.5 to SDXL directory

    Args:
        sd15_dir: Source directory with SD1.5 images
        sdxl_dir: Target SDXL directory
        force: If True, overwrite existing images

    Returns:
        Number of images copied
    """
    os.makedirs(sdxl_dir, exist_ok=True)

    png_files = glob.glob(os.path.join(sd15_dir, '*.png'))
    copied = 0
    skipped = 0

    for src in png_files:
        dst = os.path.join(sdxl_dir, os.path.basename(src))

        # Skip if destination exists and force is False
        if os.path.exists(dst) and not force:
            skipped += 1
            continue

        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            print(f"  ⚠️  Failed to copy {os.path.basename(src)}: {e}")

    if skipped > 0:
        print(f"  ℹ️  Skipped {skipped} existing images (use --force to overwrite)")

    return copied


def process_toml_config(toml_path, force=False, verbose=False):
    """
    Process a single TOML configuration file

    Returns:
        Number of images copied, or -1 on error
    """
    char_name = os.path.basename(toml_path).replace('_identity_sdxl.toml', '')
    print(f"=== Processing: {char_name} ===")

    # Extract SDXL training data path from TOML
    sdxl_base_path = extract_train_data_dir(toml_path)
    if not sdxl_base_path:
        print(f"  ❌ Could not extract train_data_dir from {toml_path}")
        return -1

    if verbose:
        print(f"  SDXL base path: {sdxl_base_path}")

    # Find SD1.5 source directory
    sd15_dir = find_sd15_training_data(sdxl_base_path)
    if not sd15_dir:
        print(f"  ⚠️  SD1.5 source not found (may already be SDXL-ready)")
        return 0

    sd15_count = len(glob.glob(os.path.join(sd15_dir, '*.png')))
    print(f"  SD1.5 source: {sd15_dir}")
    print(f"  SD1.5 images: {sd15_count}")

    # Find SDXL subdirectory
    sdxl_subdir = get_sdxl_subdir(sdxl_base_path)
    if not sdxl_subdir:
        # Create default subdirectory based on SD1.5 pattern
        sd15_subdir_name = os.path.basename(sd15_dir.rstrip('/'))
        sdxl_subdir = os.path.join(sdxl_base_path, sd15_subdir_name)
        print(f"  Creating SDXL subdir: {sdxl_subdir}")
    else:
        if verbose:
            print(f"  SDXL target: {sdxl_subdir}")

    # Copy images
    copied = copy_images(sd15_dir, sdxl_subdir, force=force)

    if copied > 0:
        print(f"  ✅ Copied {copied} images")
    elif copied == 0:
        print(f"  ℹ️  No new images to copy")

    print()
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SDXL training data by copying SD1.5 images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific TOML configs
  %(prog)s config1.toml config2.toml

  # Process all configs in a directory
  %(prog)s --config-dir configs/training/character_loras_sdxl

  # Process from batch config file
  %(prog)s --batch-config configs/training/quick_batch_6_characters.txt
        """
    )

    parser.add_argument(
        'toml_files',
        nargs='*',
        help='TOML configuration files to process'
    )
    parser.add_argument(
        '--config-dir',
        help='Directory containing TOML config files (process all *_sdxl.toml)'
    )
    parser.add_argument(
        '--batch-config',
        help='Batch config file listing TOML paths (one per line)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing images in SDXL directories'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Collect TOML files to process
    toml_configs = []

    # From direct arguments
    if args.toml_files:
        toml_configs.extend(args.toml_files)

    # From config directory
    if args.config_dir:
        pattern = os.path.join(args.config_dir, '*_sdxl.toml')
        found_configs = glob.glob(pattern)
        if not found_configs:
            print(f"⚠️  No *_sdxl.toml files found in {args.config_dir}")
        toml_configs.extend(found_configs)

    # From batch config file
    if args.batch_config:
        try:
            with open(args.batch_config, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        toml_configs.append(line)
        except FileNotFoundError:
            print(f"❌ Batch config file not found: {args.batch_config}")
            sys.exit(1)

    # Validate we have configs to process
    if not toml_configs:
        parser.print_help()
        print("\n❌ No TOML config files specified")
        sys.exit(1)

    # Process each config
    print("=" * 80)
    print("SDXL TRAINING DATA PREPARATION")
    print("=" * 80)
    print(f"Processing {len(toml_configs)} configuration(s)")
    print()

    total_copied = 0
    success_count = 0
    error_count = 0

    for toml_path in toml_configs:
        result = process_toml_config(toml_path, force=args.force, verbose=args.verbose)
        if result >= 0:
            total_copied += result
            success_count += 1
        else:
            error_count += 1

    # Summary
    print("=" * 80)
    print(f"✅ TOTAL IMAGES COPIED: {total_copied}")
    print(f"   Successful configs: {success_count}/{len(toml_configs)}")
    if error_count > 0:
        print(f"   ⚠️  Configs with errors: {error_count}")
    print("=" * 80)
    print()

    # Next steps
    if total_copied > 0:
        print("Next step: Run preprocessing to convert to 1024x1024")
        print("Command:")
        print("  conda run -n ai_env python scripts/batch/preprocess_images_for_sdxl.py \\")
        print("    --base-dir /mnt/data/ai_data/datasets/3d-anime \\")
        print("    --target-size square \\")
        print("    --report logs/preprocessing_report.json")
        print()


if __name__ == '__main__':
    main()
