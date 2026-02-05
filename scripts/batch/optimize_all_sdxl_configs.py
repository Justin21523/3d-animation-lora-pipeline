#!/usr/bin/env python3
"""
Batch optimize all SDXL character configs for 6-7 hour training
- Sets repeats to 3 (via folder renaming)
- Sets epochs to 10
- Sets save_every_n_epochs to 2
- Sets sample_every_n_epochs to 2
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple

def extract_config_info(content: str) -> Dict:
    """Extract key info from config content"""
    info = {}

    # Extract dataset info
    dataset_match = re.search(r'# Dataset: (\d+) images.*?(\d+) repeats', content)
    if dataset_match:
        info['images'] = int(dataset_match.group(1))
        info['current_repeats'] = int(dataset_match.group(2))

    # Extract training dir
    train_dir_match = re.search(r'train_data_dir\s*=\s*"([^"]+)"', content)
    if train_dir_match:
        info['train_data_dir'] = train_dir_match.group(1)

    # Extract epochs
    epochs_match = re.search(r'max_train_epochs\s*=\s*(\d+)', content)
    if epochs_match:
        info['current_epochs'] = int(epochs_match.group(1))

    return info

def update_config_content(content: str, new_repeats: int = 3, new_epochs: int = 10) -> str:
    """Update config content with new values + GPU optimization"""

    # Update comment header
    content = re.sub(
        r'# Dataset: (\d+) images × (\d+) repeats = \d+ steps/epoch\n# Target: (\d+) epochs \(\d+ total steps\).*?$',
        lambda m: f'# Dataset: {m.group(1)} images × {new_repeats} repeats = {int(m.group(1)) * new_repeats} steps/epoch\n'
                  f'# Target: {new_epochs} epochs ({int(m.group(1)) * new_repeats * new_epochs} total steps) - GPU OPTIMIZED',
        content,
        flags=re.MULTILINE
    )

    # Update epochs
    content = re.sub(
        r'max_train_epochs\s*=\s*\d+',
        f'max_train_epochs = {new_epochs}',
        content
    )

    # Update save frequency
    content = re.sub(
        r'save_every_n_epochs\s*=\s*\d+',
        'save_every_n_epochs = 2',
        content
    )

    # Update sample frequency
    content = re.sub(
        r'sample_every_n_epochs\s*=\s*\d+',
        'sample_every_n_epochs = 2',
        content
    )

    # GPU OPTIMIZATION: Increase batch size
    content = re.sub(
        r'train_batch_size\s*=\s*\d+',
        'train_batch_size = 2',
        content
    )

    # GPU OPTIMIZATION: Keep gradient accumulation
    content = re.sub(
        r'gradient_accumulation_steps\s*=\s*\d+',
        'gradient_accumulation_steps = 2',
        content
    )

    # GPU OPTIMIZATION: Increase VAE batch size
    content = re.sub(
        r'vae_batch_size\s*=\s*\d+',
        'vae_batch_size = 4',
        content
    )

    # GPU OPTIMIZATION: Increase data loader workers
    content = re.sub(
        r'max_data_loader_n_workers\s*=\s*\d+',
        'max_data_loader_n_workers = 4',
        content
    )

    return content

def rename_dataset_folder(train_data_dir: str, old_repeats: int, new_repeats: int) -> Tuple[bool, str]:
    """Rename dataset folder from old_repeats to new_repeats"""

    # Find folders matching pattern
    base_dir = Path(train_data_dir)
    if not base_dir.exists():
        return False, f"Training data dir not found: {train_data_dir}"

    # Look for folders like "10_character_name"
    old_folders = list(base_dir.glob(f"{old_repeats}_*"))

    if not old_folders:
        # Try to find any folder with repeats pattern
        all_repeat_folders = list(base_dir.glob("*_*"))
        if all_repeat_folders:
            return False, f"Found folders {[f.name for f in all_repeat_folders]}, but no {old_repeats}_* pattern"
        return False, f"No dataset folders found in {train_data_dir}"

    results = []
    for old_folder in old_folders:
        # Extract character name
        char_name = old_folder.name.split('_', 1)[1]
        new_folder = base_dir / f"{new_repeats}_{char_name}"

        if new_folder.exists():
            results.append(f"  ⚠️  {new_folder.name} already exists, skipping")
        else:
            try:
                old_folder.rename(new_folder)
                results.append(f"  ✅ {old_folder.name} → {new_folder.name}")
            except Exception as e:
                results.append(f"  ❌ Failed to rename {old_folder.name}: {e}")

    return True, "\n".join(results)

def main():
    config_dir = Path("configs/training/character_loras_sdxl")
    configs = sorted(config_dir.glob("*.toml"))

    print("=" * 80)
    print("⚡ BATCH OPTIMIZATION: All SDXL Character Configs → 3 repeats × 10 epochs")
    print("=" * 80)
    print()

    target_repeats = 3
    target_epochs = 10

    summary = []

    for i, config_path in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Processing: {config_path.name}")
        print("-" * 80)

        # Read config
        with open(config_path) as f:
            content = f.read()

        # Extract info
        info = extract_config_info(content)

        if 'images' not in info:
            print("  ⚠️  Could not extract dataset info, skipping")
            continue

        images = info['images']
        old_repeats = info.get('current_repeats', '?')
        old_epochs = info.get('current_epochs', '?')

        print(f"  Current: {images} images × {old_repeats} repeats × {old_epochs} epochs")

        # Skip if already optimized
        if old_repeats == target_repeats and old_epochs == target_epochs:
            print(f"  ✅ Already optimized, skipping")
            summary.append((config_path.name, "already optimized", images * target_repeats * target_epochs))
            continue

        # Update config content
        new_content = update_config_content(content, target_repeats, target_epochs)

        # Write back
        with open(config_path, 'w') as f:
            f.write(new_content)

        print(f"  ✅ Config updated: {target_repeats} repeats × {target_epochs} epochs")

        # Rename dataset folder
        if 'train_data_dir' in info and old_repeats != target_repeats:
            success, msg = rename_dataset_folder(info['train_data_dir'], old_repeats, target_repeats)
            print(msg)
            if success:
                summary.append((config_path.name, "updated", images * target_repeats * target_epochs))
            else:
                summary.append((config_path.name, "config updated, folder issue", images * target_repeats * target_epochs))
        else:
            summary.append((config_path.name, "config updated", images * target_repeats * target_epochs))

    # Print summary
    print("\n\n" + "=" * 80)
    print("📊 OPTIMIZATION SUMMARY")
    print("=" * 80)
    print()

    for name, status, steps in summary:
        print(f"  {name:50} | {status:25} | {steps:6,} steps")

    print()
    print("✅ Batch optimization complete!")
    print()
    print("⏱️  Estimated training time per character: 3-7 hours")
    print("   (depends on image count and GPU speed)")

if __name__ == "__main__":
    main()
