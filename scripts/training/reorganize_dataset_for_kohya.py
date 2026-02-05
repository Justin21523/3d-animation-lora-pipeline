#!/usr/bin/env python3
"""
Reorganize Inazuma Eleven dataset to match Kohya_ss expected structure.
Kohya expects: train_data_dir/REPEATS_TOKEN/image
Structure: /path/to/train_data_dir/1_character_token/image.png
"""

import shutil
from pathlib import Path

CHARACTERS = {
    "Endou Mamoru": "endou_mamoru",
    "Fudou Akio": "fudou_akio",
    "Gouenji Shuuya": "gouenji_shuuya",
    "Inamori Asuto": "inamori_asuto",
    "Matsukaze Tenma": "matsukaze_tenma",
    "Nosaka Yuuma": "nosaka_yuuma",
    "Utsunomiya Toramaru": "utsunomiya_toramaru",
}

DATASET_BASE_DIR = Path("/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented")

def reorganize_dataset():
    """Reorganize dataset for Kohya compatibility."""

    for character_name, character_id in CHARACTERS.items():
        source_dir = DATASET_BASE_DIR / character_name
        if not source_dir.exists():
            print(f"WARNING: {source_dir} does not exist, skipping {character_name}")
            continue

        # Create target directory: 1_character_token (repeat count = 1)
        target_dir = DATASET_BASE_DIR / f"1_{character_id}"

        # If target already exists from previous run, skip
        if target_dir.exists():
            file_count = len(list(target_dir.glob("*")))
            if file_count > 0:
                print(f"✓ {character_name}: already organized in {target_dir}")
                continue

        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files from source to target
        image_files = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg"))
        caption_files = list(source_dir.glob("*.txt"))

        for file in image_files + caption_files:
            shutil.copy2(file, target_dir / file.name)

        print(f"✓ {character_name}: {len(image_files)} images → {target_dir}")

if __name__ == "__main__":
    print("Reorganizing dataset for Kohya_ss compatibility...")
    print(f"Dataset base: {DATASET_BASE_DIR}\n")
    reorganize_dataset()
    print("\nDataset reorganization complete!")
