#!/usr/bin/env python3
"""
Prepare Universal and Character-Specific Training Datasets

This script creates:
1. Universal LoRAs (3): Balanced sampling across all characters
2. Character-specific LoRAs (37): All available data per character

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def scan_tier_a_data(filtered_root: Path) -> Dict:
    """Scan all tier_a data"""
    data_count = defaultdict(lambda: {"pose": 0, "action": 0, "expression": 0, "paths": {"pose": [], "action": [], "expression": []}})

    for char_dir in filtered_root.iterdir():
        if not char_dir.is_dir():
            continue

        char_name = char_dir.name

        for lora_type in ["pose", "action", "expression"]:
            tier_a_dir = char_dir / lora_type / "tier_a"

            if tier_a_dir.exists():
                images = list(tier_a_dir.glob("*.png"))
                data_count[char_name][lora_type] = len(images)
                data_count[char_name]["paths"][lora_type] = images

    return dict(data_count)


def prepare_universal_datasets(
    data_count: Dict,
    output_root: Path,
    target_per_char_per_type: int
):
    """
    Prepare 3 universal LoRA datasets with balanced sampling

    For each lora_type (pose/action/expression):
    - Sample equally from each character
    - Create single dataset for universal LoRA
    """
    print("\n" + "=" * 80)
    print("準備 Universal LoRAs (平衡採樣)")
    print("=" * 80)
    print(f"目標: 每個角色每類型 ~{target_per_char_per_type} 張（圖多的角色可以更多）")
    print()

    universal_stats = defaultdict(lambda: {"total": 0, "by_character": defaultdict(int)})

    for lora_type in ["pose", "action", "expression"]:
        # Output directory (Kohya format: 1_concept)
        output_dir = output_root / f"universal_{lora_type}" / f"1_universal_{lora_type}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📦 Universal {lora_type.upper()}")
        print("-" * 80)

        for char in sorted(data_count.keys()):
            char_images = data_count[char]["paths"][lora_type]
            # Allow more for image-rich characters (up to 150% of target)
            max_allowed = int(target_per_char_per_type * 1.5)
            target = min(len(char_images), max_allowed)

            selected_images = char_images[:target]

            # Copy images
            for img_path in selected_images:
                dest_img = output_dir / f"{char}_{img_path.name}"
                shutil.copy2(img_path, dest_img)

                # Copy caption
                caption_path = img_path.with_suffix(".txt")
                if caption_path.exists():
                    dest_caption = output_dir / f"{char}_{caption_path.name}"
                    shutil.copy2(caption_path, dest_caption)

                universal_stats[lora_type]["total"] += 1
                universal_stats[lora_type]["by_character"][char] += 1

            print(f"  {char:25s}  {len(selected_images):4d} 張")

        print(f"\n  ✅ Total: {universal_stats[lora_type]['total']} 張")

    return dict(universal_stats)


def prepare_character_datasets(
    data_count: Dict,
    output_root: Path,
    min_images_threshold: int = 50
):
    """
    Prepare character-specific LoRA datasets

    For each character:
    - Use ALL available images (no balancing)
    - Create separate dataset for each lora_type
    """
    print("\n" + "=" * 80)
    print("準備 Character-Specific LoRAs (使用所有資料)")
    print("=" * 80)
    print(f"最小圖片門檻: {min_images_threshold}")
    print()

    character_stats = defaultdict(lambda: defaultdict(int))
    skipped = []

    for char in sorted(data_count.keys()):
        print(f"\n📦 {char}")
        print("-" * 80)

        for lora_type in ["pose", "action", "expression"]:
            char_images = data_count[char]["paths"][lora_type]
            count = len(char_images)

            # Skip if below threshold
            if count < min_images_threshold:
                print(f"  {lora_type:12s}  ⏭️  跳過 ({count} < {min_images_threshold})")
                skipped.append(f"{char}_{lora_type}")
                continue

            # Output directory
            output_dir = output_root / f"{char}_{lora_type}" / f"1_{char}_{lora_type}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Copy all images
            for img_path in char_images:
                dest_img = output_dir / img_path.name
                shutil.copy2(img_path, dest_img)

                # Copy caption
                caption_path = img_path.with_suffix(".txt")
                if caption_path.exists():
                    dest_caption = output_dir / caption_path.name
                    shutil.copy2(caption_path, dest_caption)

                character_stats[char][lora_type] += 1

            print(f"  {lora_type:12s}  ✅  {len(char_images):4d} 張")

    return dict(character_stats), skipped


def main():
    parser = argparse.ArgumentParser(
        description="Prepare universal and character-specific training datasets"
    )
    parser.add_argument(
        "--filtered-root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/filtered_data"),
        help="Root directory of filtered data"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/datasets"),
        help="Output directory for training datasets"
    )
    parser.add_argument(
        "--target-per-char-universal",
        type=int,
        default=400,
        help="Target images per character for universal LoRAs (default: 400)"
    )
    parser.add_argument(
        "--min-images-character",
        type=int,
        default=50,
        help="Minimum images for character-specific LoRAs (default: 50)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("準備訓練數據集 - Universal & Character-Specific")
    print("=" * 80)

    # Scan data
    print("\n📊 掃描 Tier A 資料...")
    data_count = scan_tier_a_data(args.filtered_root)

    # Print summary
    print("\n當前資料分佈:")
    for char in sorted(data_count.keys()):
        counts = data_count[char]
        total = counts["pose"] + counts["action"] + counts["expression"]
        print(f"  {char:25s}  P:{counts['pose']:4d}  A:{counts['action']:4d}  E:{counts['expression']:4d}  Total:{total:5d}")

    # Prepare universal datasets
    universal_stats = prepare_universal_datasets(
        data_count,
        args.output_root,
        args.target_per_char_universal
    )

    # Prepare character datasets
    character_stats, skipped = prepare_character_datasets(
        data_count,
        args.output_root,
        min_images_threshold=args.min_images_character
    )

    # Final summary
    print("\n" + "=" * 80)
    print("最終統計")
    print("=" * 80)

    print("\n📊 Universal LoRAs:")
    for lora_type in ["pose", "action", "expression"]:
        total = universal_stats[lora_type]["total"]
        num_chars = len(universal_stats[lora_type]["by_character"])
        avg_per_char = total / num_chars if num_chars > 0 else 0
        print(f"  {lora_type.upper():15s}  {total:5d} 張  ({num_chars} 角色, 平均 {avg_per_char:.1f} 張/角色)")

    print("\n📊 Character-Specific LoRAs:")
    total_character_loras = 0
    for char in sorted(character_stats.keys()):
        counts = character_stats[char]
        for lora_type in ["pose", "action", "expression"]:
            if counts[lora_type] > 0:
                total_character_loras += 1
    print(f"  生成的 LoRAs: {total_character_loras} 個")
    print(f"  跳過的 LoRAs: {len(skipped)} 個")

    if skipped:
        print("\n  跳過的 LoRAs (< {} 張):".format(args.min_images_character))
        for name in skipped:
            print(f"    - {name}")

    # Save stats
    stats_file = args.output_root / "dataset_preparation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "universal": universal_stats,
            "character_specific": character_stats,
            "skipped": skipped,
            "config": {
                "target_per_char_universal": args.target_per_char_universal,
                "min_images_character": args.min_images_character
            }
        }, f, indent=2, default=str)

    print(f"\n✅ 統計已保存: {stats_file}")
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
