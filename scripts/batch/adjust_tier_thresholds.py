#!/usr/bin/env python3
"""
Adjust Tier Thresholds - Reorganize filtered data by adjusting blur score thresholds

This script:
1. Reclassifies images based on new blur_score thresholds
2. Moves images between tier_a and tier_b directories
3. Preserves original data with backup
4. Allows manual review after reclassification

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
import argparse


def reorganize_tiers(
    stats_file: Path,
    data_root: Path,
    new_tier_a_threshold: float = 100.0,
    tier_c_threshold: float = 80.0,
    backup: bool = True
):
    """
    Reorganize tier_a and tier_b folders based on new thresholds

    Args:
        stats_file: Path to filtering statistics JSON
        data_root: Root directory of filtered data
        new_tier_a_threshold: New threshold for Tier A (default: 100)
        tier_c_threshold: Threshold for Tier C (default: 80)
        backup: Create backup before reorganizing
    """
    # Load statistics
    print(f"📊 載入統計數據: {stats_file}")
    with open(stats_file) as f:
        stats = json.load(f)

    print(f"\n🔄 重新組織 Tier 資料夾")
    print(f"  新 Tier A 閾值: >= {new_tier_a_threshold}")
    print(f"  Tier B 範圍: {tier_c_threshold} - {new_tier_a_threshold}")
    print(f"  Tier C (拒絕): < {tier_c_threshold}")
    print()

    moved_to_a = 0
    moved_to_b = 0
    kept_in_a = 0
    kept_in_b = 0

    for key, report in stats["per_character_reports"].items():
        char = report["character"]
        lora_type = report["lora_type"]

        # Directories
        char_lora_dir = data_root / char / lora_type
        tier_a_dir = char_lora_dir / "tier_a"
        tier_b_dir = char_lora_dir / "tier_b"

        if not tier_a_dir.exists() or not tier_b_dir.exists():
            continue

        # Backup if requested
        if backup:
            backup_dir = char_lora_dir / "backup_original_tiers"
            if not backup_dir.exists():
                backup_dir.mkdir(parents=True)
                shutil.copytree(tier_a_dir, backup_dir / "tier_a", dirs_exist_ok=True)
                shutil.copytree(tier_b_dir, backup_dir / "tier_b", dirs_exist_ok=True)

        # Process each image
        for result in report["results"]:
            img_name = result["image"]
            blur_score = result.get("blur_score")
            current_tier = result.get("tier", "").upper()

            if blur_score is None:
                continue

            # Determine new tier
            if blur_score >= new_tier_a_threshold:
                new_tier = "A"
            elif blur_score >= tier_c_threshold:
                new_tier = "B"
            else:
                continue  # Tier C - stays rejected

            # Find current location
            img_path_a = tier_a_dir / img_name
            img_path_b = tier_b_dir / img_name
            caption_name = img_name.replace(".png", ".txt")
            caption_path_a = tier_a_dir / caption_name
            caption_path_b = tier_b_dir / caption_name

            current_img_path = None
            current_caption_path = None

            if img_path_a.exists():
                current_img_path = img_path_a
                current_caption_path = caption_path_a
                current_location = "A"
            elif img_path_b.exists():
                current_img_path = img_path_b
                current_caption_path = caption_path_b
                current_location = "B"
            else:
                continue  # File not found

            # Move if necessary
            if new_tier == "A" and current_location == "B":
                # Move from B to A
                shutil.move(str(current_img_path), str(tier_a_dir / img_name))
                if current_caption_path.exists():
                    shutil.move(str(current_caption_path), str(tier_a_dir / caption_name))
                moved_to_a += 1

            elif new_tier == "B" and current_location == "A":
                # Move from A to B
                shutil.move(str(current_img_path), str(tier_b_dir / img_name))
                if current_caption_path.exists():
                    shutil.move(str(current_caption_path), str(tier_b_dir / caption_name))
                moved_to_b += 1

            elif new_tier == "A" and current_location == "A":
                kept_in_a += 1

            elif new_tier == "B" and current_location == "B":
                kept_in_b += 1

    print("\n" + "=" * 80)
    print("📊 重組結果")
    print("=" * 80)
    print(f"  移至 Tier A: {moved_to_a} 張")
    print(f"  移至 Tier B: {moved_to_b} 張")
    print(f"  保持在 Tier A: {kept_in_a} 張")
    print(f"  保持在 Tier B: {kept_in_b} 張")
    print()

    # Count final distribution
    final_stats = defaultdict(lambda: {"tier_a": 0, "tier_b": 0})

    for key, report in stats["per_character_reports"].items():
        char = report["character"]
        lora_type = report["lora_type"]

        char_lora_dir = data_root / char / lora_type
        tier_a_dir = char_lora_dir / "tier_a"
        tier_b_dir = char_lora_dir / "tier_b"

        if tier_a_dir.exists():
            count_a = len(list(tier_a_dir.glob("*.png")))
            final_stats[lora_type]["tier_a"] += count_a

        if tier_b_dir.exists():
            count_b = len(list(tier_b_dir.glob("*.png")))
            final_stats[lora_type]["tier_b"] += count_b

    print("最終分佈:")
    for lora_type in ["pose", "action", "expression"]:
        stats_data = final_stats[lora_type]
        total = stats_data["tier_a"] + stats_data["tier_b"]
        print(f"  {lora_type.upper():12s}  Tier A: {stats_data['tier_a']:5d}  Tier B: {stats_data['tier_b']:5d}  Total: {total:5d}")

    if backup:
        print(f"\n💾 備份已保存在每個角色目錄的 backup_original_tiers/ 子目錄中")


def main():
    parser = argparse.ArgumentParser(
        description="Adjust tier thresholds and reorganize filtered data"
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/filtered_data/overall_filtering_statistics.json"),
        help="Path to filtering statistics JSON"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/filtered_data"),
        help="Root directory of filtered data"
    )
    parser.add_argument(
        "--tier-a-threshold",
        type=float,
        default=100.0,
        help="New blur score threshold for Tier A (default: 100)"
    )
    parser.add_argument(
        "--tier-c-threshold",
        type=float,
        default=80.0,
        help="Blur score threshold for Tier C (default: 80)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create backup (NOT RECOMMENDED)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Tier 閾值調整工具")
    print("=" * 80)
    print()

    if args.no_backup:
        print("⚠️  警告: 備份已禁用！")
        response = input("確定要繼續嗎? (yes/no): ")
        if response.lower() != "yes":
            print("已取消。")
            return

    reorganize_tiers(
        stats_file=args.stats_file,
        data_root=args.data_root,
        new_tier_a_threshold=args.tier_a_threshold,
        tier_c_threshold=args.tier_c_threshold,
        backup=not args.no_backup
    )

    print("\n✅ 完成！")
    print("\n接下來你可以:")
    print("  1. 手動審查 tier_a 和 tier_b 資料夾")
    print("  2. 移動圖片在 tier_a 和 tier_b 之間")
    print("  3. 使用 Web UI 進行批量審查")
    print("  4. 重新組織數據集用於訓練")


if __name__ == "__main__":
    main()
