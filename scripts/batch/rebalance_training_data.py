#!/usr/bin/env python3
"""
Rebalance Training Data - Adjust tier thresholds and balance by character

This script allows you to:
1. Adjust blur_score thresholds to reclassify Tier A/B data
2. Balance dataset by sampling evenly across characters
3. Generate new datasets for training

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


def load_filtering_statistics(stats_file: Path) -> Dict:
    """Load filtering statistics JSON"""
    with open(stats_file) as f:
        return json.load(f)


def reclassify_tiers(
    stats: Dict,
    new_tier_a_threshold: float = 100.0,
    tier_c_threshold: float = 80.0
) -> Dict:
    """
    Reclassify images into tiers based on new thresholds

    Args:
        stats: Original filtering statistics
        new_tier_a_threshold: New threshold for Tier A (default: 100)
        tier_c_threshold: Threshold for Tier C rejection (default: 80)

    Returns:
        Updated statistics with reclassified tiers
    """
    updated_stats = {
        "timestamp": stats["timestamp"],
        "reclassification_params": {
            "tier_a_threshold": new_tier_a_threshold,
            "tier_c_threshold": tier_c_threshold
        },
        "per_character_reports": {}
    }

    print(f"\n🔄 重新分類資料...")
    print(f"  新 Tier A 閾值: {new_tier_a_threshold}")
    print(f"  Tier C 閾值: {tier_c_threshold}")
    print()

    for key, report in stats["per_character_reports"].items():
        new_report = {
            "character": report["character"],
            "lora_type": report["lora_type"],
            "tier_a": 0,
            "tier_b": 0,
            "tier_c": 0,
            "results": []
        }

        for result in report["results"]:
            blur_score = result.get("blur_score")

            # Skip if blur_score is None
            if blur_score is None:
                continue

            # Reclassify based on new thresholds
            if blur_score >= new_tier_a_threshold:
                new_tier = "A"
                new_report["tier_a"] += 1
            elif blur_score >= tier_c_threshold:
                new_tier = "B"
                new_report["tier_b"] += 1
            else:
                new_tier = "C"
                new_report["tier_c"] += 1

            new_result = result.copy()
            new_result["tier"] = new_tier
            new_result["original_tier"] = result["tier"]
            new_report["results"].append(new_result)

        new_report["total_images"] = new_report["tier_a"] + new_report["tier_b"] + new_report["tier_c"]
        updated_stats["per_character_reports"][key] = new_report

        print(f"  {key:40s}  A: {new_report['tier_a']:4d}  B: {new_report['tier_b']:4d}  C: {new_report['tier_c']:4d}")

    return updated_stats


def balance_by_character(
    stats: Dict,
    target_per_character: Dict[str, int],
    use_tier_b: bool = True
) -> Dict:
    """
    Balance dataset by sampling evenly from each character

    Args:
        stats: Filtering statistics (after reclassification)
        target_per_character: Target number of images per character for each lora_type
                             e.g., {"pose": 300, "action": 350, "expression": 280}
        use_tier_b: Include Tier B images if Tier A is insufficient

    Returns:
        Balanced selection with image lists per character
    """
    balanced = defaultdict(lambda: defaultdict(list))

    print(f"\n⚖️ 角色平衡採樣...")
    print(f"  目標數量: {target_per_character}")
    print(f"  使用 Tier B: {use_tier_b}")
    print()

    for key, report in stats["per_character_reports"].items():
        char = report["character"]
        lora_type = report["lora_type"]
        target = target_per_character.get(lora_type, 0)

        if target == 0:
            continue

        # Collect Tier A images (sorted by blur score, descending)
        tier_a_images = [
            r for r in report["results"]
            if r["tier"] == "A"
        ]
        tier_a_images.sort(key=lambda x: x["blur_score"], reverse=True)

        selected = tier_a_images[:target]

        # If Tier A insufficient and use_tier_b enabled, supplement with Tier B
        if len(selected) < target and use_tier_b:
            tier_b_images = [
                r for r in report["results"]
                if r["tier"] == "B"
            ]
            tier_b_images.sort(key=lambda x: x["blur_score"], reverse=True)

            needed = target - len(selected)
            selected.extend(tier_b_images[:needed])

        balanced[char][lora_type] = selected

        tier_a_count = sum(1 for img in selected if img["tier"] == "A")
        tier_b_count = sum(1 for img in selected if img["tier"] == "B")

        print(f"  {char:20s} {lora_type:12s}  選取: {len(selected):4d} / {target:4d}  (A: {tier_a_count:4d}, B: {tier_b_count:4d})")

    return balanced


def create_balanced_datasets(
    balanced: Dict,
    source_root: Path,
    output_root: Path,
    copy_files: bool = True
) -> Dict:
    """
    Create balanced datasets by copying/linking selected images

    Args:
        balanced: Balanced selection from balance_by_character()
        source_root: Root directory of filtered data
        output_root: Output directory for balanced datasets
        copy_files: If True, copy files; if False, create symlinks

    Returns:
        Statistics about created datasets
    """
    output_root.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_images": 0,
        "by_character": defaultdict(lambda: defaultdict(int)),
        "by_lora_type": defaultdict(int)
    }

    print(f"\n📁 創建平衡數據集...")
    print(f"  源目錄: {source_root}")
    print(f"  輸出目錄: {output_root}")
    print(f"  操作: {'複製' if copy_files else '符號連結'}")
    print()

    for char, lora_types in balanced.items():
        for lora_type, images in lora_types.items():
            # Create output directory
            output_dir = output_root / f"{char}_{lora_type}" / "tier_a"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Source directory
            source_dir = source_root / f"{char}_{lora_type}" / "tier_a"
            source_b_dir = source_root / f"{char}_{lora_type}" / "tier_b"

            # Copy/link images
            for img_info in images:
                img_name = img_info["image"]

                # Try Tier A first, then Tier B
                source_path = source_dir / img_name
                if not source_path.exists():
                    source_path = source_b_dir / img_name

                if not source_path.exists():
                    print(f"  ⚠️  File not found: {img_name}")
                    continue

                dest_path = output_dir / img_name

                # Also copy caption file
                caption_source = source_path.with_suffix(".txt")
                caption_dest = dest_path.with_suffix(".txt")

                if copy_files:
                    shutil.copy2(source_path, dest_path)
                    if caption_source.exists():
                        shutil.copy2(caption_source, caption_dest)
                else:
                    dest_path.symlink_to(source_path)
                    if caption_source.exists():
                        caption_dest.symlink_to(caption_source)

                stats["total_images"] += 1
                stats["by_character"][char][lora_type] += 1
                stats["by_lora_type"][lora_type] += 1

            print(f"  ✅ {char:20s} {lora_type:12s}  {len(images):4d} 張圖片")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Rebalance training data by adjusting tiers and balancing by character"
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/filtered_data/overall_filtering_statistics.json"),
        help="Path to filtering statistics JSON"
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
        help="Blur score threshold for Tier C rejection (default: 80)"
    )
    parser.add_argument(
        "--target-pose",
        type=int,
        default=320,
        help="Target images per character for pose LoRAs (default: 320)"
    )
    parser.add_argument(
        "--target-action",
        type=int,
        default=350,
        help="Target images per character for action LoRAs (default: 350)"
    )
    parser.add_argument(
        "--target-expression",
        type=int,
        default=280,
        help="Target images per character for expression LoRAs (default: 280)"
    )
    parser.add_argument(
        "--no-tier-b",
        action="store_true",
        help="Do not use Tier B images (Tier A only)"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/filtered_data"),
        help="Root directory of filtered data"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/balanced_data"),
        help="Output directory for balanced datasets"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("數據重組工具 - 調整層級與角色平衡")
    print("=" * 80)

    # Load statistics
    print(f"\n📊 載入統計數據: {args.stats_file}")
    stats = load_filtering_statistics(args.stats_file)

    # Reclassify tiers
    updated_stats = reclassify_tiers(
        stats,
        new_tier_a_threshold=args.tier_a_threshold,
        tier_c_threshold=args.tier_c_threshold
    )

    # Balance by character
    target_per_character = {
        "pose": args.target_pose,
        "action": args.target_action,
        "expression": args.target_expression
    }

    balanced = balance_by_character(
        updated_stats,
        target_per_character=target_per_character,
        use_tier_b=not args.no_tier_b
    )

    # Create balanced datasets
    if not args.dry_run:
        dataset_stats = create_balanced_datasets(
            balanced,
            source_root=args.source_root,
            output_root=args.output_root,
            copy_files=not args.symlink
        )

        # Save updated statistics
        output_stats_file = args.output_root / "rebalance_statistics.json"
        with open(output_stats_file, 'w') as f:
            json.dump({
                "reclassification": updated_stats["reclassification_params"],
                "target_per_character": target_per_character,
                "balanced_selection": {
                    char: {lora_type: len(images) for lora_type, images in types.items()}
                    for char, types in balanced.items()
                },
                "dataset_stats": dataset_stats
            }, f, indent=2)

        print(f"\n✅ 統計已保存: {output_stats_file}")

        print("\n" + "=" * 80)
        print("📊 最終統計")
        print("=" * 80)
        print(f"總圖片數: {dataset_stats['total_images']}")
        print(f"\n按 LoRA 類型:")
        for lora_type, count in dataset_stats["by_lora_type"].items():
            print(f"  {lora_type:15s}  {count:5d} 張")
    else:
        print("\n🔍 DRY RUN - 未執行實際操作")

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
