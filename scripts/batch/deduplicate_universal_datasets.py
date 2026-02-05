#!/usr/bin/env python3
"""
Deduplicate Universal LoRA Datasets

Removes cross-dataset duplicates from universal LoRA datasets.
Strategy: Keep each image in only ONE dataset based on priority:
1. Keep in primary type (pose stays in pose, action in action, etc.)
2. If unsure, keep in the dataset with fewer total images

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple


def find_duplicates_across_datasets(
    dataset_dirs: Dict[str, Path]
) -> Dict[str, Set[str]]:
    """
    Find duplicate filenames across datasets

    Returns:
        Dict mapping filename to set of dataset names containing it
    """
    file_to_datasets = defaultdict(set)

    for dataset_name, dataset_dir in dataset_dirs.items():
        image_dir = dataset_dir / f"1_universal_{dataset_name}"
        if not image_dir.exists():
            continue

        for img_path in image_dir.glob("*.png"):
            file_to_datasets[img_path.name].add(dataset_name)

    # Filter to only duplicates (appears in multiple datasets)
    duplicates = {
        fname: datasets
        for fname, datasets in file_to_datasets.items()
        if len(datasets) > 1
    }

    return duplicates


def get_dataset_sizes(dataset_dirs: Dict[str, Path]) -> Dict[str, int]:
    """Get current size of each dataset"""
    sizes = {}
    for dataset_name, dataset_dir in dataset_dirs.items():
        image_dir = dataset_dir / f"1_universal_{dataset_name}"
        if image_dir.exists():
            sizes[dataset_name] = len(list(image_dir.glob("*.png")))
        else:
            sizes[dataset_name] = 0
    return sizes


def decide_which_dataset_to_keep(
    filename: str,
    datasets: Set[str],
    dataset_sizes: Dict[str, int]
) -> str:
    """
    Decide which dataset should keep this image

    Strategy:
    1. If filename contains dataset name, keep in that dataset
    2. Otherwise, keep in the dataset with fewer images (balance)
    """
    # Check if filename hints at primary type
    fname_lower = filename.lower()

    for dataset_name in datasets:
        # Check if this is clearly meant for this dataset
        # (e.g., pose-related keywords for pose dataset)
        if dataset_name in fname_lower:
            return dataset_name

    # Otherwise, keep in smallest dataset to balance sizes
    return min(datasets, key=lambda d: dataset_sizes[d])


def remove_duplicates(
    dataset_dirs: Dict[str, Path],
    duplicates: Dict[str, Set[str]],
    dry_run: bool = False
) -> Dict:
    """
    Remove duplicate images from datasets

    Returns stats about removed images
    """
    stats = defaultdict(lambda: {"kept": 0, "removed": 0, "removed_files": []})
    dataset_sizes = get_dataset_sizes(dataset_dirs)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}處理 {len(duplicates)} 個重複檔案...")
    print()

    for filename, datasets_with_file in duplicates.items():
        # Decide which dataset keeps this image
        keep_in = decide_which_dataset_to_keep(
            filename,
            datasets_with_file,
            dataset_sizes
        )

        # Remove from all other datasets
        for dataset_name in datasets_with_file:
            image_dir = dataset_dirs[dataset_name] / f"1_universal_{dataset_name}"
            img_path = image_dir / filename
            txt_path = img_path.with_suffix(".txt")

            if dataset_name == keep_in:
                stats[dataset_name]["kept"] += 1
            else:
                if not dry_run:
                    # Remove image
                    if img_path.exists():
                        img_path.unlink()
                    # Remove caption
                    if txt_path.exists():
                        txt_path.unlink()

                stats[dataset_name]["removed"] += 1
                stats[dataset_name]["removed_files"].append(filename)

                # Update size tracking
                dataset_sizes[dataset_name] -= 1

    return dict(stats)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Deduplicate universal LoRA datasets"
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/datasets"),
        help="Root directory containing universal datasets"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually removing files"
    )

    args = parser.parse_args()

    # Define dataset directories
    dataset_dirs = {
        "pose": args.datasets_root / "universal_pose",
        "action": args.datasets_root / "universal_action",
        "expression": args.datasets_root / "universal_expression"
    }

    print("=" * 80)
    print("Universal LoRA 資料集去重")
    print("=" * 80)

    # Get initial sizes
    initial_sizes = get_dataset_sizes(dataset_dirs)
    print("\n初始資料集大小:")
    for name, size in initial_sizes.items():
        print(f"  {name:15s}: {size:5d} 張")

    # Find duplicates
    print("\n掃描重複檔案...")
    duplicates = find_duplicates_across_datasets(dataset_dirs)

    if not duplicates:
        print("\n✅ 沒有發現重複檔案！")
        return

    # Analyze duplicates
    print(f"\n發現 {len(duplicates)} 個重複檔案:")
    overlap_stats = defaultdict(int)
    for filename, datasets in duplicates.items():
        key = " ∩ ".join(sorted(datasets))
        overlap_stats[key] += 1

    for overlap, count in sorted(overlap_stats.items()):
        print(f"  {overlap:30s}: {count:4d} 張")

    # Remove duplicates
    removal_stats = remove_duplicates(dataset_dirs, duplicates, dry_run=args.dry_run)

    print(f"\n{'[預覽] ' if args.dry_run else ''}處理結果:")
    for dataset_name in ["pose", "action", "expression"]:
        stats = removal_stats[dataset_name]
        kept = stats["kept"]
        removed = stats["removed"]
        print(f"  {dataset_name:15s}: 保留 {kept:4d} 張, 移除 {removed:4d} 張")

    # Get final sizes
    if not args.dry_run:
        final_sizes = get_dataset_sizes(dataset_dirs)
        print("\n最終資料集大小:")
        total_before = sum(initial_sizes.values())
        total_after = sum(final_sizes.values())

        for name in ["pose", "action", "expression"]:
            before = initial_sizes[name]
            after = final_sizes[name]
            diff = after - before
            print(f"  {name:15s}: {after:5d} 張 ({diff:+5d})")

        print(f"\n總計: {total_after:5d} 張 ({total_after - total_before:+5d})")

        # Save stats
        stats_file = args.datasets_root / "deduplication_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                "initial_sizes": initial_sizes,
                "final_sizes": final_sizes,
                "duplicates_found": len(duplicates),
                "removal_stats": removal_stats,
                "overlap_distribution": dict(overlap_stats)
            }, f, indent=2)

        print(f"\n✅ 統計已保存: {stats_file}")

    if args.dry_run:
        print("\n⚠️  這是預覽模式，沒有實際刪除檔案")
        print("    移除 --dry-run 參數以執行實際去重")
    else:
        print("\n✅ 去重完成！")


if __name__ == "__main__":
    main()
