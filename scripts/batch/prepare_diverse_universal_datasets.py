#!/usr/bin/env python3
"""
Prepare Diverse Universal LoRA Datasets

This script creates universal LoRAs with:
1. Fixed 300 images per character per type (or all if < 300)
2. High quality filtering (from tier_a)
3. Single person detection (face counting)
4. Maximum diversity selection (temporal or perceptual)

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

# Optional imports
try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    print("⚠️  imagehash not available, using temporal diversity instead")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  cv2 not available, skipping face detection")


def count_faces(image_path: Path) -> int:
    """Count number of faces in image using OpenCV"""
    if not CV2_AVAILABLE:
        return -1  # Unknown

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return -1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return len(faces)
    except Exception as e:
        return -1


def filter_single_person_images(image_paths: List[Path]) -> List[Path]:
    """Filter images to only include those with single person"""
    if not CV2_AVAILABLE:
        print("    ⚠️  Face detection unavailable, skipping filter")
        return image_paths

    print(f"    偵測單人圖片...")
    single_person = []
    multi_person = 0
    no_person = 0

    for img_path in image_paths:
        face_count = count_faces(img_path)

        if face_count == 1:
            single_person.append(img_path)
        elif face_count > 1:
            multi_person += 1
        elif face_count == 0:
            no_person += 1

    print(f"    ✅ 單人: {len(single_person)}, ❌ 多人: {multi_person}, ❌ 無人: {no_person}")
    return single_person


def select_diverse_temporal(image_paths: List[Path], target_count: int) -> List[Path]:
    """
    Select diverse images using simple interval sampling (FAST)

    Instead of sorting (slow), just sample every Nth image
    This is much faster and still gives good temporal diversity
    """
    if len(image_paths) <= target_count:
        return image_paths

    # Calculate sampling interval
    interval = len(image_paths) / target_count

    # Select evenly spaced images without sorting
    selected = []
    for i in range(target_count):
        idx = int(i * interval)
        if idx < len(image_paths):
            selected.append(image_paths[idx])

    return selected


def copy_image_and_caption(args):
    """Copy a single image and its caption (for parallel processing)"""
    img_path, dest_img, char_prefix = args

    try:
        # Copy image
        shutil.copy2(img_path, dest_img)

        # Copy caption
        caption_path = img_path.with_suffix(".txt")
        if caption_path.exists():
            dest_caption = dest_img.with_suffix(".txt")
            shutil.copy2(caption_path, dest_caption)

        return True
    except Exception as e:
        print(f"  ⚠️  Error copying {img_path.name}: {e}")
        return False


def calculate_image_hash(image_path: Path, hash_size: int = 16):
    """Calculate perceptual hash for an image"""
    if not IMAGEHASH_AVAILABLE:
        return None

    try:
        img = Image.open(image_path)
        return imagehash.phash(img, hash_size=hash_size)
    except Exception as e:
        print(f"  ⚠️  Error hashing {image_path.name}: {e}")
        return None


def select_diverse_images(
    image_paths: List[Path],
    target_count: int,
    hash_size: int = 16,
    use_temporal: bool = False
) -> List[Path]:
    """
    Select diverse images using perceptual hashing or temporal sampling

    Algorithm (perceptual):
    1. Calculate perceptual hash for all images (in batches to save memory)
    2. Iteratively select images that are most different from already selected
    3. Continue until target_count is reached

    Algorithm (temporal):
    1. Sort images by filename (assumes temporal ordering)
    2. Select evenly spaced images
    """
    if len(image_paths) <= target_count:
        return image_paths

    # Use temporal sampling if perceptual hashing unavailable or requested
    if use_temporal or not IMAGEHASH_AVAILABLE:
        print(f"    時間採樣 {target_count} 張（從 {len(image_paths)} 張中）...")
        return select_diverse_temporal(image_paths, target_count)

    print(f"    多樣性採樣 {target_count} 張（從 {len(image_paths)} 張中）...")

    # Process in batches to avoid OOM
    batch_size = 100
    all_hashes = []

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        for img_path in batch:
            img_hash = calculate_image_hash(img_path, hash_size)
            if img_hash is not None:
                all_hashes.append((img_path, img_hash))

    if not all_hashes:
        return image_paths[:target_count]

    # Select first image
    selected = [all_hashes[0]]
    remaining = all_hashes[1:]

    # Iteratively select most diverse images
    while len(selected) < target_count and remaining:
        max_min_distance = -1
        best_idx = 0

        for idx, (path, img_hash) in enumerate(remaining):
            # Find minimum distance to any selected image
            min_distance = min(
                img_hash - sel_hash
                for _, sel_hash in selected
            )

            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_idx = idx

        selected.append(remaining.pop(best_idx))

    return [path for path, _ in selected]


def scan_tier_a_data(filtered_root: Path) -> Dict:
    """Scan all tier_a data"""
    data_count = defaultdict(lambda: {
        "pose": 0, "action": 0, "expression": 0,
        "paths": {"pose": [], "action": [], "expression": []}
    })

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


def prepare_universal_datasets_diverse(
    data_count: Dict,
    output_root: Path,
    target_per_char: int = 300,
    enable_diversity: bool = True,
    enable_face_filter: bool = True,
    use_temporal: bool = False,
    num_workers: int = 32
):
    """
    Prepare 3 universal LoRA datasets with diverse sampling

    For each lora_type (pose/action/expression):
    - Filter to single-person images (if face detection available)
    - Each character contributes exactly target_per_char images (or all if fewer)
    - Use diversity selection to maximize visual variety
    - Parallel file copying using multiprocessing
    """
    print("\n" + "=" * 80)
    print("準備 Universal LoRAs (高品質多樣性採樣 + 並行處理)")
    print("=" * 80)
    print(f"目標: 每個角色每類型 {target_per_char} 張（不足則全部）")
    print(f"多樣性選擇: {'啟用' if enable_diversity else '禁用'}")
    print(f"單人過濾: {'啟用' if enable_face_filter and CV2_AVAILABLE else '禁用（cv2 不可用）' if enable_face_filter else '禁用'}")
    print(f"採樣方式: {'時間採樣' if use_temporal else '感知哈希'}")
    print(f"並行處理: {num_workers} workers")
    print()

    universal_stats = defaultdict(lambda: {
        "total": 0,
        "by_character": defaultdict(int),
        "filtered_stats": {"single_person": 0, "multi_person": 0, "no_person": 0}
    })

    for lora_type in ["pose", "action", "expression"]:
        # Output directory (Kohya format: 1_concept)
        output_dir = output_root / f"universal_{lora_type}" / f"1_universal_{lora_type}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📦 Universal {lora_type.upper()}")
        print("-" * 80)

        # Collect all copy tasks
        all_copy_tasks = []
        char_counts = {}

        for char in sorted(data_count.keys()):
            char_images = data_count[char]["paths"][lora_type]
            available_orig = len(char_images)

            # Step 1: Filter to single-person images
            if enable_face_filter and CV2_AVAILABLE:
                char_images_filtered = filter_single_person_images(char_images)
                filtered_count = len(char_images_filtered)
            else:
                char_images_filtered = char_images
                filtered_count = available_orig

            available = len(char_images_filtered)

            # Determine target count
            target = min(available, target_per_char)

            if target == 0:
                print(f"  ⏭️  {char:25s}  無可用圖片")
                continue

            # Step 2: Select images (with diversity if enabled and needed)
            if enable_diversity and available > target:
                selected_images = select_diverse_images(
                    char_images_filtered,
                    target,
                    use_temporal=use_temporal
                )
            else:
                selected_images = char_images_filtered[:target]

            # Prepare copy tasks
            for img_path in selected_images:
                dest_img = output_dir / f"{char}_{img_path.name}"
                all_copy_tasks.append((img_path, dest_img, char))

            char_counts[char] = len(selected_images)

        # Step 3: Parallel copy
        print(f"\n  🚀 並行複製 {len(all_copy_tasks)} 張圖片...")
        with Pool(num_workers) as pool:
            results = pool.map(copy_image_and_caption, all_copy_tasks)

        # Count successes
        success_count = sum(1 for r in results if r)

        # Update stats
        for char, count in char_counts.items():
            universal_stats[lora_type]["total"] += count
            universal_stats[lora_type]["by_character"][char] = count

            status = "✅" if data_count[char][lora_type] >= target_per_char else "⚠️ "
            print(f"  {status} {char:25s}  {count:3d} 張")

        print(f"\n  ✅ Total: {success_count} / {len(all_copy_tasks)} 張成功複製")

    return dict(universal_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare diverse universal LoRA datasets with quality filtering"
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
        "--target-per-char",
        type=int,
        default=300,
        help="Target images per character (default: 300)"
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Disable diversity selection (use first N images)"
    )
    parser.add_argument(
        "--no-face-filter",
        action="store_true",
        help="Disable single-person face filtering"
    )
    parser.add_argument(
        "--use-temporal",
        action="store_true",
        help="Use temporal sampling instead of perceptual hashing"
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=16,
        help="Perceptual hash size for diversity (default: 16)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="Number of parallel workers for file copying (default: 32)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("準備 Universal 訓練數據集 - 多樣性採樣")
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
    universal_stats = prepare_universal_datasets_diverse(
        data_count,
        args.output_root,
        target_per_char=args.target_per_char,
        enable_diversity=not args.no_diversity,
        enable_face_filter=not args.no_face_filter,
        use_temporal=args.use_temporal,
        num_workers=args.num_workers
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

    # Save stats
    stats_file = args.output_root / "universal_diverse_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "universal": universal_stats,
            "config": {
                "target_per_char": args.target_per_char,
                "diversity_enabled": not args.no_diversity,
                "face_filter_enabled": not args.no_face_filter,
                "use_temporal": args.use_temporal,
                "hash_size": args.hash_size,
                "cv2_available": CV2_AVAILABLE,
                "imagehash_available": IMAGEHASH_AVAILABLE
            }
        }, f, indent=2, default=str)

    print(f"\n✅ 統計已保存: {stats_file}")
    print("\n✅ 完成！", flush=True)


if __name__ == "__main__":
    import sys
    sys.stdout.flush()
    main()
