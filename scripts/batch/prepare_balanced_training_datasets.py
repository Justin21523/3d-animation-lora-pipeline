#!/usr/bin/env python3
"""
Prepare Balanced Training Datasets

This script:
1. Detects images with single person (using face/person detection)
2. Balances sampling across characters
3. Organizes into Kohya training format

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import cv2
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np


def detect_single_person(image_path: Path, face_cascade=None) -> bool:
    """
    Detect if image contains exactly one person using face detection

    Args:
        image_path: Path to image file
        face_cascade: OpenCV face cascade classifier

    Returns:
        True if exactly one face detected, False otherwise
    """
    if face_cascade is None:
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Return True if exactly one face detected
    return len(faces) == 1


def scan_tier_a_data(filtered_root: Path) -> Dict:
    """
    Scan all tier_a data and count images per character/type

    Returns:
        Dictionary with character -> lora_type -> count
    """
    data_count = defaultdict(lambda: {"pose": 0, "action": 0, "expression": 0})

    for char_dir in filtered_root.iterdir():
        if not char_dir.is_dir():
            continue

        char_name = char_dir.name

        for lora_type in ["pose", "action", "expression"]:
            tier_a_dir = char_dir / lora_type / "tier_a"

            if tier_a_dir.exists():
                count = len(list(tier_a_dir.glob("*.png")))
                data_count[char_name][lora_type] = count

    return dict(data_count)


def calculate_balanced_targets(data_count: Dict, target_per_type: int = 400) -> Dict:
    """
    Calculate balanced sampling targets for each character

    Args:
        data_count: Character counts from scan_tier_a_data()
        target_per_type: Target images per type for balanced characters

    Returns:
        Dictionary with character -> lora_type -> target_count
    """
    targets = {}

    # Calculate total images per character
    char_totals = []
    for char, counts in data_count.items():
        total = sum(counts.values())
        char_totals.append((char, total, counts))

    # Sort by total
    char_totals.sort(key=lambda x: x[1])

    # Find median
    median_idx = len(char_totals) // 2
    median_total = char_totals[median_idx][1]

    print(f"\n中位數角色總數: {median_total}")
    print(f"目標每類型數量: {target_per_type}")
    print()

    # Assign targets
    for char, total, counts in char_totals:
        targets[char] = {}

        if total < median_total:
            # Keep all images for below-median characters
            print(f"  {char:25s}  保留所有 ({total} 張)")
            for lora_type in ["pose", "action", "expression"]:
                targets[char][lora_type] = counts[lora_type]
        else:
            # Sample to target for above-median characters
            print(f"  {char:25s}  平衡採樣 (每類型 ~{target_per_type} 張)")
            for lora_type in ["pose", "action", "expression"]:
                targets[char][lora_type] = min(counts[lora_type], target_per_type)

    return targets


def filter_single_person_images(
    tier_a_dir: Path,
    max_samples: int,
    face_cascade
) -> List[Path]:
    """
    Filter images to only include those with single person

    Args:
        tier_a_dir: Directory containing tier_a images
        max_samples: Maximum number of images to return
        face_cascade: OpenCV face cascade

    Returns:
        List of image paths with single person detected
    """
    all_images = list(tier_a_dir.glob("*.png"))
    single_person_images = []

    for img_path in all_images:
        if detect_single_person(img_path, face_cascade):
            single_person_images.append(img_path)

        # Stop if we have enough
        if len(single_person_images) >= max_samples:
            break

    return single_person_images[:max_samples]


def organize_training_datasets(
    filtered_root: Path,
    output_root: Path,
    targets: Dict,
    enable_face_detection: bool = True
):
    """
    Organize balanced training datasets in Kohya format

    Args:
        filtered_root: Root of filtered data
        output_root: Output directory for training datasets
        targets: Target counts from calculate_balanced_targets()
        enable_face_detection: Enable face detection filtering
    """
    output_root.mkdir(parents=True, exist_ok=True)

    # Initialize face cascade if needed
    face_cascade = None
    if enable_face_detection:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        print("\n✅ 人臉偵測已啟用")
    else:
        print("\n⚠️  人臉偵測已禁用")

    print()
    print("=" * 80)
    print("組織訓練數據集")
    print("=" * 80)
    print()

    stats = {
        "total_images": 0,
        "by_character": defaultdict(lambda: defaultdict(int)),
        "by_lora_type": defaultdict(int),
        "face_detection_stats": {
            "enabled": enable_face_detection,
            "filtered_out": 0,
            "kept": 0
        }
    }

    for char in sorted(targets.keys()):
        for lora_type in ["pose", "action", "expression"]:
            target_count = targets[char][lora_type]

            if target_count == 0:
                continue

            # Source directory
            source_dir = filtered_root / char / lora_type / "tier_a"

            if not source_dir.exists():
                continue

            # Output directory (Kohya format: 1_concept_name)
            concept_name = f"{char}_{lora_type}"
            output_dir = output_root / f"{concept_name}" / "1_universal_{lora_type}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get images (with optional face detection)
            if enable_face_detection:
                selected_images = filter_single_person_images(
                    source_dir,
                    target_count,
                    face_cascade
                )
                filtered_out = target_count - len(selected_images)
                stats["face_detection_stats"]["filtered_out"] += filtered_out
                stats["face_detection_stats"]["kept"] += len(selected_images)
            else:
                all_images = list(source_dir.glob("*.png"))
                selected_images = all_images[:target_count]

            # Copy images and captions
            for img_path in selected_images:
                # Copy image
                dest_img = output_dir / img_path.name
                shutil.copy2(img_path, dest_img)

                # Copy caption
                caption_path = img_path.with_suffix(".txt")
                if caption_path.exists():
                    dest_caption = output_dir / caption_path.name
                    shutil.copy2(caption_path, dest_caption)

                stats["total_images"] += 1
                stats["by_character"][char][lora_type] += 1
                stats["by_lora_type"][lora_type] += 1

            print(f"  ✅ {char:20s} {lora_type:12s}  {len(selected_images):4d} / {target_count:4d} 張")

    # Save statistics
    stats_file = output_root / "dataset_organization_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    print()
    print("=" * 80)
    print("最終統計")
    print("=" * 80)
    print(f"總圖片數: {stats['total_images']}")
    print()
    print("按 LoRA 類型:")
    for lora_type in ["pose", "action", "expression"]:
        count = stats["by_lora_type"][lora_type]
        print(f"  {lora_type.upper():15s}  {count:5d} 張")

    if enable_face_detection:
        print()
        print("人臉偵測統計:")
        print(f"  保留: {stats['face_detection_stats']['kept']} 張")
        print(f"  過濾: {stats['face_detection_stats']['filtered_out']} 張")

    print()
    print(f"✅ 統計已保存: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare balanced training datasets with person detection"
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
        "--target-per-type",
        type=int,
        default=400,
        help="Target images per type for balanced characters (default: 400)"
    )
    parser.add_argument(
        "--no-face-detection",
        action="store_true",
        help="Disable face detection filtering"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("準備平衡訓練數據集")
    print("=" * 80)

    # Scan data
    print("\n📊 掃描 Tier A 資料...")
    data_count = scan_tier_a_data(args.filtered_root)

    # Calculate balanced targets
    print("\n⚖️  計算平衡採樣目標...")
    targets = calculate_balanced_targets(data_count, args.target_per_type)

    # Organize datasets
    organize_training_datasets(
        args.filtered_root,
        args.output_root,
        targets,
        enable_face_detection=not args.no_face_detection
    )

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
