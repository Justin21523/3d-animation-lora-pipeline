#!/usr/bin/env python3
"""
Batch YOLO + MobileSAM Segmentation
====================================

Runs two-stage segmentation (YOLO detection → MobileSAM) on multiple projects.
Output structure follows SAM2 convention for compatibility.

Output Structure:
    {project}/yolo_mobilesam_instances/
    ├── characters/          # Character crops with transparent background (RGBA)
    ├── masks/               # Binary masks
    ├── backgrounds/         # Original frames (for reference)
    └── instances_metadata.json

Naming Convention:
    {frame_basename}_inst{N}.png       - Character image (RGBA)
    {frame_basename}_inst{N}_mask.png  - Binary mask
    {frame_basename}_background.jpg    - Original frame

Usage:
    python run_yolo_mobilesam_batch.py \
        --projects astro-boy astro-kid \
        --base-dir /mnt/data/datasets/general

Author: LLMProvider Tooling
Date: 2025-12-07
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def run_segmentation(
    input_dir: Path,
    output_dir: Path,
    yolo_model: str = "yolov11n.pt",
    device: str = "cuda",
    bbox_padding: float = 0.1,
    confidence: float = 0.5,
    min_area: int = 4096
) -> Dict:
    """
    Run YOLO + MobileSAM segmentation on a directory of frames.

    Args:
        input_dir: Directory containing input frames
        output_dir: Output directory for segmented results
        yolo_model: YOLO model path
        device: cuda or cpu
        bbox_padding: Bbox expansion ratio
        confidence: YOLO confidence threshold
        min_area: Minimum mask area to keep

    Returns:
        Statistics dict
    """
    from generic.segmentation.yolo_sam_segmentation import YOLODetector, MobileSAMSegmenter

    # Create output directories
    characters_dir = output_dir / "characters"
    masks_dir = output_dir / "masks"
    backgrounds_dir = output_dir / "backgrounds"

    characters_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    backgrounds_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models
    print(f"Loading YOLO detector...")
    detector = YOLODetector(
        model_path=yolo_model,
        device=device,
        confidence_threshold=confidence,
        classes=[0]  # person class
    )

    print(f"Loading MobileSAM...")
    segmenter = MobileSAMSegmenter(device=device)

    # Get all frames
    frame_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    frame_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in frame_extensions
    ])

    print(f"Found {len(frame_paths)} frames to process")

    # Statistics
    stats = {
        "total_frames": len(frame_paths),
        "frames_with_detections": 0,
        "total_instances": 0,
        "skipped_small": 0,
        "metadata": []
    }

    # Process each frame
    for frame_path in tqdm(frame_paths, desc="Segmenting"):
        # Load frame
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_basename = frame_path.stem

        # Save background (original frame)
        bg_path = backgrounds_dir / f"{frame_basename}_background.jpg"
        cv2.imwrite(str(bg_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # YOLO detection
        bboxes = detector.detect(image)

        if not bboxes:
            continue

        stats["frames_with_detections"] += 1
        frame_instances = []

        # Process each detection
        inst_idx = 0
        for bbox in bboxes:
            # Expand bbox
            expanded = bbox.expand(bbox_padding, w, h)

            # MobileSAM segmentation
            mask = segmenter.segment(image_rgb, expanded)

            # Check mask area
            mask_area = np.sum(mask > 127)
            if mask_area < min_area:
                stats["skipped_small"] += 1
                continue

            # Save mask
            mask_filename = f"{frame_basename}_inst{inst_idx}_mask.png"
            mask_path = masks_dir / mask_filename
            cv2.imwrite(str(mask_path), mask)

            # Create character image with transparent background (RGBA)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = image_rgb
            rgba[:, :, 3] = mask

            # Crop to bbox with padding
            crop_x1 = max(0, expanded.x1 - 20)
            crop_y1 = max(0, expanded.y1 - 20)
            crop_x2 = min(w, expanded.x2 + 20)
            crop_y2 = min(h, expanded.y2 + 20)

            cropped_rgba = rgba[crop_y1:crop_y2, crop_x1:crop_x2]

            # Save character crop
            char_filename = f"{frame_basename}_inst{inst_idx}.png"
            char_path = characters_dir / char_filename
            Image.fromarray(cropped_rgba).save(char_path)

            # Record metadata
            frame_instances.append({
                "instance_id": inst_idx,
                "bbox": [expanded.x1, expanded.y1, expanded.x2, expanded.y2],
                "confidence": bbox.confidence,
                "class": bbox.class_name,
                "mask_area": int(mask_area),
                "crop_region": [crop_x1, crop_y1, crop_x2, crop_y2],
                "character_file": char_filename,
                "mask_file": mask_filename
            })

            inst_idx += 1
            stats["total_instances"] += 1

        if frame_instances:
            stats["metadata"].append({
                "frame": frame_path.name,
                "frame_size": [w, h],
                "instances": frame_instances
            })

    # Save metadata
    metadata_path = output_dir / "instances_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "created": datetime.now().isoformat(),
            "source_dir": str(input_dir),
            "config": {
                "yolo_model": yolo_model,
                "bbox_padding": bbox_padding,
                "confidence": confidence,
                "min_area": min_area
            },
            "stats": {
                "total_frames": stats["total_frames"],
                "frames_with_detections": stats["frames_with_detections"],
                "total_instances": stats["total_instances"],
                "skipped_small": stats["skipped_small"]
            },
            "frames": stats["metadata"]
        }, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Batch YOLO + MobileSAM Segmentation")
    parser.add_argument("--projects", nargs="+", required=True,
                        help="Project names to process")
    parser.add_argument("--base-dir", type=Path,
                        default=Path("/mnt/data/datasets/general"),
                        help="Base directory containing projects")
    parser.add_argument("--yolo-model", default="yolov11n.pt",
                        help="YOLO model path")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--bbox-padding", type=float, default=0.1,
                        help="Bbox expansion ratio")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="YOLO confidence threshold")
    parser.add_argument("--min-area", type=int, default=4096,
                        help="Minimum mask area")

    args = parser.parse_args()

    print("=" * 60)
    print("BATCH YOLO + MOBILESAM SEGMENTATION")
    print("=" * 60)
    print(f"Projects: {args.projects}")
    print(f"Base dir: {args.base_dir}")
    print("=" * 60)

    all_stats = {}

    for project in args.projects:
        print(f"\n{'='*60}")
        print(f"Processing: {project}")
        print("=" * 60)

        input_dir = args.base_dir / project / "frames"
        output_dir = args.base_dir / project / "yolo_mobilesam_instances"

        if not input_dir.exists():
            print(f"⚠️ Input directory not found: {input_dir}")
            continue

        stats = run_segmentation(
            input_dir=input_dir,
            output_dir=output_dir,
            yolo_model=args.yolo_model,
            device=args.device,
            bbox_padding=args.bbox_padding,
            confidence=args.confidence,
            min_area=args.min_area
        )

        all_stats[project] = stats

        print(f"\n✓ {project} completed:")
        print(f"  Frames processed: {stats['total_frames']}")
        print(f"  Frames with detections: {stats['frames_with_detections']}")
        print(f"  Total instances: {stats['total_instances']}")
        print(f"  Skipped (too small): {stats['skipped_small']}")
        print(f"  Output: {output_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_instances = 0
    for project, stats in all_stats.items():
        print(f"{project:20s}: {stats['total_instances']:5d} instances")
        total_instances += stats['total_instances']

    print("-" * 30)
    print(f"{'TOTAL':20s}: {total_instances:5d} instances")
    print("=" * 60)


if __name__ == "__main__":
    main()
