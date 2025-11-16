#!/usr/bin/env python3
"""
Reprocess Specific Frames with Custom Parameters

Allows reprocessing of specific frames that may have missed small characters
or need different segmentation parameters.

Usage:
    # Reprocess specific frames
    python reprocess_frames.py \
        --frames-dir /path/to/frames \
        --output-dir /path/to/output \
        --frame-list scene0123_pos1,scene0456_pos5 \
        --min-size 32

    # Reprocess frames from a scene range
    python reprocess_frames.py \
        --frames-dir /path/to/frames \
        --output-dir /path/to/output \
        --scene-range 100-150 \
        --min-size 48

    # Reprocess distant character frames (very small min size)
    python reprocess_frames.py \
        --frames-dir /path/to/frames \
        --output-dir /path/to/output \
        --frame-list-file distant_frames.txt \
        --min-size 32 \
        --max-instances 20
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from instance_segmentation import SAM2InstanceSegmenter, visualize_instances
from PIL import Image
import torch


def parse_frame_list(frame_list: str) -> List[str]:
    """Parse comma-separated frame list"""
    return [f.strip() for f in frame_list.split(',')]


def parse_scene_range(scene_range: str, frames_dir: Path) -> List[Path]:
    """Parse scene range like '100-150' and find matching frames"""
    start, end = map(int, scene_range.split('-'))

    frames = []
    for scene_num in range(start, end + 1):
        pattern = f"scene{scene_num:04d}_*.jpg"
        frames.extend(sorted(frames_dir.glob(pattern)))
        pattern = f"scene{scene_num:04d}_*.png"
        frames.extend(sorted(frames_dir.glob(pattern)))

    return frames


def load_frame_list_file(file_path: Path) -> List[str]:
    """Load frame names from text file (one per line)"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def reprocess_frames(
    frames_dir: Path,
    output_dir: Path,
    frame_names: List[str] = None,
    scene_range: str = None,
    frame_list_file: Path = None,
    min_size: int = 64,
    max_instances: int = 15,
    device: str = "cuda",
    save_visualization: bool = True
):
    """
    Reprocess specific frames with custom parameters

    Args:
        frames_dir: Directory containing original frames
        output_dir: Output directory for reprocessed instances
        frame_names: List of frame names to reprocess
        scene_range: Scene range like "100-150"
        frame_list_file: File containing frame names
        min_size: Minimum instance size (pixels, e.g., 32 for distant characters)
        max_instances: Maximum instances per frame
        device: cuda or cpu
        save_visualization: Save visualization images
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    # Create output directories
    instances_dir = output_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    if save_visualization:
        viz_dir = output_dir / "visualization"
        viz_dir.mkdir(exist_ok=True)

    # Determine which frames to process
    frames_to_process = []

    if frame_list_file:
        frame_names = load_frame_list_file(frame_list_file)

    if frame_names:
        # Find specific frames
        for name in frame_names:
            # Try both .jpg and .png
            for ext in ['.jpg', '.png']:
                frame_path = frames_dir / f"{name}{ext}"
                if frame_path.exists():
                    frames_to_process.append(frame_path)
                    break

    elif scene_range:
        # Process scene range
        frames_to_process = parse_scene_range(scene_range, frames_dir)

    else:
        print("❌ Error: Must specify --frame-list, --scene-range, or --frame-list-file")
        return

    if not frames_to_process:
        print("❌ No frames found to process!")
        return

    print(f"🔄 Reprocessing {len(frames_to_process)} frames...")
    print(f"   Min instance size: {min_size}×{min_size} = {min_size*min_size} pixels")
    print(f"   Max instances: {max_instances}")
    print()

    # Initialize segmenter with custom parameters
    segmenter = SAM2InstanceSegmenter(
        model_type="sam2_hiera_large",
        device=device,
        min_mask_size=min_size * min_size,
        max_instances=max_instances
    )

    stats = {
        'total_frames': len(frames_to_process),
        'total_instances': 0,
        'frames_with_instances': 0,
        'frames_without_instances': 0
    }

    # Process each frame
    for frame_path in frames_to_process:
        print(f"Processing: {frame_path.name}...", end=' ')

        try:
            # Load image
            image = Image.open(frame_path).convert("RGB")

            # Segment instances
            instances = segmenter.segment_instances(image)

            num_instances = len(instances)
            stats['total_instances'] += num_instances

            if num_instances > 0:
                stats['frames_with_instances'] += 1
                print(f"✓ {num_instances} instances")
            else:
                stats['frames_without_instances'] += 1
                print("⚠️  No instances")

            # Save instances
            for inst_idx, instance in enumerate(instances):
                # Extract instance image
                instance_image = segmenter.extract_instance_image(image, instance)

                # Generate filename
                frame_name = frame_path.stem
                instance_filename = f"{frame_name}_inst{inst_idx}.png"
                instance_path = instances_dir / instance_filename

                # Save
                instance_image.save(instance_path)

            # Save visualization
            if save_visualization and num_instances > 0:
                viz_image = visualize_instances(image, instances)
                viz_path = viz_dir / f"{frame_name}_instances.jpg"
                viz_image.save(viz_path, quality=90)

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()

    # Save report
    report = {
        'parameters': {
            'min_size': min_size,
            'min_mask_area': min_size * min_size,
            'max_instances': max_instances
        },
        'statistics': stats,
        'processed_frames': [str(f) for f in frames_to_process]
    }

    report_path = output_dir / "reprocess_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Reprocessing complete!")
    print(f"   Total frames: {stats['total_frames']}")
    print(f"   Frames with instances: {stats['frames_with_instances']}")
    print(f"   Frames without instances: {stats['frames_without_instances']}")
    print(f"   Total instances extracted: {stats['total_instances']}")
    print(f"   Output: {instances_dir}")
    print(f"   Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reprocess specific frames with custom parameters"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory containing original frames"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for reprocessed instances"
    )
    parser.add_argument(
        "--frame-list",
        type=str,
        help="Comma-separated list of frame names (without extension)"
    )
    parser.add_argument(
        "--scene-range",
        type=str,
        help="Scene range like '100-150'"
    )
    parser.add_argument(
        "--frame-list-file",
        type=str,
        help="Text file with frame names (one per line)"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=64,
        help="Minimum instance size in pixels (default: 64, use 32-48 for distant characters)"
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=15,
        help="Maximum instances per frame (default: 15)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Don't save visualization images"
    )

    args = parser.parse_args()

    reprocess_frames(
        frames_dir=Path(args.frames_dir),
        output_dir=Path(args.output_dir),
        frame_names=parse_frame_list(args.frame_list) if args.frame_list else None,
        scene_range=args.scene_range,
        frame_list_file=Path(args.frame_list_file) if args.frame_list_file else None,
        min_size=args.min_size,
        max_instances=args.max_instances,
        device=args.device,
        save_visualization=not args.no_visualization
    )


if __name__ == "__main__":
    main()
