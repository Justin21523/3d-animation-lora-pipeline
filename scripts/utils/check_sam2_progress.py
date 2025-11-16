#!/usr/bin/env python3
"""
Quick SAM2 Progress Checker

Checks the current progress of SAM2 instance segmentation.

Usage:
    python check_sam2_progress.py [INSTANCES_DIR] [TOTAL_FRAMES]

Example:
    python check_sam2_progress.py \
      /mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances \
      4323
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta


def check_progress(instances_dir: Path, total_frames: int = 4323):
    """
    Check SAM2 processing progress

    Args:
        instances_dir: Directory with instance images
        total_frames: Total number of frames to process
    """
    instances_dir = Path(instances_dir)

    # Count current instances
    if not instances_dir.exists():
        print(f"❌ Directory not found: {instances_dir}")
        return

    instance_files = list(instances_dir.glob("*.png"))
    n_instances = len(instance_files)

    # Estimate frames processed (assuming ~8 instances per frame on average)
    avg_instances_per_frame = 8
    frames_processed = n_instances // avg_instances_per_frame

    # Calculate progress
    progress_pct = (frames_processed / total_frames) * 100

    # Estimate time
    # Based on observed speed: ~150 frames/hour with optimized SAM2
    frames_per_hour = 150
    frames_remaining = total_frames - frames_processed
    hours_remaining = frames_remaining / frames_per_hour

    eta = datetime.now() + timedelta(hours=hours_remaining)

    # Display results
    print("="*60)
    print("🎬 SAM2 Instance Segmentation Progress")
    print("="*60)
    print(f"📁 Directory: {instances_dir}")
    print(f"📊 Instances generated: {n_instances:,}")
    print(f"🎞️  Frames processed: ~{frames_processed:,} / {total_frames:,}")
    print(f"📈 Progress: {progress_pct:.1f}%")
    print(f"⏱️  Estimated remaining: {hours_remaining:.1f} hours")
    print(f"🕐 Estimated completion: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Check if complete
    if frames_processed >= total_frames:
        print("✅ SAM2 processing appears to be COMPLETE!")
        print(f"   Ready for identity clustering with {n_instances:,} instances")
    elif progress_pct < 10:
        print("⚠️  Processing just started, check back in a few hours")
    elif progress_pct < 50:
        print("⏳ Processing in progress, about halfway there")
    else:
        print("🔜 Processing nearly complete, should finish soon")

    return {
        'instances': n_instances,
        'frames_processed': frames_processed,
        'total_frames': total_frames,
        'progress_pct': progress_pct,
        'hours_remaining': hours_remaining
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check SAM2 instance segmentation progress"
    )
    parser.add_argument(
        "instances_dir",
        nargs='?',
        default="/mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances",
        type=str,
        help="Directory with instance images"
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=4323,
        help="Total number of frames to process (default: 4323)"
    )

    args = parser.parse_args()

    check_progress(
        instances_dir=Path(args.instances_dir),
        total_frames=args.total_frames
    )


if __name__ == "__main__":
    main()
