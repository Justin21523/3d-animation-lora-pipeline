#!/usr/bin/env python3
"""
Direct ToonOut Segmentation on Frames
=====================================

Skip YOLO detection, directly apply ToonOut to all frames.
For 2D animation where YOLO person detection doesn't work well.

Usage:
    python run_toonout_direct.py --project wylde-pak
    python run_toonout_direct.py --project gumbell --episode ep01

Author: LLMProvider Tooling
Date: 2025-12-08
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

# Add segmentation scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "segmentation"))
from toonout_segmenter import ToonOutSegmenter


def get_episode_dirs(frames_dir: Path) -> list:
    """Get list of episode directories"""
    dirs = sorted([
        d for d in frames_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    return dirs


def count_images(directory: Path) -> int:
    """Count images in directory"""
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in extensions)


def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Direct ToonOut segmentation on frames")
    parser.add_argument("--project", "-p", required=True,
                        help="Project name (e.g., wylde-pak, gumbell)")
    parser.add_argument("--base-dir", default="/mnt/data/datasets/general",
                        help="Base directory for datasets")
    parser.add_argument("--episode", "-e", default=None,
                        help="Process specific episode only")
    parser.add_argument("--device", "-d", default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--background", "-b", default="transparent",
                        choices=["transparent", "white", "black", "gray"],
                        help="Background mode")

    args = parser.parse_args()

    # Setup paths
    project_dir = Path(args.base_dir) / args.project
    frames_dir = project_dir / "frames"
    output_dir = project_dir / "segmented_toonout"

    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        return 1

    # Get episodes to process
    episode_dirs = get_episode_dirs(frames_dir)

    if args.episode:
        # Process specific episode
        episode_dirs = [d for d in episode_dirs if d.name == args.episode]
        if not episode_dirs:
            print(f"Error: Episode '{args.episode}' not found")
            return 1

    print("=" * 60)
    print("DIRECT TOONOUT SEGMENTATION")
    print("=" * 60)
    print(f"Project:    {args.project}")
    print(f"Frames:     {frames_dir}")
    print(f"Output:     {output_dir}")
    print(f"Episodes:   {len(episode_dirs)}")
    print(f"Background: {args.background}")
    print("=" * 60)

    # Initialize segmenter once
    segmenter = ToonOutSegmenter(device=args.device)

    # Process each episode
    total_processed = 0
    total_failed = 0

    for i, ep_dir in enumerate(episode_dirs, 1):
        ep_name = ep_dir.name
        ep_output = output_dir / ep_name

        # Count images
        image_count = count_images(ep_dir)

        print(f"\n[{i}/{len(episode_dirs)}] Episode: {ep_name} ({image_count} frames)")

        if image_count == 0:
            print(f"  Skipping - no images found")
            continue

        # Process
        start_time = time.time()

        try:
            stats = segmenter.process_directory(
                input_dir=ep_dir,
                output_dir=ep_output,
                background_mode=args.background,
                show_progress=True
            )

            elapsed = time.time() - start_time
            fps = stats["successful"] / elapsed if elapsed > 0 else 0

            print(f"  ✓ Done: {stats['successful']}/{stats['total_images']} "
                  f"({elapsed:.1f}s, {fps:.1f} img/s)")

            total_processed += stats["successful"]
            total_failed += stats["failed"]

        except Exception as e:
            print(f"  ✗ Error: {e}")
            total_failed += image_count

        # Clear GPU memory between episodes
        clear_gpu_memory()

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed: {total_processed}")
    print(f"Total failed:    {total_failed}")
    print(f"Output:          {output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
