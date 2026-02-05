#!/usr/bin/env python3
"""
Batch SAM2 Instance Segmentation for 3D Animation Movies

Special handling for super-wings: removes first 1/3 of frames per episode
"""
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List
import shutil

def trim_super_wings_frames(frames_dir: Path, output_dir: Path) -> int:
    """
    For super-wings, remove first 1/3 of frames from each episode

    Args:
        frames_dir: Original frames directory
        output_dir: Output directory for trimmed frames

    Returns:
        Total frames processed
    """
    print(f"\n🎬 Special processing for super-wings: removing first 1/3 of frames per episode")
    print(f"   Input:  {frames_dir}")
    print(f"   Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_original = 0
    total_kept = 0

    # Process each episode directory
    episodes = sorted([d for d in frames_dir.iterdir() if d.is_dir()])

    for episode_dir in episodes:
        frames = sorted(list(episode_dir.glob("*.jpg")) + list(episode_dir.glob("*.png")))

        if len(frames) == 0:
            continue

        # Calculate split point (skip first 1/3)
        skip_count = len(frames) // 3
        frames_to_keep = frames[skip_count:]

        # Create output episode directory
        output_episode_dir = output_dir / episode_dir.name
        output_episode_dir.mkdir(parents=True, exist_ok=True)

        # Copy remaining frames
        for frame in frames_to_keep:
            dst = output_episode_dir / frame.name
            shutil.copy2(frame, dst)

        total_original += len(frames)
        total_kept += len(frames_to_keep)

        print(f"  {episode_dir.name}: {len(frames)} → {len(frames_to_keep)} frames (removed {skip_count})")

    print(f"\n✅ Total: {total_original:,} → {total_kept:,} frames ({total_original - total_kept:,} removed)")

    return total_kept


def run_sam2_segmentation(
    frames_dir: Path,
    output_dir: Path,
    config_path: Path,
    gpu_slots: int = 4,
    max_episodes: int = None
) -> bool:
    """
    Run SAM2 instance segmentation

    Args:
        frames_dir: Input frames directory
        output_dir: Output instances directory
        config_path: SAM2 config file
        gpu_slots: Number of GPU slots (controls memory usage)
        max_episodes: Maximum concurrent episodes (None = auto based on GPU slots)

    Returns:
        True if successful
    """
    script_path = Path(__file__).parent.parent / "generic" / "segmentation" / "instance_segmentation.py"

    # Check if frames_dir has subdirectories (episodic structure)
    episode_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])

    if len(episode_dirs) > 0:
        # Process each episode directory separately
        print(f"\n📁 Found {len(episode_dirs)} episode directories")
        print(f"   Processing episodes one by one...")

        success_count = 0
        failed_episodes = []

        for ep_idx, episode_dir in enumerate(episode_dirs, 1):
            # Check frame count
            frames = list(episode_dir.glob("*.jpg")) + list(episode_dir.glob("*.png"))
            if len(frames) == 0:
                print(f"  ⏭️  Skipping {episode_dir.name} (no frames)")
                continue

            # Episode-specific output directory
            episode_output = output_dir / episode_dir.name
            episode_output.mkdir(parents=True, exist_ok=True)

            cmd = [
                "conda", "run", "-n", "ai_env",
                "python", str(script_path),
                str(episode_dir),
                "--output-dir", str(episode_output),
                "--model", "sam2_hiera_large",
                "--min-size", "4096",
                "--device", "cuda",
                "--context-mode", "transparent"
            ]

            print(f"\n  [{ep_idx}/{len(episode_dirs)}] 🎬 {episode_dir.name}: {len(frames)} frames")

            result = subprocess.run(cmd, env={
                **subprocess.os.environ,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
            })

            if result.returncode == 0:
                success_count += 1
                print(f"  ✅ {episode_dir.name} completed")
            else:
                failed_episodes.append(episode_dir.name)
                print(f"  ❌ {episode_dir.name} failed")

        print(f"\n📊 Episode Processing Summary:")
        print(f"   Total: {len(episode_dirs)}")
        print(f"   Success: {success_count}")
        print(f"   Failed: {len(failed_episodes)}")

        if failed_episodes:
            print(f"   Failed episodes: {', '.join(failed_episodes)}")

        return len(failed_episodes) == 0

    else:
        # Flat directory structure - process directly
        print(f"\n🚀 Running SAM2 segmentation (flat directory)...")

        cmd = [
            "conda", "run", "-n", "ai_env",
            "python", str(script_path),
            str(frames_dir),
            "--output-dir", str(output_dir),
            "--model", "sam2_hiera_large",
            "--min-size", "4096",
            "--device", "cuda",
            "--context-mode", "transparent"
        ]

        print(f"   Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, env={
            **subprocess.os.environ,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        })

        return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch SAM2 segmentation for 3D animation movies"
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        choices=["super-why", "super-wings", "win-or-lose", "boss-baby", "angello", "all"],
        default=["all"],
        help="Projects to process"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/mnt/data/datasets/general"),
        help="Base directory for all projects"
    )
    parser.add_argument(
        "--gpu-slots",
        type=int,
        default=4,
        help="Number of GPU slots (affects memory usage)"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum concurrent episodes (None = auto)"
    )
    parser.add_argument(
        "--skip-trimming",
        action="store_true",
        help="Skip super-wings frame trimming (use if already done)"
    )

    args = parser.parse_args()

    # Resolve projects
    if "all" in args.projects:
        projects = ["super-why", "super-wings", "win-or-lose", "boss-baby", "angello"]
    else:
        projects = args.projects

    print("=" * 80)
    print("🎬 3D Animation Movies - SAM2 Batch Processing")
    print("=" * 80)
    print(f"Projects: {', '.join(projects)}")
    print(f"GPU slots: {args.gpu_slots}")
    print(f"Max episodes: {args.max_episodes or 'auto'}")
    print()

    success_count = 0
    failed_projects = []

    for project in projects:
        print("\n" + "=" * 80)
        print(f"📂 Processing: {project}")
        print("=" * 80)

        project_dir = args.base_dir / project
        frames_dir = project_dir / "frames"
        instances_dir = project_dir / "instances"

        # Check if frames exist
        if not frames_dir.exists():
            print(f"❌ Frames directory not found: {frames_dir}")
            failed_projects.append(f"{project} (no frames)")
            continue

        # Special handling for super-wings: trim first 1/3 of frames
        if project == "super-wings" and not args.skip_trimming:
            trimmed_dir = project_dir / "frames_trimmed"

            # Check if already trimmed
            if trimmed_dir.exists() and len(list(trimmed_dir.rglob("*.jpg"))) > 0:
                print(f"✓ Using existing trimmed frames: {trimmed_dir}")
                frames_dir = trimmed_dir
            else:
                frame_count = trim_super_wings_frames(frames_dir, trimmed_dir)
                if frame_count == 0:
                    print(f"❌ No frames after trimming")
                    failed_projects.append(f"{project} (trim failed)")
                    continue
                frames_dir = trimmed_dir

        # Run SAM2 segmentation
        success = run_sam2_segmentation(
            frames_dir=frames_dir,
            output_dir=instances_dir,
            config_path=Path("configs/3d_anime/sam2_config.yaml"),
            gpu_slots=args.gpu_slots,
            max_episodes=args.max_episodes
        )

        if success:
            print(f"✅ {project} completed successfully")
            success_count += 1
        else:
            print(f"❌ {project} failed")
            failed_projects.append(project)

    # Summary
    print("\n" + "=" * 80)
    print("📊 Batch Processing Summary")
    print("=" * 80)
    print(f"Total projects: {len(projects)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_projects)}")

    if failed_projects:
        print(f"\nFailed projects:")
        for proj in failed_projects:
            print(f"  - {proj}")

    return 0 if len(failed_projects) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
