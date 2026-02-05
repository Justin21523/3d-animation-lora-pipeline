#!/usr/bin/env python3
"""
Super Wings SAM2 Processing - Preserve Original Directory Names (LAST 1/3 FRAMES)
Processes ONLY the last 1/3 of frames from frames_dedup directories
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Set
import shutil

# Configuration
FRAMES_BASE = Path("/mnt/data/datasets/general/super-wings/frames_dedup")
INSTANCES_BASE = Path("/mnt/data/datasets/general/super-wings/instances")
MAX_CONCURRENT = 1  # Sequential processing for stability
CHECK_INTERVAL = 30
MIN_INSTANCES_THRESHOLD = 200
MAX_FRAMES_LIMIT = 600  # Adjusted for 1/3 of ~1200 frames

# Async I/O parameters
PREFETCH_SIZE = 32
SAVE_WORKERS = 8

# OOM Detection
OOM_KEYWORDS = ["CUDA out of memory", "OutOfMemoryError", "out of memory"]
MAX_OOM_RETRIES = 3

SCRIPT_PATH = "/mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/generic/segmentation/instance_segmentation.py"
LOG_DIR = Path("/tmp/super_wings_sam2_logs")
LOG_DIR.mkdir(exist_ok=True)

def remove_first_two_thirds_frames(frames_dir: Path) -> int:
    """
    Remove first 2/3 of frames from a directory, keep only last 1/3
    Returns: number of frames kept
    """
    # Get all frame files
    frame_files = sorted(frames_dir.glob("*.jpg"))
    total_frames = len(frame_files)

    if total_frames == 0:
        return 0

    # Calculate split point (keep last 1/3)
    split_point = (total_frames * 2) // 3
    frames_to_delete = frame_files[:split_point]
    frames_to_keep = frame_files[split_point:]

    print(f"    Total frames: {total_frames}")
    print(f"    Deleting first {len(frames_to_delete)} frames (2/3)")
    print(f"    Keeping last {len(frames_to_keep)} frames (1/3)")

    # Delete first 2/3
    for frame_file in frames_to_delete:
        frame_file.unlink()

    return len(frames_to_keep)

def get_all_frame_directories() -> List[Path]:
    """Get all frame directories, preserving exact names"""
    dirs = []
    for item in FRAMES_BASE.iterdir():
        if item.is_dir():
            dirs.append(item)
    return sorted(dirs)

def get_processed_episodes() -> Set[str]:
    """Get list of already processed episodes (by exact directory name)"""
    if not INSTANCES_BASE.exists():
        return set()

    processed = set()
    for instance_dir in INSTANCES_BASE.iterdir():
        if not instance_dir.is_dir():
            continue

        # Count instances
        instance_count = len(list(instance_dir.glob("**/*.png")))
        if instance_count > MIN_INSTANCES_THRESHOLD:
            processed.add(instance_dir.name)

    return processed

def launch_sam2(frames_dir: Path, output_dir: Path, episode_name: str) -> subprocess.Popen:
    """Launch SAM2 processing for an episode"""
    # Create safe log filename (replace special chars)
    safe_name = episode_name.replace(" ", "_").replace("!", "").replace("/", "_")[:100]
    log_file = LOG_DIR / f"{safe_name}_sam2.log"

    cmd = [
        "conda", "run", "-n", "ai_env",
        "python", str(SCRIPT_PATH),
        str(frames_dir),
        "--output-dir", str(output_dir),
        "--model", "sam2_hiera_large",
        "--min-size", "4096",
        "--device", "cuda",
        "--context-mode", "transparent",
        "--use-async-io",
        "--prefetch-size", str(PREFETCH_SIZE),
        "--save-workers", str(SAVE_WORKERS)
    ]

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        )

    print(f"  🚀 Launched: {episode_name[:60]}... (PID: {process.pid})")
    print(f"     Log: {log_file}")
    return process

def check_oom_error(log_file: Path) -> bool:
    """Check if log file contains OOM error"""
    if not log_file.exists():
        return False

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            last_lines = ''.join(lines[-100:])

            for keyword in OOM_KEYWORDS:
                if keyword in last_lines:
                    return True
    except Exception as e:
        print(f"  ⚠️  Error reading log {log_file}: {e}")

    return False

def main():
    print("=" * 80)
    print("Super Wings SAM2 Processing - Preserving Original Names (LAST 1/3 FRAMES)")
    print("=" * 80)

    # Backup existing instances directory
    if INSTANCES_BASE.exists():
        backup_name = f"instances_backup_{int(time.time())}"
        backup_path = INSTANCES_BASE.parent / backup_name
        print(f"\n📦 Backing up existing instances to: {backup_name}")
        shutil.move(str(INSTANCES_BASE), str(backup_path))

    # Create fresh instances directory
    INSTANCES_BASE.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created fresh instances directory")

    # Get all episodes
    all_frames = get_all_frame_directories()

    print(f"\n📊 Statistics:")
    print(f"  Total episode directories: {len(all_frames)}")
    print(f"  Max concurrent processes: {MAX_CONCURRENT}")
    print(f"  Max frames per episode: {MAX_FRAMES_LIMIT}")
    print()

    # STEP 1: Delete first 2/3 of frames from each directory
    print("\n🗑️  Deleting first 2/3 of frames from each episode (keeping last 1/3)...")
    for frames_dir in all_frames:
        episode_name = frames_dir.name
        try:
            frames_kept = remove_first_two_thirds_frames(frames_dir)
            print(f"  ✓ {episode_name[:60]}... kept {frames_kept} frames")
        except Exception as e:
            print(f"  ❌ Error processing {episode_name[:60]}...: {e}")

    print("\n✓ First 2/3 deletion complete\n")

    # Create processing queue
    to_process = []
    skipped_too_many_frames = []
    skipped_no_frames = []

    for frames_dir in all_frames:
        # Use EXACT original directory name
        episode_name = frames_dir.name
        output_dir = INSTANCES_BASE / episode_name  # Same exact name

        # Check frame count
        try:
            frame_count = len(list(frames_dir.glob("*.jpg")))

            if frame_count == 0:
                print(f"⏭️  Skipping {episode_name[:60]}... (no frames)")
                skipped_no_frames.append(episode_name)
                continue

            if frame_count > MAX_FRAMES_LIMIT:
                print(f"⏭️  Skipping {episode_name[:60]}... ({frame_count} frames > {MAX_FRAMES_LIMIT} limit)")
                skipped_too_many_frames.append((episode_name, frame_count))
                continue

            to_process.append((frames_dir, output_dir, episode_name, frame_count))

        except Exception as e:
            print(f"❌ Error checking {episode_name[:60]}...: {e}")
            continue

    if not to_process:
        print("\n⚠️  No episodes to process!")
        return

    print(f"\n🎯 Will process {len(to_process)} episodes:")
    for _, _, name, count in to_process:
        print(f"   - {name[:60]}... ({count} frames)")

    if skipped_too_many_frames:
        print(f"\n⚠️  Skipped {len(skipped_too_many_frames)} episodes (too many frames):")
        for name, count in skipped_too_many_frames:
            print(f"   - {name[:60]}... ({count} frames)")

    if skipped_no_frames:
        print(f"\n⚠️  Skipped {len(skipped_no_frames)} episodes (no frames)")

    print()

    # Process sequentially
    completed = 0
    failed = 0
    oom_retries = {}

    for idx, (frames_dir, output_dir, episode_name, frame_count) in enumerate(to_process):
        print(f"\n{'=' * 80}")
        print(f"[{idx + 1}/{len(to_process)}] Processing: {episode_name}")
        print(f"Frames: {frame_count} | Output: {output_dir.name}")
        print('=' * 80)

        retry_count = 0
        max_retries = MAX_OOM_RETRIES

        while retry_count <= max_retries:
            # Launch SAM2
            process = launch_sam2(frames_dir, output_dir, episode_name)

            # Wait for completion
            while process.poll() is None:
                time.sleep(CHECK_INTERVAL)
                print(f"⏳ Processing... (PID: {process.pid})")

            # Check result
            instance_count = len(list(output_dir.glob("**/*.png"))) if output_dir.exists() else 0
            safe_name = episode_name.replace(" ", "_").replace("!", "").replace("/", "_")[:100]
            log_file = LOG_DIR / f"{safe_name}_sam2.log"

            if process.returncode == 0:
                print(f"✅ Completed: {episode_name}")
                print(f"   Instances extracted: {instance_count}")
                completed += 1
                break
            else:
                # Check for OOM
                is_oom = check_oom_error(log_file)

                if is_oom and retry_count < max_retries:
                    retry_count += 1
                    print(f"🔄 OOM detected, retrying ({retry_count}/{max_retries})...")
                    time.sleep(10)  # Wait for GPU to clear
                else:
                    if is_oom:
                        print(f"❌ Failed (OOM) after {retry_count} retries: {episode_name}")
                    else:
                        print(f"❌ Failed (return code: {process.returncode}): {episode_name}")
                    failed += 1
                    break

    print()
    print("=" * 80)
    print(f"✅ Batch processing complete!")
    print(f"  Completed: {completed}/{len(to_process)}")
    print(f"  Failed: {failed}/{len(to_process)}")
    print(f"  Skipped (too many frames): {len(skipped_too_many_frames)}")
    print(f"  Skipped (no frames): {len(skipped_no_frames)}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
