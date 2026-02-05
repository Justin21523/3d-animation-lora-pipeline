#!/usr/bin/env python3
"""
Batch SAM2 processing for all Super-Wings episodes
Processes all episodes with sequential GPU slot management
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Set
import json

# Configuration
FRAMES_BASE = Path("/mnt/data/datasets/general/super-wings/frames_dedup")
INSTANCES_BASE = Path("/mnt/data/datasets/general/super-wings/instances")
MAX_CONCURRENT = 1  # Max concurrent SAM2 processes (SET TO 1 for absolute stability - no OOM)
CHECK_INTERVAL = 30  # Check progress every 30 seconds
MIN_INSTANCES_THRESHOLD = 200  # Consider complete if > 200 instances and no recent files
MAX_FRAMES_LIMIT = 800  # Skip episodes with more frames than this to prevent OOM

# Optimized async I/O parameters for better GPU utilization
PREFETCH_SIZE = 32      # Increased from 16 - larger prefetch buffer
SAVE_WORKERS = 8        # Increased from 4 - more async save threads

# OOM Detection and Recovery
OOM_KEYWORDS = ["CUDA out of memory", "OutOfMemoryError", "out of memory"]
MAX_OOM_RETRIES = 3     # Max retries for OOM errors

SCRIPT_PATH = "/mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/generic/segmentation/instance_segmentation.py"
LOG_DIR = Path("/tmp/super_wings_sam2_logs")
LOG_DIR.mkdir(exist_ok=True)

def remove_first_half_frames(frames_dir: Path) -> int:
    """
    Remove first half of frames from a directory
    Returns: number of frames kept
    """
    # Get all frame files
    frame_files = sorted(frames_dir.glob("*.jpg"))
    total_frames = len(frame_files)

    if total_frames == 0:
        return 0

    # Calculate split point (keep second half)
    split_point = total_frames // 2
    frames_to_delete = frame_files[:split_point]
    frames_to_keep = frame_files[split_point:]

    print(f"    Total frames: {total_frames}")
    print(f"    Deleting first {len(frames_to_delete)} frames")
    print(f"    Keeping last {len(frames_to_keep)} frames")

    # Delete first half
    for frame_file in frames_to_delete:
        frame_file.unlink()

    return len(frames_to_keep)

def get_all_frame_directories() -> List[Path]:
    """Get all frame directories"""
    return sorted([d for d in FRAMES_BASE.iterdir() if d.is_dir()])

def get_processed_episodes() -> Set[str]:
    """Get list of already processed episodes"""
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

def is_processing(output_dir: Path) -> bool:
    """Check if a directory has recent file activity"""
    if not output_dir.exists():
        return False

    # Check for files created in last 3 minutes
    recent_files = list(output_dir.glob("**/*.png"))
    if not recent_files:
        return False

    import time
    current_time = time.time()
    for file_path in recent_files[-50:]:  # Check last 50 files
        if current_time - file_path.stat().st_mtime < 180:  # 3 minutes
            return True

    return False

def check_oom_error(log_file: Path) -> bool:
    """Check if log file contains OOM error"""
    if not log_file.exists():
        return False

    try:
        with open(log_file, 'r') as f:
            # Read last 100 lines to check for OOM
            lines = f.readlines()
            last_lines = ''.join(lines[-100:])

            for keyword in OOM_KEYWORDS:
                if keyword in last_lines:
                    return True
    except Exception as e:
        print(f"  ⚠️  Error reading log {log_file}: {e}")

    return False

def launch_sam2(frames_dir: Path, output_dir: Path, episode_id: str) -> subprocess.Popen:
    """Launch SAM2 processing for an episode"""
    log_file = LOG_DIR / f"{episode_id}_sam2.log"

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

    print(f"  🚀 Launched {episode_id} (PID: {process.pid}) | Log: {log_file}")
    return process

def kill_zombie_processes():
    """Force kill any zombie SAM2 processes for super-wings"""
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "instance_segmentation.py.*super-wings"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

def main():
    print("=" * 70)
    print("Super-Wings SAM2 Batch Processing (LAST HALF FRAMES ONLY)")
    print("=" * 70)

    # Get all episodes
    all_frames = get_all_frame_directories()
    processed = get_processed_episodes()

    print(f"\n📊 Statistics:")
    print(f"  Total episodes: {len(all_frames)}")
    print(f"  Already processed: {len(processed)}")
    print(f"  To process: {len(all_frames) - len(processed)}")
    print(f"  Max concurrent: {MAX_CONCURRENT}")
    print()

    # Create processing queue
    to_process = []
    for frames_dir in all_frames:
        episode_id = frames_dir.name.replace(" ", "_").replace("!", "")[:50]  # Shorten long names
        output_dir = INSTANCES_BASE / episode_id

        # Skip if already processed
        if episode_id in processed:
            print(f"⏭️  Skipping {episode_id} (already processed)")
            continue

        # Skip if currently has substantial instances
        if output_dir.exists():
            instance_count = len(list(output_dir.glob("**/*.png")))
            if instance_count > MIN_INSTANCES_THRESHOLD:
                print(f"⏭️  Skipping {episode_id} ({instance_count} instances exist)")
                continue

        to_process.append((frames_dir, output_dir, episode_id))

    if not to_process:
        print("\n✅ All episodes already processed!")
        return

    print(f"\n🎯 Will process {len(to_process)} episodes")
    print()

    # Track running processes
    running: Dict[str, subprocess.Popen] = {}
    completed = 0
    failed = 0
    oom_retries: Dict[str, int] = {}  # Track OOM retry counts per episode

    queue_idx = 0
    last_zombie_cleanup = time.time()

    while queue_idx < len(to_process) or running:
        # Periodically kill zombie processes (every 60 seconds)
        if time.time() - last_zombie_cleanup > 60:
            kill_zombie_processes()
            last_zombie_cleanup = time.time()

        # Clean up finished processes
        finished_episodes = []
        for episode_id in list(running.keys()):
            process = running[episode_id]

            # Check if process finished
            if process.poll() is not None:
                frames_dir, output_dir, _ = [t for t in to_process if t[2] == episode_id][0]
                instance_count = len(list(output_dir.glob("**/*.png"))) if output_dir.exists() else 0
                log_file = LOG_DIR / f"{episode_id}_sam2.log"

                if process.returncode == 0:
                    print(f"✅ {episode_id} completed ({instance_count} instances)")
                    completed += 1
                    finished_episodes.append(episode_id)
                else:
                    # Check if failure was due to OOM
                    is_oom = check_oom_error(log_file)
                    retry_count = oom_retries.get(episode_id, 0)

                    if is_oom and retry_count < MAX_OOM_RETRIES:
                        # Retry with OOM recovery
                        print(f"🔄 {episode_id} failed with OOM (retry {retry_count + 1}/{MAX_OOM_RETRIES})")
                        oom_retries[episode_id] = retry_count + 1

                        # Wait a bit for GPU memory to clear + force cleanup
                        print(f"  ⏳ Waiting 10 seconds for GPU memory to clear...")
                        kill_zombie_processes()
                        time.sleep(10)

                        # Relaunch the process
                        print(f"  🚀 Relaunching {episode_id}...")
                        new_process = launch_sam2(frames_dir, output_dir, episode_id)
                        running[episode_id] = new_process
                        # Don't add to finished_episodes, keep in running dict

                    else:
                        # Max retries reached or non-OOM failure
                        if is_oom:
                            print(f"❌ {episode_id} failed with OOM after {retry_count} retries")
                        else:
                            print(f"❌ {episode_id} failed (return code: {process.returncode})")
                        failed += 1
                        finished_episodes.append(episode_id)

        # Remove finished episodes from running dict
        for episode_id in finished_episodes:
            del running[episode_id]

        # Launch new processes if slots available
        while len(running) < MAX_CONCURRENT and queue_idx < len(to_process):
            frames_dir, output_dir, episode_id = to_process[queue_idx]

            print(f"\n[{queue_idx + 1}/{len(to_process)}] Processing {episode_id}...")

            # frames_dedup already has cleaned frames - no need to delete first half
            # Just check if frames exist and launch SAM2 directly
            try:
                frame_count = len(list(frames_dir.glob("*.jpg")))

                if frame_count == 0:
                    print(f"  ⚠️  No frames found in {frames_dir}, skipping...")
                    queue_idx += 1
                    failed += 1
                    continue

                # Skip episodes with too many frames to prevent OOM
                if frame_count > MAX_FRAMES_LIMIT:
                    print(f"  ⚠️  {episode_id} has {frame_count} frames (> {MAX_FRAMES_LIMIT} limit), skipping to prevent OOM...")
                    queue_idx += 1
                    failed += 1
                    continue

                print(f"  📊 Found {frame_count} frames in {frames_dir.name}")

                # Launch SAM2 directly (no frame deletion needed)
                process = launch_sam2(frames_dir, output_dir, episode_id)
                running[episode_id] = process

            except Exception as e:
                print(f"  ❌ Error launching {episode_id}: {e}")
                failed += 1

            queue_idx += 1
            time.sleep(3)  # Small delay between launches

        # Wait before next check
        if running:
            print(f"⏳ Active: {len(running)}/{MAX_CONCURRENT} | Completed: {completed}/{len(to_process)} | Failed: {failed}")
            time.sleep(CHECK_INTERVAL)

    print()
    print("=" * 60)
    print(f"✅ Batch processing complete!")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(to_process)}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
