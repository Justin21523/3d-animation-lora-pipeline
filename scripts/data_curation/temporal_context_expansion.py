#!/usr/bin/env python3
"""
Temporal Context Expansion - Extract neighboring frames from curated dataset

Given curated frames, extract temporal neighbors from original frames directory.
This leverages the fact that adjacent frames likely contain the same character.
"""

import argparse
from pathlib import Path
import shutil
import re
from typing import List, Tuple, Dict
from collections import defaultdict
import json


def extract_frame_number(filename: str) -> int:
    """
    Extract frame number from filename

    Examples:
        frame_0123.png -> 123
        video_frame_0456_seg.png -> 456
        luca_00789.png -> 789
    """
    # Try multiple patterns
    patterns = [
        r'frame[_-]?(\d+)',
        r'_(\d+)\.',
        r'(\d{4,})',  # 4+ consecutive digits
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return -1


def find_original_frame(curated_frame: Path, original_frames_dir: Path) -> Path:
    """Find the corresponding original frame"""

    frame_num = extract_frame_number(curated_frame.name)

    if frame_num == -1:
        return None

    # Search for matching frame in original directory
    # Try various naming patterns
    patterns = [
        f"*{frame_num:04d}*",
        f"*{frame_num:05d}*",
        f"*{frame_num:06d}*",
        f"*frame*{frame_num}*",
    ]

    for pattern in patterns:
        matches = list(original_frames_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def get_temporal_neighbors(
    original_frame: Path,
    original_frames_dir: Path,
    window_size: int = 3
) -> List[Path]:
    """
    Get neighboring frames within temporal window

    Args:
        original_frame: The anchor frame
        original_frames_dir: Directory with all original frames
        window_size: How many frames before/after to include

    Returns:
        List of neighboring frame paths
    """
    center_num = extract_frame_number(original_frame.name)

    if center_num == -1:
        return []

    neighbors = []

    # Search for frames in range [center - window, center + window]
    for offset in range(-window_size, window_size + 1):
        if offset == 0:
            continue  # Skip center frame (already have it)

        target_num = center_num + offset

        # Try different naming patterns
        patterns = [
            f"*{target_num:04d}*",
            f"*{target_num:05d}*",
            f"*{target_num:06d}*",
        ]

        for pattern in patterns:
            matches = list(original_frames_dir.glob(pattern))
            if matches:
                neighbors.append(matches[0])
                break

    return neighbors


def expand_with_temporal_context(
    curated_dir: Path,
    original_frames_dir: Path,
    output_dir: Path,
    window_size: int = 3,
    copy_captions: bool = True,
    verbose: bool = True
):
    """
    Expand curated dataset with temporal neighbors

    Args:
        curated_dir: Directory with curated frames and captions
        original_frames_dir: Directory with all original frames
        output_dir: Output directory for expanded dataset
        window_size: Temporal window (frames before/after)
        copy_captions: Whether to copy captions to neighbors
        verbose: Print detailed progress
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all curated frames
    curated_frames = list(curated_dir.glob("*.png")) + list(curated_dir.glob("*.jpg"))

    print(f"🔍 Processing {len(curated_frames)} curated frames")
    print(f"   Original frames dir: {original_frames_dir}")
    print(f"   Window size: ±{window_size} frames")
    print(f"   Output: {output_dir}")

    # Copy all curated frames first
    print(f"\n📋 Copying curated frames...")
    for img_file in curated_frames:
        shutil.copy2(img_file, output_dir / img_file.name)

        # Copy caption if exists
        caption_file = img_file.with_suffix(".txt")
        if caption_file.exists():
            shutil.copy2(caption_file, output_dir / caption_file.name)

    curated_count = len(list(output_dir.glob("*.png")))

    # Expand with temporal neighbors
    print(f"\n🔄 Extracting temporal neighbors...")

    added_count = 0
    missing_count = 0
    neighbor_map = defaultdict(list)  # Track which neighbors came from which curated frame

    for idx, curated_frame in enumerate(curated_frames):
        if verbose and idx % 50 == 0:
            print(f"  Progress: {idx}/{len(curated_frames)} ({idx/len(curated_frames)*100:.1f}%)")

        # Find original frame
        original_frame = find_original_frame(curated_frame, original_frames_dir)

        if not original_frame:
            if verbose:
                print(f"  ⚠️  Could not find original for: {curated_frame.name}")
            missing_count += 1
            continue

        # Get temporal neighbors
        neighbors = get_temporal_neighbors(original_frame, original_frames_dir, window_size)

        if verbose and len(neighbors) < window_size * 2:
            print(f"  ⚠️  Only found {len(neighbors)} neighbors for {curated_frame.name} (expected {window_size*2})")

        # Copy neighbors
        for neighbor in neighbors:
            # Create unique name for neighbor
            neighbor_frame_num = extract_frame_number(neighbor.name)
            new_name = f"{curated_frame.stem}_ctx{neighbor_frame_num:06d}{neighbor.suffix}"
            output_path = output_dir / new_name

            # Copy image
            if not output_path.exists():
                shutil.copy2(neighbor, output_path)
                added_count += 1

                # Copy caption from curated frame (with note)
                if copy_captions:
                    caption_file = curated_frame.with_suffix(".txt")
                    if caption_file.exists():
                        with open(caption_file, 'r', encoding='utf-8') as f:
                            original_caption = f.read().strip()

                        # Add temporal context note
                        new_caption = f"{original_caption}, temporal neighbor of {curated_frame.stem}"

                        with open(output_path.with_suffix(".txt"), 'w', encoding='utf-8') as f:
                            f.write(new_caption)

                neighbor_map[curated_frame.stem].append(new_name)

    final_count = len(list(output_dir.glob("*.png")))

    print(f"\n✅ TEMPORAL EXPANSION COMPLETE")
    print(f"  Curated (original):  {curated_count}")
    print(f"  Neighbors added:     +{added_count}")
    print(f"  Total:               {final_count}")
    print(f"  Expansion ratio:     {final_count / curated_count:.1f}x")
    print(f"  Missing originals:   {missing_count}")
    print(f"\n📁 Output: {output_dir}")

    # Save neighbor map
    map_file = output_dir / "temporal_neighbor_map.json"
    with open(map_file, 'w') as f:
        json.dump(dict(neighbor_map), f, indent=2)
    print(f"📄 Neighbor map saved: {map_file}")

    # Save statistics
    stats = {
        "curated_count": curated_count,
        "neighbors_added": added_count,
        "final_count": final_count,
        "expansion_ratio": final_count / curated_count,
        "missing_count": missing_count,
        "window_size": window_size,
        "average_neighbors_per_frame": added_count / curated_count if curated_count > 0 else 0,
    }

    stats_file = output_dir / "expansion_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"📊 Statistics saved: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Expand curated dataset with temporal neighbors from original frames"
    )
    parser.add_argument(
        "curated_dir",
        type=Path,
        help="Directory with curated frames (372 精選幀)"
    )
    parser.add_argument(
        "original_frames_dir",
        type=Path,
        help="Directory with all original frames (完整 frame 目錄)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for expanded dataset"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Temporal window size (±N frames, default: 3)"
    )
    parser.add_argument(
        "--no-copy-captions",
        action="store_true",
        help="Don't copy captions to neighbor frames"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    if not args.curated_dir.exists():
        print(f"❌ Curated directory not found: {args.curated_dir}")
        return

    if not args.original_frames_dir.exists():
        print(f"❌ Original frames directory not found: {args.original_frames_dir}")
        return

    expand_with_temporal_context(
        curated_dir=args.curated_dir,
        original_frames_dir=args.original_frames_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        copy_captions=not args.no_copy_captions,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
