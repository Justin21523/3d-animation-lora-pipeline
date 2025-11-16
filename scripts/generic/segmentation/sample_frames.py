#!/usr/bin/env python3
"""
Frame Sampling Utility

Samples representative frames from each scene for efficient processing.
Avoids scene transition artifacts by skipping first and last frames.
"""

import argparse
from pathlib import Path
from typing import List, Set
import shutil
from tqdm import tqdm


def parse_frame_filename(filename: str) -> dict:
    """
    Parse frame filename to extract scene and position.

    Format: scene0000_pos0_frame000000_t0.08s.jpg

    Returns:
        dict with 'scene', 'position', 'frame_num', 'time'
    """
    parts = filename.replace('.jpg', '').replace('.png', '').split('_')

    return {
        'scene': parts[0],
        'position': int(parts[1].replace('pos', '')),
        'frame_num': int(parts[2].replace('frame', '')),
        'time': parts[3].replace('t', '').replace('s', ''),
        'filename': filename
    }


def sample_frames(
    input_dir: Path,
    output_dir: Path,
    positions: List[int] = [1, 5, 8],
    mode: str = 'symlink'
) -> dict:
    """
    Sample frames from each scene.

    Args:
        input_dir: Directory with all frames
        output_dir: Directory for sampled frames
        positions: Position indices to sample (e.g., [1, 5, 8])
        mode: 'symlink', 'copy', or 'list'
            - symlink: Create symbolic links (fastest, saves space)
            - copy: Copy files (slower, uses more space)
            - list: Create a text file list (for filtering)

    Returns:
        Sampling statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all frames
    image_files = sorted(
        list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    )

    print(f"📊 Found {len(image_files)} total frames")

    # Parse all frames
    frames_by_scene = {}
    for img_path in image_files:
        info = parse_frame_filename(img_path.name)
        scene = info['scene']

        if scene not in frames_by_scene:
            frames_by_scene[scene] = {}

        frames_by_scene[scene][info['position']] = img_path

    print(f"📊 Found {len(frames_by_scene)} scenes")

    # Sample frames
    sampled_frames = []
    stats = {
        'total_scenes': len(frames_by_scene),
        'positions_per_scene': len(positions),
        'total_sampled': 0,
        'missing_positions': 0
    }

    for scene, scene_frames in tqdm(frames_by_scene.items(), desc="Sampling frames"):
        for pos in positions:
            if pos in scene_frames:
                sampled_frames.append(scene_frames[pos])
                stats['total_sampled'] += 1
            else:
                stats['missing_positions'] += 1
                print(f"⚠️  Missing {scene}_pos{pos}")

    print(f"\n✅ Sampled {stats['total_sampled']} frames from {stats['total_scenes']} scenes")

    # Process based on mode
    if mode == 'symlink':
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"🔗 Creating symbolic links in {output_dir}...")

        for frame_path in tqdm(sampled_frames, desc="Creating symlinks"):
            symlink_path = output_dir / frame_path.name
            if not symlink_path.exists():
                symlink_path.symlink_to(frame_path.resolve())

    elif mode == 'copy':
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📋 Copying frames to {output_dir}...")

        for frame_path in tqdm(sampled_frames, desc="Copying files"):
            dest_path = output_dir / frame_path.name
            if not dest_path.exists():
                shutil.copy2(frame_path, dest_path)

    elif mode == 'list':
        list_file = output_dir / 'sampled_frames.txt'
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📝 Writing frame list to {list_file}...")
        with open(list_file, 'w') as f:
            for frame_path in sampled_frames:
                f.write(f"{frame_path}\n")

        print(f"✅ Saved {len(sampled_frames)} frame paths to {list_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Sample representative frames from each scene"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory with all frames"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for sampled frames"
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs='+',
        default=[1, 5, 8],
        help="Position indices to sample (default: 1 5 8)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['symlink', 'copy', 'list'],
        default='symlink',
        help="Sampling mode: symlink (default), copy, or list"
    )

    args = parser.parse_args()

    # Sample frames
    stats = sample_frames(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        positions=args.positions,
        mode=args.mode
    )

    print(f"\n📊 Sampling Statistics:")
    print(f"   Total scenes: {stats['total_scenes']}")
    print(f"   Positions per scene: {stats['positions_per_scene']}")
    print(f"   Total sampled: {stats['total_sampled']}")
    print(f"   Missing positions: {stats['missing_positions']}")


if __name__ == "__main__":
    main()
