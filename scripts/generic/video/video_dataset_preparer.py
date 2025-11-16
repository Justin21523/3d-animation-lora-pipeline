#!/usr/bin/env python3
"""
Video Generation Dataset Preparer
Prepares datasets for AnimateDiff/SVD training from layered frames
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import argparse
from tqdm import tqdm
import re


class VideoDatasetPreparer:
    """Prepare video generation training datasets"""

    def __init__(
        self,
        layered_frames_dir: Path,
        output_dir: Path,
        sequence_length: int = 16,
        min_fps: float = 4.0,
        max_gap_seconds: float = 1.0,
        use_hard_links: bool = True
    ):
        """
        Initialize dataset preparer

        Args:
            layered_frames_dir: Directory with layered frames
            output_dir: Output directory for dataset
            sequence_length: Number of frames per sequence
            min_fps: Minimum FPS for sequence selection
            max_gap_seconds: Maximum time gap between consecutive frames
            use_hard_links: Use hard links instead of copying
        """
        self.layered_frames_dir = Path(layered_frames_dir)
        self.output_dir = Path(output_dir)
        self.sequence_length = sequence_length
        self.min_fps = min_fps
        self.max_gap_seconds = max_gap_seconds
        self.use_hard_links = use_hard_links

        print(f"🎬 Video Dataset Preparer")
        print(f"  Input: {layered_frames_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Sequence Length: {sequence_length} frames")
        print(f"  Min FPS: {min_fps}")
        print(f"  Max Gap: {max_gap_seconds}s")
        print(f"  Link Mode: {'Hard Links' if use_hard_links else 'Copy'}")

    def parse_filename(self, filename: str) -> Dict:
        """
        Parse frame filename to extract metadata

        Args:
            filename: Frame filename (e.g., scene0001_frame000003_t10.92s_character.png)

        Returns:
            Dictionary with scene_id, frame_id, timestamp
        """
        pattern = r'scene(\d+)_frame(\d+)_t([\d.]+)s'
        match = re.match(pattern, filename)

        if not match:
            return None

        return {
            'scene_id': int(match.group(1)),
            'frame_id': int(match.group(2)),
            'timestamp': float(match.group(3)),
            'filename': filename
        }

    def extract_sequences_from_episode(
        self,
        episode_dir: Path
    ) -> List[List[Dict]]:
        """
        Extract valid frame sequences from an episode

        Args:
            episode_dir: Episode directory

        Returns:
            List of frame sequences
        """
        # Load segmentation metadata
        metadata_path = episode_dir / "segmentation_results.json"
        if not metadata_path.exists():
            print(f"  ⚠️  No metadata found in {episode_dir.name}")
            return []

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Group frames by scene
        scenes = defaultdict(list)

        for frame_info in metadata['processed_frames']:
            input_name = frame_info['input']
            parsed = self.parse_filename(input_name)

            if parsed:
                parsed['outputs'] = frame_info['outputs']
                scenes[parsed['scene_id']].append(parsed)

        # Sort frames within each scene by timestamp
        for scene_id in scenes:
            scenes[scene_id].sort(key=lambda x: x['timestamp'])

        # Extract sequences from each scene
        sequences = []

        for scene_id, frames in scenes.items():
            if len(frames) < self.sequence_length:
                continue

            # Sliding window to extract sequences
            for i in range(len(frames) - self.sequence_length + 1):
                sequence = frames[i:i + self.sequence_length]

                # Check if sequence is valid
                if self._is_valid_sequence(sequence):
                    sequences.append(sequence)

        return sequences

    def _is_valid_sequence(self, sequence: List[Dict]) -> bool:
        """
        Check if a frame sequence is valid for training

        Args:
            sequence: List of frame metadata

        Returns:
            True if valid
        """
        # Check FPS
        time_span = sequence[-1]['timestamp'] - sequence[0]['timestamp']
        if time_span == 0:
            return False

        fps = (len(sequence) - 1) / time_span
        if fps < self.min_fps:
            return False

        # Check maximum gap between consecutive frames
        for i in range(len(sequence) - 1):
            gap = sequence[i + 1]['timestamp'] - sequence[i]['timestamp']
            if gap > self.max_gap_seconds:
                return False

        return True

    def prepare_character_sequences(self):
        """Prepare character animation sequences"""
        output_char_dir = self.output_dir / "character_sequences"
        output_char_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Extracting character animation sequences...")

        # Find all episodes
        episodes = sorted([d for d in self.layered_frames_dir.iterdir() if d.is_dir()])

        total_sequences = 0
        all_sequences = []

        for episode_dir in tqdm(episodes, desc="Episodes"):
            sequences = self.extract_sequences_from_episode(episode_dir)

            for seq_idx, sequence in enumerate(sequences):
                sequence_info = {
                    'episode': episode_dir.name,
                    'sequence_id': f"{episode_dir.name}_scene{sequence[0]['scene_id']:04d}_seq{seq_idx:04d}",
                    'frames': sequence,
                    'fps': (len(sequence) - 1) / (sequence[-1]['timestamp'] - sequence[0]['timestamp']),
                    'duration': sequence[-1]['timestamp'] - sequence[0]['timestamp']
                }
                all_sequences.append(sequence_info)

        # Save sequences
        print(f"\n💾 Saving {len(all_sequences)} sequences...")

        for seq_info in tqdm(all_sequences, desc="Saving"):
            seq_dir = output_char_dir / seq_info['sequence_id']
            seq_dir.mkdir(parents=True, exist_ok=True)

            # Save frames
            episode_dir = self.layered_frames_dir / seq_info['episode']

            for frame_idx, frame in enumerate(seq_info['frames']):
                src_path = episode_dir / "character" / frame['outputs']['character']
                dst_path = seq_dir / f"frame_{frame_idx:04d}.png"

                if self.use_hard_links and 'ai_warehouse' in str(self.output_dir):
                    try:
                        dst_path.hardlink_to(src_path)
                    except Exception:
                        shutil.copy2(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)

            # Save metadata
            metadata = {
                'sequence_id': seq_info['sequence_id'],
                'episode': seq_info['episode'],
                'scene_id': seq_info['frames'][0]['scene_id'],
                'num_frames': len(seq_info['frames']),
                'fps': seq_info['fps'],
                'duration': seq_info['duration'],
                'timestamps': [f['timestamp'] for f in seq_info['frames']]
            }

            with open(seq_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

        # Save summary
        summary = {
            'total_sequences': len(all_sequences),
            'sequence_length': self.sequence_length,
            'min_fps': self.min_fps,
            'max_gap_seconds': self.max_gap_seconds,
            'sequences': [
                {
                    'id': s['sequence_id'],
                    'episode': s['episode'],
                    'fps': s['fps'],
                    'duration': s['duration']
                }
                for s in all_sequences
            ]
        }

        with open(output_char_dir / "sequences_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Character sequences prepared!")
        print(f"  Total: {len(all_sequences)} sequences")
        print(f"  Output: {output_char_dir}")

        return all_sequences

    def prepare_background_sequences(self):
        """Prepare background sequences for scene generation"""
        output_bg_dir = self.output_dir / "background_sequences"
        output_bg_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🏞️  Extracting background sequences...")

        episodes = sorted([d for d in self.layered_frames_dir.iterdir() if d.is_dir()])

        all_sequences = []

        for episode_dir in tqdm(episodes, desc="Episodes"):
            sequences = self.extract_sequences_from_episode(episode_dir)

            for seq_idx, sequence in enumerate(sequences):
                sequence_info = {
                    'episode': episode_dir.name,
                    'sequence_id': f"{episode_dir.name}_scene{sequence[0]['scene_id']:04d}_seq{seq_idx:04d}",
                    'frames': sequence,
                    'fps': (len(sequence) - 1) / (sequence[-1]['timestamp'] - sequence[0]['timestamp'])
                }
                all_sequences.append(sequence_info)

        # Save background sequences
        print(f"\n💾 Saving {len(all_sequences)} background sequences...")

        for seq_info in tqdm(all_sequences, desc="Saving"):
            seq_dir = output_bg_dir / seq_info['sequence_id']
            seq_dir.mkdir(parents=True, exist_ok=True)

            episode_dir = self.layered_frames_dir / seq_info['episode']

            for frame_idx, frame in enumerate(seq_info['frames']):
                src_path = episode_dir / "background" / frame['outputs']['background']
                dst_path = seq_dir / f"frame_{frame_idx:04d}.jpg"

                if self.use_hard_links and 'ai_warehouse' in str(self.output_dir):
                    try:
                        dst_path.hardlink_to(src_path)
                    except Exception:
                        shutil.copy2(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)

            # Save metadata
            metadata = {
                'sequence_id': seq_info['sequence_id'],
                'episode': seq_info['episode'],
                'scene_id': seq_info['frames'][0]['scene_id'],
                'num_frames': len(seq_info['frames']),
                'fps': seq_info['fps'],
                'timestamps': [f['timestamp'] for f in seq_info['frames']]
            }

            with open(seq_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"\n✅ Background sequences prepared!")
        print(f"  Total: {len(all_sequences)} sequences")
        print(f"  Output: {output_bg_dir}")

        return all_sequences

    def prepare_all_datasets(self):
        """Prepare all video generation datasets"""
        print(f"\n{'='*80}")
        print(f"Preparing Video Generation Datasets")
        print(f"{'='*80}\n")

        # Prepare character sequences
        char_sequences = self.prepare_character_sequences()

        # Prepare background sequences
        bg_sequences = self.prepare_background_sequences()

        # Save overall summary
        overall_summary = {
            'timestamp': Path(__file__).stat().st_mtime,
            'input_dir': str(self.layered_frames_dir),
            'output_dir': str(self.output_dir),
            'config': {
                'sequence_length': self.sequence_length,
                'min_fps': self.min_fps,
                'max_gap_seconds': self.max_gap_seconds,
                'use_hard_links': self.use_hard_links
            },
            'stats': {
                'character_sequences': len(char_sequences),
                'background_sequences': len(bg_sequences)
            }
        }

        with open(self.output_dir / "dataset_summary.json", 'w') as f:
            json.dump(overall_summary, f, indent=2)

        print(f"\n{'='*80}")
        print(f"✅ All Datasets Prepared!")
        print(f"{'='*80}")
        print(f"Character Sequences: {len(char_sequences)}")
        print(f"Background Sequences: {len(bg_sequences)}")
        print(f"Output Directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare video generation datasets from layered frames"
    )
    parser.add_argument(
        "layered_frames_dir",
        type=Path,
        help="Directory with layered frames"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="Number of frames per sequence (default: 16)"
    )
    parser.add_argument(
        "--min-fps",
        type=float,
        default=4.0,
        help="Minimum FPS for sequences (default: 4.0)"
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=1.0,
        help="Maximum time gap between frames in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of using hard links"
    )

    args = parser.parse_args()

    preparer = VideoDatasetPreparer(
        layered_frames_dir=args.layered_frames_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        min_fps=args.min_fps,
        max_gap_seconds=args.max_gap,
        use_hard_links=not args.copy
    )

    preparer.prepare_all_datasets()


if __name__ == "__main__":
    main()
