#!/usr/bin/env python3
"""
Temporal Coherence Analyzer

Purpose: Analyze temporal consistency across frames and scenes
Features: Character tracking, appearance consistency, transition detection
Use Cases: Quality control, dataset validation, temporal anomaly detection

Usage:
    python temporal_coherence.py \
        --frames-dir /path/to/frames \
        --clusters-dir /path/to/clusters \
        --output-dir /path/to/analysis \
        --project luca
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from collections import defaultdict


@dataclass
class TemporalCoherenceConfig:
    """Configuration for temporal coherence analysis"""
    consistency_threshold: float = 0.8  # SSIM threshold for consistency
    track_characters: bool = True  # Track character appearances
    detect_transitions: bool = True  # Detect scene transitions
    check_color_consistency: bool = True  # Check color consistency
    window_size: int = 5  # Frames to look around for coherence


class TemporalCoherenceAnalyzer:
    """Analyze temporal consistency"""

    def __init__(self, config: TemporalCoherenceConfig):
        """
        Initialize analyzer

        Args:
            config: Analysis configuration
        """
        self.config = config

    def parse_frame_number(self, frame_name: str) -> Optional[int]:
        """Parse frame number from filename"""
        import re
        numbers = re.findall(r'\d+', frame_name)
        if numbers:
            return int(numbers[0])
        return None

    def load_character_clusters(self, clusters_dir: Path) -> Dict[str, List[int]]:
        """
        Load character identity clusters

        Args:
            clusters_dir: Directory with clusters

        Returns:
            Dictionary mapping character to frame numbers
        """
        clusters_dir = Path(clusters_dir)
        clusters = {}

        for char_dir in clusters_dir.iterdir():
            if char_dir.is_dir() and not char_dir.name.startswith('.'):
                character = char_dir.name
                frame_nums = []

                for img in char_dir.glob("*.png"):
                    num = self.parse_frame_number(img.name)
                    if num is not None:
                        frame_nums.append(num)

                if frame_nums:
                    clusters[character] = sorted(frame_nums)

        return clusters

    def compute_ssim(self, img1_path: Path, img2_path: Path) -> float:
        """Compute SSIM between two images"""
        from skimage.metrics import structural_similarity as ssim

        img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return 0.0

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        score, _ = ssim(img1, img2, full=True)
        return float(score)

    def analyze_character_tracking(
        self,
        character_clusters: Dict[str, List[int]],
        total_frames: int
    ) -> Dict:
        """
        Analyze character appearance tracking

        Args:
            character_clusters: Character frame mappings
            total_frames: Total number of frames

        Returns:
            Tracking analysis
        """
        print(f"\n👤 Analyzing character tracking...")

        tracking = {}

        for character, frame_nums in character_clusters.items():
            # Find gaps in appearances
            gaps = []

            for i in range(len(frame_nums) - 1):
                gap_size = frame_nums[i+1] - frame_nums[i] - 1
                if gap_size > 0:
                    gaps.append({
                        "start": frame_nums[i],
                        "end": frame_nums[i+1],
                        "size": gap_size
                    })

            # Compute appearance density
            appearance_frames = len(frame_nums)
            density = appearance_frames / total_frames if total_frames > 0 else 0

            # Find continuous segments
            segments = []
            if frame_nums:
                segment_start = frame_nums[0]
                prev_frame = frame_nums[0]

                for frame_num in frame_nums[1:]:
                    if frame_num - prev_frame > 10:  # Gap threshold
                        segments.append({
                            "start": segment_start,
                            "end": prev_frame,
                            "length": prev_frame - segment_start + 1
                        })
                        segment_start = frame_num

                    prev_frame = frame_num

                # Last segment
                segments.append({
                    "start": segment_start,
                    "end": prev_frame,
                    "length": prev_frame - segment_start + 1
                })

            tracking[character] = {
                "total_appearances": appearance_frames,
                "density": density,
                "num_gaps": len(gaps),
                "gaps": gaps[:10],  # Top 10 gaps
                "num_segments": len(segments),
                "segments": segments,
            }

            print(f"   {character}: {appearance_frames} frames, {len(segments)} segments, density {density:.2%}")

        return tracking

    def detect_scene_transitions(
        self,
        frames: List[Path]
    ) -> List[Dict]:
        """
        Detect abrupt scene transitions

        Args:
            frames: List of frame paths

        Returns:
            List of detected transitions
        """
        print(f"\n🎬 Detecting scene transitions...")

        transitions = []

        for i in tqdm(range(1, len(frames)), desc="Analyzing transitions"):
            ssim_score = self.compute_ssim(frames[i-1], frames[i])

            # Transition detected if similarity is very low
            if ssim_score < 0.3:
                transitions.append({
                    "frame": i,
                    "prev_frame": frames[i-1].name,
                    "curr_frame": frames[i].name,
                    "ssim": ssim_score,
                    "type": "cut" if ssim_score < 0.1 else "dissolve"
                })

        print(f"   Found {len(transitions)} transitions")

        return transitions

    def check_color_consistency(
        self,
        frames: List[Path]
    ) -> Dict:
        """
        Check color consistency across frames

        Args:
            frames: List of frame paths

        Returns:
            Color consistency metrics
        """
        print(f"\n🎨 Checking color consistency...")

        # Sample frames
        sample_size = min(100, len(frames))
        sample_indices = np.linspace(0, len(frames)-1, sample_size, dtype=int)
        sample_frames = [frames[i] for i in sample_indices]

        color_stats = []

        for frame_path in tqdm(sample_frames, desc="Analyzing colors"):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            # Compute color statistics
            mean_color = cv2.mean(img)[:3]
            color_stats.append(mean_color)

        if not color_stats:
            return {}

        color_stats = np.array(color_stats)

        # Compute consistency metrics
        mean_per_channel = np.mean(color_stats, axis=0)
        std_per_channel = np.std(color_stats, axis=0)
        coeff_of_variation = std_per_channel / (mean_per_channel + 1e-8)

        consistency = {
            "mean_color": mean_per_channel.tolist(),
            "std_color": std_per_channel.tolist(),
            "coefficient_of_variation": coeff_of_variation.tolist(),
            "consistency_score": float(1.0 - np.mean(coeff_of_variation)),
        }

        print(f"   Color consistency score: {consistency['consistency_score']:.2f}")

        return consistency

    def analyze_temporal_coherence(
        self,
        frames_dir: Path,
        clusters_dir: Optional[Path],
        output_dir: Path
    ) -> Dict:
        """
        Main temporal coherence analysis

        Args:
            frames_dir: Directory with frames
            clusters_dir: Directory with character clusters (optional)
            output_dir: Output directory

        Returns:
            Analysis results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Temporal Coherence Analysis")
        print(f"   Frames: {frames_dir}")
        if clusters_dir:
            print(f"   Clusters: {clusters_dir}")
        print(f"   Output: {output_dir}")

        # Find frames
        frames = sorted(
            list(frames_dir.glob("frame_*.jpg")) +
            list(frames_dir.glob("frame_*.png"))
        )

        print(f"   Total frames: {len(frames)}")

        results = {
            "frames_dir": str(frames_dir),
            "total_frames": len(frames),
            "timestamp": datetime.now().isoformat(),
        }

        # Character tracking
        if self.config.track_characters and clusters_dir:
            clusters = self.load_character_clusters(clusters_dir)
            if clusters:
                results["character_tracking"] = self.analyze_character_tracking(
                    clusters,
                    len(frames)
                )

        # Transition detection
        if self.config.detect_transitions:
            results["transitions"] = self.detect_scene_transitions(frames)

        # Color consistency
        if self.config.check_color_consistency:
            results["color_consistency"] = self.check_color_consistency(frames)

        # Save results
        results_path = output_dir / "temporal_coherence.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved: {results_path}")

        # Save visualization
        if results.get("character_tracking"):
            self.save_character_timeline(
                results["character_tracking"],
                len(frames),
                output_dir
            )

        return results

    def save_character_timeline(
        self,
        tracking: Dict,
        total_frames: int,
        output_dir: Path
    ):
        """Save character appearance timeline"""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(15, max(6, len(tracking))))

            characters = sorted(tracking.keys())
            colors = plt.cm.Set3(np.linspace(0, 1, len(characters)))

            for i, character in enumerate(characters):
                segments = tracking[character]["segments"]

                for seg in segments:
                    ax.barh(
                        i,
                        seg["length"],
                        left=seg["start"],
                        height=0.8,
                        color=colors[i],
                        edgecolor='black',
                        linewidth=0.5
                    )

            ax.set_yticks(range(len(characters)))
            ax.set_yticklabels(characters)
            ax.set_xlabel("Frame Number", fontsize=12)
            ax.set_title("Character Appearance Timeline", fontsize=14, fontweight='bold')
            ax.set_xlim(0, total_frames)
            ax.grid(True, axis='x', alpha=0.3)

            plt.tight_layout()

            viz_path = output_dir / "character_timeline.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   Character timeline saved: {viz_path}")

        except ImportError:
            print("   ⚠️ Matplotlib not available")


def main():
    parser = argparse.ArgumentParser(
        description="Temporal Coherence Analysis (Film-Agnostic)"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory with frames"
    )
    parser.add_argument(
        "--clusters-dir",
        type=str,
        help="Directory with character clusters (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--no-character-tracking",
        action="store_true",
        help="Disable character tracking"
    )
    parser.add_argument(
        "--no-transitions",
        action="store_true",
        help="Disable transition detection"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name"
    )

    args = parser.parse_args()

    config = TemporalCoherenceConfig(
        track_characters=not args.no_character_tracking,
        detect_transitions=not args.no_transitions,
        check_color_consistency=True,
    )

    analyzer = TemporalCoherenceAnalyzer(config)
    results = analyzer.analyze_temporal_coherence(
        frames_dir=Path(args.frames_dir),
        clusters_dir=Path(args.clusters_dir) if args.clusters_dir else None,
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
