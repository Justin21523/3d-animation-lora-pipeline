#!/usr/bin/env python3
"""
Character Voice Mapper

Purpose: Map speaker segments to character identities
Method: Visual-audio synchronization, face tracking, temporal alignment
Use Cases: Associate voices with characters for multi-modal datasets

Usage:
    python character_voice_mapper.py \
        --diarization-file /path/to/diarization.json \
        --frames-dir /path/to/frames \
        --face-clusters-dir /path/to/clusters \
        --output-dir /path/to/mapping \
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


@dataclass
class VoiceMappingConfig:
    """Configuration for character voice mapping"""
    fps: float = 24.0  # Video frame rate
    face_detection_threshold: float = 0.7  # Confidence threshold
    temporal_window: float = 2.0  # Seconds to look around segment
    min_overlap_ratio: float = 0.3  # Minimum overlap to consider match
    use_lip_sync: bool = False  # Use lip sync detection (advanced)
    confidence_threshold: float = 0.5  # Minimum confidence for mapping


class CharacterVoiceMapper:
    """Map speaker segments to character identities"""

    def __init__(self, config: VoiceMappingConfig):
        """
        Initialize mapper

        Args:
            config: Mapping configuration
        """
        self.config = config

    def load_diarization(self, diarization_file: Path) -> Dict:
        """
        Load speaker diarization results

        Args:
            diarization_file: Path to diarization JSON

        Returns:
            Diarization data
        """
        with open(diarization_file) as f:
            return json.load(f)

    def load_face_clusters(self, clusters_dir: Path) -> Dict[str, List[str]]:
        """
        Load face identity clusters

        Args:
            clusters_dir: Directory with clustered faces

        Returns:
            Dictionary mapping character to frame files
        """
        clusters_dir = Path(clusters_dir)
        clusters = {}

        # Find character directories
        for char_dir in clusters_dir.iterdir():
            if char_dir.is_dir() and not char_dir.name.startswith('.'):
                character_name = char_dir.name

                # Find frames for this character
                frames = []
                for img in char_dir.glob("*.png"):
                    frames.append(img.name)

                if frames:
                    clusters[character_name] = frames

        print(f"   Loaded {len(clusters)} character clusters")
        for char, frames in clusters.items():
            print(f"     {char}: {len(frames)} frames")

        return clusters

    def parse_frame_number(self, frame_name: str) -> Optional[int]:
        """
        Parse frame number from filename

        Args:
            frame_name: Frame filename

        Returns:
            Frame number or None
        """
        import re
        numbers = re.findall(r'\d+', frame_name)
        if numbers:
            return int(numbers[0])
        return None

    def time_to_frame(self, time_seconds: float) -> int:
        """
        Convert time to frame number

        Args:
            time_seconds: Time in seconds

        Returns:
            Frame number
        """
        return int(time_seconds * self.config.fps)

    def frame_to_time(self, frame_num: int) -> float:
        """
        Convert frame number to time

        Args:
            frame_num: Frame number

        Returns:
            Time in seconds
        """
        return frame_num / self.config.fps

    def find_frames_in_segment(
        self,
        segment: Dict,
        character_frames: List[str]
    ) -> List[str]:
        """
        Find character frames within speaker segment

        Args:
            segment: Speaker segment dict
            character_frames: List of frame filenames for character

        Returns:
            List of matching frame names
        """
        start_frame = self.time_to_frame(segment["start"] - self.config.temporal_window)
        end_frame = self.time_to_frame(segment["end"] + self.config.temporal_window)

        matching = []

        for frame_name in character_frames:
            frame_num = self.parse_frame_number(frame_name)

            if frame_num is not None:
                if start_frame <= frame_num <= end_frame:
                    matching.append(frame_name)

        return matching

    def compute_character_overlap(
        self,
        segments: List[Dict],
        character_frames: List[str]
    ) -> float:
        """
        Compute total overlap between segments and character appearances

        Args:
            segments: Speaker segments
            character_frames: Character frame filenames

        Returns:
            Total overlap duration in seconds
        """
        total_overlap = 0.0

        for segment in segments:
            matching_frames = self.find_frames_in_segment(segment, character_frames)

            if matching_frames:
                # Estimate overlap duration based on number of matching frames
                overlap_duration = len(matching_frames) / self.config.fps
                total_overlap += min(overlap_duration, segment["duration"])

        return total_overlap

    def map_speakers_to_characters(
        self,
        diarization: Dict,
        face_clusters: Dict[str, List[str]]
    ) -> Dict:
        """
        Map speaker IDs to character names

        Args:
            diarization: Diarization results
            face_clusters: Face identity clusters

        Returns:
            Mapping dictionary
        """
        print(f"\n🔗 Mapping speakers to characters...")

        segments = diarization["segments"]
        speakers = diarization["statistics"]["speakers"]

        mappings = {}

        # For each speaker, find best matching character
        for speaker_id, speaker_info in speakers.items():
            speaker_segments = speaker_info["segments"]

            print(f"\n   Analyzing {speaker_id} ({len(speaker_segments)} segments, {speaker_info['total_duration']:.1f}s)...")

            character_scores = {}

            # Compute overlap with each character
            for character, char_frames in face_clusters.items():
                overlap = self.compute_character_overlap(speaker_segments, char_frames)

                if overlap > 0:
                    # Compute confidence score
                    # Higher overlap and more frames = higher confidence
                    confidence = min(1.0, overlap / speaker_info["total_duration"])
                    character_scores[character] = {
                        "overlap_duration": overlap,
                        "confidence": confidence,
                        "num_matching_segments": sum(
                            1 for seg in speaker_segments
                            if self.find_frames_in_segment(seg, char_frames)
                        )
                    }

            # Select best match
            if character_scores:
                best_character = max(
                    character_scores.items(),
                    key=lambda x: x[1]["confidence"]
                )

                character_name, score = best_character

                if score["confidence"] >= self.config.confidence_threshold:
                    mappings[speaker_id] = {
                        "character": character_name,
                        "confidence": score["confidence"],
                        "overlap_duration": score["overlap_duration"],
                        "num_matching_segments": score["num_matching_segments"],
                        "alternative_matches": {
                            char: s["confidence"]
                            for char, s in character_scores.items()
                            if char != character_name and s["confidence"] > 0.1
                        }
                    }

                    print(f"     ✅ Mapped to: {character_name} (confidence: {score['confidence']:.2f})")

                    if mappings[speaker_id]["alternative_matches"]:
                        print(f"     📊 Alternatives: {mappings[speaker_id]['alternative_matches']}")
                else:
                    mappings[speaker_id] = {
                        "character": "UNKNOWN",
                        "confidence": 0.0,
                        "reason": f"Low confidence ({score['confidence']:.2f} < {self.config.confidence_threshold})",
                    }
                    print(f"     ⚠️ No confident match (best: {character_name} at {score['confidence']:.2f})")
            else:
                mappings[speaker_id] = {
                    "character": "UNKNOWN",
                    "confidence": 0.0,
                    "reason": "No temporal overlap with any character",
                }
                print(f"     ⚠️ No overlap with any character")

        return mappings

    def generate_character_timeline(
        self,
        segments: List[Dict],
        mappings: Dict
    ) -> List[Dict]:
        """
        Generate timeline with character labels

        Args:
            segments: Speaker segments
            mappings: Speaker to character mappings

        Returns:
            Timeline with character annotations
        """
        timeline = []

        for seg in segments:
            speaker = seg["speaker"]
            mapping = mappings.get(speaker, {})

            timeline.append({
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "speaker_id": speaker,
                "character": mapping.get("character", "UNKNOWN"),
                "confidence": mapping.get("confidence", 0.0),
            })

        return timeline

    def save_visualization(
        self,
        timeline: List[Dict],
        output_path: Path,
        audio_duration: float
    ):
        """
        Save timeline visualization with character colors

        Args:
            timeline: Character timeline
            output_path: Output path
            audio_duration: Total audio duration
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, ax = plt.subplots(figsize=(15, 6))

            # Get unique characters
            characters = sorted(set(seg["character"] for seg in timeline))
            colors = plt.cm.Set3(np.linspace(0, 1, len(characters)))
            char_colors = {c: colors[i] for i, c in enumerate(characters)}

            # Plot segments
            for seg in timeline:
                color = char_colors[seg["character"]]
                alpha = 0.3 + 0.7 * seg["confidence"]  # Transparency based on confidence

                ax.barh(
                    0,
                    seg["duration"],
                    left=seg["start"],
                    height=0.8,
                    color=color,
                    alpha=alpha,
                    edgecolor='black',
                    linewidth=0.5
                )

                # Add text label
                if seg["duration"] > 2:  # Only if segment is long enough
                    ax.text(
                        seg["start"] + seg["duration"]/2,
                        0,
                        seg["character"],
                        ha='center',
                        va='center',
                        fontsize=8,
                        fontweight='bold'
                    )

            # Formatting
            ax.set_xlim(0, audio_duration)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel("Time (seconds)", fontsize=12)
            ax.set_yticks([])
            ax.set_title("Character Voice Timeline", fontsize=14, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)

            # Legend
            patches = [
                mpatches.Patch(color=char_colors[c], label=c)
                for c in characters
            ]
            ax.legend(handles=patches, loc='upper right')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   Timeline visualization saved: {output_path}")

        except ImportError:
            print("   ⚠️ Matplotlib not available, skipping visualization")

    def map_voices(
        self,
        diarization_file: Path,
        face_clusters_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main voice mapping pipeline

        Args:
            diarization_file: Path to diarization JSON
            face_clusters_dir: Directory with face clusters
            output_dir: Output directory

        Returns:
            Mapping results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Character Voice Mapping")
        print(f"   Diarization: {diarization_file}")
        print(f"   Face clusters: {face_clusters_dir}")
        print(f"   Output: {output_dir}")

        # Load data
        print(f"\n📂 Loading data...")
        diarization = self.load_diarization(diarization_file)
        face_clusters = self.load_face_clusters(face_clusters_dir)

        # Map speakers to characters
        mappings = self.map_speakers_to_characters(diarization, face_clusters)

        # Generate character timeline
        timeline = self.generate_character_timeline(
            diarization["segments"],
            mappings
        )

        # Compute statistics
        character_stats = {}
        for seg in timeline:
            char = seg["character"]
            if char not in character_stats:
                character_stats[char] = {
                    "total_duration": 0.0,
                    "num_segments": 0,
                    "avg_confidence": 0.0,
                }

            character_stats[char]["total_duration"] += seg["duration"]
            character_stats[char]["num_segments"] += 1
            character_stats[char]["avg_confidence"] += seg["confidence"]

        # Compute averages
        for char in character_stats:
            n = character_stats[char]["num_segments"]
            if n > 0:
                character_stats[char]["avg_confidence"] /= n

        print(f"\n📊 Character Voice Statistics:")
        for char, stats in sorted(character_stats.items(), key=lambda x: x[1]["total_duration"], reverse=True):
            print(f"   {char}: {stats['total_duration']:.1f}s, {stats['num_segments']} segments, confidence: {stats['avg_confidence']:.2f}")

        # Save results
        results = {
            "diarization_file": str(diarization_file),
            "face_clusters_dir": str(face_clusters_dir),
            "config": {
                "fps": self.config.fps,
                "temporal_window": self.config.temporal_window,
                "confidence_threshold": self.config.confidence_threshold,
            },
            "speaker_to_character": mappings,
            "timeline": timeline,
            "character_statistics": character_stats,
            "timestamp": datetime.now().isoformat(),
        }

        # Save JSON
        results_path = output_dir / "voice_mapping.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved: {results_path}")

        # Save visualization
        viz_path = output_dir / "character_timeline.png"
        self.save_visualization(
            timeline,
            viz_path,
            diarization["audio_duration"]
        )

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Map speaker voices to characters (Film-Agnostic)"
    )
    parser.add_argument(
        "--diarization-file",
        type=str,
        required=True,
        help="Path to diarization JSON file"
    )
    parser.add_argument(
        "--face-clusters-dir",
        type=str,
        required=True,
        help="Directory with face identity clusters"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for mapping results"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Video frame rate (default: 24.0)"
    )
    parser.add_argument(
        "--temporal-window",
        type=float,
        default=2.0,
        help="Temporal window around segments in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence for mapping (default: 0.5)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (for logging)"
    )

    args = parser.parse_args()

    # Create config
    config = VoiceMappingConfig(
        fps=args.fps,
        temporal_window=args.temporal_window,
        confidence_threshold=args.confidence_threshold,
        use_lip_sync=False,
    )

    # Run mapping
    mapper = CharacterVoiceMapper(config)
    results = mapper.map_voices(
        diarization_file=Path(args.diarization_file),
        face_clusters_dir=Path(args.face_clusters_dir),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
