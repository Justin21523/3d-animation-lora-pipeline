#!/usr/bin/env python3
"""
Speaker Diarization

Purpose: Identify who speaks when in audio tracks
Methods: PyAnnote-audio, speaker embeddings, clustering
Use Cases: Character voice identification, multi-character scene analysis

Usage:
    python speaker_diarization.py \
        --audio-path /path/to/audio.wav \
        --output-dir /path/to/diarization \
        --min-speakers 2 \
        --max-speakers 5 \
        --project luca
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization"""
    min_speakers: Optional[int] = None  # Minimum number of speakers
    max_speakers: Optional[int] = None  # Maximum number of speakers
    num_speakers: Optional[int] = None  # Exact number (if known)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_auth_token: Optional[str] = None  # HuggingFace token for pyannote
    min_segment_duration: float = 0.5  # Minimum segment duration in seconds
    merge_threshold: float = 0.3  # Merge segments within this threshold
    save_visualization: bool = True
    save_segments_audio: bool = False


class SpeakerDiarizer:
    """Speaker diarization using PyAnnote"""

    def __init__(self, config: DiarizationConfig):
        """
        Initialize diarizer

        Args:
            config: Diarization configuration
        """
        self.config = config
        self.pipeline = None

        # Try to load PyAnnote pipeline
        try:
            from pyannote.audio import Pipeline

            print(f"\n🔧 Loading PyAnnote diarization pipeline...")

            if config.use_auth_token:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",
                    use_auth_token=config.use_auth_token
                ).to(torch.device(config.device))
            else:
                print("⚠️ No HuggingFace token provided. Will use fallback method.")
                self.pipeline = None

            if self.pipeline:
                print(f"✅ Pipeline loaded on {config.device}")

        except ImportError:
            print("⚠️ PyAnnote not installed. Install with: pip install pyannote.audio")
            self.pipeline = None
        except Exception as e:
            print(f"⚠️ Could not load PyAnnote pipeline: {e}")
            self.pipeline = None

    def diarize_pyannote(
        self,
        audio_path: Path
    ) -> List[Dict]:
        """
        Perform diarization using PyAnnote

        Args:
            audio_path: Path to audio file

        Returns:
            List of speaker segments
        """
        if not self.pipeline:
            return []

        print(f"\n🎤 Running speaker diarization...")

        # Run diarization
        diarization = self.pipeline(
            str(audio_path),
            num_speakers=self.config.num_speakers,
            min_speakers=self.config.min_speakers,
            max_speakers=self.config.max_speakers,
        )

        # Extract segments
        segments = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = {
                "start": turn.start,
                "end": turn.end,
                "duration": turn.end - turn.start,
                "speaker": speaker,
            }

            # Filter by minimum duration
            if segment["duration"] >= self.config.min_segment_duration:
                segments.append(segment)

        print(f"   Found {len(segments)} speaker segments")

        return segments

    def diarize_fallback(
        self,
        audio_path: Path
    ) -> List[Dict]:
        """
        Fallback diarization using simple VAD + clustering

        Args:
            audio_path: Path to audio file

        Returns:
            List of speaker segments
        """
        print(f"\n🎤 Running fallback diarization (VAD + clustering)...")

        import librosa
        from scipy.signal import medfilt

        # Load audio
        y, sr = librosa.load(str(audio_path), sr=16000)

        # Voice Activity Detection using energy
        frame_length = 2048
        hop_length = 512

        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        energy_threshold = np.percentile(energy, 30)

        # Create voice activity mask
        voice_activity = energy > energy_threshold

        # Smooth with median filter
        voice_activity = medfilt(voice_activity, kernel_size=5).astype(bool)

        # Convert to time segments
        segments = []
        in_segment = False
        start_frame = 0

        for i, is_voice in enumerate(voice_activity):
            if is_voice and not in_segment:
                start_frame = i
                in_segment = True
            elif not is_voice and in_segment:
                start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
                end_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
                duration = end_time - start_time

                if duration >= self.config.min_segment_duration:
                    segments.append({
                        "start": float(start_time),
                        "end": float(end_time),
                        "duration": float(duration),
                        "speaker": "SPEAKER_00",  # Fallback: single speaker
                    })

                in_segment = False

        print(f"   Found {len(segments)} voice segments (single speaker assumed)")

        return segments

    def merge_close_segments(
        self,
        segments: List[Dict]
    ) -> List[Dict]:
        """
        Merge segments that are close together

        Args:
            segments: List of segments

        Returns:
            Merged segments
        """
        if not segments:
            return []

        # Sort by start time
        segments = sorted(segments, key=lambda x: x["start"])

        merged = [segments[0]]

        for seg in segments[1:]:
            last = merged[-1]

            # Same speaker and close in time
            if (seg["speaker"] == last["speaker"] and
                seg["start"] - last["end"] <= self.config.merge_threshold):

                # Merge
                last["end"] = seg["end"]
                last["duration"] = last["end"] - last["start"]
            else:
                merged.append(seg)

        return merged

    def get_speaker_statistics(
        self,
        segments: List[Dict]
    ) -> Dict:
        """
        Compute speaker statistics

        Args:
            segments: List of segments

        Returns:
            Statistics dictionary
        """
        speakers = {}

        for seg in segments:
            speaker = seg["speaker"]

            if speaker not in speakers:
                speakers[speaker] = {
                    "total_duration": 0.0,
                    "num_segments": 0,
                    "segments": [],
                }

            speakers[speaker]["total_duration"] += seg["duration"]
            speakers[speaker]["num_segments"] += 1
            speakers[speaker]["segments"].append(seg)

        # Sort speakers by total duration
        sorted_speakers = sorted(
            speakers.items(),
            key=lambda x: x[1]["total_duration"],
            reverse=True
        )

        return {
            "num_speakers": len(speakers),
            "speakers": dict(sorted_speakers),
            "total_speech_duration": sum(s["total_duration"] for s in speakers.values()),
        }

    def save_visualization(
        self,
        segments: List[Dict],
        output_path: Path,
        audio_duration: float
    ):
        """
        Save timeline visualization

        Args:
            segments: Speaker segments
            output_path: Output path for visualization
            audio_duration: Total audio duration
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, ax = plt.subplots(figsize=(15, 4))

            # Get unique speakers
            speakers = sorted(set(seg["speaker"] for seg in segments))
            colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
            speaker_colors = {s: colors[i] for i, s in enumerate(speakers)}

            # Plot segments
            for seg in segments:
                ax.barh(
                    0,
                    seg["duration"],
                    left=seg["start"],
                    height=0.8,
                    color=speaker_colors[seg["speaker"]],
                    edgecolor='black',
                    linewidth=0.5
                )

            # Formatting
            ax.set_xlim(0, audio_duration)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel("Time (seconds)", fontsize=12)
            ax.set_yticks([])
            ax.set_title("Speaker Diarization Timeline", fontsize=14, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)

            # Legend
            patches = [mpatches.Patch(color=speaker_colors[s], label=s) for s in speakers]
            ax.legend(handles=patches, loc='upper right')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   Visualization saved: {output_path}")

        except ImportError:
            print("   ⚠️ Matplotlib not available, skipping visualization")

    def extract_segment_audio(
        self,
        audio_path: Path,
        segment: Dict,
        output_path: Path
    ):
        """
        Extract audio segment to file

        Args:
            audio_path: Source audio path
            segment: Segment dict with start/end times
            output_path: Output path
        """
        import librosa
        import soundfile as sf

        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None)

        # Extract segment
        start_sample = int(segment["start"] * sr)
        end_sample = int(segment["end"] * sr)
        segment_audio = y[start_sample:end_sample]

        # Save
        sf.write(str(output_path), segment_audio, sr)

    def diarize(
        self,
        audio_path: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main diarization pipeline

        Args:
            audio_path: Path to audio file
            output_dir: Output directory

        Returns:
            Diarization results
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Speaker Diarization")
        print(f"   Audio: {audio_path}")
        print(f"   Output: {output_dir}")

        # Get audio duration
        import librosa
        duration = librosa.get_duration(path=str(audio_path))
        print(f"   Duration: {duration:.1f}s")

        # Run diarization
        if self.pipeline:
            segments = self.diarize_pyannote(audio_path)
        else:
            segments = self.diarize_fallback(audio_path)

        # Merge close segments
        segments = self.merge_close_segments(segments)
        print(f"   After merging: {len(segments)} segments")

        # Get statistics
        stats = self.get_speaker_statistics(segments)

        print(f"\n📊 Speaker Statistics:")
        print(f"   Number of speakers: {stats['num_speakers']}")
        for speaker, info in stats["speakers"].items():
            print(f"   {speaker}: {info['total_duration']:.1f}s ({info['num_segments']} segments)")

        # Save results
        results = {
            "audio_path": str(audio_path),
            "audio_duration": duration,
            "config": {
                "min_speakers": self.config.min_speakers,
                "max_speakers": self.config.max_speakers,
                "num_speakers": self.config.num_speakers,
                "min_segment_duration": self.config.min_segment_duration,
            },
            "segments": segments,
            "statistics": stats,
            "timestamp": datetime.now().isoformat(),
        }

        # Save to JSON
        results_path = output_dir / "diarization.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved: {results_path}")

        # Save visualization
        if self.config.save_visualization:
            viz_path = output_dir / "diarization_timeline.png"
            self.save_visualization(segments, viz_path, duration)

        # Save segment audio files
        if self.config.save_segments_audio:
            segments_dir = output_dir / "segments"
            segments_dir.mkdir(exist_ok=True)

            print(f"\n💾 Extracting segment audio files...")

            for i, seg in enumerate(tqdm(segments, desc="Extracting")):
                seg_path = segments_dir / f"segment_{i:04d}_{seg['speaker']}_{seg['start']:.2f}s.wav"
                self.extract_segment_audio(audio_path, seg, seg_path)

            print(f"   Saved to: {segments_dir}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Speaker Diarization (Film-Agnostic)"
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for diarization results"
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Exact number of speakers (if known)"
    )
    parser.add_argument(
        "--min-segment-duration",
        type=float,
        default=0.5,
        help="Minimum segment duration in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.3,
        help="Merge segments within this threshold (default: 0.3s)"
    )
    parser.add_argument(
        "--save-segments",
        action="store_true",
        help="Save individual segment audio files"
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        help="HuggingFace auth token for PyAnnote models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda if available)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (for logging)"
    )

    args = parser.parse_args()

    # Create config
    config = DiarizationConfig(
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        num_speakers=args.num_speakers,
        device=args.device,
        use_auth_token=args.auth_token,
        min_segment_duration=args.min_segment_duration,
        merge_threshold=args.merge_threshold,
        save_visualization=True,
        save_segments_audio=args.save_segments,
    )

    # Run diarization
    diarizer = SpeakerDiarizer(config)
    results = diarizer.diarize(
        audio_path=Path(args.audio_path),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
