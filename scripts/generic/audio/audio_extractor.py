#!/usr/bin/env python3
"""
Audio Extraction Tool
=====================
Extracts audio tracks from video files for voice cloning and analysis.

Features:
- Batch extraction from multiple episodes
- Multiple audio formats (WAV, FLAC, MP3)
- Configurable sample rate and channels
- Episode pattern matching (flexible for different series)
- Parallel processing with multiple workers

Usage:
    python audio_extractor.py /path/to/videos --format wav --sample-rate 44100
    python audio_extractor.py /path/to/videos --episode-pattern "E(\d+)" --workers 8
"""

import subprocess
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import argparse
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json


class AudioFormat(Enum):
    """Supported audio output formats"""
    WAV = "wav"      # Uncompressed, best for processing
    FLAC = "flac"    # Lossless compression
    MP3 = "mp3"      # Lossy compression, smaller size


@dataclass
class AudioConfig:
    """Configuration for audio extraction"""
    format: AudioFormat = AudioFormat.WAV
    sample_rate: int = 44100      # 44.1kHz standard, 16kHz for speech
    channels: int = 1             # 1=mono, 2=stereo (mono recommended for voice)
    bitrate: Optional[str] = None # For MP3: "192k", "320k"
    normalize: bool = True        # Normalize audio levels
    remove_silence: bool = False  # Trim silence from beginning/end

    def to_ffmpeg_args(self) -> List[str]:
        """Convert config to FFmpeg arguments"""
        args = [
            "-vn",  # No video
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
        ]

        # Format-specific settings
        if self.format == AudioFormat.WAV:
            args.extend(["-acodec", "pcm_s16le"])
        elif self.format == AudioFormat.FLAC:
            args.extend(["-acodec", "flac"])
        elif self.format == AudioFormat.MP3:
            args.extend(["-acodec", "libmp3lame"])
            if self.bitrate:
                args.extend(["-b:a", self.bitrate])

        # Audio normalization
        if self.normalize:
            args.extend(["-af", "loudnorm"])

        return args


class AudioExtractor:
    """Extract audio from video files"""

    def __init__(self, config: AudioConfig, episode_pattern: str = r"(\d+)"):
        """
        Initialize audio extractor

        Args:
            config: Audio extraction configuration
            episode_pattern: Regex pattern to extract episode number
                            Default: r"(\d+)" - matches any number
                            Examples: r"E(\d+)", r"S\d+\.(\d+)"
        """
        self.config = config
        self.episode_pattern = re.compile(episode_pattern)

    def extract_episode_number(self, video_path: Path) -> Optional[int]:
        """
        Extract episode number from video filename

        Args:
            video_path: Path to video file

        Returns:
            Episode number or None if not found
        """
        match = self.episode_pattern.search(video_path.stem)
        if match:
            return int(match.group(1))
        return None

    def extract_audio(self, video_path: Path, output_path: Path) -> bool:
        """
        Extract audio from a single video file

        Args:
            video_path: Path to input video
            output_path: Path to output audio file

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Build FFmpeg command
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-y",  # Overwrite output
                *self.config.to_ffmpeg_args(),
                str(output_path)
            ]

            # Run extraction with suppressed output
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode == 0 and output_path.exists():
                return True
            else:
                print(f"Error extracting {video_path.name}: {result.stderr}")
                return False

        except Exception as e:
            print(f"Exception extracting {video_path.name}: {e}")
            return False

    def get_audio_info(self, audio_path: Path) -> dict:
        """
        Get information about extracted audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with audio metadata
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(audio_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)

            # Extract relevant information
            audio_stream = next(
                (s for s in data.get("streams", []) if s["codec_type"] == "audio"),
                None
            )

            if audio_stream:
                return {
                    "duration": float(data["format"].get("duration", 0)),
                    "sample_rate": int(audio_stream.get("sample_rate", 0)),
                    "channels": int(audio_stream.get("channels", 0)),
                    "codec": audio_stream.get("codec_name", "unknown"),
                    "bitrate": int(data["format"].get("bit_rate", 0)) // 1000,  # kbps
                    "size_mb": int(data["format"].get("size", 0)) / (1024 * 1024)
                }

        except Exception as e:
            print(f"Error getting audio info: {e}")

        return {}


def extract_single_video(args) -> tuple:
    """
    Worker function for parallel processing

    Args:
        args: Tuple of (video_path, output_dir, config, episode_pattern)

    Returns:
        Tuple of (video_name, success, episode_number)
    """
    video_path, output_dir, config, episode_pattern = args

    extractor = AudioExtractor(config, episode_pattern)
    episode_num = extractor.extract_episode_number(video_path)

    if episode_num is None:
        print(f"Warning: Could not extract episode number from {video_path.name}")
        output_name = f"{video_path.stem}.{config.format.value}"
    else:
        output_name = f"episode_{episode_num:03d}.{config.format.value}"

    output_path = output_dir / output_name
    success = extractor.extract_audio(video_path, output_path)

    return video_path.name, success, episode_num, output_path


def batch_extract_audio(
    video_dir: Path,
    output_dir: Path,
    config: AudioConfig,
    episode_pattern: str = r"(\d+)",
    workers: int = 4,
    video_extensions: List[str] = [".mp4", ".mkv", ".avi", ".mov"]
) -> dict:
    """
    Extract audio from all videos in a directory

    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted audio
        config: Audio extraction configuration
        episode_pattern: Regex pattern for episode number extraction
        workers: Number of parallel workers
        video_extensions: List of video file extensions to process

    Returns:
        Dictionary with extraction statistics
    """
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))

    video_files = sorted(video_files)

    if not video_files:
        print(f"No video files found in {video_dir}")
        return {}

    print(f"Found {len(video_files)} video files")
    print(f"Output directory: {output_dir}")
    print(f"Audio format: {config.format.value} @ {config.sample_rate}Hz")
    print(f"Workers: {workers}")
    print()

    # Prepare arguments for parallel processing
    args_list = [
        (video_path, output_dir, config, episode_pattern)
        for video_path in video_files
    ]

    # Process videos in parallel
    results = []
    with Pool(workers) as pool:
        with tqdm(total=len(video_files), desc="Extracting audio") as pbar:
            for result in pool.imap_unordered(extract_single_video, args_list):
                results.append(result)
                pbar.update(1)

    # Collect statistics
    successful = sum(1 for _, success, _, _ in results if success)
    failed = len(results) - successful

    stats = {
        "total_videos": len(video_files),
        "successful": successful,
        "failed": failed,
        "output_dir": str(output_dir),
        "format": config.format.value,
        "sample_rate": config.sample_rate,
        "channels": config.channels
    }

    # Get info from first extracted file for verification
    if successful > 0:
        first_success = next(
            (output_path for _, success, _, output_path in results if success),
            None
        )
        if first_success:
            extractor = AudioExtractor(config)
            audio_info = extractor.get_audio_info(first_success)
            stats["sample_audio_info"] = audio_info

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio from video files for voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction to WAV
  python audio_extractor.py /path/to/videos

  # Extract to FLAC with custom sample rate
  python audio_extractor.py /path/to/videos --format flac --sample-rate 48000

  # For Yokai Watch episodes (S1.01 format)
  python audio_extractor.py /path/to/videos --episode-pattern "S\\d+\\.(\\d+)"

  # High-quality mono extraction for voice cloning
  python audio_extractor.py /path/to/videos --format wav --sample-rate 44100 --channels 1

  # Use 16 workers for faster processing
  python audio_extractor.py /path/to/videos --workers 16
        """
    )

    # Required arguments
    parser.add_argument(
        "video_dir",
        type=Path,
        help="Directory containing video files"
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for audio files (default: video_dir/../audio)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["wav", "flac", "mp3"],
        default="wav",
        help="Audio output format (default: wav)"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Audio sample rate in Hz (default: 44100)"
    )

    parser.add_argument(
        "--channels",
        type=int,
        choices=[1, 2],
        default=1,
        help="Number of audio channels: 1=mono, 2=stereo (default: 1)"
    )

    parser.add_argument(
        "--bitrate",
        type=str,
        help="Bitrate for MP3 format (e.g., '192k', '320k')"
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable audio normalization"
    )

    parser.add_argument(
        "--episode-pattern",
        type=str,
        default=r"(\d+)",
        help="Regex pattern to extract episode number (default: '(\\d+)')"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count() // 2,
        help=f"Number of parallel workers (default: {cpu_count() // 2})"
    )

    parser.add_argument(
        "--video-extensions",
        type=str,
        nargs="+",
        default=[".mp4", ".mkv", ".avi", ".mov"],
        help="Video file extensions to process (default: .mp4 .mkv .avi .mov)"
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.video_dir.exists():
        print(f"Error: Video directory not found: {args.video_dir}")
        return 1

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.video_dir.parent / "audio"

    # Create audio configuration
    config = AudioConfig(
        format=AudioFormat(args.format),
        sample_rate=args.sample_rate,
        channels=args.channels,
        bitrate=args.bitrate,
        normalize=not args.no_normalize
    )

    # Run extraction
    print("=" * 60)
    print("Audio Extraction Tool")
    print("=" * 60)
    print()

    stats = batch_extract_audio(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        config=config,
        episode_pattern=args.episode_pattern,
        workers=args.workers,
        video_extensions=args.video_extensions
    )

    # Print results
    print()
    print("=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print()
    print(f"Total videos: {stats['total_videos']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Output directory: {stats['output_dir']}")
    print()

    if "sample_audio_info" in stats:
        info = stats["sample_audio_info"]
        print("Sample audio info:")
        print(f"  Duration: {info.get('duration', 0):.1f} seconds")
        print(f"  Sample rate: {info.get('sample_rate', 0)} Hz")
        print(f"  Channels: {info.get('channels', 0)}")
        print(f"  Codec: {info.get('codec', 'unknown')}")
        print(f"  Size: {info.get('size_mb', 0):.2f} MB")
        print()

    print("Next steps:")
    print(f"  1. Check output: ls {stats['output_dir']}")
    print(f"  2. Verify audio: ffprobe {stats['output_dir']}/episode_001.{stats['format']}")
    print("  3. Run voice separation: python voice_separator.py <audio_dir>")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
