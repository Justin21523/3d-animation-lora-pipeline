#!/usr/bin/env python3
"""
Voice Separation Tool
=====================
Separates vocals from background music and sound effects using Demucs.

Features:
- Uses Demucs v4 (htdemucs) for state-of-the-art separation
- Batch processing of multiple audio files
- GPU acceleration support
- Separates into vocals, bass, drums, other tracks
- Automatic quality assessment

Usage:
    python voice_separator.py /path/to/audio --model htdemucs
    python voice_separator.py /path/to/audio --two-stems vocals --workers 4
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil


class DemucsModel(Enum):
    """Available Demucs models"""
    HTDEMUCS = "htdemucs"           # Best quality, slower
    HTDEMUCS_FT = "htdemucs_ft"     # Fine-tuned version
    HTDEMUCS_6S = "htdemucs_6s"     # 6-stem separation
    MDXNET = "mdx_extra"            # Fast, good quality


@dataclass
class SeparationConfig:
    """Configuration for voice separation"""
    model: DemucsModel = DemucsModel.HTDEMUCS
    two_stems: bool = True          # Only extract vocals (faster)
    shifts: int = 1                 # Number of random shifts (higher=better, slower)
    overlap: float = 0.25           # Overlap between splits
    split: bool = True              # Split audio for GPU memory
    segment: Optional[int] = None   # Segment length (None=auto)
    device: str = "cuda"            # cuda or cpu
    jobs: int = 0                   # CPU threads (0=auto)
    float32: bool = False           # Use float32 (more memory, slightly better)

    def to_demucs_args(self) -> List[str]:
        """Convert config to Demucs command arguments"""
        args = ["-n", self.model.value]

        if self.two_stems:
            args.extend(["--two-stems", "vocals"])

        args.extend(["--shifts", str(self.shifts)])
        args.extend(["--overlap", str(self.overlap)])

        if self.device == "cuda":
            args.append("--device")
            args.append("cuda")
        else:
            args.append("--device")
            args.append("cpu")

        if self.jobs > 0:
            args.extend(["-j", str(self.jobs)])

        if self.float32:
            args.append("--float32")

        return args


class VoiceSeparator:
    """Separate vocals from music and sound effects"""

    def __init__(self, config: SeparationConfig):
        """
        Initialize voice separator

        Args:
            config: Separation configuration
        """
        self.config = config
        self._check_demucs_installed()

    def _check_demucs_installed(self) -> bool:
        """Check if Demucs is installed"""
        try:
            result = subprocess.run(
                ["demucs", "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return result.returncode == 0
        except FileNotFoundError:
            print("Error: Demucs not found!")
            print("Install with: pip install demucs")
            return False

    def separate_audio(
        self,
        audio_path: Path,
        output_dir: Path
    ) -> Tuple[bool, Optional[Path]]:
        """
        Separate vocals from a single audio file

        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save separated tracks

        Returns:
            Tuple of (success, vocals_path)
        """
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Build Demucs command
            cmd = [
                "demucs",
                *self.config.to_demucs_args(),
                "-o", str(output_dir),
                str(audio_path)
            ]

            # Run separation
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print(f"Error separating {audio_path.name}: {result.stderr}")
                return False, None

            # Find the vocals file
            # Demucs creates: output_dir/model_name/audio_stem/vocals.wav
            model_dir = output_dir / self.config.model.value / audio_path.stem
            vocals_path = model_dir / "vocals.wav"

            if vocals_path.exists():
                return True, vocals_path
            else:
                print(f"Warning: Vocals file not found at {vocals_path}")
                return False, None

        except Exception as e:
            print(f"Exception separating {audio_path.name}: {e}")
            return False, None

    def get_audio_stats(self, audio_path: Path) -> dict:
        """
        Get basic statistics about audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with audio statistics
        """
        try:
            import librosa
            import numpy as np

            # Load audio
            y, sr = librosa.load(str(audio_path), sr=None, mono=True)

            # Calculate statistics
            duration = len(y) / sr
            rms = np.sqrt(np.mean(y**2))
            peak = np.max(np.abs(y))
            zero_crossings = np.sum(librosa.zero_crossings(y))

            return {
                "duration": duration,
                "rms_level": float(rms),
                "peak_level": float(peak),
                "zero_crossings": int(zero_crossings),
                "sample_rate": sr
            }

        except ImportError:
            print("Note: Install librosa for audio statistics: pip install librosa")
            return {}
        except Exception as e:
            print(f"Error getting audio stats: {e}")
            return {}


def separate_single_audio(args) -> Tuple[str, bool, Optional[Path]]:
    """
    Worker function for parallel processing

    Args:
        args: Tuple of (audio_path, output_dir, config)

    Returns:
        Tuple of (audio_name, success, vocals_path)
    """
    audio_path, output_dir, config = args

    separator = VoiceSeparator(config)
    success, vocals_path = separator.separate_audio(audio_path, output_dir)

    return audio_path.name, success, vocals_path


def organize_separated_vocals(
    output_dir: Path,
    model_name: str,
    organized_dir: Path
) -> int:
    """
    Organize separated vocals into a flat directory structure

    Args:
        output_dir: Demucs output directory (with model subdirs)
        model_name: Name of Demucs model used
        organized_dir: Target directory for organized vocals

    Returns:
        Number of files organized
    """
    organized_dir.mkdir(parents=True, exist_ok=True)

    # Find all vocals files
    model_dir = output_dir / model_name
    vocals_files = list(model_dir.glob("*/vocals.wav"))

    count = 0
    for vocals_path in vocals_files:
        # Get episode name from parent directory
        episode_name = vocals_path.parent.name
        target_path = organized_dir / f"{episode_name}_vocals.wav"

        # Copy vocals file
        shutil.copy2(vocals_path, target_path)
        count += 1

    return count


def batch_separate_vocals(
    audio_dir: Path,
    output_dir: Path,
    config: SeparationConfig,
    audio_extensions: List[str] = [".wav", ".flac", ".mp3"],
    sequential: bool = False
) -> dict:
    """
    Separate vocals from all audio files in a directory

    Args:
        audio_dir: Directory containing audio files
        output_dir: Directory to save separated vocals
        config: Separation configuration
        audio_extensions: List of audio file extensions to process
        sequential: Process files sequentially (for GPU processing)

    Returns:
        Dictionary with separation statistics
    """
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))

    audio_files = sorted(audio_files)

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return {}

    print(f"Found {len(audio_files)} audio files")
    print(f"Output directory: {output_dir}")
    print(f"Model: {config.model.value}")
    print(f"Device: {config.device}")
    print(f"Mode: {'Sequential' if sequential else 'Parallel'}")
    print()

    # Process audio files
    results = []

    if sequential or config.device == "cuda":
        # Sequential processing (required for GPU)
        print("Processing sequentially (GPU mode)...")
        for audio_path in tqdm(audio_files, desc="Separating vocals"):
            separator = VoiceSeparator(config)
            success, vocals_path = separator.separate_audio(audio_path, output_dir)
            results.append((audio_path.name, success, vocals_path))
    else:
        # Parallel processing (CPU only)
        args_list = [
            (audio_path, output_dir, config)
            for audio_path in audio_files
        ]

        with Pool(min(cpu_count() // 2, len(audio_files))) as pool:
            with tqdm(total=len(audio_files), desc="Separating vocals") as pbar:
                for result in pool.imap_unordered(separate_single_audio, args_list):
                    results.append(result)
                    pbar.update(1)

    # Collect statistics
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful

    stats = {
        "total_files": len(audio_files),
        "successful": successful,
        "failed": failed,
        "model": config.model.value,
        "device": config.device,
        "output_dir": str(output_dir)
    }

    # Organize vocals into flat directory
    organized_dir = output_dir / "vocals"
    organized_count = organize_separated_vocals(
        output_dir,
        config.model.value,
        organized_dir
    )
    stats["organized_vocals"] = organized_count
    stats["vocals_dir"] = str(organized_dir)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Separate vocals from background music using Demucs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic separation (vocals only)
  python voice_separator.py /path/to/audio

  # Use fine-tuned model with higher quality
  python voice_separator.py /path/to/audio --model htdemucs_ft --shifts 5

  # CPU processing with multiple threads
  python voice_separator.py /path/to/audio --device cpu --jobs 8

  # Full 4-stem separation (vocals, bass, drums, other)
  python voice_separator.py /path/to/audio --full-stems

Notes:
  - GPU processing is faster but uses more memory
  - Higher --shifts value = better quality but slower
  - Use --full-stems if you need all instrument tracks
        """
    )

    # Required arguments
    parser.add_argument(
        "audio_dir",
        type=Path,
        help="Directory containing audio files"
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for separated audio (default: audio_dir/../separated)"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"],
        default="htdemucs",
        help="Demucs model to use (default: htdemucs)"
    )

    parser.add_argument(
        "--full-stems",
        action="store_true",
        help="Extract all stems (vocals, bass, drums, other) instead of vocals only"
    )

    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="Number of random shifts for better quality (default: 1, higher=better)"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Processing device (default: cuda)"
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Number of CPU threads (default: 0=auto)"
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process files sequentially (automatic for GPU)"
    )

    parser.add_argument(
        "--audio-extensions",
        type=str,
        nargs="+",
        default=[".wav", ".flac", ".mp3"],
        help="Audio file extensions to process (default: .wav .flac .mp3)"
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.audio_dir.exists():
        print(f"Error: Audio directory not found: {args.audio_dir}")
        return 1

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.audio_dir.parent / "separated"

    # Create separation configuration
    config = SeparationConfig(
        model=DemucsModel(args.model),
        two_stems=not args.full_stems,
        shifts=args.shifts,
        device=args.device,
        jobs=args.jobs
    )

    # Run separation
    print("=" * 60)
    print("Voice Separation Tool (Demucs)")
    print("=" * 60)
    print()

    stats = batch_separate_vocals(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        config=config,
        audio_extensions=args.audio_extensions,
        sequential=args.sequential or args.device == "cuda"
    )

    # Print results
    print()
    print("=" * 60)
    print("Separation Complete!")
    print("=" * 60)
    print()
    print(f"Total files: {stats['total_files']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Organized vocals: {stats['organized_vocals']}")
    print()
    print(f"Vocals directory: {stats['vocals_dir']}")
    print()

    print("Next steps:")
    print(f"  1. Check vocals: ls {stats['vocals_dir']}")
    print("  2. Run voice analysis: python voice_analyzer.py <vocals_dir>")
    print("  3. Speaker diarization: python speaker_diarization.py <vocals_dir>")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
