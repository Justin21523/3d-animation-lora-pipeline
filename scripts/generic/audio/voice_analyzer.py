#!/usr/bin/env python3
"""
Voice Analysis Tool
===================
Comprehensive voice characteristic extraction and analysis.

Features:
- Fundamental frequency (F0/pitch) analysis
- Formant extraction (F1, F2, F3, F4, F5)
- Spectral features (MFCCs, spectral centroid, bandwidth)
- Temporal features (speaking rate, jitter, shimmer)
- Prosody analysis (intonation, rhythm, stress)
- Voice embeddings (speaker identification)
- Visualization (spectrograms, pitch contours, formant plots)

This tool helps understand voice characteristics to distinguish different speakers.

Usage:
    python voice_analyzer.py /path/to/vocals --visualize
    python voice_analyzer.py /path/to/vocals --feature all --output-csv results.csv
"""

import numpy as np
import librosa
import librosa.display
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
import json
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VoiceFeatures:
    """Container for extracted voice features"""
    # File information
    filename: str
    duration: float

    # Fundamental frequency (pitch)
    f0_mean: float
    f0_std: float
    f0_min: float
    f0_max: float
    f0_range: float

    # Formants (vocal tract resonances)
    f1_mean: Optional[float] = None
    f2_mean: Optional[float] = None
    f3_mean: Optional[float] = None

    # Spectral features
    spectral_centroid_mean: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0
    zero_crossing_rate: float = 0.0

    # MFCCs (Mel-frequency cepstral coefficients)
    mfcc_mean: Optional[List[float]] = None

    # Temporal features
    speaking_rate: Optional[float] = None  # syllables per second
    pause_ratio: Optional[float] = None    # % of silence

    # Energy
    rms_mean: float = 0.0
    rms_std: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values"""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


class VoiceAnalyzer:
    """Analyze voice characteristics for speaker identification"""

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mfcc: int = 13
    ):
        """
        Initialize voice analyzer

        Args:
            sample_rate: Audio sample rate (Hz)
            hop_length: Number of samples between frames
            n_fft: FFT window size
            n_mfcc: Number of MFCC coefficients
        """
        self.sr = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc

    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, float]:
        """
        Load audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_array, duration)
        """
        y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
        duration = len(y) / sr
        return y, duration

    def extract_f0(self, y: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Extract fundamental frequency (pitch)

        This is the most important feature for voice identification.
        Different speakers have different pitch ranges.

        Args:
            y: Audio signal

        Returns:
            Tuple of (f0_array, f0_stats)
        """
        # Use pYIN algorithm (probabilistic YIN)
        # fmin: minimum expected pitch (typically 80Hz for male, 165Hz for female)
        # fmax: maximum expected pitch (typically 400Hz)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=self.sr,
            hop_length=self.hop_length
        )

        # Remove unvoiced frames (NaN values)
        f0_voiced = f0[~np.isnan(f0)]

        if len(f0_voiced) > 0:
            stats = {
                'mean': float(np.mean(f0_voiced)),
                'std': float(np.std(f0_voiced)),
                'min': float(np.min(f0_voiced)),
                'max': float(np.max(f0_voiced)),
                'range': float(np.max(f0_voiced) - np.min(f0_voiced)),
                'median': float(np.median(f0_voiced))
            }
        else:
            stats = {
                'mean': 0.0, 'std': 0.0, 'min': 0.0,
                'max': 0.0, 'range': 0.0, 'median': 0.0
            }

        return f0, stats

    def extract_formants(self, y: np.ndarray) -> Optional[dict]:
        """
        Extract formant frequencies (requires praat-parselmouth)

        Formants are resonances of the vocal tract.
        F1: related to tongue height (vowel openness)
        F2: related to tongue position (front/back)
        F3-F5: voice quality and speaker characteristics

        Args:
            y: Audio signal

        Returns:
            Dictionary of formant statistics or None if library not available
        """
        try:
            import parselmouth
            from parselmouth.praat import call

            # Create Praat Sound object
            sound = parselmouth.Sound(y, sampling_frequency=self.sr)

            # Extract formants
            formant = sound.to_formant_burg(
                time_step=0.01,
                max_number_of_formants=5,
                maximum_formant=5500,
                window_length=0.025,
                pre_emphasis_from=50
            )

            # Get formant values over time
            duration = sound.duration
            num_frames = formant.get_number_of_frames()

            f1_values = []
            f2_values = []
            f3_values = []

            for i in range(1, num_frames + 1):
                t = formant.get_time_from_frame_number(i)
                f1 = formant.get_value_at_time(1, t)
                f2 = formant.get_value_at_time(2, t)
                f3 = formant.get_value_at_time(3, t)

                if not np.isnan(f1):
                    f1_values.append(f1)
                if not np.isnan(f2):
                    f2_values.append(f2)
                if not np.isnan(f3):
                    f3_values.append(f3)

            formants = {}
            if f1_values:
                formants['f1_mean'] = float(np.mean(f1_values))
                formants['f1_std'] = float(np.std(f1_values))
            if f2_values:
                formants['f2_mean'] = float(np.mean(f2_values))
                formants['f2_std'] = float(np.std(f2_values))
            if f3_values:
                formants['f3_mean'] = float(np.mean(f3_values))
                formants['f3_std'] = float(np.std(f3_values))

            return formants

        except ImportError:
            print("Note: Install praat-parselmouth for formant extraction:")
            print("  pip install praat-parselmouth")
            return None
        except Exception as e:
            print(f"Warning: Formant extraction failed: {e}")
            return None

    def extract_spectral_features(self, y: np.ndarray) -> dict:
        """
        Extract spectral characteristics

        Spectral features describe the frequency distribution of the voice.
        - Spectral centroid: "brightness" of sound
        - Spectral bandwidth: range of frequencies
        - Spectral rolloff: frequency below which 85% of energy is contained
        - Zero crossing rate: number of sign changes (roughness)

        Args:
            y: Audio signal

        Returns:
            Dictionary of spectral features
        """
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, hop_length=self.hop_length
        )

        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=self.sr, hop_length=self.hop_length
        )

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=self.sr, hop_length=self.hop_length
        )

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y, hop_length=self.hop_length
        )

        return {
            'spectral_centroid_mean': float(np.mean(centroid)),
            'spectral_centroid_std': float(np.std(centroid)),
            'spectral_bandwidth_mean': float(np.mean(bandwidth)),
            'spectral_bandwidth_std': float(np.std(bandwidth)),
            'spectral_rolloff_mean': float(np.mean(rolloff)),
            'spectral_rolloff_std': float(np.std(rolloff)),
            'zero_crossing_rate': float(np.mean(zcr))
        }

    def extract_mfcc(self, y: np.ndarray) -> dict:
        """
        Extract Mel-frequency cepstral coefficients

        MFCCs are a compact representation of the spectral envelope.
        Widely used in speech recognition and speaker identification.

        Args:
            y: Audio signal

        Returns:
            Dictionary of MFCC statistics
        """
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )

        # Calculate statistics for each coefficient
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        return {
            'mfcc_mean': mfcc_mean.tolist(),
            'mfcc_std': mfcc_std.tolist()
        }

    def extract_temporal_features(self, y: np.ndarray) -> dict:
        """
        Extract temporal characteristics

        - Speaking rate: estimated syllables per second
        - Pause ratio: percentage of silence

        Args:
            y: Audio signal

        Returns:
            Dictionary of temporal features
        """
        # Detect onsets (potential syllable boundaries)
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self.sr,
            hop_length=self.hop_length
        )

        duration = len(y) / self.sr
        speaking_rate = len(onsets) / duration if duration > 0 else 0

        # Detect silence (pause ratio)
        # Use RMS energy to identify silent regions
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        threshold = np.mean(rms) * 0.1
        silence_frames = np.sum(rms < threshold)
        pause_ratio = silence_frames / len(rms) if len(rms) > 0 else 0

        return {
            'speaking_rate': float(speaking_rate),
            'pause_ratio': float(pause_ratio),
            'num_onsets': len(onsets)
        }

    def extract_energy_features(self, y: np.ndarray) -> dict:
        """
        Extract energy characteristics

        RMS (root mean square) energy indicates loudness and vocal effort.

        Args:
            y: Audio signal

        Returns:
            Dictionary of energy features
        """
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

        return {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'rms_max': float(np.max(rms)),
            'rms_min': float(np.min(rms))
        }

    def analyze_audio(
        self,
        audio_path: Path,
        extract_formants: bool = False
    ) -> VoiceFeatures:
        """
        Comprehensive voice analysis

        Args:
            audio_path: Path to audio file
            extract_formants: Whether to extract formants (slow)

        Returns:
            VoiceFeatures object with all extracted features
        """
        # Load audio
        y, duration = self.load_audio(audio_path)

        # Extract fundamental frequency
        f0, f0_stats = self.extract_f0(y)

        # Extract spectral features
        spectral = self.extract_spectral_features(y)

        # Extract MFCCs
        mfcc = self.extract_mfcc(y)

        # Extract temporal features
        temporal = self.extract_temporal_features(y)

        # Extract energy features
        energy = self.extract_energy_features(y)

        # Create features object
        features = VoiceFeatures(
            filename=audio_path.name,
            duration=duration,
            f0_mean=f0_stats['mean'],
            f0_std=f0_stats['std'],
            f0_min=f0_stats['min'],
            f0_max=f0_stats['max'],
            f0_range=f0_stats['range'],
            spectral_centroid_mean=spectral['spectral_centroid_mean'],
            spectral_bandwidth_mean=spectral['spectral_bandwidth_mean'],
            spectral_rolloff_mean=spectral['spectral_rolloff_mean'],
            zero_crossing_rate=spectral['zero_crossing_rate'],
            mfcc_mean=mfcc['mfcc_mean'],
            speaking_rate=temporal['speaking_rate'],
            pause_ratio=temporal['pause_ratio'],
            rms_mean=energy['rms_mean'],
            rms_std=energy['rms_std']
        )

        # Extract formants if requested
        if extract_formants:
            formants = self.extract_formants(y)
            if formants:
                features.f1_mean = formants.get('f1_mean')
                features.f2_mean = formants.get('f2_mean')
                features.f3_mean = formants.get('f3_mean')

        return features

    def visualize_voice(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None
    ):
        """
        Create visualizations of voice characteristics

        Creates:
        1. Waveform
        2. Spectrogram
        3. Pitch contour
        4. MFCCs

        Args:
            audio_path: Path to audio file
            output_dir: Directory to save visualizations (None = display only)
        """
        # Load audio
        y, duration = self.load_audio(audio_path)

        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        fig.suptitle(f"Voice Analysis: {audio_path.name}", fontsize=14, fontweight='bold')

        # 1. Waveform
        librosa.display.waveshow(y, sr=self.sr, ax=axes[0])
        axes[0].set_title("Waveform")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")

        # 2. Spectrogram
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(
            S_db, sr=self.sr, hop_length=self.hop_length,
            x_axis='time', y_axis='hz', ax=axes[1]
        )
        axes[1].set_title("Spectrogram")
        axes[1].set_ylabel("Frequency (Hz)")
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

        # 3. Pitch contour
        f0, _ = self.extract_f0(y)
        times = librosa.times_like(f0, sr=self.sr, hop_length=self.hop_length)
        axes[2].plot(times, f0, linewidth=1, color='blue')
        axes[2].set_title("Pitch Contour (F0)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Frequency (Hz)")
        axes[2].set_ylim(0, 500)
        axes[2].grid(True, alpha=0.3)

        # 4. MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        img = librosa.display.specshow(
            mfcc, sr=self.sr, hop_length=self.hop_length,
            x_axis='time', ax=axes[3]
        )
        axes[3].set_title("MFCCs (Mel-frequency cepstral coefficients)")
        axes[3].set_ylabel("MFCC")
        fig.colorbar(img, ax=axes[3])

        plt.tight_layout()

        # Save or display
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{audio_path.stem}_analysis.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization: {output_path}")
        else:
            plt.show()

        plt.close()


def batch_analyze(
    audio_dir: Path,
    output_csv: Optional[Path] = None,
    extract_formants: bool = False,
    visualize: bool = False,
    viz_output_dir: Optional[Path] = None
) -> List[VoiceFeatures]:
    """
    Analyze all audio files in a directory

    Args:
        audio_dir: Directory containing audio files
        output_csv: Path to save CSV results
        extract_formants: Extract formant features (slower)
        visualize: Create visualizations for each file
        viz_output_dir: Directory to save visualizations

    Returns:
        List of VoiceFeatures for all files
    """
    analyzer = VoiceAnalyzer()

    # Find audio files
    audio_files = list(audio_dir.glob("*.wav"))
    audio_files.extend(audio_dir.glob("*.flac"))
    audio_files.extend(audio_dir.glob("*.mp3"))
    audio_files = sorted(audio_files)

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return []

    print(f"Found {len(audio_files)} audio files")
    print(f"Extracting formants: {extract_formants}")
    print()

    # Analyze each file
    all_features = []
    for audio_path in tqdm(audio_files, desc="Analyzing voices"):
        try:
            features = analyzer.analyze_audio(audio_path, extract_formants)
            all_features.append(features)

            # Create visualization if requested
            if visualize:
                analyzer.visualize_voice(audio_path, viz_output_dir)

        except Exception as e:
            print(f"Error analyzing {audio_path.name}: {e}")

    # Save to CSV if requested
    if output_csv and all_features:
        save_to_csv(all_features, output_csv)
        print(f"\nResults saved to: {output_csv}")

    return all_features


def save_to_csv(features_list: List[VoiceFeatures], output_path: Path):
    """Save voice features to CSV file"""
    if not features_list:
        return

    # Get all field names (some may be None)
    sample_dict = features_list[0].to_dict()
    fieldnames = list(sample_dict.keys())

    # Remove MFCC fields (too many columns for CSV)
    fieldnames = [f for f in fieldnames if not f.startswith('mfcc')]

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for features in features_list:
            row = features.to_dict()
            # Remove MFCC data
            row = {k: v for k, v in row.items() if not k.startswith('mfcc')}
            writer.writerow(row)


def compare_voices(features_list: List[VoiceFeatures]):
    """
    Compare voices and group by similarity

    Prints a summary showing pitch ranges and other characteristics
    to help identify different speakers.
    """
    if not features_list:
        return

    print("\n" + "=" * 60)
    print("Voice Comparison Summary")
    print("=" * 60)
    print()

    # Sort by pitch (F0)
    sorted_features = sorted(features_list, key=lambda f: f.f0_mean)

    print(f"{'Filename':<40} {'Pitch (Hz)':<15} {'Range':<10} {'Brightness'}")
    print("-" * 80)

    for features in sorted_features:
        pitch_str = f"{features.f0_mean:.1f} Hz"
        range_str = f"{features.f0_range:.1f}"
        brightness = features.spectral_centroid_mean / 1000  # kHz

        print(f"{features.filename:<40} {pitch_str:<15} {range_str:<10} {brightness:.2f} kHz")

    print()
    print("Interpretation:")
    print("  - Lower pitch (< 180 Hz): Typically male voices")
    print("  - Higher pitch (> 200 Hz): Typically female/child voices")
    print("  - Larger range: More expressive/emotional delivery")
    print("  - Higher brightness: Sharper, crisper voice quality")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze voice characteristics for speaker identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis of all vocals
  python voice_analyzer.py /path/to/vocals

  # Full analysis with formants and visualization
  python voice_analyzer.py /path/to/vocals --formants --visualize

  # Save results to CSV
  python voice_analyzer.py /path/to/vocals --output-csv results.csv

  # Create visualizations for all files
  python voice_analyzer.py /path/to/vocals --visualize --viz-dir ./visualizations
        """
    )

    parser.add_argument(
        "audio_dir",
        type=Path,
        help="Directory containing vocal audio files"
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Save analysis results to CSV file"
    )

    parser.add_argument(
        "--formants",
        action="store_true",
        help="Extract formant features (requires praat-parselmouth)"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations for each audio file"
    )

    parser.add_argument(
        "--viz-dir",
        type=Path,
        help="Directory to save visualizations (default: audio_dir/visualizations)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.audio_dir.exists():
        print(f"Error: Directory not found: {args.audio_dir}")
        return 1

    # Set visualization directory
    if args.visualize and args.viz_dir is None:
        args.viz_dir = args.audio_dir / "visualizations"

    # Run analysis
    print("=" * 60)
    print("Voice Analysis Tool")
    print("=" * 60)
    print()

    features_list = batch_analyze(
        audio_dir=args.audio_dir,
        output_csv=args.output_csv,
        extract_formants=args.formants,
        visualize=args.visualize,
        viz_output_dir=args.viz_dir
    )

    # Print comparison
    if features_list:
        compare_voices(features_list)

    print("Next steps:")
    print("  1. Review voice characteristics to identify different speakers")
    print("  2. Group similar voices together")
    print("  3. Label audio files by character name")
    print("  4. Run speaker diarization for automatic grouping")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
