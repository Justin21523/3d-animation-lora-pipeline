#!/usr/bin/env python3
"""
Frame Preprocessing Pipeline

Purpose: Main preprocessing orchestrator that coordinates deduplication, quality filtering, and sampling
Features: Configurable pipeline stages, parallel processing support, comprehensive reporting
Use Cases: Prepare raw extracted frames for segmentation

Usage:
    python preprocess_frames.py \
        --input-dir /path/to/raw_frames \
        --output-dir /path/to/preprocessed \
        --project luca \
        --deduplicate \
        --quality-filter \
        --sample \
        --target-count 1000
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import shutil


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    # Deduplication
    enable_deduplication: bool = True
    dedup_method: str = "phash"
    dedup_threshold: int = 12

    # Quality filtering
    enable_quality_filter: bool = True
    min_sharpness: float = 100.0
    max_blur: float = 0.15
    min_brightness: int = 20
    max_brightness: int = 235
    min_contrast: float = 30.0
    max_noise: float = 0.10

    # Sampling
    enable_sampling: bool = True
    sampling_method: str = "hybrid"
    target_count: int = 500
    target_ratio: Optional[float] = None

    # General
    use_hardlinks: bool = False
    keep_intermediates: bool = True
    project: Optional[str] = None


class FramePreprocessor:
    """Orchestrates frame preprocessing pipeline"""

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize preprocessor

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.script_dir = Path(__file__).parent

    def run_deduplication(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Run frame deduplication

        Args:
            input_dir: Input directory
            output_dir: Output directory

        Returns:
            Statistics dictionary
        """
        print("\n" + "="*60)
        print("STAGE 1: DEDUPLICATION")
        print("="*60)

        script_path = self.script_dir / "deduplicate_frames.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--method", self.config.dedup_method,
            "--phash-threshold", str(self.config.dedup_threshold),
        ]

        if self.config.use_hardlinks:
            cmd.append("--hardlinks")

        if self.config.project:
            cmd.extend(["--project", self.config.project])

        result = subprocess.run(cmd, check=True)

        # Load statistics
        report_path = output_dir / "deduplication_report.json"
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)
        return {}

    def run_quality_filter(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Run quality filtering

        Args:
            input_dir: Input directory
            output_dir: Output directory

        Returns:
            Statistics dictionary
        """
        print("\n" + "="*60)
        print("STAGE 2: QUALITY FILTERING")
        print("="*60)

        script_path = self.script_dir / "quality_filter.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--min-sharpness", str(self.config.min_sharpness),
            "--max-blur", str(self.config.max_blur),
            "--min-brightness", str(self.config.min_brightness),
            "--max-brightness", str(self.config.max_brightness),
            "--min-contrast", str(self.config.min_contrast),
            "--max-noise", str(self.config.max_noise),
        ]

        if self.config.use_hardlinks:
            cmd.append("--hardlinks")

        if self.config.project:
            cmd.extend(["--project", self.config.project])

        result = subprocess.run(cmd, check=True)

        # Load statistics
        report_path = output_dir / "quality_filter_report.json"
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)
        return {}

    def run_sampling(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Run adaptive sampling

        Args:
            input_dir: Input directory
            output_dir: Output directory

        Returns:
            Statistics dictionary
        """
        print("\n" + "="*60)
        print("STAGE 3: ADAPTIVE SAMPLING")
        print("="*60)

        script_path = self.script_dir / "adaptive_sampler.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--method", self.config.sampling_method,
            "--target-count", str(self.config.target_count),
        ]

        if self.config.target_ratio:
            cmd.extend(["--target-ratio", str(self.config.target_ratio)])

        if self.config.use_hardlinks:
            cmd.append("--hardlinks")

        if self.config.project:
            cmd.extend(["--project", self.config.project])

        result = subprocess.run(cmd, check=True)

        # Load statistics
        report_path = output_dir / "sampling_report.json"
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)
        return {}

    def preprocess(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main preprocessing pipeline

        Args:
            input_dir: Directory with raw extracted frames
            output_dir: Directory to save preprocessed frames

        Returns:
            Overall statistics dictionary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create intermediate directories
        intermediate_dir = output_dir / "intermediate"
        if self.config.keep_intermediates:
            intermediate_dir.mkdir(exist_ok=True)

        print("\n" + "="*70)
        print("FRAME PREPROCESSING PIPELINE")
        print("="*70)
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        if self.config.project:
            print(f"Project: {self.config.project}")
        print("="*70)

        # Count input frames
        input_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        print(f"\n📊 Input frames: {len(input_files)}")

        stats = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "project": self.config.project,
            "stages": {},
            "start_time": datetime.now().isoformat(),
        }

        current_input = input_dir
        final_output = output_dir / "final"

        # Stage 1: Deduplication
        if self.config.enable_deduplication:
            dedup_output = intermediate_dir / "01_deduplicated" if self.config.keep_intermediates else final_output
            dedup_stats = self.run_deduplication(current_input, dedup_output)
            stats["stages"]["deduplication"] = dedup_stats
            current_input = dedup_output

            if dedup_stats:
                print(f"\n   ✅ Deduplication: {dedup_stats['unique_frames_kept']} frames kept")
                print(f"      (Removed {dedup_stats['duplicates_removed']} duplicates)")
        else:
            print("\n   ⏭️  Skipping deduplication")

        # Stage 2: Quality Filtering
        if self.config.enable_quality_filter:
            quality_output = intermediate_dir / "02_quality_filtered" if self.config.keep_intermediates else final_output
            quality_stats = self.run_quality_filter(current_input, quality_output)
            stats["stages"]["quality_filter"] = quality_stats
            current_input = quality_output

            if quality_stats:
                print(f"\n   ✅ Quality Filter: {quality_stats['accepted_frames']} frames kept")
                print(f"      (Rejected {quality_stats['rejected_frames']} low-quality)")
        else:
            print("\n   ⏭️  Skipping quality filtering")

        # Stage 3: Sampling
        if self.config.enable_sampling:
            sampling_output = final_output
            sampling_stats = self.run_sampling(current_input, sampling_output)
            stats["stages"]["sampling"] = sampling_stats

            if sampling_stats:
                print(f"\n   ✅ Sampling: {sampling_stats['sampled_frames']} frames selected")
                print(f"      (Method: {self.config.sampling_method})")
        else:
            # No sampling - copy current_input to final_output
            print("\n   ⏭️  Skipping sampling")
            final_output.mkdir(parents=True, exist_ok=True)

            if current_input != final_output:
                current_files = list(current_input.glob("*.jpg")) + list(current_input.glob("*.png"))
                print(f"   Copying {len(current_files)} frames to final output...")
                for src in current_files:
                    dst = final_output / src.name
                    if self.config.use_hardlinks:
                        try:
                            dst.hardlink_to(src)
                        except:
                            shutil.copy2(src, dst)
                    else:
                        shutil.copy2(src, dst)

        # Count final frames
        final_files = list(final_output.glob("*.jpg")) + list(final_output.glob("*.png"))

        stats["end_time"] = datetime.now().isoformat()
        stats["summary"] = {
            "input_frames": len(input_files),
            "output_frames": len(final_files),
            "reduction_rate": 1 - (len(final_files) / len(input_files)) if len(input_files) > 0 else 0,
            "stages_executed": [k for k, v in {
                "deduplication": self.config.enable_deduplication,
                "quality_filter": self.config.enable_quality_filter,
                "sampling": self.config.enable_sampling,
            }.items() if v]
        }

        # Save overall report
        report_path = output_dir / "preprocessing_report.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        print(f"Input frames:  {stats['summary']['input_frames']}")
        print(f"Output frames: {stats['summary']['output_frames']}")
        print(f"Reduction:     {stats['summary']['reduction_rate']:.1%}")
        print(f"\nFinal frames: {final_output}")
        print(f"Report:       {report_path}")

        if self.config.keep_intermediates:
            print(f"Intermediates: {intermediate_dir}")

        print("="*70)

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Frame Preprocessing Pipeline (Film-Agnostic)"
    )

    # I/O
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with raw extracted frames"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save preprocessed frames"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name"
    )

    # Pipeline stages
    parser.add_argument(
        "--skip-deduplication",
        action="store_true",
        help="Skip deduplication stage"
    )
    parser.add_argument(
        "--skip-quality-filter",
        action="store_true",
        help="Skip quality filtering stage"
    )
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Skip sampling stage"
    )

    # Deduplication settings
    parser.add_argument(
        "--dedup-method",
        type=str,
        default="phash",
        choices=["phash", "dhash", "ahash", "ssim"],
        help="Deduplication method (default: phash)"
    )
    parser.add_argument(
        "--dedup-threshold",
        type=int,
        default=12,
        help="Deduplication threshold (default: 12)"
    )

    # Quality filter settings
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=100.0,
        help="Minimum sharpness (default: 100)"
    )
    parser.add_argument(
        "--max-blur",
        type=float,
        default=0.15,
        help="Maximum blur ratio (default: 0.15)"
    )
    parser.add_argument(
        "--min-brightness",
        type=int,
        default=20,
        help="Minimum brightness (default: 20)"
    )
    parser.add_argument(
        "--max-brightness",
        type=int,
        default=235,
        help="Maximum brightness (default: 235)"
    )
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=30.0,
        help="Minimum contrast (default: 30)"
    )
    parser.add_argument(
        "--max-noise",
        type=float,
        default=0.10,
        help="Maximum noise level (default: 0.10)"
    )

    # Sampling settings
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="hybrid",
        choices=["clustering", "temporal", "quality", "hybrid"],
        help="Sampling method (default: hybrid)"
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=500,
        help="Target frame count after sampling (default: 500)"
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        help="Alternative: target ratio (0-1)"
    )

    # General settings
    parser.add_argument(
        "--hardlinks",
        action="store_true",
        help="Use hardlinks instead of copying (saves space)"
    )
    parser.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Don't keep intermediate results (saves space)"
    )

    args = parser.parse_args()

    # Create config
    config = PreprocessingConfig(
        enable_deduplication=not args.skip_deduplication,
        dedup_method=args.dedup_method,
        dedup_threshold=args.dedup_threshold,
        enable_quality_filter=not args.skip_quality_filter,
        min_sharpness=args.min_sharpness,
        max_blur=args.max_blur,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
        min_contrast=args.min_contrast,
        max_noise=args.max_noise,
        enable_sampling=not args.skip_sampling,
        sampling_method=args.sampling_method,
        target_count=args.target_count,
        target_ratio=args.target_ratio,
        use_hardlinks=args.hardlinks,
        keep_intermediates=not args.no_intermediates,
        project=args.project
    )

    # Run preprocessing
    preprocessor = FramePreprocessor(config)
    stats = preprocessor.preprocess(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
