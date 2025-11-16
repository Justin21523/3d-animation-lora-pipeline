#!/usr/bin/env python3
"""
Frame Quality Filter

Purpose: Filter low-quality frames based on multiple quality metrics
Metrics: Sharpness, blur detection, brightness, contrast, noise level
Use Cases: Preprocessing frames before segmentation or training

Usage:
    python quality_filter.py \
        --input-dir /path/to/frames \
        --output-dir /path/to/filtered \
        --min-sharpness 100 \
        --max-blur 0.15 \
        --min-brightness 20 \
        --max-brightness 235
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
import shutil


@dataclass
class QualityConfig:
    """Configuration for quality filtering"""
    min_sharpness: float = 100.0  # Laplacian variance threshold
    max_blur: float = 0.15  # Blur ratio threshold (0-1)
    min_brightness: int = 20  # Min average brightness (0-255)
    max_brightness: int = 235  # Max average brightness (0-255)
    min_contrast: float = 30.0  # Min standard deviation
    max_noise: float = 0.10  # Max noise ratio (0-1)
    min_size: Tuple[int, int] = (256, 256)  # Min resolution (width, height)
    check_corruption: bool = True  # Check for corrupted images
    save_rejected: bool = False  # Save rejected frames to separate folder
    create_hardlinks: bool = False  # Use hardlinks instead of copying


class FrameQualityFilter:
    """Filter frames based on quality metrics"""

    def __init__(self, config: QualityConfig):
        """
        Initialize quality filter

        Args:
            config: Quality filtering configuration
        """
        self.config = config

    def compute_sharpness(self, image_path: Path) -> float:
        """
        Compute sharpness using Laplacian variance

        Higher values = sharper image
        Typical ranges:
        - < 100: Very blurry
        - 100-300: Acceptable
        - > 300: Sharp

        Args:
            image_path: Path to image

        Returns:
            Sharpness score (Laplacian variance)
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0

        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)

    def compute_blur_ratio(self, image_path: Path) -> float:
        """
        Compute blur ratio using FFT high-frequency content

        Lower values = sharper image
        Higher values = more blur

        Args:
            image_path: Path to image

        Returns:
            Blur ratio (0-1, lower is better)
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 1.0

        # Apply FFT
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Compute ratio of high frequency to total
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        radius = min(center_h, center_w) // 3

        # Create high-frequency mask
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_w)**2 + (y - center_h)**2) > radius**2

        high_freq = magnitude[mask].sum()
        total = magnitude.sum()

        # Blur ratio = 1 - (high_freq / total)
        blur_ratio = 1.0 - (high_freq / total) if total > 0 else 1.0
        return float(blur_ratio)

    def compute_brightness(self, image_path: Path) -> float:
        """
        Compute average brightness

        Args:
            image_path: Path to image

        Returns:
            Average brightness (0-255)
        """
        with Image.open(image_path) as img:
            if img.mode != 'L':
                img = img.convert('L')
            arr = np.array(img)
            return float(arr.mean())

    def compute_contrast(self, image_path: Path) -> float:
        """
        Compute contrast (standard deviation of pixel values)

        Higher values = more contrast

        Args:
            image_path: Path to image

        Returns:
            Contrast score (std deviation)
        """
        with Image.open(image_path) as img:
            if img.mode != 'L':
                img = img.convert('L')
            arr = np.array(img)
            return float(arr.std())

    def compute_noise_level(self, image_path: Path) -> float:
        """
        Estimate noise level using high-frequency content

        Args:
            image_path: Path to image

        Returns:
            Noise ratio (0-1, lower is better)
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 1.0

        # Use median filter to estimate noise
        denoised = cv2.medianBlur(img, 5)
        noise = cv2.absdiff(img, denoised)
        noise_level = noise.mean() / 255.0
        return float(noise_level)

    def check_image_size(self, image_path: Path) -> bool:
        """
        Check if image meets minimum size requirements

        Args:
            image_path: Path to image

        Returns:
            True if size is acceptable
        """
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                min_w, min_h = self.config.min_size
                return w >= min_w and h >= min_h
        except Exception:
            return False

    def check_corruption(self, image_path: Path) -> bool:
        """
        Check if image is corrupted or can't be loaded

        Args:
            image_path: Path to image

        Returns:
            True if image is valid
        """
        try:
            with Image.open(image_path) as img:
                img.verify()
            # Re-open after verify (verify closes the file)
            with Image.open(image_path) as img:
                img.load()
            return True
        except Exception:
            return False

    def evaluate_quality(self, image_path: Path) -> Tuple[bool, Dict]:
        """
        Evaluate all quality metrics for an image

        Args:
            image_path: Path to image

        Returns:
            (is_acceptable, metrics_dict)
        """
        metrics = {
            "sharpness": None,
            "blur_ratio": None,
            "brightness": None,
            "contrast": None,
            "noise_level": None,
            "size_ok": None,
            "not_corrupted": None,
        }

        rejection_reasons = []

        try:
            # Check corruption first
            if self.config.check_corruption:
                not_corrupted = self.check_corruption(image_path)
                metrics["not_corrupted"] = not_corrupted
                if not not_corrupted:
                    rejection_reasons.append("corrupted")
                    return False, {"metrics": metrics, "reasons": rejection_reasons}

            # Check size
            size_ok = self.check_image_size(image_path)
            metrics["size_ok"] = size_ok
            if not size_ok:
                rejection_reasons.append(f"too_small (min: {self.config.min_size})")

            # Compute quality metrics
            sharpness = self.compute_sharpness(image_path)
            metrics["sharpness"] = sharpness
            if sharpness < self.config.min_sharpness:
                rejection_reasons.append(f"low_sharpness ({sharpness:.1f} < {self.config.min_sharpness})")

            blur_ratio = self.compute_blur_ratio(image_path)
            metrics["blur_ratio"] = blur_ratio
            if blur_ratio > self.config.max_blur:
                rejection_reasons.append(f"too_blurry ({blur_ratio:.3f} > {self.config.max_blur})")

            brightness = self.compute_brightness(image_path)
            metrics["brightness"] = brightness
            if brightness < self.config.min_brightness:
                rejection_reasons.append(f"too_dark ({brightness:.1f} < {self.config.min_brightness})")
            elif brightness > self.config.max_brightness:
                rejection_reasons.append(f"too_bright ({brightness:.1f} > {self.config.max_brightness})")

            contrast = self.compute_contrast(image_path)
            metrics["contrast"] = contrast
            if contrast < self.config.min_contrast:
                rejection_reasons.append(f"low_contrast ({contrast:.1f} < {self.config.min_contrast})")

            noise_level = self.compute_noise_level(image_path)
            metrics["noise_level"] = noise_level
            if noise_level > self.config.max_noise:
                rejection_reasons.append(f"too_noisy ({noise_level:.3f} > {self.config.max_noise})")

            is_acceptable = len(rejection_reasons) == 0

            return is_acceptable, {"metrics": metrics, "reasons": rejection_reasons}

        except Exception as e:
            metrics["error"] = str(e)
            rejection_reasons.append(f"evaluation_error: {e}")
            return False, {"metrics": metrics, "reasons": rejection_reasons}

    def filter_frames(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main quality filtering pipeline

        Args:
            input_dir: Directory with input frames
            output_dir: Directory to save filtered frames

        Returns:
            Statistics dictionary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create rejected folder if needed
        if self.config.save_rejected:
            rejected_dir = output_dir / "rejected"
            rejected_dir.mkdir(exist_ok=True)

        # Find all image files
        image_files = sorted(
            list(input_dir.glob("*.jpg")) +
            list(input_dir.glob("*.png"))
        )

        print(f"\n📊 Found {len(image_files)} frames in {input_dir}")
        print(f"🔍 Evaluating quality metrics...")

        accepted_files = []
        rejected_files = []
        rejection_stats = {}
        all_metrics = []

        for img_path in tqdm(image_files, desc="Filtering frames"):
            is_acceptable, result = self.evaluate_quality(img_path)

            result["filename"] = img_path.name
            all_metrics.append(result)

            if is_acceptable:
                accepted_files.append(img_path)
                dst_path = output_dir / img_path.name

                if self.config.create_hardlinks:
                    try:
                        dst_path.hardlink_to(img_path)
                    except:
                        shutil.copy2(img_path, dst_path)
                else:
                    shutil.copy2(img_path, dst_path)

            else:
                rejected_files.append(img_path)

                # Count rejection reasons
                for reason in result["reasons"]:
                    rejection_stats[reason] = rejection_stats.get(reason, 0) + 1

                # Optionally save rejected frames
                if self.config.save_rejected:
                    dst_path = rejected_dir / img_path.name
                    if self.config.create_hardlinks:
                        try:
                            dst_path.hardlink_to(img_path)
                        except:
                            shutil.copy2(img_path, dst_path)
                    else:
                        shutil.copy2(img_path, dst_path)

        # Compute aggregate statistics
        accepted_metrics = [m for m in all_metrics if len(m["reasons"]) == 0]

        def safe_mean(values):
            valid = [v for v in values if v is not None]
            return float(np.mean(valid)) if valid else None

        aggregate_metrics = {}
        if accepted_metrics:
            for key in ["sharpness", "blur_ratio", "brightness", "contrast", "noise_level"]:
                values = [m["metrics"][key] for m in accepted_metrics if m["metrics"].get(key) is not None]
                if values:
                    aggregate_metrics[f"avg_{key}"] = safe_mean(values)
                    aggregate_metrics[f"min_{key}"] = float(np.min(values))
                    aggregate_metrics[f"max_{key}"] = float(np.max(values))

        # Save statistics
        stats = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "config": {
                "min_sharpness": self.config.min_sharpness,
                "max_blur": self.config.max_blur,
                "min_brightness": self.config.min_brightness,
                "max_brightness": self.config.max_brightness,
                "min_contrast": self.config.min_contrast,
                "max_noise": self.config.max_noise,
                "min_size": self.config.min_size,
            },
            "total_input_frames": len(image_files),
            "accepted_frames": len(accepted_files),
            "rejected_frames": len(rejected_files),
            "acceptance_rate": len(accepted_files) / len(image_files) if len(image_files) > 0 else 0,
            "rejection_reasons": rejection_stats,
            "aggregate_metrics": aggregate_metrics,
            "timestamp": datetime.now().isoformat()
        }

        # Save metadata
        metadata_path = output_dir / "quality_filter_report.json"
        with open(metadata_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Save detailed metrics
        detailed_path = output_dir / "detailed_metrics.json"
        with open(detailed_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\n✅ Quality filtering complete!")
        print(f"   Input frames: {len(image_files)}")
        print(f"   Accepted: {len(accepted_files)}")
        print(f"   Rejected: {len(rejected_files)}")
        print(f"   Acceptance rate: {stats['acceptance_rate']:.1%}")

        if rejection_stats:
            print(f"\n📊 Rejection reasons:")
            for reason, count in sorted(rejection_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"   - {reason}: {count}")

        print(f"\n📄 Reports saved to:")
        print(f"   - {metadata_path}")
        print(f"   - {detailed_path}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter low-quality frames (Film-Agnostic)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with input frames"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save filtered frames"
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=100.0,
        help="Minimum sharpness (Laplacian variance, default: 100)"
    )
    parser.add_argument(
        "--max-blur",
        type=float,
        default=0.15,
        help="Maximum blur ratio (0-1, default: 0.15)"
    )
    parser.add_argument(
        "--min-brightness",
        type=int,
        default=20,
        help="Minimum brightness (0-255, default: 20)"
    )
    parser.add_argument(
        "--max-brightness",
        type=int,
        default=235,
        help="Maximum brightness (0-255, default: 235)"
    )
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=30.0,
        help="Minimum contrast (std dev, default: 30)"
    )
    parser.add_argument(
        "--max-noise",
        type=float,
        default=0.10,
        help="Maximum noise level (0-1, default: 0.10)"
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=256,
        help="Minimum image width (default: 256)"
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=256,
        help="Minimum image height (default: 256)"
    )
    parser.add_argument(
        "--no-corruption-check",
        action="store_true",
        help="Skip corruption checking (faster but less safe)"
    )
    parser.add_argument(
        "--save-rejected",
        action="store_true",
        help="Save rejected frames to output_dir/rejected/"
    )
    parser.add_argument(
        "--hardlinks",
        action="store_true",
        help="Use hardlinks instead of copying (saves space)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (for logging)"
    )

    args = parser.parse_args()

    # Create config
    config = QualityConfig(
        min_sharpness=args.min_sharpness,
        max_blur=args.max_blur,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
        min_contrast=args.min_contrast,
        max_noise=args.max_noise,
        min_size=(args.min_width, args.min_height),
        check_corruption=not args.no_corruption_check,
        save_rejected=args.save_rejected,
        create_hardlinks=args.hardlinks
    )

    # Run filtering
    filter_obj = FrameQualityFilter(config)
    stats = filter_obj.filter_frames(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
