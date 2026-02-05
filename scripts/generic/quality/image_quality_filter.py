#!/usr/bin/env python3
"""
Image Quality Filtering System

Production-grade quality control for synthetic datasets:
- Blur detection (Laplacian variance)
- Duplicate detection (perceptual hashing)
- NSFW filtering (CLIP-based safety classifier)
- Quality tier classification (excellent/good/acceptable/rejected)
- Checkpoint/resume capability for large batches
- Comprehensive quality reports

Part of Module 3: Quality Filtering System
Author: LLMProvider Tooling
Date: 2025-11-30
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm

# Import our filter modules
import sys
sys.path.append(str(Path(__file__).parent))
from blur_detector import BlurDetector
from duplicate_detector import DuplicateDetector
from nsfw_detector import NSFWDetector

# Import checkpoint manager
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.core.utils.checkpoint_manager import IndexCheckpointManager


@dataclass
class ImageQualityMetrics:
    """Quality metrics for a single image"""
    path: str
    blur_score: Optional[float]
    is_blurry: bool
    is_duplicate: bool
    duplicate_of: Optional[str]
    perceptual_hash: Optional[str]
    nsfw_score: Optional[float]
    is_nsfw: bool
    overall_quality: str  # 'excellent', 'good', 'acceptable', 'rejected'
    rejection_reasons: List[str]


@dataclass
class FilterConfig:
    """Configuration for quality filtering"""
    # Blur detection
    blur_threshold: float = 100.0
    enable_blur_detection: bool = True

    # Duplicate detection
    duplicate_threshold: int = 8  # Hamming distance
    hash_algorithm: str = 'phash'
    enable_duplicate_detection: bool = True

    # NSFW filtering
    nsfw_threshold: float = 0.3
    enable_nsfw_filtering: bool = True

    # Quality tiers
    excellent_blur_threshold: float = 250.0  # Sharp images
    acceptable_blur_threshold: float = 150.0  # Acceptable quality

    # Processing
    batch_size: int = 100
    checkpoint_interval: int = 500


@dataclass
class FilteringReport:
    """Report for completed filtering job"""
    total_images: int
    images_passed: int
    images_rejected: int
    rejection_breakdown: Dict[str, int]  # {reason: count}
    quality_distribution: Dict[str, int]  # {tier: count}
    duplicates_found: int
    start_time: str
    end_time: str
    duration_seconds: float
    config: Dict[str, Any]


class ImageQualityFilter:
    """
    Main quality filtering orchestrator

    Integrates blur, duplicate, and NSFW detection with
    checkpoint/resume capability for large batches.
    """

    def __init__(
        self,
        config: FilterConfig,
        checkpoint_dir: Path,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = device
        self.checkpoint_mgr = IndexCheckpointManager(
            checkpoint_dir,
            filename="filtering_checkpoint.json"
        )

        # Initialize detectors
        print("\n" + "="*80)
        print("🔍 IMAGE QUALITY FILTER - Initializing")
        print("="*80)

        if config.enable_blur_detection:
            print(f"📊 Loading blur detector (threshold={config.blur_threshold})...")
            self.blur_detector = BlurDetector(threshold=config.blur_threshold)

        if config.enable_duplicate_detection:
            print(f"🔄 Loading duplicate detector ({config.hash_algorithm}, threshold={config.duplicate_threshold})...")
            self.duplicate_detector = DuplicateDetector(
                hash_algorithm=config.hash_algorithm,
                hamming_threshold=config.duplicate_threshold
            )

        if config.enable_nsfw_filtering:
            print(f"🛡️  Loading NSFW detector (threshold={config.nsfw_threshold})...")
            self.nsfw_detector = NSFWDetector(
                threshold=config.nsfw_threshold,
                device=device
            )

        print("="*80)
        print("✅ Filter ready")
        print("="*80 + "\n")

    def classify_quality_tier(self, metrics: ImageQualityMetrics) -> str:
        """
        Classify image into quality tier based on metrics

        Args:
            metrics: Image quality metrics

        Returns:
            Quality tier: 'excellent', 'good', 'acceptable', 'rejected'
        """
        # Reject if failed any filter
        if metrics.rejection_reasons:
            return 'rejected'

        # Classify by blur score
        if metrics.blur_score is None:
            return 'acceptable'

        if metrics.blur_score >= self.config.excellent_blur_threshold:
            return 'excellent'
        elif metrics.blur_score >= self.config.acceptable_blur_threshold:
            return 'good'
        else:
            return 'acceptable'

    def analyze_single_image(
        self,
        image_path: Path,
        duplicate_hashes: Dict[str, Path]
    ) -> ImageQualityMetrics:
        """
        Analyze single image with all enabled filters

        Args:
            image_path: Path to image
            duplicate_hashes: Dictionary of {hash: path} for duplicate detection

        Returns:
            ImageQualityMetrics for the image
        """
        rejection_reasons = []
        blur_score = None
        is_blurry = False
        is_duplicate = False
        duplicate_of = None
        perceptual_hash = None
        nsfw_score = None
        is_nsfw = False

        # Blur detection
        if self.config.enable_blur_detection:
            try:
                blur_score = self.blur_detector.compute_blur_score(image_path)
                is_blurry = blur_score < self.config.blur_threshold
                if is_blurry:
                    rejection_reasons.append('blurry')
            except Exception as e:
                print(f"Warning: Blur detection failed for {image_path.name}: {e}")

        # Duplicate detection
        if self.config.enable_duplicate_detection:
            try:
                img_hash = self.duplicate_detector.compute_hash(image_path)
                perceptual_hash = str(img_hash)

                # Check if duplicate
                for existing_hash, existing_path in duplicate_hashes.items():
                    distance = self.duplicate_detector.compute_hamming_distance(
                        img_hash,
                        existing_hash
                    )
                    if distance <= self.config.duplicate_threshold:
                        is_duplicate = True
                        duplicate_of = str(existing_path)
                        rejection_reasons.append('duplicate')
                        break

                # Add to hash database if not duplicate
                if not is_duplicate:
                    duplicate_hashes[perceptual_hash] = image_path

            except Exception as e:
                print(f"Warning: Duplicate detection failed for {image_path.name}: {e}")

        # NSFW detection
        if self.config.enable_nsfw_filtering:
            try:
                nsfw_score = self.nsfw_detector.compute_nsfw_score(image_path)
                is_nsfw = nsfw_score > self.config.nsfw_threshold
                if is_nsfw:
                    rejection_reasons.append('nsfw')
            except Exception as e:
                print(f"Warning: NSFW detection failed for {image_path.name}: {e}")

        # Create metrics
        metrics = ImageQualityMetrics(
            path=str(image_path),
            blur_score=blur_score,
            is_blurry=is_blurry,
            is_duplicate=is_duplicate,
            duplicate_of=duplicate_of,
            perceptual_hash=perceptual_hash,
            nsfw_score=nsfw_score,
            is_nsfw=is_nsfw,
            overall_quality='',  # Will be set below
            rejection_reasons=rejection_reasons
        )

        # Classify quality tier
        metrics.overall_quality = self.classify_quality_tier(metrics)

        return metrics

    def filter_batch(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> FilteringReport:
        """
        Filter batch of images with checkpointing

        Args:
            input_dir: Directory containing images to filter
            output_dir: Directory to organize filtered images

        Returns:
            FilteringReport with statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output subdirectories
        (output_dir / 'excellent').mkdir(exist_ok=True)
        (output_dir / 'good').mkdir(exist_ok=True)
        (output_dir / 'acceptable').mkdir(exist_ok=True)
        (output_dir / 'rejected' / 'blurry').mkdir(parents=True, exist_ok=True)
        (output_dir / 'rejected' / 'duplicates').mkdir(parents=True, exist_ok=True)
        (output_dir / 'rejected' / 'nsfw').mkdir(parents=True, exist_ok=True)

        start_time = datetime.now()

        # Find all images
        image_paths = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        image_paths = sorted(image_paths)

        print(f"📁 Found {len(image_paths)} images to filter\n")

        # Check for checkpoint
        checkpoint = self.checkpoint_mgr.load()
        if checkpoint:
            start_idx = checkpoint['last_completed_index'] + 1
            duplicate_hashes = {
                h: Path(p) for h, p in checkpoint.get('duplicate_hashes', {}).items()
            }
            print(f"📂 Resuming from checkpoint: index {start_idx}/{len(image_paths)}\n")
        else:
            start_idx = 0
            duplicate_hashes = {}
            print(f"🆕 Starting fresh filtering\n")

        # Track metrics
        images_passed = 0
        images_rejected = 0
        rejection_breakdown = {
            'blurry': 0,
            'duplicate': 0,
            'nsfw': 0
        }
        quality_distribution = {
            'excellent': 0,
            'good': 0,
            'acceptable': 0,
            'rejected': 0
        }
        duplicates_found = 0

        # Process images
        all_metrics = []
        pbar = tqdm(
            range(start_idx, len(image_paths)),
            desc="Filtering images",
            initial=start_idx,
            total=len(image_paths)
        )

        for idx in pbar:
            image_path = image_paths[idx]

            # Analyze image
            metrics = self.analyze_single_image(image_path, duplicate_hashes)
            all_metrics.append(metrics)

            # Update statistics
            quality_distribution[metrics.overall_quality] += 1

            if metrics.overall_quality == 'rejected':
                images_rejected += 1
                for reason in metrics.rejection_reasons:
                    rejection_breakdown[reason] += 1
            else:
                images_passed += 1

            if metrics.is_duplicate:
                duplicates_found += 1

            # Copy image to appropriate folder
            try:
                if metrics.overall_quality == 'rejected':
                    # Primary rejection reason
                    if 'blurry' in metrics.rejection_reasons:
                        dest = output_dir / 'rejected' / 'blurry' / image_path.name
                    elif 'duplicate' in metrics.rejection_reasons:
                        dest = output_dir / 'rejected' / 'duplicates' / image_path.name
                    elif 'nsfw' in metrics.rejection_reasons:
                        dest = output_dir / 'rejected' / 'nsfw' / image_path.name
                    else:
                        dest = output_dir / 'rejected' / image_path.name
                else:
                    dest = output_dir / metrics.overall_quality / image_path.name

                shutil.copy2(image_path, dest)
            except Exception as e:
                print(f"\n⚠️  Failed to copy {image_path.name}: {e}")

            # Checkpoint periodically
            if (idx + 1) % self.config.checkpoint_interval == 0:
                self.checkpoint_mgr.save(
                    last_completed_index=idx,
                    total_items=len(image_paths),
                    duplicate_hashes={h: str(p) for h, p in duplicate_hashes.items()}
                )

                pbar.set_postfix({
                    'passed': images_passed,
                    'rejected': images_rejected,
                    'duplicates': duplicates_found
                })

        # Final checkpoint
        self.checkpoint_mgr.save(
            last_completed_index=len(image_paths) - 1,
            total_items=len(image_paths),
            duplicate_hashes={h: str(p) for h, p in duplicate_hashes.items()}
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Save detailed metrics
        metrics_file = output_dir / "image_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump([asdict(m) for m in all_metrics], f, indent=2)

        # Generate report
        report = FilteringReport(
            total_images=len(image_paths),
            images_passed=images_passed,
            images_rejected=images_rejected,
            rejection_breakdown=rejection_breakdown,
            quality_distribution=quality_distribution,
            duplicates_found=duplicates_found,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            config=asdict(self.config)
        )

        # Save report
        report_file = output_dir / "filtering_report.json"
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)

        # Clear checkpoint
        self.checkpoint_mgr.clear()

        # Print summary
        print(f"\n{'='*80}")
        print(f"✅ FILTERING COMPLETE")
        print(f"{'='*80}")
        print(f"  Total images:     {len(image_paths)}")
        print(f"  Passed:           {images_passed} ({images_passed/len(image_paths)*100:.1f}%)")
        print(f"  Rejected:         {images_rejected} ({images_rejected/len(image_paths)*100:.1f}%)")
        print(f"\n  Rejection breakdown:")
        for reason, count in rejection_breakdown.items():
            if count > 0:
                print(f"    - {reason:12s}: {count}")
        print(f"\n  Quality distribution:")
        for tier, count in quality_distribution.items():
            if count > 0:
                print(f"    - {tier:12s}: {count}")
        print(f"\n  Duplicates found: {duplicates_found}")
        print(f"  Duration:         {duration/60:.1f} minutes")
        print(f"  Output:           {output_dir}")
        print(f"{'='*80}\n")

        return report


def main():
    """CLI for quality filtering"""
    import argparse

    parser = argparse.ArgumentParser(description="Image Quality Filtering System")

    # Required arguments
    parser.add_argument("input_dir", type=str, help="Input directory with images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Checkpoint directory")

    # Filter configuration
    parser.add_argument("--blur-threshold", type=float, default=100.0, help="Blur threshold (default: 100)")
    parser.add_argument("--duplicate-threshold", type=int, default=8, help="Duplicate Hamming distance (default: 8)")
    parser.add_argument("--nsfw-threshold", type=float, default=0.3, help="NSFW score threshold (default: 0.3)")
    parser.add_argument("--hash-algorithm", type=str, default="phash",
                       choices=['phash', 'dhash', 'average_hash'], help="Hash algorithm (default: phash)")

    # Enable/disable filters
    parser.add_argument("--disable-blur", action="store_true", help="Disable blur detection")
    parser.add_argument("--disable-duplicates", action="store_true", help="Disable duplicate detection")
    parser.add_argument("--disable-nsfw", action="store_true", help="Disable NSFW filtering")

    # Processing
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size (default: 100)")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Checkpoint interval (default: 500)")
    parser.add_argument("--device", type=str, default="cuda", choices=['cuda', 'cpu'], help="Device (default: cuda)")

    args = parser.parse_args()

    # Create config
    config = FilterConfig(
        blur_threshold=args.blur_threshold,
        enable_blur_detection=not args.disable_blur,
        duplicate_threshold=args.duplicate_threshold,
        hash_algorithm=args.hash_algorithm,
        enable_duplicate_detection=not args.disable_duplicates,
        nsfw_threshold=args.nsfw_threshold,
        enable_nsfw_filtering=not args.disable_nsfw,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval
    )

    # Initialize filter
    filter_system = ImageQualityFilter(
        config=config,
        checkpoint_dir=Path(args.checkpoint_dir),
        device=args.device
    )

    # Run filtering
    report = filter_system.filter_batch(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir)
    )

    print(f"✅ Report saved to {args.output_dir}/filtering_report.json")
    print(f"✅ Metrics saved to {args.output_dir}/image_metrics.json")


if __name__ == "__main__":
    main()
