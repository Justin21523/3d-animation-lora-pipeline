#!/usr/bin/env python3
"""
Synthetic Quality Filter Pipeline

Three-stage quality filtering for synthetic training images (optimized for multiprocessing):
1. Blur Detection (Laplacian variance)
2. Size Validation (resolution check)
3. Deduplication (perceptual hash)

NOTE: Character Presence (YOLOv8) removed to avoid model loading in each worker.
      Synthetic images are pre-filtered for character presence during generation.

Quality Tiers:
- Tier A (Premium): Pass all + high blur score (>120)
- Tier B (Good): Pass all + medium blur (80-120)
- Tier C (Rejected): Fail any stage

Author: LLMProvider Tooling
Date: 2025-12-04 (Updated: removed Stage 4 for multiprocessing optimization)
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

# Import existing filters
import sys
sys.path.append(str(Path(__file__).parent))

from blur_filter import BlurFilter
from size_filter import SizeFilter
from perceptual_hash_deduplicator import PerceptualHashDeduplicator
# CharacterPresenceFilter removed for multiprocessing optimization


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global worker function for multiprocessing (must be at module level for pickle)
def _filter_image_worker(args):
    """Worker function for parallel 2-stage image filtering (blur, size).

    Deduplication must happen in the parent process so we can keep shared state
    (otherwise multiprocessing workers would each have an empty hash set and the
    dedup stage becomes a no-op).
    """
    img_path, blur_threshold, blur_premium_threshold, device = args

    # Import filters inside worker to avoid pickle issues
    from blur_filter import BlurFilter
    from size_filter import SizeFilter

    # Initialize filters (no YOLOv8 to avoid memory overhead)
    blur_filter = BlurFilter(config={'threshold': blur_threshold, 'method': 'laplacian'})
    size_filter = SizeFilter(config={'min_resolution': 768, 'max_resolution': 1280})

    try:
        # Load image
        img_pil = Image.open(img_path)
        img = np.array(img_pil)

        # Stage 1: Blur
        blur_variance = blur_filter.compute_laplacian_variance(img)
        passed_blur, blur_reason = blur_filter.filter_single(img)
        if not passed_blur:
            return ('C', blur_reason, blur_variance, img_path)

        # Stage 2: Size
        passed_size, size_reason = size_filter.filter_single(img)
        if not passed_size:
            return (False, size_reason, blur_variance, img_path)

        # Passed blur + size
        return (True, None, blur_variance, img_path)

    except Exception as e:
        logger.error(f"Error filtering {img_path}: {e}")
        return (False, f"Error: {str(e)}", None, img_path)


class SyntheticQualityPipeline:
    """
    Multi-stage quality filtering pipeline for synthetic training data.

    Processes images through 3 stages (blur, size, dedup) and categorizes into quality tiers.
    Stage 4 (character presence) removed for multiprocessing optimization.
    """

    def __init__(
        self,
        blur_threshold: float = 80.0,
        blur_premium_threshold: float = 120.0,
        min_resolution: int = 768,
        max_resolution: int = 1280,
        target_resolution: int = 1024,
        dedup_hamming_threshold: int = 8,
        character_confidence: float = 0.5,
        device: str = "cuda",
        num_workers: int = None
    ):
        """
        Initialize quality pipeline.

        Args:
            blur_threshold: Laplacian variance threshold (reject below)
            blur_premium_threshold: Premium quality threshold
            min_resolution: Minimum acceptable resolution
            max_resolution: Maximum resolution before resize
            target_resolution: Target output resolution
            dedup_hamming_threshold: Hamming distance for dedup
            character_confidence: YOLOv8 confidence threshold
            device: Device for models (cuda/cpu)
        """
        self.blur_threshold = blur_threshold
        self.blur_premium_threshold = blur_premium_threshold
        self.target_resolution = target_resolution
        self.num_workers = num_workers or cpu_count()

        logger.info("Initializing Quality Filter Pipeline...")
        logger.info(f"Using {self.num_workers} parallel workers")

        # Stage 1: Blur Detection
        self.blur_filter = BlurFilter(config={
            'threshold': blur_threshold,
            'method': 'laplacian'
        })
        logger.info(f"  ✓ Stage 1: Blur Filter (threshold: {blur_threshold})")

        # Stage 2: Size Validation
        self.size_filter = SizeFilter(config={
            'min_resolution': min_resolution,
            'max_resolution': max_resolution
        })
        logger.info(f"  ✓ Stage 2: Size Filter ({min_resolution}-{max_resolution}px)")

        # Stage 3: Deduplication
        self.dedup_filter = PerceptualHashDeduplicator(config={
            'hash_method': 'perceptual',
            'hash_size': 16,
            'hamming_threshold': dedup_hamming_threshold
        })
        logger.info(f"  ✓ Stage 3: Deduplication (Hamming > {dedup_hamming_threshold})")

        # NOTE: Stage 4 (Character Presence with YOLOv8) removed for multiprocessing optimization
        # Synthetic images are pre-verified for character presence during generation

        # Statistics
        self.stats = {
            'total_processed': 0,
            'tier_a': 0,
            'tier_b': 0,
            'tier_c_rejected': 0,
            'rejection_reasons': defaultdict(int)
        }

        logger.info("✓ Quality Pipeline initialized\n")

    def filter_single_image(
        self,
        image_path: Path,
        caption_path: Optional[Path] = None
    ) -> Tuple[str, Optional[str], Optional[float]]:
        """
        Filter a single image through all stages.

        Args:
            image_path: Path to image
            caption_path: Path to caption (optional, will copy if passed)

        Returns:
            (tier, rejection_reason, blur_score)
            tier: 'A', 'B', or 'C'
        """
        self.stats['total_processed'] += 1

        try:
            # Load image
            img_pil = Image.open(image_path)
            img = np.array(img_pil)

            # Stage 1: Blur Detection
            # Get blur score for tier assignment
            blur_variance = self.blur_filter.compute_laplacian_variance(img)

            passed_blur, blur_reason = self.blur_filter.filter_single(img)

            if not passed_blur:
                self.stats['tier_c_rejected'] += 1
                self.stats['rejection_reasons']['blur'] += 1
                return 'C', blur_reason, blur_variance

            # Stage 2: Size Validation
            passed_size, size_reason = self.size_filter.filter_single(img)

            if not passed_size:
                self.stats['tier_c_rejected'] += 1
                self.stats['rejection_reasons']['size'] += 1
                return 'C', size_reason, blur_variance

            # Stage 3: Deduplication
            passed_dedup, dedup_reason = self.dedup_filter.filter_single(img)

            if not passed_dedup:
                self.stats['tier_c_rejected'] += 1
                self.stats['rejection_reasons']['duplicate'] += 1
                return 'C', dedup_reason, blur_variance

            # NOTE: Stage 4 (Character Presence) removed for multiprocessing optimization
            # Synthetic images are pre-verified during generation

            # Passed all 3 stages - assign tier based on blur score
            if blur_variance >= self.blur_premium_threshold:
                tier = 'A'
                self.stats['tier_a'] += 1
            else:
                tier = 'B'
                self.stats['tier_b'] += 1

            return tier, None, blur_variance

        except Exception as e:
            logger.error(f"Error filtering {image_path}: {e}")
            self.stats['tier_c_rejected'] += 1
            self.stats['rejection_reasons']['error'] += 1
            return 'C', f"Processing error: {str(e)}", None

    def process_character_type(
        self,
        input_dir: Path,
        output_dir: Path,
        character: str,
        lora_type: str,
        copy_files: bool = True
    ) -> Dict:
        """
        Process all images for a character/type combination.

        Args:
            input_dir: Input directory with captioned images
            output_dir: Output directory for filtered images
            character: Character name
            lora_type: LoRA type
            copy_files: Copy files to tier directories

        Returns:
            Processing statistics
        """
        logger.info(f"\nProcessing {character} {lora_type}...")

        # Input paths (check for /generated subdirectory first)
        char_input_dir = input_dir / character / lora_type
        if (char_input_dir / "generated").exists():
            char_input_dir = char_input_dir / "generated"
        # Support synthetic round-robin generator layout: {character}/{lora_type}/images/*.png
        elif (char_input_dir / "images").exists():
            char_input_dir = char_input_dir / "images"

        if not char_input_dir.exists():
            logger.warning(f"Input directory not found: {char_input_dir}")
            return {}

        # Output paths
        char_output_dir = output_dir / character / lora_type
        tier_a_dir = char_output_dir / "tier_a"
        tier_b_dir = char_output_dir / "tier_b"
        tier_c_dir = char_output_dir / "tier_c"

        for tier_dir in [tier_a_dir, tier_b_dir, tier_c_dir]:
            tier_dir.mkdir(parents=True, exist_ok=True)

        # Get all images
        images = sorted(char_input_dir.glob("*.png"))

        if len(images) == 0:
            logger.warning(f"No images found in {char_input_dir}")
            return {}

        # Process each image in parallel
        # Prepare arguments for worker function
        worker_args = [(img, self.blur_threshold, self.blur_premium_threshold, 'cpu') for img in images]

        # Use multiprocessing pool
        with Pool(processes=self.num_workers) as pool:
            raw_results = list(tqdm(
                pool.imap(_filter_image_worker, worker_args),
                total=len(images),
                desc=f"{character} {lora_type}"
            ))

        # Convert results to dict format
        results = []
        for passed, reason, blur_score, img_path in raw_results:
            results.append({
                'image': img_path.name,
                'passed': bool(passed),
                'reason': reason,
                'blur_score': blur_score,
                'img_path': img_path,  # Keep Path object for file operations
                'caption_path': img_path.with_suffix('.txt') if img_path.with_suffix('.txt').exists() else None  # Keep Path object
            })

        # Dedup must be done with shared state (per character/type)
        self.dedup_filter.reset()
        passed_results = [r for r in results if r["passed"] and r["blur_score"] is not None]

        # Prefer keeping higher-quality (sharper) images when duplicates exist
        passed_results.sort(key=lambda r: float(r["blur_score"]), reverse=True)

        for r in passed_results:
            ok, dedup_reason = self.dedup_filter.filter_single(r["img_path"])
            if not ok:
                r["passed"] = False
                r["reason"] = dedup_reason or "Duplicate"

        # Assign final tiers
        for r in results:
            if not r["passed"]:
                r["tier"] = "C"
                continue
            blur_score = float(r["blur_score"] or 0.0)
            r["tier"] = "A" if blur_score >= self.blur_premium_threshold else "B"

        # Copy files after processing
        for result in results:
            tier = result['tier']
            img_path = result['img_path']
            caption_path = result['caption_path']

            # Copy to tier directory if requested
            if copy_files and tier in ['A', 'B']:
                target_dir = tier_a_dir if tier == 'A' else tier_b_dir

                # Copy image
                shutil.copy2(img_path, target_dir / img_path.name)

                # Copy caption if exists
                if caption_path and caption_path.exists():
                    shutil.copy2(caption_path, target_dir / caption_path.name)

            elif copy_files and tier == 'C':
                # Optionally copy rejected for debugging
                shutil.copy2(img_path, tier_c_dir / img_path.name)

        # Generate report (with JSON-safe results)
        json_safe_results = []
        for r in results:
            json_safe_results.append({
                'image': r['image'],
                'tier': r['tier'],
                'reason': r['reason'],
                'blur_score': r['blur_score']
                # Omit img_path and caption_path (Path objects)
            })

        report = {
            'character': character,
            'lora_type': lora_type,
            'total_images': len(images),
            'tier_a': sum(1 for r in results if r['tier'] == 'A'),
            'tier_b': sum(1 for r in results if r['tier'] == 'B'),
            'tier_c': sum(1 for r in results if r['tier'] == 'C'),
            'avg_blur_score': sum(r['blur_score'] for r in results if r['blur_score']) / len(results),
            'results': json_safe_results
        }

        # Save report
        report_path = char_output_dir / "filtering_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(
            f"  ✓ {character} {lora_type}: "
            f"A={report['tier_a']}, B={report['tier_b']}, C={report['tier_c']} "
            f"(avg blur: {report['avg_blur_score']:.1f})"
        )

        return report

    def process_all_characters(
        self,
        input_dir: Path,
        output_dir: Path,
        characters: List[str],
        lora_types: List[str],
        copy_files: bool = True
    ):
        """
        Process all characters and LoRA types.

        Args:
            input_dir: Input directory with captioned images
            output_dir: Output directory for filtered images
            characters: List of character names
            lora_types: List of LoRA types
            copy_files: Copy files to tier directories
        """
        logger.info(f"\n{'='*60}")
        logger.info("QUALITY FILTERING - ALL CHARACTERS")
        logger.info(f"{'='*60}\n")

        all_reports = {}

        for lora_type in lora_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {lora_type.upper()}")
            logger.info(f"{'='*60}")

            for character in characters:
                report = self.process_character_type(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    character=character,
                    lora_type=lora_type,
                    copy_files=copy_files
                )

                if report:
                    all_reports[f"{character}_{lora_type}"] = report

        # Save overall statistics
        self._save_overall_stats(output_dir, all_reports)

        # Print summary
        self._print_summary(all_reports)

    def _save_overall_stats(self, output_dir: Path, reports: Dict):
        """Save overall filtering statistics."""
        stats_file = output_dir / "overall_filtering_statistics.json"

        overall = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_stats': self.stats,
            'per_character_reports': reports
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(overall, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✓ Overall statistics saved to {stats_file}")

    def _print_summary(self, reports: Dict):
        """Print filtering summary."""
        logger.info(f"\n{'='*60}")
        logger.info("QUALITY FILTERING SUMMARY")
        logger.info(f"{'='*60}\n")

        total_images = sum(r['total_images'] for r in reports.values())
        total_tier_a = sum(r['tier_a'] for r in reports.values())
        total_tier_b = sum(r['tier_b'] for r in reports.values())
        total_tier_c = sum(r['tier_c'] for r in reports.values())

        logger.info(f"Total images processed: {total_images}")
        logger.info(f"Tier A (Premium): {total_tier_a} ({total_tier_a/total_images*100:.1f}%)")
        logger.info(f"Tier B (Good): {total_tier_b} ({total_tier_b/total_images*100:.1f}%)")
        logger.info(f"Tier C (Rejected): {total_tier_c} ({total_tier_c/total_images*100:.1f}%)")
        logger.info(f"\nPass rate: {(total_tier_a + total_tier_b)/total_images*100:.1f}%")

        logger.info(f"\nRejection breakdown:")
        for reason, count in self.stats['rejection_reasons'].items():
            logger.info(f"  {reason}: {count} ({count/total_images*100:.1f}%)")

        logger.info(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Quality filter pipeline for synthetic training data"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default='/mnt/data/ai_data/synthetic_lora_data/captioned_data',
        help='Input directory with captioned images'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='/mnt/data/ai_data/synthetic_lora_data/filtered_data',
        help='Output directory for filtered images'
    )
    parser.add_argument(
        '--characters',
        nargs='+',
        default=['alberto', 'bryce', 'caleb', 'elio', 'giulia', 'ian_lightfoot',
                 'luca', 'miguel', 'orion', 'russell', 'tyler',
                 'alberto_seamonster', 'luca_seamonster', 'barley_lightfoot'],
        help='Character names to process'
    )
    parser.add_argument(
        '--lora-types',
        nargs='+',
        default=['pose', 'action', 'expression'],
        help='LoRA types to process'
    )
    parser.add_argument(
        '--blur-threshold',
        type=float,
        default=80.0,
        help='Blur detection threshold (3D optimized)'
    )
    parser.add_argument(
        '--blur-premium',
        type=float,
        default=120.0,
        help='Premium quality blur threshold'
    )
    parser.add_argument(
        '--no-copy',
        action='store_true',
        help='Do not copy files (report only)'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device for models (cuda/cpu)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='Number of parallel workers (default: 16, use 0 for CPU count)'
    )

    args = parser.parse_args()

    # Initialize pipeline
    num_workers = args.num_workers if args.num_workers > 0 else None
    pipeline = SyntheticQualityPipeline(
        blur_threshold=args.blur_threshold,
        blur_premium_threshold=args.blur_premium,
        device=args.device,
        num_workers=num_workers
    )

    # Process all characters
    pipeline.process_all_characters(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        characters=args.characters,
        lora_types=args.lora_types,
        copy_files=not args.no_copy
    )

    logger.info("✓ Quality filtering complete!")


if __name__ == '__main__':
    main()
