#!/usr/bin/env python3
"""
Smart Sampling for Caption Review

Intelligently samples images from filtered dataset for manual caption review.
Ensures balanced representation across:
- Characters
- LoRA types (pose/action/expression)
- Quality tiers (A/B)
- Blur score distribution

Usage:
    python sample_for_review.py \\
        --input-dir /path/to/filtered_data \\
        --output-dir /path/to/review_samples \\
        --target-samples 600 \\
        --sampling-strategy balanced

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import argparse
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm


@dataclass
class SampleConfig:
    """Configuration for sampling strategy"""
    target_samples: int = 600
    min_per_character: int = 20
    max_per_character: int = 100
    samples_per_type: int = 15  # Per character per LoRA type
    tier_ratio: Tuple[float, float] = (0.7, 0.3)  # Tier A:B ratio
    blur_score_bins: int = 5  # Number of blur score bins for stratified sampling
    random_seed: int = 42


@dataclass
class SampleResult:
    """Result of a single sampled image"""
    character: str
    lora_type: str
    tier: str
    image_path: str
    caption_path: str
    blur_score: float
    sample_id: int


class SmartSampler:
    """
    Intelligent sampler for caption review

    Ensures balanced coverage across all dimensions:
    - Character distribution
    - LoRA type distribution
    - Quality tier distribution
    - Blur score stratification
    """

    def __init__(self, config: SampleConfig):
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    def load_filtering_report(self, filtered_dir: Path) -> Dict:
        """Load overall filtering statistics"""
        stats_file = filtered_dir / "overall_filtering_statistics.json"
        if not stats_file.exists():
            raise FileNotFoundError(f"Filtering report not found: {stats_file}")

        with open(stats_file) as f:
            return json.load(f)

    def build_image_index(self, filtered_dir: Path, stats: Dict) -> Dict[str, List[Dict]]:
        """
        Build comprehensive index of all available images

        Returns:
            Dict mapping "{character}_{lora_type}" to list of image metadata
        """
        index = defaultdict(list)

        for key, report in tqdm(stats['per_character_reports'].items(), desc="Building index"):
            character = report['character']
            lora_type = report['lora_type']
            group_key = f"{character}_{lora_type}"

            # Get tier directories
            char_dir = filtered_dir / character / lora_type

            for tier in ['tier_a', 'tier_b']:
                tier_dir = char_dir / tier
                if not tier_dir.exists():
                    continue

                # Find all images in this tier
                for img_path in tier_dir.glob("*.png"):
                    caption_path = img_path.with_suffix('.txt')
                    if not caption_path.exists():
                        logging.warning(f"Missing caption: {caption_path}")
                        continue

                    # Find blur score from report
                    blur_score = None
                    for result in report.get('results', []):
                        if result['image'] == img_path.name:
                            blur_score = result['blur_score']
                            break

                    if blur_score is None:
                        logging.warning(f"Missing blur score for: {img_path}")
                        blur_score = 100.0  # Default

                    index[group_key].append({
                        'character': character,
                        'lora_type': lora_type,
                        'tier': tier,
                        'image_path': str(img_path),
                        'caption_path': str(caption_path),
                        'blur_score': blur_score
                    })

        return index

    def stratified_sample_by_blur(
        self,
        images: List[Dict],
        n_samples: int
    ) -> List[Dict]:
        """
        Stratified sampling by blur score

        Ensures samples cover the full blur score distribution
        """
        if len(images) <= n_samples:
            return images

        # Get blur scores
        blur_scores = np.array([img['blur_score'] for img in images])

        # Create bins
        percentiles = np.linspace(0, 100, self.config.blur_score_bins + 1)
        bins = np.percentile(blur_scores, percentiles)

        # Assign each image to a bin
        bin_assignments = np.digitize(blur_scores, bins[1:-1])

        # Sample proportionally from each bin
        samples_per_bin = max(1, n_samples // self.config.blur_score_bins)
        sampled = []

        for bin_idx in range(self.config.blur_score_bins):
            bin_images = [img for i, img in enumerate(images)
                         if bin_assignments[i] == bin_idx]

            if not bin_images:
                continue

            # Sample from this bin
            k = min(samples_per_bin, len(bin_images))
            sampled.extend(random.sample(bin_images, k))

        # If we need more samples, randomly sample the remainder
        if len(sampled) < n_samples:
            remaining = [img for img in images if img not in sampled]
            k = min(n_samples - len(sampled), len(remaining))
            sampled.extend(random.sample(remaining, k))

        # If we have too many, randomly drop some
        if len(sampled) > n_samples:
            sampled = random.sample(sampled, n_samples)

        return sampled

    def balanced_sampling(
        self,
        index: Dict[str, List[Dict]]
    ) -> List[SampleResult]:
        """
        Balanced sampling strategy

        Ensures equal representation across characters and LoRA types
        """
        # Count groups
        n_groups = len(index)
        samples_per_group = max(
            self.config.min_per_character // 3,  # Divide by 3 types
            self.config.target_samples // n_groups
        )

        logging.info(f"Sampling {samples_per_group} images per group ({n_groups} groups)")

        all_samples = []
        sample_id = 0

        for group_key, images in tqdm(index.items(), desc="Sampling"):
            # Separate by tier
            tier_a = [img for img in images if img['tier'] == 'tier_a']
            tier_b = [img for img in images if img['tier'] == 'tier_b']

            # Calculate tier samples
            n_tier_a = int(samples_per_group * self.config.tier_ratio[0])
            n_tier_b = samples_per_group - n_tier_a

            # Stratified sample from each tier
            sampled_a = self.stratified_sample_by_blur(tier_a, n_tier_a)
            sampled_b = self.stratified_sample_by_blur(tier_b, n_tier_b)

            # Combine and create results
            for img in sampled_a + sampled_b:
                all_samples.append(SampleResult(
                    character=img['character'],
                    lora_type=img['lora_type'],
                    tier=img['tier'],
                    image_path=img['image_path'],
                    caption_path=img['caption_path'],
                    blur_score=img['blur_score'],
                    sample_id=sample_id
                ))
                sample_id += 1

        # Shuffle for review
        random.shuffle(all_samples)

        # Limit to target
        if len(all_samples) > self.config.target_samples:
            all_samples = all_samples[:self.config.target_samples]

        return all_samples

    def copy_samples(
        self,
        samples: List[SampleResult],
        output_dir: Path
    ) -> None:
        """Copy sampled images and captions to output directory"""
        output_dir.mkdir(parents=True, exist_ok=True)

        for sample in tqdm(samples, desc="Copying samples"):
            # Create output paths
            img_out = output_dir / f"sample_{sample.sample_id:04d}.png"
            cap_out = output_dir / f"sample_{sample.sample_id:04d}.txt"
            meta_out = output_dir / f"sample_{sample.sample_id:04d}.json"

            # Copy files
            shutil.copy2(sample.image_path, img_out)
            shutil.copy2(sample.caption_path, cap_out)

            # Save metadata
            with open(meta_out, 'w') as f:
                json.dump(asdict(sample), f, indent=2)

    def generate_report(
        self,
        samples: List[SampleResult],
        output_dir: Path
    ) -> None:
        """Generate sampling report"""
        # Overall stats
        stats = {
            'total_samples': len(samples),
            'by_character': defaultdict(int),
            'by_lora_type': defaultdict(int),
            'by_tier': defaultdict(int),
            'blur_score_distribution': {
                'min': min(s.blur_score for s in samples),
                'max': max(s.blur_score for s in samples),
                'mean': np.mean([s.blur_score for s in samples]),
                'median': np.median([s.blur_score for s in samples]),
                'std': np.std([s.blur_score for s in samples])
            }
        }

        for sample in samples:
            stats['by_character'][sample.character] += 1
            stats['by_lora_type'][sample.lora_type] += 1
            stats['by_tier'][sample.tier] += 1

        # Convert defaultdicts to regular dicts
        stats['by_character'] = dict(stats['by_character'])
        stats['by_lora_type'] = dict(stats['by_lora_type'])
        stats['by_tier'] = dict(stats['by_tier'])

        # Save report
        report_path = output_dir / "sampling_report.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logging.info(f"\n{'='*60}")
        logging.info("Sampling Report")
        logging.info(f"{'='*60}")
        logging.info(f"Total samples: {stats['total_samples']}")
        logging.info(f"\nBy character:")
        for char, count in sorted(stats['by_character'].items()):
            logging.info(f"  {char}: {count}")
        logging.info(f"\nBy LoRA type:")
        for lora_type, count in sorted(stats['by_lora_type'].items()):
            logging.info(f"  {lora_type}: {count}")
        logging.info(f"\nBy tier:")
        for tier, count in sorted(stats['by_tier'].items()):
            logging.info(f"  {tier}: {count}")
        logging.info(f"\nBlur score distribution:")
        logging.info(f"  Min: {stats['blur_score_distribution']['min']:.2f}")
        logging.info(f"  Max: {stats['blur_score_distribution']['max']:.2f}")
        logging.info(f"  Mean: {stats['blur_score_distribution']['mean']:.2f}")
        logging.info(f"  Median: {stats['blur_score_distribution']['median']:.2f}")
        logging.info(f"  Std: {stats['blur_score_distribution']['std']:.2f}")
        logging.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Smart sampling for caption review",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help="Input directory with filtered data"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help="Output directory for review samples"
    )
    parser.add_argument(
        '--target-samples',
        type=int,
        default=600,
        help="Target number of samples (actual may vary slightly)"
    )
    parser.add_argument(
        '--min-per-character',
        type=int,
        default=20,
        help="Minimum samples per character"
    )
    parser.add_argument(
        '--tier-a-ratio',
        type=float,
        default=0.7,
        help="Ratio of Tier A samples (0.0-1.0)"
    )
    parser.add_argument(
        '--blur-bins',
        type=int,
        default=5,
        help="Number of blur score bins for stratified sampling"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create config
    config = SampleConfig(
        target_samples=args.target_samples,
        min_per_character=args.min_per_character,
        tier_ratio=(args.tier_a_ratio, 1.0 - args.tier_a_ratio),
        blur_score_bins=args.blur_bins,
        random_seed=args.seed
    )

    # Initialize sampler
    sampler = SmartSampler(config)

    # Load filtering report
    logging.info("Loading filtering statistics...")
    stats = sampler.load_filtering_report(args.input_dir)

    # Build image index
    logging.info("Building image index...")
    index = sampler.build_image_index(args.input_dir, stats)
    logging.info(f"Indexed {sum(len(v) for v in index.values())} images across {len(index)} groups")

    # Perform balanced sampling
    logging.info("Performing balanced sampling...")
    samples = sampler.balanced_sampling(index)
    logging.info(f"Sampled {len(samples)} images")

    # Copy samples to output
    logging.info("Copying samples to output directory...")
    sampler.copy_samples(samples, args.output_dir)

    # Generate report
    sampler.generate_report(samples, args.output_dir)

    logging.info(f"\n✅ Sampling complete!")
    logging.info(f"Review samples saved to: {args.output_dir}")
    logging.info(f"Next step: Run interactive caption review UI")


if __name__ == '__main__':
    main()
