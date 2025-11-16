#!/usr/bin/env python3
"""
Intelligent Dataset Curator

Selects the BEST N images from a large candidate pool for LoRA training.

Quality Criteria:
- Image quality (sharpness, lighting, resolution)
- Diversity (pose, view angle, background complexity)
- Deduplication (remove near-duplicates)
- Balance (ensure even distribution across categories)

Typical Usage:
    # From 4400 candidates → Select best 400 for training
    python intelligent_dataset_curator.py \
      candidate_pool/ \
      --output-dir final_training_set/ \
      --target-size 400 \
      --ensure-diversity
"""

import argparse
import cv2
import json
import numpy as np
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
import yaml

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.frame_decision_engine import FrameDecisionEngine


@dataclass
class ImageScore:
    """Quality and diversity scores for an image"""
    path: Path

    # Quality scores (0-1, higher = better)
    sharpness: float
    lighting: float
    resolution: float
    overall_quality: float

    # Diversity metrics
    complexity: float  # Background complexity
    occlusion: float   # Occlusion level

    # Metadata
    source_strategy: str  # keep_full, segment, create_occlusion, enhance_segment
    category: str         # close-up, medium, full-body, etc.

    # Final score (weighted combination)
    final_score: float


class DiversitySampler:
    """Ensures diverse sampling across categories"""

    def __init__(self, target_distribution: Optional[Dict] = None):
        """
        Initialize sampler

        Args:
            target_distribution: Desired category distribution
                e.g., {'close-up': 0.3, 'medium': 0.3, 'full-body': 0.4}
        """
        self.target_distribution = target_distribution or {
            'close-up': 0.30,
            'medium': 0.25,
            'full-body': 0.25,
            'far-shot': 0.10,
            'other': 0.10
        }

    def categorize_image(self, img_path: Path, analysis: 'FrameAnalysis') -> str:
        """
        Categorize image by shot type

        Args:
            img_path: Image path
            analysis: Frame analysis

        Returns:
            Category name
        """
        # Read image to check dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            return 'other'

        h, w = img.shape[:2]
        aspect = w / h if h > 0 else 1.0

        # Analyze caption if available
        caption_path = img_path.with_suffix('.txt')
        caption = ""
        if caption_path.exists():
            caption = caption_path.read_text().lower()

        # Category detection
        if 'close' in caption or 'portrait' in caption or 'face' in caption:
            return 'close-up'
        elif 'full body' in caption or 'full-body' in caption or 'standing' in caption:
            return 'full-body'
        elif 'medium' in caption or 'waist' in caption:
            return 'medium'
        elif 'far' in caption or 'distant' in caption or 'wide' in caption:
            return 'far-shot'

        # Heuristic based on image analysis
        # High complexity + low quality often means far shot
        if analysis.complexity > 0.6 and analysis.quality_score < 0.6:
            return 'far-shot'
        # High quality + simple background often means close-up
        elif analysis.quality_score > 0.7 and analysis.complexity < 0.3:
            return 'close-up'
        # Medium complexity = medium shot
        elif 0.3 < analysis.complexity < 0.6:
            return 'medium'
        else:
            return 'full-body'

    def balance_selection(
        self,
        scored_images: List[ImageScore],
        target_count: int
    ) -> List[ImageScore]:
        """
        Select images with balanced category distribution

        Args:
            scored_images: All scored images
            target_count: Desired final count

        Returns:
            Balanced selection
        """
        # Group by category
        by_category = {}
        for img in scored_images:
            cat = img.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(img)

        # Sort each category by score
        for cat in by_category:
            by_category[cat].sort(key=lambda x: x.final_score, reverse=True)

        # Calculate target counts per category
        target_counts = {}
        for cat, ratio in self.target_distribution.items():
            if cat in by_category:
                target_counts[cat] = int(target_count * ratio)

        # Adjust for categories with fewer images than target
        selected = []
        remaining_count = target_count

        for cat, target in target_counts.items():
            available = len(by_category.get(cat, []))
            take = min(target, available)

            selected.extend(by_category[cat][:take])
            remaining_count -= take

        # Fill remaining slots with highest-scored images
        if remaining_count > 0:
            all_remaining = []
            for cat, images in by_category.items():
                taken = target_counts.get(cat, 0)
                all_remaining.extend(images[taken:])

            all_remaining.sort(key=lambda x: x.final_score, reverse=True)
            selected.extend(all_remaining[:remaining_count])

        return selected


class ImageDeduplicator:
    """Remove near-duplicate images"""

    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize deduplicator

        Args:
            similarity_threshold: Images above this similarity are duplicates
        """
        self.similarity_threshold = similarity_threshold

    def compute_hash(self, img_path: Path) -> str:
        """Compute perceptual hash"""
        import hashlib

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""

        # Resize to 8x8
        resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)

        # Compute average
        avg = resized.mean()

        # Generate hash
        hash_str = ""
        for pixel in resized.flatten():
            hash_str += "1" if pixel > avg else "0"

        return hash_str

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Compute Hamming distance between hashes"""
        if len(hash1) != len(hash2):
            return 64  # Max distance

        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def deduplicate(
        self,
        images: List[ImageScore],
        keep_highest_score: bool = True
    ) -> List[ImageScore]:
        """
        Remove near-duplicates

        Args:
            images: List of scored images
            keep_highest_score: Keep image with highest score among duplicates

        Returns:
            Deduplicated list
        """
        # Compute hashes
        print("🔍 Computing perceptual hashes...")
        hashes = {}
        for img in tqdm(images):
            hashes[img.path] = self.compute_hash(img.path)

        # Find duplicates
        print("🔍 Finding duplicates...")
        duplicates = set()

        for i, img1 in enumerate(tqdm(images)):
            if img1.path in duplicates:
                continue

            for img2 in images[i+1:]:
                if img2.path in duplicates:
                    continue

                # Compute similarity
                dist = self.hamming_distance(hashes[img1.path], hashes[img2.path])
                similarity = 1.0 - (dist / 64.0)

                if similarity >= self.similarity_threshold:
                    # Mark lower-scored image as duplicate
                    if keep_highest_score:
                        if img1.final_score >= img2.final_score:
                            duplicates.add(img2.path)
                        else:
                            duplicates.add(img1.path)
                    else:
                        duplicates.add(img2.path)

        # Remove duplicates
        result = [img for img in images if img.path not in duplicates]

        print(f"✓ Removed {len(duplicates)} near-duplicates")

        return result


class IntelligentDatasetCurator:
    """Main curator orchestrator"""

    def __init__(
        self,
        decision_engine: Optional[FrameDecisionEngine] = None,
        diversity_sampler: Optional[DiversitySampler] = None,
        deduplicator: Optional[ImageDeduplicator] = None
    ):
        """
        Initialize curator

        Args:
            decision_engine: Frame analyzer
            diversity_sampler: Category balancer
            deduplicator: Duplicate remover
        """
        self.decision_engine = decision_engine or FrameDecisionEngine()
        self.diversity_sampler = diversity_sampler or DiversitySampler()
        self.deduplicator = deduplicator or ImageDeduplicator()

    def score_image(self, img_path: Path) -> ImageScore:
        """
        Score a single image

        Args:
            img_path: Image path

        Returns:
            ImageScore object
        """
        # Analyze frame
        analysis = self.decision_engine.analyze_frame(img_path)

        # Quality scores
        quality_score = (
            analysis.quality_score * 0.4 +
            analysis.sharpness * 0.3 +
            analysis.lighting_quality * 0.2 +
            analysis.contrast * 0.1
        )

        # Determine source strategy from path
        source_strategy = "unknown"
        if 'keep_full' in str(img_path):
            source_strategy = 'keep_full'
        elif 'segment' in str(img_path):
            source_strategy = 'segment'
        elif 'occlusion' in str(img_path):
            source_strategy = 'create_occlusion'
        elif 'enhance' in str(img_path):
            source_strategy = 'enhance_segment'

        # Categorize
        category = self.diversity_sampler.categorize_image(img_path, analysis)

        # Diversity bonus for underrepresented categories
        diversity_bonus = 0.0
        if category in ['far-shot', 'medium']:
            diversity_bonus = 0.1  # Boost these categories

        # Final score
        final_score = quality_score + diversity_bonus

        return ImageScore(
            path=img_path,
            sharpness=analysis.sharpness,
            lighting=analysis.lighting_quality,
            resolution=1.0,  # Placeholder
            overall_quality=quality_score,
            complexity=analysis.complexity,
            occlusion=analysis.occlusion_level,
            source_strategy=source_strategy,
            category=category,
            final_score=final_score
        )

    def curate(
        self,
        candidate_dir: Path,
        output_dir: Path,
        target_size: int = 400,
        enable_dedup: bool = True,
        ensure_diversity: bool = True
    ) -> Dict:
        """
        Curate dataset from candidates

        Args:
            candidate_dir: Directory with candidate images
            output_dir: Output directory
            target_size: Target dataset size
            enable_dedup: Remove duplicates
            ensure_diversity: Balance categories

        Returns:
            Curation report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all candidate images
        print(f"\n🔍 Scanning candidates in {candidate_dir}...")

        image_extensions = {'.png', '.jpg', '.jpeg'}
        candidates = []

        # Search in all strategy subdirectories
        for strategy_dir in candidate_dir.iterdir():
            if strategy_dir.is_dir():
                images_dir = strategy_dir / 'images'
                if images_dir.exists():
                    candidates.extend([
                        p for p in images_dir.iterdir()
                        if p.suffix.lower() in image_extensions
                    ])

        print(f"✓ Found {len(candidates)} candidate images")

        if len(candidates) == 0:
            print("❌ No candidates found!")
            return {}

        # Score all images
        print(f"\n📊 Scoring images...")
        scored_images = []

        for img_path in tqdm(candidates):
            try:
                score = self.score_image(img_path)
                scored_images.append(score)
            except Exception as e:
                print(f"⚠️ Failed to score {img_path.name}: {e}")

        print(f"✓ Scored {len(scored_images)} images")

        # Deduplicate
        if enable_dedup:
            print(f"\n🔄 Deduplicating...")
            scored_images = self.deduplicator.deduplicate(scored_images)

        # Balance diversity
        if ensure_diversity:
            print(f"\n⚖️ Balancing category distribution...")
            selected = self.diversity_sampler.balance_selection(
                scored_images,
                target_size
            )
        else:
            # Just take top N by score
            scored_images.sort(key=lambda x: x.final_score, reverse=True)
            selected = scored_images[:target_size]

        print(f"✓ Selected {len(selected)} images")

        # Copy selected images + captions
        print(f"\n📦 Copying to {output_dir}...")

        images_dir = output_dir / 'images'
        captions_dir = output_dir / 'captions'
        images_dir.mkdir(exist_ok=True)
        captions_dir.mkdir(exist_ok=True)

        for img_score in tqdm(selected):
            # Copy image
            dst_img = images_dir / img_score.path.name
            shutil.copy2(img_score.path, dst_img)

            # Copy caption if exists
            caption_src = img_score.path.with_suffix('.txt')
            if caption_src.exists():
                dst_caption = captions_dir / caption_src.name
                shutil.copy2(caption_src, dst_caption)

        # Generate report
        report = self._generate_report(
            candidates=len(candidates),
            scored=len(scored_images),
            selected=len(selected),
            selected_images=selected
        )

        # Save report
        report_path = output_dir / 'curation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📊 Report saved: {report_path}")

        # Print summary
        self._print_summary(report)

        return report

    def _generate_report(
        self,
        candidates: int,
        scored: int,
        selected: int,
        selected_images: List[ImageScore]
    ) -> Dict:
        """Generate curation report"""
        # Category distribution
        by_category = {}
        for img in selected_images:
            cat = img.category
            by_category[cat] = by_category.get(cat, 0) + 1

        # Strategy distribution
        by_strategy = {}
        for img in selected_images:
            strat = img.source_strategy
            by_strategy[strat] = by_strategy.get(strat, 0) + 1

        # Quality stats
        qualities = [img.overall_quality for img in selected_images]

        return {
            'summary': {
                'candidates': candidates,
                'scored': scored,
                'selected': selected,
                'reduction_ratio': f"{selected/candidates:.1%}" if candidates > 0 else "N/A"
            },
            'category_distribution': by_category,
            'strategy_distribution': by_strategy,
            'quality_stats': {
                'mean': np.mean(qualities),
                'min': np.min(qualities),
                'max': np.max(qualities),
                'std': np.std(qualities)
            },
            'top_10_images': [
                {
                    'path': str(img.path),
                    'score': img.final_score,
                    'category': img.category,
                    'quality': img.overall_quality
                }
                for img in sorted(selected_images, key=lambda x: x.final_score, reverse=True)[:10]
            ]
        }

    def _print_summary(self, report: Dict):
        """Print curation summary"""
        summary = report['summary']

        print("\n" + "="*60)
        print("  📊 CURATION SUMMARY")
        print("="*60)
        print(f"\n✅ Candidates:  {summary['candidates']}")
        print(f"✅ Scored:      {summary['scored']}")
        print(f"✅ Selected:    {summary['selected']}")
        print(f"   Reduction:   {summary['reduction_ratio']}")

        print(f"\n📈 Category Distribution:")
        for cat, count in report['category_distribution'].items():
            pct = (count / summary['selected'] * 100) if summary['selected'] > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"   {cat:12s}: {count:3d} ({pct:5.1f}%) {bar}")

        print(f"\n📈 Strategy Distribution:")
        for strat, count in report['strategy_distribution'].items():
            pct = (count / summary['selected'] * 100) if summary['selected'] > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"   {strat:18s}: {count:3d} ({pct:5.1f}%) {bar}")

        print(f"\n📊 Quality Statistics:")
        stats = report['quality_stats']
        print(f"   Mean:  {stats['mean']:.3f}")
        print(f"   Range: {stats['min']:.3f} - {stats['max']:.3f}")
        print(f"   Std:   {stats['std']:.3f}")

        print("\n" + "="*60 + "\n")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Curate optimal training dataset from candidates"
    )

    parser.add_argument(
        'candidate_dir',
        type=Path,
        help='Directory with candidate images (intelligent processor output)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for curated dataset'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=400,
        help='Target dataset size (default: 400)'
    )
    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Disable deduplication'
    )
    parser.add_argument(
        '--no-diversity',
        action='store_true',
        help='Disable diversity balancing (just take top N)'
    )
    parser.add_argument(
        '--decision-config',
        type=Path,
        help='Path to decision thresholds config'
    )

    args = parser.parse_args()

    # Initialize curator
    decision_engine = None
    if args.decision_config and args.decision_config.exists():
        decision_engine = FrameDecisionEngine(args.decision_config)

    curator = IntelligentDatasetCurator(decision_engine=decision_engine)

    # Curate
    report = curator.curate(
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        enable_dedup=not args.no_dedup,
        ensure_diversity=not args.no_diversity
    )

    print(f"\n✅ Curation complete!")
    print(f"📁 Final dataset: {args.output_dir}")
    print(f"📊 {report['summary']['selected']} images ready for training")


if __name__ == "__main__":
    main()
