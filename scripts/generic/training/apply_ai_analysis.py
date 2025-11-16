#!/usr/bin/env python3
"""
Apply AI Deep Analysis Results
Use AI analysis data to curate training datasets, filter quality frames, and organize by scene/mood
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AIAnalysisApplicator:
    """Apply AI analysis results for practical dataset curation"""

    def __init__(self, analysis_json_path: Path):
        """
        Initialize with AI analysis results

        Args:
            analysis_json_path: Path to ai_deep_analysis.json
        """
        logger.info(f"Loading AI analysis from {analysis_json_path}")
        with open(analysis_json_path, 'r') as f:
            self.analysis_data = json.load(f)

        logger.info(f"Loaded analysis for {len(self.analysis_data)} frames")

        # Build indices for fast lookup
        self._build_indices()

    def _build_indices(self):
        """Build lookup indices for efficient filtering"""
        self.by_scene = defaultdict(list)
        self.by_mood = defaultdict(list)
        self.by_style = defaultdict(list)
        self.by_episode = defaultdict(list)

        for item in self.analysis_data:
            self.by_scene[item['scene_type']].append(item)
            self.by_mood[item['mood']].append(item)
            self.by_style[item['visual_style']].append(item)
            self.by_episode[item['episode']].append(item)

        logger.info(f"Built indices: {len(self.by_scene)} scenes, {len(self.by_mood)} moods, "
                   f"{len(self.by_style)} styles, {len(self.by_episode)} episodes")

    def filter_by_quality(
        self,
        min_quality: float = 0.3,
        min_aesthetic: float = 0.5,
        top_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Filter frames by quality and aesthetic scores

        Args:
            min_quality: Minimum quality_score (0-1)
            min_aesthetic: Minimum aesthetic_score (0-1)
            top_n: If specified, return only top N frames by combined score

        Returns:
            List of filtered frame data dictionaries
        """
        filtered = [
            item for item in self.analysis_data
            if item['quality_score'] >= min_quality and item['aesthetic_score'] >= min_aesthetic
        ]

        logger.info(f"Quality filter: {len(filtered)}/{len(self.analysis_data)} frames "
                   f"(quality≥{min_quality}, aesthetic≥{min_aesthetic})")

        if top_n:
            # Sort by combined score
            filtered.sort(key=lambda x: x['quality_score'] + x['aesthetic_score'], reverse=True)
            filtered = filtered[:top_n]
            logger.info(f"Selected top {top_n} frames")

        return filtered

    def filter_by_scene(
        self,
        scene_types: List[str],
        min_confidence: float = 0.4
    ) -> List[Dict]:
        """
        Filter frames by scene type

        Args:
            scene_types: List of scene types to include (e.g., ['battle scene', 'indoor scene'])
            min_confidence: Minimum scene classification confidence

        Returns:
            List of filtered frame data dictionaries
        """
        filtered = []
        for scene_type in scene_types:
            if scene_type in self.by_scene:
                scene_frames = [
                    item for item in self.by_scene[scene_type]
                    if item['scene_confidence'] >= min_confidence
                ]
                filtered.extend(scene_frames)

        logger.info(f"Scene filter: {len(filtered)} frames for scenes {scene_types} "
                   f"(confidence≥{min_confidence})")

        return filtered

    def filter_by_mood(
        self,
        moods: List[str],
        min_confidence: float = 0.4
    ) -> List[Dict]:
        """
        Filter frames by mood

        Args:
            moods: List of moods to include (e.g., ['energetic dynamic', 'exciting action'])
            min_confidence: Minimum mood classification confidence

        Returns:
            List of filtered frame data dictionaries
        """
        filtered = []
        for mood in moods:
            if mood in self.by_mood:
                mood_frames = [
                    item for item in self.by_mood[mood]
                    if item['mood_confidence'] >= min_confidence
                ]
                filtered.extend(mood_frames)

        logger.info(f"Mood filter: {len(filtered)} frames for moods {moods} "
                   f"(confidence≥{min_confidence})")

        return filtered

    def get_diverse_sample(
        self,
        n_samples: int,
        min_quality: float = 0.25,
        balance_by: str = 'scene'  # 'scene', 'mood', 'style', or 'episode'
    ) -> List[Dict]:
        """
        Get diverse sample of frames balanced across categories

        Args:
            n_samples: Total number of samples to retrieve
            min_quality: Minimum quality threshold
            balance_by: Category to balance by

        Returns:
            List of diverse frame data dictionaries
        """
        # Get index to balance by
        if balance_by == 'scene':
            index = self.by_scene
        elif balance_by == 'mood':
            index = self.by_mood
        elif balance_by == 'style':
            index = self.by_style
        elif balance_by == 'episode':
            index = self.by_episode
        else:
            raise ValueError(f"Unknown balance_by: {balance_by}")

        # Calculate samples per category
        n_categories = len(index)
        samples_per_category = n_samples // n_categories
        remainder = n_samples % n_categories

        selected = []

        for i, (category, items) in enumerate(sorted(index.items())):
            # Filter by quality
            quality_items = [
                item for item in items
                if item['quality_score'] >= min_quality
            ]

            if not quality_items:
                logger.warning(f"No quality frames in {balance_by}={category}")
                continue

            # Sort by combined score
            quality_items.sort(
                key=lambda x: x['quality_score'] + x['aesthetic_score'],
                reverse=True
            )

            # Get samples for this category
            n_take = samples_per_category + (1 if i < remainder else 0)
            n_take = min(n_take, len(quality_items))

            selected.extend(quality_items[:n_take])
            logger.info(f"  {category}: selected {n_take}/{len(quality_items)} frames")

        logger.info(f"Diverse sample: {len(selected)} frames balanced by {balance_by}")

        return selected

    def copy_selected_frames(
        self,
        selected_frames: List[Dict],
        output_dir: Path,
        source_root: Path,
        organize_by: Optional[str] = None,  # 'scene', 'mood', 'style', 'episode', or None
        copy_mode: str = 'copy'  # 'copy' or 'symlink'
    ):
        """
        Copy or symlink selected frames to output directory

        Args:
            selected_frames: List of frame data dictionaries
            output_dir: Output directory
            source_root: Root directory containing source frames
            organize_by: If specified, organize into subdirectories by this category
            copy_mode: 'copy' to copy files, 'symlink' to create symbolic links
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Copying {len(selected_frames)} frames to {output_dir}")
        logger.info(f"Mode: {copy_mode}, Organize by: {organize_by or 'flat'}")

        stats = defaultdict(int)

        for item in tqdm(selected_frames, desc="Copying frames"):
            # Construct source path
            # Files are stored in episode_xxx/character/ subdirectory
            source_path = source_root / item['episode'] / 'character' / item['frame_name']

            if not source_path.exists():
                logger.warning(f"Source not found: {source_path}")
                stats['not_found'] += 1
                continue

            # Determine destination
            if organize_by:
                category = item.get(organize_by, 'unknown')
                dest_dir = output_dir / category
                dest_dir.mkdir(parents=True, exist_ok=True)
            else:
                dest_dir = output_dir

            dest_path = dest_dir / item['frame_name']

            # Copy or symlink
            try:
                if copy_mode == 'copy':
                    shutil.copy2(source_path, dest_path)
                elif copy_mode == 'symlink':
                    if dest_path.exists():
                        dest_path.unlink()
                    dest_path.symlink_to(source_path.resolve())
                else:
                    raise ValueError(f"Unknown copy_mode: {copy_mode}")

                stats['copied'] += 1

            except Exception as e:
                logger.error(f"Failed to copy {source_path}: {e}")
                stats['failed'] += 1

        logger.info(f"Copy complete: {stats['copied']} copied, {stats['not_found']} not found, "
                   f"{stats['failed']} failed")

        return stats

    def generate_statistics_report(self, output_path: Path):
        """Generate comprehensive statistics report"""
        stats = {
            'total_frames': len(self.analysis_data),

            'quality_distribution': self._compute_distribution('quality_score', bins=10),
            'aesthetic_distribution': self._compute_distribution('aesthetic_score', bins=10),

            'scene_counts': {k: len(v) for k, v in self.by_scene.items()},
            'mood_counts': {k: len(v) for k, v in self.by_mood.items()},
            'style_counts': {k: len(v) for k, v in self.by_style.items()},
            'episode_counts': {k: len(v) for k, v in self.by_episode.items()},

            'avg_quality': sum(x['quality_score'] for x in self.analysis_data) / len(self.analysis_data),
            'avg_aesthetic': sum(x['aesthetic_score'] for x in self.analysis_data) / len(self.analysis_data),

            'high_quality_frames': len([x for x in self.analysis_data if x['quality_score'] >= 0.4]),
            'high_aesthetic_frames': len([x for x in self.analysis_data if x['aesthetic_score'] >= 0.6]),
        }

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics report saved to {output_path}")

        return stats

    def _compute_distribution(self, key: str, bins: int = 10) -> Dict:
        """Compute distribution histogram for a numeric field"""
        values = [item[key] for item in self.analysis_data]
        min_val = min(values)
        max_val = max(values)
        bin_width = (max_val - min_val) / bins

        histogram = defaultdict(int)
        for val in values:
            bin_idx = min(int((val - min_val) / bin_width), bins - 1)
            bin_label = f"{min_val + bin_idx * bin_width:.2f}-{min_val + (bin_idx + 1) * bin_width:.2f}"
            histogram[bin_label] += 1

        return dict(histogram)


def main():
    parser = argparse.ArgumentParser(description="Apply AI Deep Analysis Results")
    parser.add_argument('analysis_json', type=Path, help='Path to ai_deep_analysis.json')
    parser.add_argument('--source-root', type=Path, required=True,
                       help='Root directory containing source frames')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for curated dataset')

    # Filtering options
    parser.add_argument('--min-quality', type=float, default=0.3,
                       help='Minimum quality score (0-1)')
    parser.add_argument('--min-aesthetic', type=float, default=0.5,
                       help='Minimum aesthetic score (0-1)')
    parser.add_argument('--top-n', type=int, help='Select only top N frames')

    # Scene/mood filtering
    parser.add_argument('--scenes', nargs='+', help='Filter by scene types')
    parser.add_argument('--moods', nargs='+', help='Filter by moods')
    parser.add_argument('--min-confidence', type=float, default=0.4,
                       help='Minimum classification confidence')

    # Diverse sampling
    parser.add_argument('--diverse-sample', type=int,
                       help='Get diverse sample of N frames')
    parser.add_argument('--balance-by', choices=['scene', 'mood', 'style', 'episode'],
                       default='scene', help='Balance diverse sample by category')

    # Output options
    parser.add_argument('--organize-by', choices=['scene', 'mood', 'style', 'episode'],
                       help='Organize output into subdirectories')
    parser.add_argument('--symlink', action='store_true',
                       help='Create symlinks instead of copying files')
    parser.add_argument('--stats-only', action='store_true',
                       help='Generate statistics report only, no copying')

    args = parser.parse_args()

    # Initialize applicator
    applicator = AIAnalysisApplicator(args.analysis_json)

    # Generate statistics if requested
    if args.stats_only:
        stats_path = args.output_dir / 'analysis_stats.json'
        applicator.generate_statistics_report(stats_path)
        logger.info(f"Statistics saved to {stats_path}")
        return

    # Apply filters
    selected = applicator.analysis_data  # Start with all

    if args.diverse_sample:
        # Diverse sampling mode
        selected = applicator.get_diverse_sample(
            n_samples=args.diverse_sample,
            min_quality=args.min_quality,
            balance_by=args.balance_by
        )
    else:
        # Quality filtering
        selected = applicator.filter_by_quality(
            min_quality=args.min_quality,
            min_aesthetic=args.min_aesthetic,
            top_n=args.top_n
        )

        # Scene filtering
        if args.scenes:
            selected = [
                item for item in selected
                if item['scene_type'] in args.scenes
                and item['scene_confidence'] >= args.min_confidence
            ]
            logger.info(f"After scene filter: {len(selected)} frames")

        # Mood filtering
        if args.moods:
            selected = [
                item for item in selected
                if item['mood'] in args.moods
                and item['mood_confidence'] >= args.min_confidence
            ]
            logger.info(f"After mood filter: {len(selected)} frames")

    # Copy selected frames
    if selected:
        copy_mode = 'symlink' if args.symlink else 'copy'
        stats = applicator.copy_selected_frames(
            selected_frames=selected,
            output_dir=args.output_dir,
            source_root=args.source_root,
            organize_by=args.organize_by,
            copy_mode=copy_mode
        )

        # Save selection metadata
        metadata_path = args.output_dir / 'selection_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(selected, f, indent=2)
        logger.info(f"Selection metadata saved to {metadata_path}")

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Dataset curation complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Selected frames: {len(selected)}")
        logger.info(f"Copied: {stats['copied']}")
        logger.info(f"Output: {args.output_dir}")
    else:
        logger.warning("No frames selected after filtering")


if __name__ == '__main__':
    main()
