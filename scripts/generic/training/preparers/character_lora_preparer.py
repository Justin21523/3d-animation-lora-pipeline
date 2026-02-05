"""
Character identity LoRA data preparer.

Prepares training data for character identity LoRA by:
1. Extracting visual features from character images
2. Clustering by identity
3. Filtering low-quality images
4. Generating captions
5. Assembling final dataset in Kohya format
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import logging
import json
import shutil
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from base.feature_extractor import BaseFeatureExtractor
from base.clusterer import BaseClusterer
from base.caption_engine import BaseCaptionEngine
from base.quality_filter import BaseQualityFilter, CompositeQualityFilter

# Import implementations
from feature_extractors import CLIPFeatureExtractor
from clusterers import HDBSCANClusterer
from caption_engines import Qwen2VLCaptionEngine, TemplateCaptionEngine
from quality_filters import BlurFilter, SizeFilter, PerceptualHashDeduplicator


class CharacterLoRAPreparer:
    """
    Character identity LoRA data preparation pipeline.

    This preparer orchestrates the full workflow:
    - Feature extraction (CLIP/DINOv2/etc.)
    - Quality filtering (blur, size, dedup)
    - Identity clustering (HDBSCAN/KMeans/etc.)
    - Caption generation (VLM/Template/etc.)
    - Dataset assembly (Kohya format)

    All components are configurable via config dict.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        character_name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize character LoRA preparer.

        Args:
            input_dir: Directory containing character images
            output_dir: Output directory for prepared dataset
            character_name: Character name for captions
            config: Configuration dictionary with component settings
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.character_name = character_name
        self.config = config or {}

        self.logger = logging.getLogger(self.__class__.__name__)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize feature extractor, clusterer, filters, and caption engine."""

        # Feature extractor
        extractor_config = self.config.get('feature_extractor', {})
        extractor_type = extractor_config.get('type', 'clip')
        device = self.config.get('device', 'cuda')

        if extractor_type == 'clip':
            self.feature_extractor = CLIPFeatureExtractor(extractor_config, device)
        else:
            # Dynamically import other extractors
            from feature_extractors import (
                EVACLIPFeatureExtractor,
                DINOv2FeatureExtractor,
                SigLIPFeatureExtractor,
                InternVL2FeatureExtractor
            )

            extractors = {
                'eva_clip': EVACLIPFeatureExtractor,
                'dinov2': DINOv2FeatureExtractor,
                'siglip': SigLIPFeatureExtractor,
                'internvl2': InternVL2FeatureExtractor,
            }

            if extractor_type not in extractors:
                raise ValueError(
                    f"Unknown extractor type '{extractor_type}'. "
                    f"Supported: {list(extractors.keys())}"
                )

            self.feature_extractor = extractors[extractor_type](extractor_config, device)

        self.logger.info(f"✓ Feature extractor initialized: {self.feature_extractor}")

        # Clusterer
        clusterer_config = self.config.get('clusterer', {})
        clusterer_type = clusterer_config.get('type', 'hdbscan')

        if clusterer_type == 'hdbscan':
            self.clusterer = HDBSCANClusterer(clusterer_config)
        elif clusterer_type == 'kmeans':
            from clusterers import KMeansClusterer
            self.clusterer = KMeansClusterer(clusterer_config)
        elif clusterer_type == 'spectral':
            from clusterers import SpectralClusterer
            self.clusterer = SpectralClusterer(clusterer_config)
        else:
            raise ValueError(f"Unknown clusterer type '{clusterer_type}'")

        self.logger.info(f"✓ Clusterer initialized: {self.clusterer}")

        # Quality filters
        filter_configs = self.config.get('quality_filters', [])

        if not filter_configs:
            # Default filters for 3D character LoRA
            filter_configs = [
                {'type': 'blur', 'threshold': 100.0},
                {'type': 'size', 'min_width': 256, 'min_height': 256},
                {'type': 'dedup', 'threshold': 8}
            ]

        filters = []
        for fconfig in filter_configs:
            ftype = fconfig.get('type')

            if ftype == 'blur':
                filters.append(BlurFilter(fconfig))
            elif ftype == 'size':
                filters.append(SizeFilter(fconfig))
            elif ftype == 'dedup':
                filters.append(PerceptualHashDeduplicator(fconfig))
            else:
                self.logger.warning(f"Unknown filter type '{ftype}', skipping")

        if filters:
            self.quality_filter = CompositeQualityFilter(filters, mode='all')
            self.logger.info(f"✓ Quality filters initialized: {len(filters)} filters")
        else:
            self.quality_filter = None
            self.logger.info("✓ No quality filters configured")

        # Caption engine
        caption_config = self.config.get('caption_engine', {})
        caption_type = caption_config.get('type', 'template')

        # Add character name to caption config
        caption_config['character_name'] = self.character_name

        if caption_type == 'template':
            self.caption_engine = TemplateCaptionEngine(caption_config)
        elif caption_type == 'qwen2_vl':
            self.caption_engine = Qwen2VLCaptionEngine(caption_config, device)
        elif caption_type == 'internvl2':
            from caption_engines import InternVL2CaptionEngine
            self.caption_engine = InternVL2CaptionEngine(caption_config, device)
        elif caption_type == 'llm_provider':
            from caption_engines import LLMProviderAPICaptionEngine
            self.caption_engine = LLMProviderAPICaptionEngine(caption_config)
        else:
            raise ValueError(f"Unknown caption engine type '{caption_type}'")

        self.logger.info(f"✓ Caption engine initialized: {self.caption_engine}")

    def prepare(self) -> Dict[str, Any]:
        """
        Execute the full character LoRA data preparation pipeline.

        Returns:
            Dictionary with preparation results and metadata
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Character LoRA Preparation: {self.character_name}")
        self.logger.info("=" * 60)

        start_time = datetime.now()

        # Step 1: Load images
        image_paths = self._load_images()
        self.logger.info(f"Step 1: Loaded {len(image_paths)} images")

        # Step 2: Quality filtering
        if self.quality_filter:
            image_paths = self._filter_images(image_paths)
            self.logger.info(f"Step 2: {len(image_paths)} images passed quality filters")
        else:
            self.logger.info("Step 2: Quality filtering skipped")

        if len(image_paths) == 0:
            raise ValueError("No images remaining after quality filtering")

        # Step 3: Feature extraction
        features = self._extract_features(image_paths)
        self.logger.info(f"Step 3: Extracted features with shape {features.shape}")

        # Step 4: Clustering
        labels = self._cluster_features(features)
        self.logger.info(f"Step 4: Clustered into {self.clusterer.n_clusters_} clusters")

        # Step 5: Organize by cluster
        clusters = self._organize_clusters(image_paths, labels)
        self.logger.info(f"Step 5: Organized {len(clusters)} clusters")

        # Step 6: Generate captions
        self._generate_captions(clusters)
        self.logger.info(f"Step 6: Generated captions for all clusters")

        # Step 7: Assemble dataset
        dataset_info = self._assemble_dataset(clusters)
        self.logger.info(f"Step 7: Assembled dataset in {self.output_dir}")

        # Step 8: Save metadata
        metadata = self._save_metadata(dataset_info, start_time)
        self.logger.info(f"Step 8: Saved metadata")

        # Cleanup
        self.feature_extractor.cleanup()
        self.caption_engine.cleanup()

        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info("=" * 60)
        self.logger.info(f"✓ Preparation completed in {elapsed:.1f}s")
        self.logger.info("=" * 60)

        return metadata

    def _load_images(self) -> List[Path]:
        """Load all image files from input directory."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        image_paths = [
            f for f in self.input_dir.glob('*')
            if f.suffix.lower() in image_extensions
        ]

        if not image_paths:
            raise ValueError(f"No images found in {self.input_dir}")

        return sorted(image_paths)

    def _filter_images(self, image_paths: List[Path]) -> List[Path]:
        """Apply quality filters to images."""
        self.logger.info("Applying quality filters...")

        results = self.quality_filter.filter_batch(image_paths, show_progress=True)

        # Keep images that passed all filters
        filtered_paths = [
            path for path, (passed, reason) in zip(image_paths, results)
            if passed
        ]

        # Log filter statistics
        stats = self.quality_filter.get_filter_stats()
        self.logger.info(
            f"Quality filtering: {stats['total_passed']}/{stats['total_processed']} "
            f"passed ({stats['pass_rate']:.1f}%)"
        )

        return filtered_paths

    def _extract_features(self, image_paths: List[Path]) -> np.ndarray:
        """Extract features from images."""
        self.logger.info("Extracting features...")

        batch_size = self.config.get('batch_size', 32)
        features = self.feature_extractor.extract_batch(
            image_paths,
            batch_size=batch_size,
            show_progress=True
        )

        return features

    def _cluster_features(self, features: np.ndarray) -> np.ndarray:
        """Cluster features by identity."""
        self.logger.info("Clustering features...")

        labels = self.clusterer.fit_predict(features)

        # Log cluster info
        cluster_info = self.clusterer.get_cluster_info()
        self.logger.info(f"Clustering results: {cluster_info}")

        return labels

    def _organize_clusters(
        self,
        image_paths: List[Path],
        labels: np.ndarray
    ) -> Dict[int, List[Path]]:
        """Organize images by cluster label."""

        clusters = {}

        for path, label in zip(image_paths, labels):
            if label == -1:  # Noise
                continue

            if label not in clusters:
                clusters[label] = []

            clusters[label].append(path)

        # Sort clusters by size (largest first)
        clusters = dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True))

        return clusters

    def _generate_captions(self, clusters: Dict[int, List[Path]]):
        """Generate captions for all images in all clusters."""
        self.logger.info("Generating captions...")

        total_images = sum(len(images) for images in clusters.values())

        # Process all images
        all_images = []
        for cluster_id, images in clusters.items():
            all_images.extend(images)

        # Generate captions in batch
        batch_size = self.config.get('caption_batch_size', 8)
        captions = self.caption_engine.generate_batch(
            all_images,
            batch_size=batch_size,
            show_progress=True
        )

        # Save captions as .txt files next to images
        for image_path, caption in zip(all_images, captions):
            caption_path = image_path.with_suffix('.txt')
            caption_path.write_text(caption, encoding='utf-8')

    def _assemble_dataset(self, clusters: Dict[int, List[Path]]) -> Dict[str, Any]:
        """Assemble final dataset in Kohya format."""
        self.logger.info("Assembling dataset...")

        # Kohya format: output_dir / {repeats}_{character_name} / images and captions
        repeats = self.config.get('repeats', 10)
        dataset_dir = self.output_dir / f"{repeats}_{self.character_name}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Copy all images and captions
        total_images = 0

        for cluster_id, image_paths in clusters.items():
            for image_path in image_paths:
                # Copy image
                dest_image = dataset_dir / image_path.name
                shutil.copy2(image_path, dest_image)

                # Copy caption
                caption_path = image_path.with_suffix('.txt')
                if caption_path.exists():
                    dest_caption = dataset_dir / caption_path.name
                    shutil.copy2(caption_path, dest_caption)

                total_images += 1

        dataset_info = {
            'dataset_dir': str(dataset_dir),
            'num_images': total_images,
            'num_clusters': len(clusters),
            'repeats': repeats,
            'cluster_sizes': {int(k): len(v) for k, v in clusters.items()}
        }

        return dataset_info

    def _save_metadata(
        self,
        dataset_info: Dict[str, Any],
        start_time: datetime
    ) -> Dict[str, Any]:
        """Save preparation metadata."""

        metadata = {
            'character_name': self.character_name,
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': (datetime.now() - start_time).total_seconds(),
            'config': self.config,
            'dataset_info': dataset_info,
            'components': {
                'feature_extractor': str(self.feature_extractor),
                'clusterer': str(self.clusterer),
                'caption_engine': str(self.caption_engine),
            }
        }

        metadata_path = self.output_dir / 'preparation_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return metadata


# CLI entry point
def main():
    """Command-line interface for character LoRA preparer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare character identity LoRA training data"
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Input directory containing character images'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for prepared dataset'
    )

    parser.add_argument(
        '--character-name',
        type=str,
        required=True,
        help='Character name for captions'
    )

    parser.add_argument(
        '--feature-extractor',
        type=str,
        default='clip',
        choices=['clip', 'eva_clip', 'dinov2', 'siglip', 'internvl2'],
        help='Feature extractor to use (default: clip)'
    )

    parser.add_argument(
        '--clusterer',
        type=str,
        default='hdbscan',
        choices=['hdbscan', 'kmeans', 'spectral'],
        help='Clustering algorithm (default: hdbscan)'
    )

    parser.add_argument(
        '--caption-engine',
        type=str,
        default='template',
        choices=['template', 'qwen2_vl', 'internvl2', 'llm_provider'],
        help='Caption generation engine (default: template)'
    )

    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=10,
        help='Minimum cluster size for HDBSCAN (default: 10)'
    )

    parser.add_argument(
        '--repeats',
        type=int,
        default=10,
        help='Kohya repeats value (default: 10)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (default: cuda)'
    )

    args = parser.parse_args()

    # Build config
    config = {
        'device': args.device,
        'repeats': args.repeats,
        'feature_extractor': {
            'type': args.feature_extractor,
            'model_name': 'clip-vit-l14' if args.feature_extractor == 'clip' else None
        },
        'clusterer': {
            'type': args.clusterer,
            'min_cluster_size': args.min_cluster_size
        },
        'caption_engine': {
            'type': args.caption_engine
        }
    }

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run preparer
    preparer = CharacterLoRAPreparer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        character_name=args.character_name,
        config=config
    )

    metadata = preparer.prepare()

    print("\n" + "=" * 60)
    print("✓ Character LoRA preparation completed!")
    print("=" * 60)
    print(f"Dataset: {metadata['dataset_info']['dataset_dir']}")
    print(f"Images: {metadata['dataset_info']['num_images']}")
    print(f"Clusters: {metadata['dataset_info']['num_clusters']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
