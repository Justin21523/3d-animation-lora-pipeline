"""
Style LoRA data preparer.

Prepares training data for style-specific LoRA by:
1. Extracting stylistic features (rendering style, lighting, materials)
2. Clustering by visual style similarity
3. Filtering low-quality images
4. Generating style-aware captions
5. Assembling final dataset in Kohya format

This preparer is designed for learning rendering styles (Pixar, DreamWorks, etc.),
lighting techniques, material properties, and overall visual aesthetics.
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
from clusterers import KMeansClusterer
from caption_engines import TemplateCaptionEngine
from quality_filters import BlurFilter, SizeFilter, PerceptualHashDeduplicator


class StyleLoRAPreparer:
    """
    Style LoRA data preparation pipeline.

    This preparer organizes images by visual style characteristics:
    - Style feature extraction (CLIP/DINOv2 for aesthetics)
    - Style clustering (rendering techniques, lighting setups, materials)
    - Quality filtering with deduplication
    - Style-aware caption generation (materials, lighting, rendering)
    - Dataset assembly (Kohya format)

    Useful for capturing specific studio styles (Pixar, Disney, DreamWorks),
    time periods, or artistic techniques.

    All components are configurable via config dict.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        style_name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize style LoRA preparer.

        Args:
            input_dir: Directory containing images with target style
            output_dir: Output directory for prepared dataset
            style_name: Style name for captions (e.g., "pixar_style")
            config: Configuration dictionary with component settings
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.style_name = style_name
        self.config = config or {}

        self.logger = logging.getLogger(self.__class__.__name__)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize feature extractor, clusterer, filters, and caption engine."""

        # Feature extractor (CLIP for style/aesthetic understanding)
        extractor_config = self.config.get('feature_extractor', {})
        extractor_type = extractor_config.get('type', 'clip')
        device = self.config.get('device', 'cuda')

        if extractor_type == 'clip':
            self.feature_extractor = CLIPFeatureExtractor(extractor_config, device)
        else:
            from feature_extractors import (
                EVACLIPFeatureExtractor,
                DINOv2FeatureExtractor,
                SigLIPFeatureExtractor,
            )

            extractors = {
                'eva_clip': EVACLIPFeatureExtractor,
                'dinov2': DINOv2FeatureExtractor,
                'siglip': SigLIPFeatureExtractor,
            }

            if extractor_type not in extractors:
                raise ValueError(
                    f"Unknown extractor type '{extractor_type}'. "
                    f"Supported: {list(extractors.keys())}"
                )

            self.feature_extractor = extractors[extractor_type](extractor_config, device)

        self.logger.info(f"✓ Feature extractor initialized: {self.feature_extractor}")

        # Clusterer (style-based clustering - often use KMeans for fixed style buckets)
        clusterer_config = self.config.get('clusterer', {})
        clusterer_type = clusterer_config.get('type', 'kmeans')

        if clusterer_type == 'kmeans':
            self.clusterer = KMeansClusterer(clusterer_config)
        elif clusterer_type == 'hdbscan':
            from clusterers import HDBSCANClusterer
            self.clusterer = HDBSCANClusterer(clusterer_config)
        elif clusterer_type == 'spectral':
            from clusterers import SpectralClusterer
            self.clusterer = SpectralClusterer(clusterer_config)
        else:
            raise ValueError(f"Unknown clusterer type '{clusterer_type}'")

        self.logger.info(f"✓ Clusterer initialized: {self.clusterer}")

        # Quality filters (with deduplication)
        filter_configs = self.config.get('quality_filters', [])

        if not filter_configs:
            # Default filters for style LoRA
            filter_configs = [
                {'type': 'blur', 'threshold': 100.0},
                {'type': 'size', 'min_width': 512, 'min_height': 512},
                {'type': 'dedup', 'threshold': 8},  # Moderate deduplication
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

        # Caption engine (style-aware templates)
        caption_config = self.config.get('caption_engine', {})
        caption_type = caption_config.get('type', 'template')

        # Add style name to caption config
        caption_config['character_name'] = self.style_name  # Reuse for style name
        # Add style-specific prefix
        if 'prefix' not in caption_config:
            caption_config['prefix'] = f'{self.style_name}, 3d animation'

        if caption_type == 'template':
            self.caption_engine = TemplateCaptionEngine(caption_config)
        elif caption_type == 'qwen2_vl':
            from caption_engines import Qwen2VLCaptionEngine
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
        Execute the full style LoRA data preparation pipeline.

        Returns:
            Dictionary with preparation results and metadata
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Style LoRA Preparation: {self.style_name}")
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

        # Step 3: Feature extraction (style features)
        features = self._extract_features(image_paths)
        self.logger.info(f"Step 3: Extracted features with shape {features.shape}")

        # Step 4: Clustering by style similarity
        labels = self._cluster_features(features)
        self.logger.info(f"Step 4: Clustered into {self.clusterer.n_clusters_} style groups")

        # Step 5: Organize by style cluster
        style_groups = self._organize_clusters(image_paths, labels)
        self.logger.info(f"Step 5: Organized {len(style_groups)} style groups")

        # Step 6: Generate style-aware captions
        self._generate_captions(style_groups)
        self.logger.info(f"Step 6: Generated captions for all style groups")

        # Step 7: Assemble dataset
        dataset_info = self._assemble_dataset(style_groups)
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
        self.logger.info("Extracting style features...")

        batch_size = self.config.get('batch_size', 32)
        features = self.feature_extractor.extract_batch(
            image_paths,
            batch_size=batch_size,
            show_progress=True
        )

        return features

    def _cluster_features(self, features: np.ndarray) -> np.ndarray:
        """Cluster features by style similarity."""
        self.logger.info("Clustering by style...")

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
        """Organize images by style cluster label."""

        style_groups = {}

        for path, label in zip(image_paths, labels):
            if label == -1:  # Noise
                continue

            if label not in style_groups:
                style_groups[label] = []

            style_groups[label].append(path)

        # Sort by size (largest first)
        style_groups = dict(sorted(style_groups.items(), key=lambda x: len(x[1]), reverse=True))

        return style_groups

    def _generate_captions(self, style_groups: Dict[int, List[Path]]):
        """Generate style-aware captions for all images."""
        self.logger.info("Generating style-aware captions...")

        # Process all images
        all_images = []
        for group_id, images in style_groups.items():
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

    def _assemble_dataset(self, style_groups: Dict[int, List[Path]]) -> Dict[str, Any]:
        """Assemble final dataset in Kohya format."""
        self.logger.info("Assembling style dataset...")

        # Kohya format: output_dir / {repeats}_{style_name} / images and captions
        repeats = self.config.get('repeats', 8)  # Moderate repeats for style
        dataset_dir = self.output_dir / f"{repeats}_{self.style_name}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Copy all images and captions
        total_images = 0

        for group_id, image_paths in style_groups.items():
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
            'num_style_groups': len(style_groups),
            'repeats': repeats,
            'group_sizes': {int(k): len(v) for k, v in style_groups.items()}
        }

        return dataset_info

    def _save_metadata(
        self,
        dataset_info: Dict[str, Any],
        start_time: datetime
    ) -> Dict[str, Any]:
        """Save preparation metadata."""

        metadata = {
            'preparer_type': 'style_lora',
            'style_name': self.style_name,
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
    """Command-line interface for style LoRA preparer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare style-specific LoRA training data"
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Input directory containing images with target style'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for prepared dataset'
    )

    parser.add_argument(
        '--style-name',
        type=str,
        required=True,
        help='Style name for captions (e.g., "pixar_style")'
    )

    parser.add_argument(
        '--feature-extractor',
        type=str,
        default='clip',
        choices=['clip', 'eva_clip', 'dinov2', 'siglip'],
        help='Feature extractor to use (default: clip)'
    )

    parser.add_argument(
        '--clusterer',
        type=str,
        default='kmeans',
        choices=['kmeans', 'hdbscan', 'spectral'],
        help='Clustering algorithm (default: kmeans)'
    )

    parser.add_argument(
        '--n-clusters',
        type=int,
        default=5,
        help='Number of style clusters for KMeans (default: 5)'
    )

    parser.add_argument(
        '--caption-engine',
        type=str,
        default='template',
        choices=['template', 'qwen2_vl', 'internvl2', 'llm_provider'],
        help='Caption generation engine (default: template)'
    )

    parser.add_argument(
        '--repeats',
        type=int,
        default=8,
        help='Kohya repeats value (default: 8 for style)'
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
        },
        'clusterer': {
            'type': args.clusterer,
            'n_clusters': args.n_clusters
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
    preparer = StyleLoRAPreparer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        style_name=args.style_name,
        config=config
    )

    metadata = preparer.prepare()

    print("\n" + "=" * 60)
    print("✓ Style LoRA preparation completed!")
    print("=" * 60)
    print(f"Dataset: {metadata['dataset_info']['dataset_dir']}")
    print(f"Images: {metadata['dataset_info']['num_images']}")
    print(f"Style groups: {metadata['dataset_info']['num_style_groups']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
