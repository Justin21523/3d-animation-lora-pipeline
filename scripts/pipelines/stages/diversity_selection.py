"""
Stage 4: Diversity Selection

Uses multi-modal diversity metrics to select balanced dataset:
- RTM-Pose keypoints (pose diversity)
- Face angle classification (view diversity)
- CLIP embeddings (semantic diversity)
- Background complexity
- Scale variety
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import shutil
from tqdm import tqdm
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.pipelines.stages.base_stage import BaseStage
from scripts.core.diversity.diversity_metrics import DiversityMetrics


class DiversitySelectionStage(BaseStage):
    """Stage 4: Select diverse subset for training"""

    def validate_config(self) -> bool:
        """Validate configuration"""
        required_keys = ['input_dir', 'output_dir', 'target_size']

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        input_dir = Path(self.config['input_dir'])
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")

        return True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute diversity selection stage"""

        self.logger.info("="*60)
        self.logger.info("STAGE 4: Diversity Selection")
        self.logger.info("="*60)

        # Setup paths
        input_dir = Path(self.config['input_dir'])
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect input images
        input_images = sorted(
            list(input_dir.glob('*.jpg')) +
            list(input_dir.glob('*.png')) +
            list(input_dir.glob('*.jpeg'))
        )

        self.logger.info(f"Processing {len(input_images)} images...")

        # Get target size
        target_size = self.config['target_size']
        n_clusters = self.config.get('n_clusters', 8)

        if len(input_images) <= target_size:
            self.logger.warning(
                f"Input size ({len(input_images)}) <= target size ({target_size}). "
                f"Copying all images without selection."
            )

            # Copy all images
            copied_paths = []
            for img_path in tqdm(input_images, desc="Copying images"):
                output_path = output_dir / img_path.name
                shutil.copy2(img_path, output_path)
                copied_paths.append(output_path)

            metadata = {
                'input_count': len(input_images),
                'target_size': target_size,
                'selected_count': len(copied_paths),
                'selection_applied': False,
                'output_dir': str(output_dir)
            }

        else:
            # Perform diversity selection
            self.logger.info(
                f"Selecting {target_size} diverse images from {len(input_images)} candidates..."
            )

            diversity = DiversityMetrics(device=self.config.get('device', 'cuda'))

            # Perform stratified sampling with quality weighting
            selected_paths, selection_metadata = diversity.stratified_sample(
                image_paths=input_images,
                n_samples=target_size,
                n_clusters=n_clusters,
                quality_weight=self.config.get('quality_weight', 0.3),
                diversity_weight=self.config.get('diversity_weight', 0.7)
            )

            self.logger.info(f"Selected {len(selected_paths)} diverse images")

            # Copy selected images
            self.logger.info("Copying selected images to output directory...")
            copied_paths = []

            for img_path in tqdm(selected_paths, desc="Copying selected"):
                output_path = output_dir / img_path.name
                shutil.copy2(img_path, output_path)
                copied_paths.append(output_path)

            # Build metadata
            metadata = {
                'input_count': len(input_images),
                'target_size': target_size,
                'selected_count': len(selected_paths),
                'selection_applied': True,
                'n_clusters': n_clusters,
                'quality_weight': self.config.get('quality_weight', 0.3),
                'diversity_weight': self.config.get('diversity_weight', 0.7),
                'cluster_distribution': selection_metadata.get('cluster_info', {}),
                'diversity_features': {
                    'clip_embeddings': True,
                    'pose_features': True,
                    'face_angles': True,
                    'background_complexity': True,
                    'scale_variety': True
                },
                'output_dir': str(output_dir)
            }

        # Save results
        results_file = output_dir / 'diversity_selection_results.json'
        with open(results_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save detailed analysis
        analysis_file = output_dir / 'diversity_analysis.json'
        analysis = self._analyze_selection(copied_paths, metadata)
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Analysis saved to {analysis_file}")

        # Print summary
        self.logger.info("\nDiversity Selection Summary:")
        self.logger.info(f"  Input images: {len(input_images)}")
        self.logger.info(f"  Target size: {target_size}")
        self.logger.info(f"  Selected: {len(copied_paths)}")
        if metadata.get('selection_applied'):
            self.logger.info(f"  Clusters: {n_clusters}")
            self.logger.info(f"  Quality weight: {metadata['quality_weight']}")
            self.logger.info(f"  Diversity weight: {metadata['diversity_weight']}")
        self.logger.info(f"  Output directory: {output_dir}")

        # Print cluster distribution if available
        if 'cluster_distribution' in metadata:
            self.logger.info("\n  Cluster distribution:")
            for cluster_id, info in metadata['cluster_distribution'].items():
                self.logger.info(
                    f"    {cluster_id}: {info['selected']}/{info['total_size']} images "
                    f"(avg quality: {info['avg_quality']:.3f})"
                )

        return {
            'output_dir': output_dir,
            'selected_images': copied_paths,
            'metadata': metadata,
            'num_selected': len(copied_paths)
        }

    def _analyze_selection(
        self,
        selected_paths: List[Path],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze selected dataset

        Args:
            selected_paths: List of selected image paths
            metadata: Selection metadata

        Returns:
            Analysis dictionary
        """
        analysis = {
            'total_selected': len(selected_paths),
            'selection_strategy': 'stratified_sampling' if metadata.get('selection_applied') else 'all_images',
            'diversity_metrics': metadata.get('diversity_features', {}),
            'recommendations': []
        }

        # Add recommendations based on selection
        target_size = metadata.get('target_size', 400)
        selected_count = len(selected_paths)

        if selected_count < target_size * 0.8:
            analysis['recommendations'].append(
                f"Selected count ({selected_count}) is significantly below target ({target_size}). "
                "Consider relaxing quality filters or adding more input data."
            )
        elif selected_count > target_size * 1.2:
            analysis['recommendations'].append(
                f"Selected count ({selected_count}) exceeds target ({target_size}). "
                "Consider increasing diversity weight or reducing target size."
            )
        else:
            analysis['recommendations'].append(
                "Selection meets target size. Dataset is ready for captioning."
            )

        # Check cluster balance if available
        if 'cluster_distribution' in metadata:
            cluster_sizes = [
                info['selected']
                for info in metadata['cluster_distribution'].values()
            ]

            if len(cluster_sizes) > 0:
                min_size = min(cluster_sizes)
                max_size = max(cluster_sizes)
                imbalance_ratio = max_size / min_size if min_size > 0 else float('inf')

                analysis['cluster_balance'] = {
                    'min_cluster_size': min_size,
                    'max_cluster_size': max_size,
                    'imbalance_ratio': imbalance_ratio
                }

                if imbalance_ratio > 2.0:
                    analysis['recommendations'].append(
                        f"Cluster imbalance detected (ratio: {imbalance_ratio:.1f}). "
                        "Some diversity buckets are underrepresented. Consider adjusting cluster count."
                    )

        return analysis
