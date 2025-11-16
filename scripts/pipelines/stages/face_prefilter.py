"""
Stage 1: Face Prefilter

Uses InsightFace ArcFace to match SAM2 instances against reference faces.
Implements weighted sampling from multiple input directories.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import random
import shutil
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.pipelines.stages.base_stage import BaseStage
from scripts.core.face_matching.arcface_matcher import ArcFaceMatcher


class FacePrefilterStage(BaseStage):
    """Stage 1: Filter instances by face matching"""

    def validate_config(self) -> bool:
        """Validate configuration"""
        required_keys = [
            'reference_faces_dir',
            'input_dirs',
            'output_dir',
            'similarity_threshold'
        ]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Validate paths exist
        ref_dir = Path(self.config['reference_faces_dir'])
        if not ref_dir.exists():
            raise ValueError(f"Reference faces directory not found: {ref_dir}")

        for input_cfg in self.config['input_dirs']:
            input_dir = Path(input_cfg['path'])
            if not input_dir.exists():
                raise ValueError(f"Input directory not found: {input_dir}")

        return True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute face prefilter stage"""

        self.logger.info("="*60)
        self.logger.info("STAGE 1: Face Prefilter")
        self.logger.info("="*60)

        # Setup paths
        reference_dir = Path(self.config['reference_faces_dir'])
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize face matcher
        self.logger.info(f"Loading reference faces from {reference_dir}")
        matcher = ArcFaceMatcher(
            model_name=self.config.get('model_name', 'buffalo_l'),
            device=self.config.get('device', 'cuda'),
            similarity_threshold=self.config['similarity_threshold']
        )

        # Build reference embeddings
        max_references = self.config.get('max_references', None)
        num_references = matcher.build_reference_embeddings(
            reference_dir,
            max_references=max_references
        )

        if num_references == 0:
            raise ValueError("No valid reference faces found")

        # Collect all input images with weighted sampling
        self.logger.info("Collecting input images with weighted sampling...")
        input_images = self._collect_weighted_samples()

        self.logger.info(f"Total samples to process: {len(input_images)}")

        # Match faces
        self.logger.info("Matching faces against references...")
        matched_images = []
        match_scores = []

        for img_path in tqdm(input_images, desc="Matching faces"):
            is_match, similarity = matcher.match_face(img_path, return_similarity=True)

            if is_match:
                matched_images.append(img_path)
                match_scores.append(similarity)

        self.logger.info(f"Matched {len(matched_images)}/{len(input_images)} images "
                        f"({100*len(matched_images)/len(input_images):.1f}%)")

        # Copy matched images to output
        self.logger.info("Copying matched images to output directory...")
        copied_paths = []

        for img_path, score in tqdm(
            zip(matched_images, match_scores),
            total=len(matched_images),
            desc="Copying files"
        ):
            # Preserve original filename but add source indicator
            source_name = img_path.parent.name
            new_name = f"{source_name}_{img_path.name}"
            output_path = output_dir / new_name

            shutil.copy2(img_path, output_path)
            copied_paths.append(output_path)

        # Save metadata
        metadata = {
            'num_references': num_references,
            'similarity_threshold': self.config['similarity_threshold'],
            'input_samples': len(input_images),
            'matched_samples': len(matched_images),
            'match_rate': len(matched_images) / len(input_images) if input_images else 0,
            'avg_similarity': float(sum(match_scores) / len(match_scores)) if match_scores else 0,
            'min_similarity': float(min(match_scores)) if match_scores else 0,
            'max_similarity': float(max(match_scores)) if match_scores else 0,
            'output_dir': str(output_dir),
            'input_dirs': [
                {
                    'path': str(cfg['path']),
                    'weight': cfg['weight'],
                    'description': cfg.get('description', '')
                }
                for cfg in self.config['input_dirs']
            ]
        }

        # Save detailed results
        results_file = output_dir / 'face_prefilter_results.json'
        with open(results_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Results saved to {results_file}")

        # Print summary
        self.logger.info("\nFace Prefilter Summary:")
        self.logger.info(f"  Reference faces: {num_references}")
        self.logger.info(f"  Input samples: {len(input_images)}")
        self.logger.info(f"  Matched samples: {len(matched_images)}")
        self.logger.info(f"  Match rate: {100*metadata['match_rate']:.1f}%")
        self.logger.info(f"  Avg similarity: {metadata['avg_similarity']:.3f}")
        self.logger.info(f"  Output directory: {output_dir}")

        return {
            'output_dir': output_dir,
            'matched_images': copied_paths,
            'metadata': metadata,
            'num_matched': len(matched_images)
        }

    def _collect_weighted_samples(self) -> List[Path]:
        """
        Collect samples from multiple directories with weighted sampling

        Returns:
            List of sampled image paths
        """
        all_samples = []
        dir_samples = {}

        # Collect images from each directory
        for input_cfg in self.config['input_dirs']:
            input_dir = Path(input_cfg['path'])
            weight = input_cfg['weight']
            description = input_cfg.get('description', '')

            # Find all images
            images = sorted(
                list(input_dir.glob('*.jpg')) +
                list(input_dir.glob('*.png')) +
                list(input_dir.glob('*.jpeg'))
            )

            dir_samples[str(input_dir)] = {
                'images': images,
                'weight': weight,
                'description': description
            }

            self.logger.info(f"  {input_dir.name}: {len(images)} images (weight: {weight})")

        # Calculate total samples to draw
        max_samples = self.config.get('max_samples', None)

        if max_samples is None:
            # Use all images
            for dir_info in dir_samples.values():
                all_samples.extend(dir_info['images'])
        else:
            # Weighted sampling
            total_weight = sum(info['weight'] for info in dir_samples.values())

            for dir_path, dir_info in dir_samples.items():
                images = dir_info['images']
                weight = dir_info['weight']

                # Calculate number of samples for this directory
                n_samples = int(max_samples * (weight / total_weight))
                n_samples = min(n_samples, len(images))

                # Random sample
                sampled = random.sample(images, n_samples)
                all_samples.extend(sampled)

                self.logger.info(f"  Sampled {n_samples} from {Path(dir_path).name}")

        # Shuffle to mix sources
        random.shuffle(all_samples)

        return all_samples
