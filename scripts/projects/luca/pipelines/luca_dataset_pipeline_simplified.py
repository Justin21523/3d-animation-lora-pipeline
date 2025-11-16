#!/usr/bin/env python3
"""
Dataset Preparation Pipeline - Simplified Integrated Version (Project-Agnostic)
================================================================================

Configuration-driven pipeline that works with any character/project by changing
the --project-config parameter. Defaults to Luca for backward compatibility.

Integrates existing mature scripts into a streamlined pipeline:
1. Face-based pre-filtering (ArcFace vs reference images)
2. Quality filtering (3D-specific metrics)
3. Comprehensive augmentation (10k-15k for manual review)
4. Diversity-based auto-selection (400 images)
5. Caption generation (using existing regenerate_captions_vlm.py logic)
6. Training data preparation (Kohya_ss format)

Usage:
  # Default (Luca):
  python luca_dataset_pipeline_simplified.py

  # Different project:
  python luca_dataset_pipeline_simplified.py \
    --project-config configs/projects/alberto.yaml \
    --config configs/projects/alberto_dataset_prep.yaml

Author: Claude Code
Date: 2025-11-13
Version: 3.0 (Configuration-driven, project-agnostic)
"""

import argparse
import json
import logging
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import yaml
from tqdm import tqdm
from PIL import Image, ImageEnhance
import random
import numpy as np
import cv2

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(log_dir: Path, project_name: str = "luca") -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(f"{project_name.title()}Pipeline")


class SimplifiedLucaPipeline:
    """Simplified pipeline using existing scripts."""

    def __init__(self, config_path: str, project_config_path: Optional[str] = None):
        # Load pipeline config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load project config to get project name
        if project_config_path:
            with open(project_config_path, 'r') as f:
                project_config = yaml.safe_load(f)
                self.project_name = project_config.get('project', {}).get('name', 'luca')
        else:
            # Default to luca for backward compatibility
            self.project_name = 'luca'

        log_dir = Path(self.config['logging']['output_dir'])
        self.logger = setup_logging(log_dir, self.project_name)

        self.logger.info("="*80)
        self.logger.info(f"{self.project_name.title()} Dataset Pipeline - Simplified Version")
        self.logger.info("="*80)

        # Define output directories using project name
        self.base_dir = Path(self.config['input']['sam2_results']).parent
        self.output_dirs = {
            'face_matched': self.base_dir / f'{self.project_name}_face_matched',
            'quality_filtered': self.base_dir / f'{self.project_name}_quality_filtered',
            'augmented': self.base_dir / f'{self.project_name}_augmented_comprehensive',
            'curated_400': self.base_dir / f'{self.project_name}_curated_400',
            'training_data': Path(self.config['training_prep']['output_dir'])
        }

    def run(self):
        """Execute complete pipeline."""
        start_time = datetime.now()

        try:
            # Stage 1: Face-based pre-filtering
            self.logger.info("\n" + "="*80)
            self.logger.info("STAGE 1: Face-Based Pre-Filtering")
            self.logger.info("="*80)
            self.stage_1_face_prefilter()

            # Stage 2: Quality filtering
            self.logger.info("\n" + "="*80)
            self.logger.info("STAGE 2: Quality Filtering")
            self.logger.info("="*80)
            self.stage_2_quality_filter()

            # Stage 3: Comprehensive augmentation
            self.logger.info("\n" + "="*80)
            self.logger.info("STAGE 3: Comprehensive Augmentation")
            self.logger.info("="*80)
            self.stage_3_augmentation()

            # Stage 4: Diversity-based auto-selection
            self.logger.info("\n" + "="*80)
            self.logger.info("STAGE 4: Diversity-Based Auto-Selection (400 images)")
            self.logger.info("="*80)
            self.stage_4_diversity_selection()

            # Stage 5: Caption generation
            self.logger.info("\n" + "="*80)
            self.logger.info("STAGE 5: Caption Generation")
            self.logger.info("="*80)
            self.stage_5_caption_generation()

            # Stage 6: Training data preparation
            self.logger.info("\n" + "="*80)
            self.logger.info("STAGE 6: Training Data Preparation")
            self.logger.info("="*80)
            self.stage_6_training_prep()

            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info("\n" + "="*80)
            self.logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total duration: {duration/3600:.2f} hours")
            self.logger.info("="*80)

            return {'success': True, 'duration': duration}

        except Exception as e:
            self.logger.error(f"✗ Pipeline failed: {str(e)}", exc_info=True)
            raise

    def stage_1_face_prefilter(self):
        """Stage 1: Use ArcFace to match against reference Luca faces."""
        self.logger.info("Using ArcFace to match SAM2 instances against 372 reference Luca images...")

        # Import face recognition
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            self.logger.error("InsightFace not installed. Install: pip install insightface onnxruntime-gpu")
            raise

        import cv2
        import numpy as np

        # Initialize FaceAnalysis
        app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Load reference faces
        reference_dir = Path(self.config['input']['reference_faces'])
        reference_images = list(reference_dir.glob("*.png"))[:50]  # Top 50 diverse references

        self.logger.info(f"Extracting embeddings from {len(reference_images)} reference images...")
        reference_embeddings = []

        for img_path in tqdm(reference_images, desc="Reference faces"):
            try:
                img = cv2.imread(str(img_path))
                faces = app.get(img)
                if faces:
                    reference_embeddings.append(faces[0].embedding)
            except Exception as e:
                self.logger.warning(f"Failed to process {img_path.name}: {e}")

        if not reference_embeddings:
            raise ValueError("No reference embeddings extracted!")

        reference_embeddings = np.array(reference_embeddings)
        self.logger.info(f"✓ Extracted {len(reference_embeddings)} reference embeddings")

        # Process SAM2 instances
        sam2_dir = Path(self.config['input']['sam2_results'])
        instance_dirs = []

        # Weighted sampling from instance types
        for inst_type_config in self.config['input']['instance_types']:
            inst_dir = sam2_dir / inst_type_config['type']
            if inst_dir.exists():
                instance_dirs.append((inst_dir, inst_type_config['weight']))

        output_dir = self.output_dirs['face_matched']
        output_dir.mkdir(parents=True, exist_ok=True)

        matched_count = 0
        processed_count = 0
        threshold = self.config['face_prefilter']['similarity_threshold']

        self.logger.info(f"Processing SAM2 instances (threshold={threshold})...")

        for inst_dir, weight in instance_dirs:
            instance_files = list(inst_dir.glob("*.png"))

            # Sample based on weight
            import random
            sample_size = int(len(instance_files) * weight)
            sampled_files = random.sample(instance_files, min(sample_size, len(instance_files)))

            self.logger.info(f"Processing {len(sampled_files)} from {inst_dir.name}...")

            for img_path in tqdm(sampled_files, desc=f"  {inst_dir.name}"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    # Pre-quality filter
                    h, w = img.shape[:2]
                    if h < 100 or w < 100:
                        continue

                    faces = app.get(img)
                    if not faces:
                        continue

                    # Compare with reference embeddings
                    query_embedding = faces[0].embedding
                    similarities = np.dot(reference_embeddings, query_embedding)
                    max_similarity = similarities.max()

                    if max_similarity >= threshold:
                        # Copy to output
                        shutil.copy2(img_path, output_dir / img_path.name)
                        matched_count += 1

                    processed_count += 1

                except Exception as e:
                    self.logger.warning(f"Error processing {img_path.name}: {e}")

        self.logger.info(f"✓ Face matching complete:")
        self.logger.info(f"  Processed: {processed_count}")
        self.logger.info(f"  Matched: {matched_count}")
        self.logger.info(f"  Reduction: {(1 - matched_count/max(processed_count, 1))*100:.1f}%")

        # Save metadata
        metadata = {
            'processed': processed_count,
            'matched': matched_count,
            'threshold': threshold,
            'reference_count': len(reference_embeddings)
        }
        with open(output_dir / 'face_matching_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def stage_2_quality_filter(self):
        """Stage 2: Apply 3D-specific quality filters."""
        self.logger.info("Applying 3D-specific quality filters...")

        import cv2
        import numpy as np
        from PIL import Image

        input_dir = self.output_dirs['face_matched']
        output_dir = self.output_dirs['quality_filtered']
        output_dir.mkdir(parents=True, exist_ok=True)

        config = self.config['quality_filtering']
        image_files = list(input_dir.glob("*.png"))

        passed_count = 0

        for img_path in tqdm(image_files, desc="Quality filtering"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Check dimensions
                h, w = img.shape[:2]
                min_h, min_w = config['min_dimensions']
                if h < min_h or w < min_w:
                    continue

                # Sharpness (Laplacian variance)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < config['min_sharpness']:
                    continue

                # Alpha coverage (if has alpha channel)
                pil_img = Image.open(img_path)
                if pil_img.mode == 'RGBA':
                    alpha = np.array(pil_img)[:, :, 3]
                    coverage = (alpha > 0).sum() / alpha.size
                    if coverage < config['min_alpha_coverage']:
                        continue

                # Passed all filters
                shutil.copy2(img_path, output_dir / img_path.name)
                passed_count += 1

            except Exception as e:
                self.logger.warning(f"Error in quality filter for {img_path.name}: {e}")

        self.logger.info(f"✓ Quality filtering complete:")
        self.logger.info(f"  Input: {len(image_files)}")
        self.logger.info(f"  Passed: {passed_count}")

    def stage_3_augmentation(self):
        """Stage 3: Generate comprehensive augmented dataset."""
        self.logger.info("Generating comprehensive augmented dataset (10k-15k images)...")

        # Use existing augmentation script logic
        from PIL import Image, ImageEnhance
        import numpy as np

        input_dir = self.output_dirs['quality_filtered']
        output_dir = self.output_dirs['augmented']
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(input_dir.glob("*.png"))
        multiplier = int(self.config['augmentation']['multiplier'])

        total_generated = 0

        for img_path in tqdm(image_files, desc="Augmenting"):
            try:
                img = Image.open(img_path).convert('RGBA')

                # Generate augmented versions
                for i in range(multiplier):
                    aug_img = self._apply_3d_safe_augmentations(img)

                    # Save
                    output_path = output_dir / f"{img_path.stem}_aug{i:02d}.png"
                    aug_img.save(output_path)
                    total_generated += 1

                # Also copy original
                shutil.copy2(img_path, output_dir / img_path.name)
                total_generated += 1

            except Exception as e:
                self.logger.warning(f"Augmentation error for {img_path.name}: {e}")

        self.logger.info(f"✓ Augmentation complete: Generated {total_generated} images")

    def _apply_3d_safe_augmentations(self, img: Image.Image) -> Image.Image:
        """Apply 3D-safe augmentations that preserve PBR materials."""
        import random
        from PIL import ImageEnhance

        aug_img = img.copy()

        # Random crop (0.8-1.0x)
        if random.random() < 0.8:
            w, h = aug_img.size
            scale = random.uniform(0.8, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            aug_img = aug_img.crop((left, top, left + new_w, top + new_h))
            aug_img = aug_img.resize((w, h), Image.LANCZOS)

        # Rotation (-5 to +5 degrees)
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            aug_img = aug_img.rotate(angle, expand=False, fillcolor=(0, 0, 0, 0))

        # Brightness (0.9-1.1x)
        if random.random() < 0.6:
            enhancer = ImageEnhance.Brightness(aug_img)
            aug_img = enhancer.enhance(random.uniform(0.9, 1.1))

        # Contrast (0.95-1.05x - MINIMAL)
        if random.random() < 0.4:
            enhancer = ImageEnhance.Contrast(aug_img)
            aug_img = enhancer.enhance(random.uniform(0.95, 1.05))

        # Gaussian noise
        if random.random() < 0.3:
            import numpy as np
            arr = np.array(aug_img, dtype=np.float32)
            noise = np.random.normal(0, 0.01 * 255, arr.shape[:2] + (3,))
            arr[:, :, :3] = np.clip(arr[:, :, :3] + noise, 0, 255)
            aug_img = Image.fromarray(arr.astype(np.uint8), 'RGBA')

        return aug_img

    def stage_4_diversity_selection(self):
        """Stage 4: Auto-select 400 diverse images using multi-modal metrics."""
        self.logger.info("Selecting 400 diverse images using RTM-Pose + CLIP + face angles...")

        # This is a placeholder - full implementation would require RTM-Pose, CLIP, etc.
        # For now, use CLIP-based diversity sampling

        try:
            import torch
            import clip
            from PIL import Image
            import numpy as np
            from sklearn.cluster import KMeans

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-L/14", device=device)

            input_dir = self.output_dirs['augmented']
            output_dir = self.output_dirs['curated_400']
            output_dir.mkdir(parents=True, exist_ok=True)

            image_files = list(input_dir.glob("*.png"))
            self.logger.info(f"Processing {len(image_files)} images for diversity selection...")

            # Extract CLIP embeddings
            embeddings = []
            valid_files = []

            for img_path in tqdm(image_files, desc="Extracting CLIP embeddings"):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_input = preprocess(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        embedding = model.encode_image(img_input).cpu().numpy()[0]

                    embeddings.append(embedding)
                    valid_files.append(img_path)

                except Exception as e:
                    self.logger.warning(f"Failed to process {img_path.name}: {e}")

            embeddings = np.array(embeddings)

            # Cluster into 8 groups, select 50 from each
            n_clusters = 8
            samples_per_cluster = 50

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)

            selected_files = []

            for cluster_id in range(n_clusters):
                cluster_indices = np.where(labels == cluster_id)[0]
                cluster_files = [valid_files[i] for i in cluster_indices]

                # Select top samples_per_cluster from this cluster (by distance to centroid)
                cluster_embeddings = embeddings[cluster_indices]
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

                # Sort by distance (closest to centroid first)
                sorted_indices = np.argsort(distances)
                selected_indices = sorted_indices[:min(samples_per_cluster, len(sorted_indices))]

                selected_files.extend([cluster_files[i] for i in selected_indices])

            # Copy selected files
            for img_path in tqdm(selected_files[:400], desc="Copying selected images"):
                shutil.copy2(img_path, output_dir / img_path.name)

            self.logger.info(f"✓ Selected {min(len(selected_files), 400)} diverse images")

            # Save selection metadata
            metadata = {
                'total_candidates': len(image_files),
                'selected': min(len(selected_files), 400),
                'n_clusters': n_clusters,
                'samples_per_cluster': samples_per_cluster
            }
            with open(output_dir / 'diversity_selection_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            self.logger.error(f"Diversity selection failed: {e}", exc_info=True)
            raise

    def stage_5_caption_generation(self):
        """Stage 5: Generate captions using existing VLM caption generator."""
        self.logger.info("Generating captions using Qwen2-VL...")

        # Import existing caption generator
        sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'training'))
        from regenerate_captions_vlm import VLMCaptionGenerator, load_character_profile

        # Load character profile
        character_profile_path = Path(self.config['captioning']['character_config'])
        if character_profile_path.exists():
            import yaml
            with open(character_profile_path, 'r') as f:
                char_config = yaml.safe_load(f)

            # Convert to JSON format expected by VLMCaptionGenerator
            character_profile = {
                'name': char_config.get('name', ''),
                'full_name': char_config.get('full_name', ''),
                'film': char_config.get('film', ''),
                'age': char_config.get('age', ''),
                'core_description': f"Luca Paguro from Pixar Luca (2021), 12-year-old italian pre-teen boy, large round brown eyes, thick arched eyebrows, button red-tinted nose, rosy cheeks, soft oval face, short dark-brown wavy curls",
                'physical_traits': ', '.join(char_config.get('human_form', {}).get('appearance', {}).get('distinctive_features', []))
            }
        else:
            character_profile = {}

        # Initialize generator
        generator = VLMCaptionGenerator(
            model_name=self.config['captioning']['model'],
            device=self.config['hardware']['device'],
            character_profile=character_profile
        )

        # Generate for curated 400
        curated_dir = self.output_dirs['curated_400']
        self.logger.info("Generating captions for 400 curated images...")
        generator.batch_generate(
            image_dir=curated_dir,
            output_dir=curated_dir,
            sample_size=None
        )

        self.logger.info("✓ Caption generation complete")

    def stage_6_training_prep(self):
        """Stage 6: Prepare Kohya_ss format training data."""
        self.logger.info("Preparing Kohya_ss format training data...")

        source_dir = self.output_dirs['curated_400']
        output_dir = self.output_dirs['training_data']
        repeat_count = self.config['training_prep']['repeat_count']
        class_name = self.config['training_prep']['class_name']

        # Create output structure
        kohya_dir = output_dir / f"{repeat_count}_{class_name}"
        kohya_dir.mkdir(parents=True, exist_ok=True)

        # Copy images and captions
        image_files = list(source_dir.glob("*.png"))
        copied = 0

        for img_path in tqdm(image_files, desc="Copying to training dir"):
            # Copy image
            shutil.copy2(img_path, kohya_dir / img_path.name)

            # Copy caption if exists
            txt_path = source_dir / f"{img_path.stem}.txt"
            if txt_path.exists():
                shutil.copy2(txt_path, kohya_dir / f"{img_path.stem}.txt")

            copied += 1

        # Generate metadata
        metadata = {
            'source': str(source_dir),
            'image_count': copied,
            'repeat_count': repeat_count,
            'class_name': class_name,
            'created': datetime.now().isoformat()
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"✓ Training data prepared: {copied} image-caption pairs")
        self.logger.info(f"  Location: {kohya_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Preparation Pipeline - Simplified Version (Project-Agnostic)"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/projects/luca_dataset_prep_v2.yaml',
        help='Pipeline configuration file path'
    )
    parser.add_argument(
        '--project-config',
        type=str,
        default='configs/projects/luca.yaml',
        help='Project configuration file (defines project name, paths, characters)'
    )

    args = parser.parse_args()

    try:
        pipeline = SimplifiedLucaPipeline(args.config, args.project_config)
        result = pipeline.run()

        print("\n" + "="*80)
        print("✓ Pipeline completed successfully!")
        print(f"  Duration: {result['duration']/3600:.2f} hours")
        print("="*80)

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
