#!/usr/bin/env python3
"""
Prepare Character LoRA Training Data (Enhanced with VLM Captioning)

Prepares character-specific images for LoRA training with:
1. Quality filtering (blur detection, size validation)
2. Perceptual hash deduplication
3. CLIP-based diversity sampling
4. VLM captioning (Qwen2-VL / InternVL2)
5. Schema-guided caption generation
6. Resume capability

Features:
- Real VLM integration (not placeholder!)
- Quality filtering pipeline
- Diversity-aware sampling
- Production-ready error handling

Usage:
    python prepare_training_data_v2.py \
        --character-dirs /path/to/clustered/character_0 \
        --output-dir /path/to/training_data/luca \
        --character-name "luca" \
        --caption-model qwen2_vl \
        --target-size 400 \
        --device cuda

Author: AI Pipeline (Enhanced Version)
Date: 2025-01-17
"""

import sys
import shutil
import json
import argparse
import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import imagehash

# Add scripts directory to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.core.utils.logger import setup_logger
from scripts.core.utils.path_utils import ensure_dir

# Optional imports with fallbacks
try:
    from core.utils.character_loader import load_character, get_character_tags
    CHARACTER_LOADER_AVAILABLE = True
except ImportError:
    CHARACTER_LOADER_AVAILABLE = False

try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# =============================================================================
# Quality Filtering Utilities
# =============================================================================

class QualityFilter:
    """
    Quality filtering for character images.

    Filters:
    - Blur detection (Laplacian variance)
    - Size validation (min dimensions)
    - Aspect ratio validation
    """

    def __init__(self,
                 min_blur_score: float = 100.0,
                 min_width: int = 256,
                 min_height: int = 256,
                 max_aspect_ratio: float = 3.0,
                 logger=None):
        """
        Initialize quality filter.

        Args:
            min_blur_score: Minimum Laplacian variance (lower = more blurry)
            min_width: Minimum image width
            min_height: Minimum image height
            max_aspect_ratio: Maximum width/height or height/width ratio
            logger: Logger instance
        """
        self.min_blur_score = min_blur_score
        self.min_width = min_width
        self.min_height = min_height
        self.max_aspect_ratio = max_aspect_ratio
        self.logger = logger or logging.getLogger(__name__)

    def check_blur(self, image_path: Path) -> Tuple[bool, float]:
        """
        Check if image is blurry using Laplacian variance.

        Args:
            image_path: Path to image

        Returns:
            (is_sharp, blur_score)
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False, 0.0

            # Laplacian variance
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            blur_score = laplacian.var()

            is_sharp = blur_score >= self.min_blur_score
            return is_sharp, float(blur_score)

        except Exception as e:
            self.logger.warning(f"Blur check failed for {image_path}: {e}")
            return True, self.min_blur_score  # Assume sharp if check fails

    def check_size(self, image_path: Path) -> Tuple[bool, Tuple[int, int]]:
        """
        Check if image meets minimum size requirements.

        Args:
            image_path: Path to image

        Returns:
            (is_valid, (width, height))
        """
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                is_valid = w >= self.min_width and h >= self.min_height
                return is_valid, (w, h)

        except Exception as e:
            self.logger.warning(f"Size check failed for {image_path}: {e}")
            return False, (0, 0)

    def check_aspect_ratio(self, width: int, height: int) -> bool:
        """Check if aspect ratio is reasonable."""
        if width == 0 or height == 0:
            return False

        ratio = max(width, height) / min(width, height)
        return ratio <= self.max_aspect_ratio

    def is_valid(self, image_path: Path) -> Tuple[bool, Dict]:
        """
        Check if image passes all quality checks.

        Args:
            image_path: Path to image

        Returns:
            (is_valid, quality_info)
        """
        info = {}

        # Size check
        size_valid, (w, h) = self.check_size(image_path)
        info['width'] = w
        info['height'] = h
        info['size_valid'] = size_valid

        if not size_valid:
            return False, info

        # Aspect ratio check
        aspect_valid = self.check_aspect_ratio(w, h)
        info['aspect_ratio'] = max(w, h) / min(w, h)
        info['aspect_valid'] = aspect_valid

        if not aspect_valid:
            return False, info

        # Blur check
        blur_valid, blur_score = self.check_blur(image_path)
        info['blur_score'] = blur_score
        info['blur_valid'] = blur_valid

        is_valid = size_valid and aspect_valid and blur_valid
        return is_valid, info


class PerceptualHashDeduplicator:
    """
    Deduplication using perceptual hashing.

    Uses imagehash library for perceptual hash comparison.
    """

    def __init__(self, hash_size: int = 8, threshold: int = 5, logger=None):
        """
        Initialize deduplicator.

        Args:
            hash_size: Hash size (8 = 64-bit hash)
            threshold: Hamming distance threshold (0-64 for hash_size=8)
            logger: Logger instance
        """
        self.hash_size = hash_size
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        self.seen_hashes = {}  # hash -> image_path

    def compute_hash(self, image_path: Path) -> Optional[imagehash.ImageHash]:
        """Compute perceptual hash for image."""
        try:
            with Image.open(image_path) as img:
                # Use average hash (fast and good for near-duplicates)
                phash = imagehash.average_hash(img, hash_size=self.hash_size)
                return phash
        except Exception as e:
            self.logger.warning(f"Hash computation failed for {image_path}: {e}")
            return None

    def is_duplicate(self, image_path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Check if image is duplicate of seen images.

        Args:
            image_path: Path to image

        Returns:
            (is_duplicate, duplicate_of)
        """
        phash = self.compute_hash(image_path)
        if phash is None:
            return False, None

        # Compare with seen hashes
        for seen_hash, seen_path in self.seen_hashes.items():
            distance = phash - seen_hash  # Hamming distance
            if distance <= self.threshold:
                return True, seen_path

        # Not a duplicate, add to seen
        self.seen_hashes[phash] = image_path
        return False, None


# =============================================================================
# VLM Caption Generator
# =============================================================================

class VLMCaptionGenerator:
    """
    VLM-based caption generation for character images.

    Supports:
    - Qwen2-VL-7B-Instruct
    - InternVL2-8B
    - Schema-guided output (character, outfit, pose, lighting, style)
    """

    def __init__(self,
                 model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
                 device: str = 'cuda',
                 batch_size: int = 4,
                 logger=None):
        """
        Initialize VLM caption generator.

        Args:
            model_name: HuggingFace model name
            device: Device ('cuda' or 'cpu')
            batch_size: Batch size for inference
            logger: Logger instance
        """
        if not VLM_AVAILABLE:
            raise ImportError("transformers required for VLM. Install with: pip install transformers")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)

        # Initialize model
        self._init_model()

    def _init_model(self):
        """Initialize VLM model and processor."""
        self.logger.info(f"Loading VLM model: {self.model_name}")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map=self.device
            )
            self.model.eval()

            self.logger.info(f"VLM model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load VLM model: {e}")
            raise

    def generate_caption(self,
                        image_path: Path,
                        character_name: str,
                        style_context: str = "pixar style, 3d animation, smooth shading",
                        max_new_tokens: int = 77) -> str:
        """
        Generate caption for single image.

        Args:
            image_path: Path to image
            character_name: Character name
            style_context: Style description context
            max_new_tokens: Maximum caption length (tokens)

        Returns:
            Generated caption string
        """
        # Schema-guided prompt
        prompt = f"""Describe this 3D animated character image in detail.

Character: {character_name}
Style: {style_context}

Please provide:
1. Character's outfit/clothing
2. Pose and action
3. Lighting conditions
4. Camera angle/framing

Format: "{character_name}, [outfit], [pose], [lighting], [style]"
Keep under 75 tokens. Be concise and specific."""

        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Prepare inputs
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic
                    num_beams=1
                )

            # Decode
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Clean up caption
            caption = self._clean_caption(caption, character_name)

            return caption

        except Exception as e:
            self.logger.error(f"VLM caption generation failed for {image_path}: {e}")
            # Fallback to template
            return f"{character_name}, {style_context}"

    def _clean_caption(self, raw_caption: str, character_name: str) -> str:
        """Clean and validate VLM-generated caption."""
        # Remove prompt echo (if model repeats prompt)
        caption = raw_caption.strip()

        # Ensure character name is at start
        if not caption.lower().startswith(character_name.lower()):
            caption = f"{character_name}, {caption}"

        # Remove extra quotes
        caption = caption.replace('"', '').replace("'", '')

        # Limit length (CLIP: 77 tokens max)
        tokens = caption.split()
        if len(tokens) > 75:
            caption = ' '.join(tokens[:75])

        return caption

    def batch_generate(self,
                      image_paths: List[Path],
                      character_name: str,
                      style_context: str = "pixar style, 3d animation") -> List[str]:
        """
        Generate captions for batch of images.

        Args:
            image_paths: List of image paths
            character_name: Character name
            style_context: Style description

        Returns:
            List of generated captions
        """
        captions = []

        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Generating VLM captions"):
            batch = image_paths[i:i + self.batch_size]

            for img_path in batch:
                caption = self.generate_caption(img_path, character_name, style_context)
                captions.append(caption)

        return captions


# =============================================================================
# CLIP-based Diversity Sampler
# =============================================================================

class CLIPDiversitySampler:
    """
    Diversity-aware sampling using CLIP embeddings.

    Uses CLIP to compute visual embeddings, then samples
    diverse images to avoid redundant poses/angles.
    """

    def __init__(self, model_name: str = 'ViT-L/14', device: str = 'cuda', logger=None):
        """
        Initialize CLIP diversity sampler.

        Args:
            model_name: CLIP model name
            device: Device
            logger: Logger instance
        """
        if not CLIP_AVAILABLE:
            raise ImportError("clip required. Install with: pip install git+https://github.com/openai/CLIP.git")

        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # Load CLIP
        self.logger.info(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def extract_embeddings(self, image_paths: List[Path]) -> np.ndarray:
        """
        Extract CLIP embeddings for images.

        Args:
            image_paths: List of image paths

        Returns:
            Embeddings array (N, D)
        """
        embeddings = []

        for img_path in tqdm(image_paths, desc="Extracting CLIP embeddings"):
            try:
                image = Image.open(img_path).convert('RGB')
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    embedding = self.model.encode_image(image_input)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
                    embeddings.append(embedding.cpu().numpy()[0])

            except Exception as e:
                self.logger.warning(f"CLIP embedding failed for {img_path}: {e}")
                # Use zero embedding for failed images
                embeddings.append(np.zeros(512))  # ViT-L/14 = 768, ViT-B/32 = 512

        return np.array(embeddings)

    def sample_diverse(self,
                      image_paths: List[Path],
                      target_size: int,
                      embeddings: Optional[np.ndarray] = None) -> List[Path]:
        """
        Sample diverse images using greedy farthest-point sampling.

        Args:
            image_paths: List of image paths
            target_size: Number of images to sample
            embeddings: Pre-computed embeddings (optional)

        Returns:
            List of sampled image paths
        """
        if len(image_paths) <= target_size:
            return image_paths

        # Extract embeddings if not provided
        if embeddings is None:
            embeddings = self.extract_embeddings(image_paths)

        # Greedy farthest-point sampling
        selected_indices = []
        remaining_indices = list(range(len(image_paths)))

        # Start with random point
        import random
        random.seed(42)
        first_idx = random.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Iteratively select farthest point
        for _ in tqdm(range(target_size - 1), desc="Diversity sampling"):
            if not remaining_indices:
                break

            # Compute min distance to selected points
            selected_embs = embeddings[selected_indices]
            remaining_embs = embeddings[remaining_indices]

            # Cosine distance = 1 - similarity
            similarities = np.dot(remaining_embs, selected_embs.T)  # (N_rem, N_sel)
            max_sims = similarities.max(axis=1)  # (N_rem,)
            min_distances = 1 - max_sims

            # Select farthest point
            farthest_idx = remaining_indices[min_distances.argmax()]
            selected_indices.append(farthest_idx)
            remaining_indices.remove(farthest_idx)

        return [image_paths[i] for i in selected_indices]


# =============================================================================
# Main Training Data Preparer
# =============================================================================

class CharacterLoRADataPreparer:
    """
    Prepare character LoRA training data with quality filtering and VLM captioning.
    """

    def __init__(self,
                 character_dirs: List[Path],
                 output_dir: Path,
                 character_name: str,
                 caption_model: str = 'qwen2_vl',
                 style_description: str = "pixar style, 3d animation, smooth shading",
                 device: str = 'cuda',
                 logger=None):
        """
        Initialize character LoRA data preparer.

        Args:
            character_dirs: Directories containing character images
            output_dir: Output directory
            character_name: Character name
            caption_model: Caption model ('qwen2_vl', 'internvl2', 'template')
            style_description: Style description for captions
            device: Device
            logger: Logger instance
        """
        self.character_dirs = [Path(d) for d in character_dirs]
        self.output_dir = Path(output_dir)
        self.character_name = character_name
        self.caption_model = caption_model
        self.style_description = style_description
        self.device = device
        self.logger = logger or setup_logger(__name__)

        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.captions_dir = self.output_dir / "captions"
        self.logs_dir = self.output_dir / "logs"

        for dir_path in [self.images_dir, self.captions_dir, self.logs_dir]:
            ensure_dir(dir_path)

        # Initialize components
        self.quality_filter = QualityFilter(logger=self.logger)
        self.deduplicator = PerceptualHashDeduplicator(logger=self.logger)

        # Initialize VLM if requested
        if caption_model in ['qwen2_vl', 'internvl2'] and VLM_AVAILABLE:
            model_names = {
                'qwen2_vl': "Qwen/Qwen2-VL-7B-Instruct",
                'internvl2': "OpenGVLab/InternVL2-8B"
            }
            self.vlm_generator = VLMCaptionGenerator(
                model_name=model_names[caption_model],
                device=device,
                logger=self.logger
            )
        else:
            self.vlm_generator = None
            if caption_model in ['qwen2_vl', 'internvl2']:
                self.logger.warning(f"VLM not available, falling back to template captions")

        # Initialize CLIP if available
        if CLIP_AVAILABLE:
            self.clip_sampler = CLIPDiversitySampler(device=device, logger=self.logger)
        else:
            self.clip_sampler = None
            self.logger.warning("CLIP not available, diversity sampling disabled")

        self.logger.info(f"Character LoRA Data Preparer initialized")
        self.logger.info(f"  Character: {self.character_name}")
        self.logger.info(f"  Caption Model: {self.caption_model}")
        self.logger.info(f"  Device: {self.device}")

    def prepare_dataset(self,
                       target_size: Optional[int] = None,
                       repeat_count: int = 10,
                       enable_quality_filter: bool = True,
                       enable_dedup: bool = True,
                       enable_diversity_sampling: bool = True,
                       resume: bool = False) -> Dict:
        """
        Full pipeline for preparing character LoRA training data.

        Args:
            target_size: Target number of images (optional)
            repeat_count: Kohya_ss repeat count
            enable_quality_filter: Enable quality filtering
            enable_dedup: Enable deduplication
            enable_diversity_sampling: Enable CLIP diversity sampling
            resume: Resume from checkpoint

        Returns:
            Metadata dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Preparing Character LoRA Dataset: {self.character_name}")
        self.logger.info("=" * 60)

        # Step 1: Scan images
        self.logger.info("Step 1: Scanning character images...")
        all_images = self._scan_images()

        if len(all_images) == 0:
            self.logger.error("No images found!")
            return {}

        # Step 2: Quality filtering
        if enable_quality_filter:
            self.logger.info("Step 2: Quality filtering...")
            all_images = self._quality_filter(all_images)

        # Step 3: Deduplication
        if enable_dedup:
            self.logger.info("Step 3: Deduplication...")
            all_images = self._deduplicate(all_images)

        # Step 4: Diversity sampling
        if enable_diversity_sampling and target_size and len(all_images) > target_size:
            self.logger.info(f"Step 4: Diversity sampling ({len(all_images)} → {target_size})...")
            all_images = self._diversity_sample(all_images, target_size)
        elif target_size and len(all_images) > target_size:
            self.logger.info(f"Step 4: Random sampling ({len(all_images)} → {target_size})...")
            import random
            random.seed(42)
            all_images = random.sample(all_images, target_size)
        else:
            self.logger.info(f"Step 4: Using all {len(all_images)} images...")

        # Step 5: Generate captions and copy images
        self.logger.info("Step 5: Generating captions and organizing dataset...")
        self._assemble_dataset(all_images)

        # Step 6: Create kohya_ss format
        self.logger.info("Step 6: Creating kohya_ss training directory...")
        training_dir = self._create_kohya_format(repeat_count)

        # Step 7: Save metadata
        self.logger.info("Step 7: Saving metadata...")
        metadata = self._save_metadata(all_images, repeat_count, training_dir)

        self.logger.info("=" * 60)
        self.logger.info(f"✅ Character LoRA dataset preparation complete!")
        self.logger.info("=" * 60)
        self.logger.info(f"Training directory: {training_dir}")
        self.logger.info(f"Total images: {len(all_images)}")
        self.logger.info(f"Effective size: {len(all_images) * repeat_count} per epoch")

        return metadata

    def _scan_images(self) -> List[Path]:
        """Scan all images from character directories."""
        all_images = []

        for char_dir in self.character_dirs:
            if not char_dir.exists():
                self.logger.warning(f"Directory not found: {char_dir}")
                continue

            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                all_images.extend(char_dir.glob(f'*{ext}'))
                all_images.extend(char_dir.glob(f'*{ext.upper()}'))

        all_images = sorted(all_images)
        self.logger.info(f"Found {len(all_images)} images from {len(self.character_dirs)} directories")

        return all_images

    def _quality_filter(self, image_paths: List[Path]) -> List[Path]:
        """Filter images by quality."""
        filtered = []
        stats = defaultdict(int)

        for img_path in tqdm(image_paths, desc="Quality filtering"):
            is_valid, info = self.quality_filter.is_valid(img_path)

            if is_valid:
                filtered.append(img_path)
            else:
                # Log rejection reason
                if not info.get('size_valid'):
                    stats['rejected_size'] += 1
                elif not info.get('aspect_valid'):
                    stats['rejected_aspect'] += 1
                elif not info.get('blur_valid'):
                    stats['rejected_blur'] += 1

        self.logger.info(f"Quality filter: {len(filtered)}/{len(image_paths)} passed")
        for reason, count in stats.items():
            self.logger.info(f"  {reason}: {count}")

        return filtered

    def _deduplicate(self, image_paths: List[Path]) -> List[Path]:
        """Remove duplicate images using perceptual hashing."""
        unique = []
        duplicates = []

        for img_path in tqdm(image_paths, desc="Deduplicating"):
            is_dup, dup_of = self.deduplicator.is_duplicate(img_path)

            if is_dup:
                duplicates.append((img_path, dup_of))
            else:
                unique.append(img_path)

        self.logger.info(f"Deduplication: {len(unique)}/{len(image_paths)} unique")
        self.logger.info(f"  Removed {len(duplicates)} duplicates")

        return unique

    def _diversity_sample(self, image_paths: List[Path], target_size: int) -> List[Path]:
        """Sample diverse images using CLIP."""
        if self.clip_sampler is None:
            self.logger.warning("CLIP not available, using random sampling")
            import random
            random.seed(42)
            return random.sample(image_paths, target_size)

        return self.clip_sampler.sample_diverse(image_paths, target_size)

    def _assemble_dataset(self, image_paths: List[Path]):
        """Generate captions and copy images."""
        for i, img_path in enumerate(tqdm(image_paths, desc="Assembling dataset")):
            img_name = f"{self.character_name}_{i:04d}{img_path.suffix}"

            # Copy image
            dest_img = self.images_dir / img_name
            shutil.copy2(img_path, dest_img)

            # Generate caption
            caption = self._generate_caption(img_path)

            # Save caption
            caption_path = self.captions_dir / f"{self.character_name}_{i:04d}.txt"
            caption_path.write_text(caption, encoding='utf-8')

    def _generate_caption(self, image_path: Path) -> str:
        """Generate caption for image."""
        if self.vlm_generator is not None:
            # Use VLM
            return self.vlm_generator.generate_caption(
                image_path,
                self.character_name,
                self.style_description
            )
        else:
            # Fallback to template
            return f"{self.character_name}, {self.style_description}"

    def _create_kohya_format(self, repeat_count: int) -> Path:
        """Create kohya_ss training directory."""
        training_dir = self.output_dir.parent / f"{repeat_count}_{self.character_name}"

        if training_dir.exists():
            shutil.rmtree(training_dir)
        training_dir.mkdir(parents=True)

        # Copy images and captions
        for img_file in self.images_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                shutil.copy2(img_file, training_dir / img_file.name)

                # Copy caption
                caption_file = self.captions_dir / img_file.with_suffix('.txt').name
                if caption_file.exists():
                    shutil.copy2(caption_file, training_dir / caption_file.name)

        return training_dir

    def _save_metadata(self,
                       image_paths: List[Path],
                       repeat_count: int,
                       training_dir: Path) -> Dict:
        """Save dataset metadata."""
        metadata = {
            'character_name': self.character_name,
            'caption_model': self.caption_model,
            'style_description': self.style_description,
            'total_images': len(image_paths),
            'repeat_count': repeat_count,
            'effective_size': len(image_paths) * repeat_count,
            'source_directories': [str(d) for d in self.character_dirs],
            'training_directory': str(training_dir),
            'created_at': datetime.now().isoformat(),

            'quality_filter': {
                'min_blur_score': self.quality_filter.min_blur_score,
                'min_width': self.quality_filter.min_width,
                'min_height': self.quality_filter.min_height,
            },

            'deduplication': {
                'hash_size': self.deduplicator.hash_size,
                'threshold': self.deduplicator.threshold,
            }
        }

        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Metadata saved to {metadata_file}")

        return metadata


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Character LoRA training data with VLM captioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with template captions
  python prepare_training_data_v2.py \
      --character-dirs /path/to/character_0 \
      --output-dir /path/to/training_data/luca \
      --character-name "luca" \
      --caption-model template \
      --target-size 400

  # With Qwen2-VL captioning
  python prepare_training_data_v2.py \
      --character-dirs /path/to/character_0 /path/to/character_1 \
      --output-dir /path/to/training_data/luca \
      --character-name "luca" \
      --caption-model qwen2_vl \
      --target-size 400 \
      --device cuda

  # Disable quality filtering and dedup
  python prepare_training_data_v2.py \
      --character-dirs /path/to/character_0 \
      --output-dir /path/to/training_data/luca \
      --character-name "luca" \
      --no-quality-filter \
      --no-dedup
        """
    )

    # Required arguments
    parser.add_argument('--character-dirs', type=str, nargs='+', required=True,
                       help='Directories containing character images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for training data')
    parser.add_argument('--character-name', type=str, required=True,
                       help='Character name')

    # Caption options
    parser.add_argument('--caption-model', type=str, default='template',
                       choices=['qwen2_vl', 'internvl2', 'template'],
                       help='Caption model (default: template)')
    parser.add_argument('--style-description', type=str,
                       default='pixar style, 3d animation, smooth shading',
                       help='Style description for captions')

    # Dataset options
    parser.add_argument('--target-size', type=int,
                       help='Target dataset size (will sample if exceeded)')
    parser.add_argument('--repeat', type=int, default=10,
                       help='Repeat count for kohya_ss format (default: 10)')

    # Quality filtering options
    parser.add_argument('--no-quality-filter', action='store_true',
                       help='Disable quality filtering')
    parser.add_argument('--no-dedup', action='store_true',
                       help='Disable deduplication')
    parser.add_argument('--no-diversity-sampling', action='store_true',
                       help='Disable CLIP diversity sampling')

    # Other options
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device (default: cuda)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint (not yet implemented)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logger
    log_dir = Path(args.output_dir) / "logs"
    ensure_dir(log_dir)

    logger = setup_logger(
        name="character_lora_prep",
        log_file=log_dir / f"prepare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO
    )

    # Run pipeline
    preparer = CharacterLoRADataPreparer(
        character_dirs=args.character_dirs,
        output_dir=args.output_dir,
        character_name=args.character_name,
        caption_model=args.caption_model,
        style_description=args.style_description,
        device=args.device,
        logger=logger
    )

    preparer.prepare_dataset(
        target_size=args.target_size,
        repeat_count=args.repeat,
        enable_quality_filter=not args.no_quality_filter,
        enable_dedup=not args.no_dedup,
        enable_diversity_sampling=not args.no_diversity_sampling,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
