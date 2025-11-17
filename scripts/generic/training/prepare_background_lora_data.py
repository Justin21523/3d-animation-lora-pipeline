#!/usr/bin/env python3
"""
Prepare Background LoRA Training Data (Deep Learning Version)

Prepares clean background images for background/scene LoRA training by:
1. Loading SAM2 background layers (with characters removed)
2. Loading LaMa inpainted backgrounds (character regions filled)
3. Extracting deep learning features (CLIP ViT-L/14 or DINOv2)
4. Scene classification (Places365)
5. Clustering similar scenes (HDBSCAN)
6. Generating VLM-assisted scene captions (Qwen2-VL)
7. Quality filtering (blur detection, deduplication)
8. Organizing into kohya_ss training format

Features:
- CLIP ViT-L/14 or DINOv2-g visual embeddings
- Places365 scene type classification
- HDBSCAN density-based clustering
- Qwen2-VL scene description generation
- Perceptual hash deduplication
- Blur and quality filtering

Usage:
    python prepare_background_lora_data.py \\
        --sam2-backgrounds /path/to/sam2/backgrounds/ \\
        --lama-backgrounds /path/to/lama/cleaned/ \\
        --output-dir /path/to/training_data/scene_name/ \\
        --scene-name "portorosso" \\
        --target-size 300 \\
        --enable-vlm-captioning \\
        --device cuda

Author: AI Pipeline
Date: 2025-01-17
"""

import sys
import shutil
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import imagehash

# Transformers for VLM
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    CLIPModel,
    CLIPProcessor
)

# UMAP and HDBSCAN for clustering
try:
    import umap
    import hdbscan
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    print("Warning: umap-learn or hdbscan not installed. Clustering disabled.")

# Add scripts directory to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.core.utils.logger import setup_logger
from scripts.core.utils.path_utils import ensure_dir


# Places365 scene categories (subset of most common outdoor/indoor scenes)
PLACES365_COMMON_SCENES = {
    # Outdoor
    'beach', 'coast', 'forest', 'mountain', 'ocean', 'river', 'lake',
    'field', 'desert', 'canyon', 'valley', 'waterfall', 'sky',
    'street', 'alley', 'plaza', 'park', 'garden', 'yard',

    # Indoor
    'bedroom', 'living_room', 'kitchen', 'bathroom', 'dining_room',
    'office', 'classroom', 'library', 'lobby', 'corridor',
    'restaurant', 'cafe', 'bar', 'store', 'mall',

    # Special
    'underwater', 'inside_bus', 'inside_car', 'cockpit',
    'castle', 'church', 'tower', 'ruins'
}


class DeepLearningFeatureExtractor:
    """
    Extract deep learning features for scene images.

    Supports:
    - CLIP ViT-L/14 (default)
    - DINOv2-g (optional)
    """

    def __init__(self,
                 model_name: str = "openai/clip-vit-large-patch14",
                 device: str = 'cuda',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize feature extractor.

        Args:
            model_name: Model identifier (CLIP or DINOv2)
            device: Device ('cuda' or 'cpu')
            logger: Logger instance
        """
        self.model_name = model_name
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info(f"Loading feature extractor: {model_name}")

        # Load CLIP model
        if 'clip' in model_name.lower():
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            self.feature_dim = self.model.config.vision_config.hidden_size

        elif 'dinov2' in model_name.lower():
            # DINOv2 support (optional)
            try:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(model_name).to(device)
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.model.eval()
                self.feature_dim = self.model.config.hidden_size
            except Exception as e:
                self.logger.error(f"Failed to load DINOv2: {e}")
                raise
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.logger.info(f"Feature extractor loaded (dim={self.feature_dim})")

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract visual features from image.

        Args:
            image: PIL Image

        Returns:
            Feature vector (numpy array)
        """
        # Preprocess image
        if 'clip' in self.model_name.lower():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            features = outputs.cpu().numpy()[0]

        elif 'dinov2' in self.model_name.lower():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # Use CLS token
            features = outputs.last_hidden_state[:, 0].cpu().numpy()[0]

        # L2 normalize
        features = features / (np.linalg.norm(features) + 1e-8)

        return features

    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract features from batch of images.

        Args:
            images: List of PIL Images

        Returns:
            Feature array (N, feature_dim)
        """
        if 'clip' in self.model_name.lower():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            features = outputs.cpu().numpy()

        elif 'dinov2' in self.model_name.lower():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0].cpu().numpy()

        # L2 normalize
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

        return features


class Places365SceneClassifier:
    """
    Scene type classification using Places365 labels.

    Classifies scenes into categories like:
    - Outdoor (beach, forest, mountain, etc.)
    - Indoor (bedroom, kitchen, restaurant, etc.)
    - Special (underwater, inside_vehicle, etc.)
    """

    def __init__(self,
                 device: str = 'cuda',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize scene classifier.

        Args:
            device: Device ('cuda' or 'cpu')
            logger: Logger instance
        """
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # For now, we'll use CLIP zero-shot classification
        # Can be upgraded to actual Places365 model later
        self.logger.info("Initializing Places365 scene classifier (CLIP zero-shot)")

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()

        # Scene category templates
        self.scene_categories = [
            # Outdoor natural
            "a photo of a beach", "a photo of a forest", "a photo of a mountain",
            "a photo of an ocean", "a photo of a river", "a photo of a lake",
            "a photo of a field", "a photo of a desert", "a photo of the sky",

            # Outdoor urban
            "a photo of a street", "a photo of a plaza", "a photo of a park",
            "a photo of a garden", "a photo of a building exterior",

            # Indoor residential
            "a photo of a bedroom", "a photo of a living room", "a photo of a kitchen",
            "a photo of a bathroom", "a photo of a dining room",

            # Indoor public
            "a photo of an office", "a photo of a classroom", "a photo of a restaurant",
            "a photo of a cafe", "a photo of a store", "a photo of a lobby",

            # Special
            "an underwater scene", "inside a vehicle", "a castle interior"
        ]

        self.logger.info(f"Scene classifier initialized ({len(self.scene_categories)} categories)")

    @torch.no_grad()
    def classify(self, image: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Classify scene type.

        Args:
            image: PIL Image
            top_k: Number of top predictions to return

        Returns:
            List of (scene_type, confidence) tuples
        """
        # CLIP zero-shot classification
        inputs = self.clip_processor(
            text=self.scene_categories,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

        # Get top-k predictions
        top_indices = np.argsort(probs)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            scene_label = self.scene_categories[idx].replace("a photo of ", "").replace("an ", "")
            confidence = float(probs[idx])
            results.append((scene_label, confidence))

        return results


class VLMSceneCaptionGenerator:
    """
    VLM-based scene caption generation for background images.

    Uses Qwen2-VL or InternVL2 to generate detailed scene descriptions
    that emphasize lighting, atmosphere, and environmental details.
    """

    def __init__(self,
                 model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
                 device: str = 'cuda',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize VLM caption generator.

        Args:
            model_name: VLM model identifier
            device: Device for inference
            logger: Logger instance
        """
        self.model_name = model_name
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info(f"Loading VLM for scene captioning: {model_name}")

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )

            self.model.eval()
            self.logger.info("VLM scene caption generator loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load VLM: {e}")
            self.logger.warning("Scene captioning will fall back to templates")
            self.processor = None
            self.model = None

    @torch.no_grad()
    def generate_caption(self,
                        image_path: Union[str, Path],
                        scene_name: str,
                        scene_type: Optional[str] = None,
                        max_tokens: int = 77) -> str:
        """
        Generate scene-specific caption.

        Args:
            image_path: Path to background image
            scene_name: Scene identifier (e.g., "portorosso", "underwater")
            scene_type: Detected scene type (e.g., "beach", "forest")
            max_tokens: Maximum caption length

        Returns:
            Generated caption string
        """
        if self.model is None:
            # Fallback to template
            return self._template_caption(scene_name, scene_type)

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Schema-guided prompt for scene descriptions
            prompt = f"""Describe this 3D animated scene background in detail.

Scene: {scene_name}
{f'Type: {scene_type}' if scene_type else ''}

Please provide:
1. Main environment/location (what kind of place is this?)
2. Lighting conditions (time of day, light quality, shadows)
3. Atmosphere and mood
4. Key environmental details (architecture, nature, objects)

Format the output as:
"{scene_name}, [environment type], [lighting description], [atmosphere], pixar style, 3d animation"

Keep under {max_tokens} tokens. Be concise and specific. Focus on visual/lighting qualities."""

            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(self.device)

            # Generate caption
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.7,
                top_p=0.9
            )

            generated_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            # Extract caption from response
            caption = self._extract_caption(generated_text)

            # Validate caption
            if not caption or len(caption.split()) < 5:
                self.logger.warning(f"VLM caption too short, using template fallback")
                return self._template_caption(scene_name, scene_type)

            return caption

        except Exception as e:
            self.logger.error(f"VLM caption generation failed: {e}")
            return self._template_caption(scene_name, scene_type)

    def _extract_caption(self, generated_text: str) -> str:
        """Extract clean caption from VLM response."""
        # VLM response often includes the full conversation
        # Extract just the generated caption

        lines = generated_text.strip().split('\n')

        # Find lines with quotes (likely the formatted caption)
        for line in lines:
            if '"' in line:
                # Extract text between quotes
                start = line.find('"')
                end = line.rfind('"')
                if start != -1 and end != -1 and end > start:
                    return line[start+1:end].strip()

        # Fallback: return last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()

        return ""

    def _template_caption(self, scene_name: str, scene_type: Optional[str] = None) -> str:
        """Generate template-based caption as fallback."""
        parts = [scene_name]

        if scene_type:
            parts.append(scene_type)

        parts.extend([
            "3d animated background",
            "pixar style",
            "detailed environment",
            "professional rendering",
            "high quality"
        ])

        return ", ".join(parts)


class BackgroundQualityFilter:
    """
    Quality filtering for background images.

    Filters:
    - Blur detection (Laplacian variance)
    - Size validation
    - Aspect ratio checks
    """

    def __init__(self,
                 min_blur_score: float = 80.0,
                 min_width: int = 512,
                 min_height: int = 512,
                 max_aspect_ratio: float = 3.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize quality filter.

        Args:
            min_blur_score: Minimum Laplacian variance (lower = more blur)
            min_width: Minimum image width
            min_height: Minimum image height
            max_aspect_ratio: Maximum width/height ratio
            logger: Logger instance
        """
        self.min_blur_score = min_blur_score
        self.min_width = min_width
        self.min_height = min_height
        self.max_aspect_ratio = max_aspect_ratio
        self.logger = logger or logging.getLogger(__name__)

    def check_blur(self, image_path: Union[str, Path]) -> Tuple[bool, float]:
        """
        Check if image is too blurry.

        Args:
            image_path: Path to image

        Returns:
            (is_acceptable, blur_score)
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            return False, 0.0

        # Compute Laplacian variance
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        blur_score = laplacian.var()

        is_acceptable = blur_score >= self.min_blur_score

        return is_acceptable, blur_score

    def check_size(self, image_path: Union[str, Path]) -> Tuple[bool, int, int]:
        """
        Check if image meets size requirements.

        Args:
            image_path: Path to image

        Returns:
            (is_acceptable, width, height)
        """
        img = Image.open(image_path)
        width, height = img.size

        is_acceptable = (width >= self.min_width and height >= self.min_height)

        return is_acceptable, width, height

    def check_aspect_ratio(self, width: int, height: int) -> bool:
        """Check if aspect ratio is acceptable."""
        aspect_ratio = max(width, height) / (min(width, height) + 1e-8)
        return aspect_ratio <= self.max_aspect_ratio

    def filter_image(self, image_path: Union[str, Path]) -> Tuple[bool, Dict]:
        """
        Run all quality checks.

        Args:
            image_path: Path to image

        Returns:
            (passes_all_checks, check_results_dict)
        """
        results = {}

        # Size check
        size_ok, width, height = self.check_size(image_path)
        results['size_ok'] = size_ok
        results['width'] = width
        results['height'] = height

        # Aspect ratio check
        aspect_ok = self.check_aspect_ratio(width, height)
        results['aspect_ok'] = aspect_ok

        # Blur check
        blur_ok, blur_score = self.check_blur(image_path)
        results['blur_ok'] = blur_ok
        results['blur_score'] = blur_score

        # Overall pass/fail
        passes = size_ok and aspect_ok and blur_ok

        return passes, results


class PerceptualHashDeduplicator:
    """
    Deduplication using perceptual hashing.

    Uses imagehash library (pHash/aHash) to detect near-duplicate images.
    """

    def __init__(self,
                 hash_size: int = 16,
                 threshold: int = 8,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize deduplicator.

        Args:
            hash_size: Hash size (larger = more sensitive)
            threshold: Hamming distance threshold (lower = stricter)
            logger: Logger instance
        """
        self.hash_size = hash_size
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)

        # Track seen hashes
        self.seen_hashes = {}  # hash -> image_path

    def compute_hash(self, image_path: Union[str, Path]) -> imagehash.ImageHash:
        """
        Compute perceptual hash.

        Args:
            image_path: Path to image

        Returns:
            ImageHash object
        """
        img = Image.open(image_path)
        phash = imagehash.average_hash(img, hash_size=self.hash_size)
        return phash

    def is_duplicate(self, image_path: Union[str, Path]) -> Tuple[bool, Optional[Path]]:
        """
        Check if image is duplicate of previously seen image.

        Args:
            image_path: Path to image

        Returns:
            (is_duplicate, original_image_path)
        """
        phash = self.compute_hash(image_path)

        # Check against seen hashes
        for seen_hash, seen_path in self.seen_hashes.items():
            distance = phash - seen_hash  # Hamming distance

            if distance <= self.threshold:
                # Duplicate found
                return True, seen_path

        # Not a duplicate, add to seen hashes
        self.seen_hashes[phash] = image_path

        return False, None

    def reset(self):
        """Clear seen hashes."""
        self.seen_hashes.clear()


class BackgroundLoRADataPreparer:
    """
    Prepare background/scene LoRA training data with deep learning features.
    """

    def __init__(self,
                 sam2_backgrounds_dir: Path,
                 lama_backgrounds_dir: Optional[Path] = None,
                 output_dir: Path = None,
                 scene_name: str = "background",
                 feature_model: str = "openai/clip-vit-large-patch14",
                 enable_vlm_captioning: bool = False,
                 vlm_model: str = "Qwen/Qwen2-VL-7B-Instruct",
                 device: str = 'cuda',
                 logger=None):
        """
        Initialize preparer.

        Args:
            sam2_backgrounds_dir: Directory containing SAM2 background layers
            lama_backgrounds_dir: Directory containing LaMa inpainted backgrounds (optional)
            output_dir: Output directory for training data
            scene_name: Name/identifier for the scene
            feature_model: Feature extraction model (CLIP or DINOv2)
            enable_vlm_captioning: Use VLM for caption generation
            vlm_model: VLM model identifier
            device: Device ('cuda' or 'cpu')
            logger: Logger instance
        """
        self.sam2_backgrounds_dir = Path(sam2_backgrounds_dir)
        self.lama_backgrounds_dir = Path(lama_backgrounds_dir) if lama_backgrounds_dir else None
        self.output_dir = Path(output_dir)
        self.scene_name = scene_name
        self.device = device
        self.logger = logger or setup_logger(__name__)

        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.captions_dir = self.output_dir / "captions"
        self.vis_dir = self.output_dir / "visualizations"
        self.logs_dir = self.output_dir / "logs"

        for dir_path in [self.images_dir, self.captions_dir, self.vis_dir, self.logs_dir]:
            ensure_dir(dir_path)

        # Initialize components
        self.logger.info("Initializing deep learning feature extractor...")
        self.feature_extractor = DeepLearningFeatureExtractor(
            model_name=feature_model,
            device=device,
            logger=self.logger
        )

        self.logger.info("Initializing scene classifier...")
        self.scene_classifier = Places365SceneClassifier(
            device=device,
            logger=self.logger
        )

        # VLM caption generator (optional)
        self.vlm_caption_generator = None
        if enable_vlm_captioning:
            self.logger.info("Initializing VLM caption generator...")
            self.vlm_caption_generator = VLMSceneCaptionGenerator(
                model_name=vlm_model,
                device=device,
                logger=self.logger
            )

        # Quality filter
        self.logger.info("Initializing quality filter...")
        self.quality_filter = BackgroundQualityFilter(
            min_blur_score=80.0,
            min_width=512,
            min_height=512,
            logger=self.logger
        )

        # Deduplicator
        self.logger.info("Initializing deduplicator...")
        self.deduplicator = PerceptualHashDeduplicator(
            hash_size=16,
            threshold=8,
            logger=self.logger
        )

        self.logger.info(f"Background LoRA Data Preparer initialized")
        self.logger.info(f"  SAM2 Backgrounds: {self.sam2_backgrounds_dir}")
        if self.lama_backgrounds_dir:
            self.logger.info(f"  LaMa Backgrounds: {self.lama_backgrounds_dir}")
        self.logger.info(f"  Output: {self.output_dir}")
        self.logger.info(f"  Scene: {self.scene_name}")
        self.logger.info(f"  Feature Model: {feature_model}")
        self.logger.info(f"  VLM Captioning: {'Enabled' if enable_vlm_captioning else 'Disabled'}")

    def scan_backgrounds(self) -> List[Tuple[Path, Optional[Path]]]:
        """
        Scan background directories and match SAM2 + LaMa pairs.

        Returns:
            List of (sam2_path, lama_path) tuples
        """
        self.logger.info("Scanning background images...")

        # Get SAM2 backgrounds
        sam2_files = sorted(list(self.sam2_backgrounds_dir.glob("*.[jp][pn]g")))
        self.logger.info(f"  Found {len(sam2_files)} SAM2 backgrounds")

        # Match with LaMa backgrounds if available
        pairs = []
        if self.lama_backgrounds_dir and self.lama_backgrounds_dir.exists():
            lama_files = {f.stem: f for f in self.lama_backgrounds_dir.glob("*.[jp][pn]g")}
            self.logger.info(f"  Found {len(lama_files)} LaMa backgrounds")

            for sam2_file in sam2_files:
                # Try to find matching LaMa file
                lama_file = lama_files.get(sam2_file.stem)
                pairs.append((sam2_file, lama_file))

            matched = sum(1 for _, lama in pairs if lama is not None)
            self.logger.info(f"  Matched {matched}/{len(pairs)} backgrounds with LaMa inpainting")
        else:
            # No LaMa backgrounds, use SAM2 only
            pairs = [(f, None) for f in sam2_files]
            self.logger.info("  No LaMa backgrounds provided, using SAM2 only")

        return pairs

    def load_background_image(self, sam2_path: Path, lama_path: Optional[Path]) -> Image.Image:
        """
        Load background image, preferring LaMa inpainted version.

        Args:
            sam2_path: Path to SAM2 background
            lama_path: Path to LaMa background (optional)

        Returns:
            PIL Image
        """
        # Prefer LaMa inpainted version (cleaner, no character residue)
        if lama_path and lama_path.exists():
            return Image.open(lama_path).convert("RGB")
        else:
            return Image.open(sam2_path).convert("RGB")

    def prepare_dataset(self,
                       target_size: int = 300,
                       enable_clustering: bool = True,
                       min_cluster_size: int = 5,
                       min_samples: int = 2,
                       quality_filtering: bool = True):
        """
        Full pipeline for preparing background LoRA dataset.

        Args:
            target_size: Target number of images
            enable_clustering: Enable HDBSCAN clustering
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
            quality_filtering: Enable quality filtering
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Preparing {self.scene_name} Background LoRA Dataset")
        self.logger.info("=" * 60)

        # Step 1: Scan backgrounds
        self.logger.info("Step 1: Scanning background images...")
        background_pairs = self.scan_backgrounds()

        if len(background_pairs) == 0:
            self.logger.error("No background images found!")
            return

        # Step 2: Quality filtering
        if quality_filtering:
            self.logger.info("Step 2: Quality filtering...")
            background_pairs = self._filter_quality(background_pairs)

        if len(background_pairs) == 0:
            self.logger.error("No backgrounds passed quality filtering!")
            return

        # Step 3: Extract features
        self.logger.info("Step 3: Extracting deep learning features...")
        features, valid_pairs = self._extract_features(background_pairs)

        # Step 4: Scene classification
        self.logger.info("Step 4: Classifying scenes...")
        scene_types = self._classify_scenes(valid_pairs)

        # Step 5: Clustering (optional)
        clusters = None
        if enable_clustering and CLUSTERING_AVAILABLE:
            self.logger.info("Step 5: Clustering scenes with HDBSCAN...")
            clusters = self._cluster_scenes(
                features,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples
            )
        else:
            self.logger.info("Step 5: Skipping clustering")

        # Step 6: Deduplication
        self.logger.info("Step 6: Deduplicating backgrounds...")
        deduplicated_pairs, dedup_indices = self._deduplicate(valid_pairs)

        # Update features and scene types after deduplication
        features = features[dedup_indices]
        scene_types = [scene_types[i] for i in dedup_indices]
        if clusters is not None:
            clusters = [clusters[i] for i in dedup_indices]

        # Step 7: Sample to target size
        if len(deduplicated_pairs) > target_size:
            self.logger.info(f"Step 7: Sampling {target_size} diverse backgrounds from {len(deduplicated_pairs)}...")
            sampled_pairs, sampled_indices = self._sample_diverse(
                deduplicated_pairs,
                features,
                target_size
            )

            # Update metadata
            scene_types = [scene_types[i] for i in sampled_indices]
            if clusters is not None:
                clusters = [clusters[i] for i in sampled_indices]
        else:
            self.logger.info(f"Step 7: Using all {len(deduplicated_pairs)} backgrounds...")
            sampled_pairs = deduplicated_pairs

        # Step 8: Generate captions and assemble dataset
        self.logger.info("Step 8: Generating captions and assembling dataset...")
        self._assemble_dataset(
            sampled_pairs,
            scene_types,
            clusters
        )

        # Step 9: Save metadata
        self.logger.info("Step 9: Saving metadata...")
        self._save_metadata(sampled_pairs, scene_types, clusters)

        self.logger.info("=" * 60)
        self.logger.info(f"✅ Background LoRA dataset preparation complete!")
        self.logger.info("=" * 60)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Total images: {len(sampled_pairs)}")
        self.logger.info(f"Scene: {self.scene_name}")

    def _filter_quality(self, background_pairs: List[Tuple[Path, Optional[Path]]]) -> List[Tuple[Path, Optional[Path]]]:
        """Filter backgrounds by quality."""
        filtered_pairs = []
        filter_stats = defaultdict(int)

        for sam2_path, lama_path in tqdm(background_pairs, desc="Quality filtering"):
            # Use LaMa path if available, otherwise SAM2
            check_path = lama_path if lama_path else sam2_path

            passes, results = self.quality_filter.filter_image(check_path)

            if passes:
                filtered_pairs.append((sam2_path, lama_path))
                filter_stats['passed'] += 1
            else:
                if not results['size_ok']:
                    filter_stats['size_failed'] += 1
                if not results['aspect_ok']:
                    filter_stats['aspect_failed'] += 1
                if not results['blur_ok']:
                    filter_stats['blur_failed'] += 1

        self.logger.info(f"Quality filtering results:")
        self.logger.info(f"  Passed: {filter_stats['passed']}")
        self.logger.info(f"  Failed (size): {filter_stats['size_failed']}")
        self.logger.info(f"  Failed (aspect): {filter_stats['aspect_failed']}")
        self.logger.info(f"  Failed (blur): {filter_stats['blur_failed']}")

        return filtered_pairs

    def _extract_features(self, background_pairs: List[Tuple[Path, Optional[Path]]]) -> Tuple[np.ndarray, List]:
        """Extract deep learning features."""
        features_list = []
        valid_pairs = []

        for sam2_path, lama_path in tqdm(background_pairs, desc="Extracting features"):
            try:
                img = self.load_background_image(sam2_path, lama_path)
                features = self.feature_extractor.extract_features(img)
                features_list.append(features)
                valid_pairs.append((sam2_path, lama_path))
            except Exception as e:
                self.logger.warning(f"Failed to extract features from {sam2_path.name}: {e}")
                continue

        features_array = np.array(features_list)
        self.logger.info(f"Extracted features: {features_array.shape}")

        return features_array, valid_pairs

    def _classify_scenes(self, background_pairs: List[Tuple[Path, Optional[Path]]]) -> List[str]:
        """Classify scene types."""
        scene_types = []

        for sam2_path, lama_path in tqdm(background_pairs, desc="Classifying scenes"):
            try:
                img = self.load_background_image(sam2_path, lama_path)
                predictions = self.scene_classifier.classify(img, top_k=1)

                # Use top prediction
                scene_type = predictions[0][0] if predictions else "unknown"
                scene_types.append(scene_type)

            except Exception as e:
                self.logger.warning(f"Failed to classify {sam2_path.name}: {e}")
                scene_types.append("unknown")

        # Log scene type distribution
        type_counts = defaultdict(int)
        for st in scene_types:
            type_counts[st] += 1

        self.logger.info("Scene type distribution:")
        for st, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
            self.logger.info(f"  {st}: {count}")

        return scene_types

    def _cluster_scenes(self, features: np.ndarray, min_cluster_size: int, min_samples: int) -> List[int]:
        """Cluster scenes using HDBSCAN."""
        self.logger.info("Reducing dimensionality with UMAP...")
        reducer = umap.UMAP(
            n_components=50,
            metric='cosine',
            random_state=42
        )
        reduced_features = reducer.fit_transform(features)

        self.logger.info("Clustering with HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        cluster_labels = clusterer.fit_predict(reduced_features)

        # Count clusters
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        self.logger.info(f"Clustering results:")
        self.logger.info(f"  Clusters: {n_clusters}")
        self.logger.info(f"  Noise: {n_noise}")

        # Log cluster sizes
        cluster_counts = defaultdict(int)
        for label in cluster_labels:
            if label != -1:
                cluster_counts[label] += 1

        for label, count in sorted(cluster_counts.items()):
            self.logger.info(f"  Cluster {label}: {count} images")

        return cluster_labels.tolist()

    def _deduplicate(self, background_pairs: List[Tuple[Path, Optional[Path]]]) -> Tuple[List, List[int]]:
        """Deduplicate backgrounds."""
        deduplicated_pairs = []
        dedup_indices = []
        dup_count = 0

        self.deduplicator.reset()

        for idx, (sam2_path, lama_path) in enumerate(tqdm(background_pairs, desc="Deduplicating")):
            check_path = lama_path if lama_path else sam2_path

            is_dup, original_path = self.deduplicator.is_duplicate(check_path)

            if not is_dup:
                deduplicated_pairs.append((sam2_path, lama_path))
                dedup_indices.append(idx)
            else:
                dup_count += 1

        self.logger.info(f"Deduplication: removed {dup_count} duplicates, kept {len(deduplicated_pairs)}")

        return deduplicated_pairs, dedup_indices

    def _sample_diverse(self,
                       background_pairs: List[Tuple[Path, Optional[Path]]],
                       features: np.ndarray,
                       target_size: int) -> Tuple[List, List[int]]:
        """Sample diverse backgrounds using greedy farthest-point sampling."""
        n_samples = len(background_pairs)

        if n_samples <= target_size:
            return background_pairs, list(range(n_samples))

        # Greedy farthest-point sampling
        sampled_indices = []
        remaining_indices = list(range(n_samples))

        # Start with random seed
        first_idx = np.random.randint(0, n_samples)
        sampled_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        for _ in tqdm(range(target_size - 1), desc="Diversity sampling"):
            selected_features = features[sampled_indices]
            remaining_features = features[remaining_indices]

            # Compute cosine similarities
            similarities = np.dot(remaining_features, selected_features.T)

            # Find farthest point (minimum max similarity)
            max_sims = similarities.max(axis=1)
            min_distances = 1 - max_sims
            farthest_idx = remaining_indices[min_distances.argmax()]

            sampled_indices.append(farthest_idx)
            remaining_indices.remove(farthest_idx)

        sampled_pairs = [background_pairs[i] for i in sampled_indices]

        return sampled_pairs, sampled_indices

    def _assemble_dataset(self,
                         background_pairs: List[Tuple[Path, Optional[Path]]],
                         scene_types: List[str],
                         clusters: Optional[List[int]]):
        """Generate captions and copy images to output directory."""
        for i, (sam2_path, lama_path) in enumerate(tqdm(background_pairs, desc="Assembling dataset")):
            # Load image
            img = self.load_background_image(sam2_path, lama_path)

            # Generate output filename
            scene_type = scene_types[i]
            cluster_id = clusters[i] if clusters else 0

            if clusters:
                output_name = f"{self.scene_name}_cluster{cluster_id}_{i:05d}"
            else:
                output_name = f"{self.scene_name}_{i:05d}"

            output_img_path = self.images_dir / f"{output_name}.jpg"
            output_cap_path = self.captions_dir / f"{output_name}.txt"

            # Save image
            img.save(output_img_path, quality=95)

            # Generate caption
            if self.vlm_caption_generator:
                # Use VLM
                check_path = lama_path if lama_path else sam2_path
                caption = self.vlm_caption_generator.generate_caption(
                    check_path,
                    self.scene_name,
                    scene_type
                )
            else:
                # Use template
                caption = self._template_caption(scene_type)

            # Save caption
            output_cap_path.write_text(caption, encoding='utf-8')

    def _template_caption(self, scene_type: str) -> str:
        """Generate template-based caption."""
        parts = [self.scene_name]

        if scene_type and scene_type != "unknown":
            parts.append(scene_type)

        parts.extend([
            "3d animated background",
            "pixar style",
            "detailed environment",
            "professional rendering",
            "high quality"
        ])

        return ", ".join(parts)

    def _save_metadata(self,
                      background_pairs: List[Tuple[Path, Optional[Path]]],
                      scene_types: List[str],
                      clusters: Optional[List[int]]):
        """Save dataset metadata."""
        # Count scene types
        type_counts = defaultdict(int)
        for st in scene_types:
            type_counts[st] += 1

        # Count clusters
        cluster_counts = defaultdict(int)
        if clusters:
            for cid in clusters:
                if cid != -1:
                    cluster_counts[cid] += 1

        metadata = {
            'scene_name': self.scene_name,
            'total_images': len(background_pairs),
            'device': self.device,
            'feature_model': self.feature_extractor.model_name,
            'vlm_captioning': self.vlm_caption_generator is not None,
            'created_at': datetime.now().isoformat(),

            'scene_type_distribution': dict(type_counts),
            'cluster_distribution': dict(cluster_counts) if clusters else {},

            'sam2_backgrounds_dir': str(self.sam2_backgrounds_dir),
            'lama_backgrounds_dir': str(self.lama_backgrounds_dir) if self.lama_backgrounds_dir else None
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Metadata saved to {metadata_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Background LoRA training data (Deep Learning Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with CLIP features
  python prepare_background_lora_data.py \\
      --sam2-backgrounds /path/to/sam2/backgrounds/ \\
      --output-dir /path/to/training_data/portorosso/ \\
      --scene-name "portorosso" \\
      --target-size 300 \\
      --device cuda

  # With VLM captioning
  python prepare_background_lora_data.py \\
      --sam2-backgrounds /path/to/sam2/backgrounds/ \\
      --lama-backgrounds /path/to/lama/cleaned/ \\
      --output-dir /path/to/training_data/underwater/ \\
      --scene-name "underwater" \\
      --enable-vlm-captioning \\
      --target-size 250 \\
      --device cuda

  # With DINOv2 features
  python prepare_background_lora_data.py \\
      --sam2-backgrounds /path/to/sam2/backgrounds/ \\
      --output-dir /path/to/training_data/forest/ \\
      --scene-name "forest" \\
      --feature-model "facebook/dinov2-giant" \\
      --target-size 200 \\
      --device cuda
        """
    )

    # Required arguments
    parser.add_argument('--sam2-backgrounds', type=str, required=True,
                       help='Directory containing SAM2 background layers')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for training dataset')
    parser.add_argument('--scene-name', type=str, required=True,
                       help='Scene name/identifier (e.g., "portorosso", "underwater")')

    # Optional input
    parser.add_argument('--lama-backgrounds', type=str, default=None,
                       help='Directory containing LaMa inpainted backgrounds (optional)')

    # Feature extraction
    parser.add_argument('--feature-model', type=str,
                       default='openai/clip-vit-large-patch14',
                       help='Feature extraction model (default: CLIP ViT-L/14)')

    # VLM captioning
    parser.add_argument('--enable-vlm-captioning', action='store_true',
                       help='Enable VLM-based scene caption generation')
    parser.add_argument('--vlm-model', type=str,
                       default='Qwen/Qwen2-VL-7B-Instruct',
                       help='VLM model for captioning (default: Qwen2-VL-7B)')

    # Clustering
    parser.add_argument('--enable-clustering', action='store_true',
                       help='Enable HDBSCAN scene clustering')
    parser.add_argument('--min-cluster-size', type=int, default=5,
                       help='Minimum cluster size for HDBSCAN (default: 5)')
    parser.add_argument('--min-samples', type=int, default=2,
                       help='Minimum samples for HDBSCAN (default: 2)')

    # Dataset size
    parser.add_argument('--target-size', type=int, default=300,
                       help='Target number of training images (default: 300)')

    # Quality filtering
    parser.add_argument('--no-quality-filtering', action='store_true',
                       help='Disable quality filtering')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for processing (default: cuda)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logger
    log_dir = Path(args.output_dir) / "logs"
    ensure_dir(log_dir)

    logger = setup_logger(
        name="background_lora_prep",
        log_file=log_dir / f"prepare_background_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO
    )

    # Check UMAP/HDBSCAN if clustering enabled
    if args.enable_clustering and not CLUSTERING_AVAILABLE:
        logger.error("Clustering enabled but umap-learn or hdbscan not installed!")
        logger.error("Install with: pip install umap-learn hdbscan")
        return

    # Run pipeline
    preparer = BackgroundLoRADataPreparer(
        sam2_backgrounds_dir=args.sam2_backgrounds,
        lama_backgrounds_dir=args.lama_backgrounds,
        output_dir=args.output_dir,
        scene_name=args.scene_name,
        feature_model=args.feature_model,
        enable_vlm_captioning=args.enable_vlm_captioning,
        vlm_model=args.vlm_model,
        device=args.device,
        logger=logger
    )

    preparer.prepare_dataset(
        target_size=args.target_size,
        enable_clustering=args.enable_clustering,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        quality_filtering=not args.no_quality_filtering
    )


if __name__ == '__main__':
    main()
