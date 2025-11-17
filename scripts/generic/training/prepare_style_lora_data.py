#!/usr/bin/env python3
"""
Style LoRA Training Data Preparation

Prepares training datasets for Style LoRA models by:
1. Extracting comprehensive style features (color, lighting, texture, rendering)
2. Clustering images by visual style consistency
3. Filtering style outliers and quality issues
4. Generating style-aware captions with VLM
5. Assembling final training dataset with metadata

Style LoRA captures:
- Lighting style (3-point, Rembrandt, volumetric, ambient)
- Color grading (palette, temperature, saturation, LUT)
- Rendering quality (PBR materials, smooth shading, anti-aliasing)
- Post-processing (DoF, bloom, lens effects, vignette)

Usage:
    # From character instances (captures character-consistent style)
    python prepare_style_lora_data.py \\
        --character-instances /path/to/instances/ \\
        --output-dir /path/to/style_lora/ \\
        --style-name "pixar_warm_indoor" \\
        --target-size 300 \\
        --device cuda

    # From mixed sources (backgrounds + characters)
    python prepare_style_lora_data.py \\
        --mixed-sources /path/to/backgrounds /path/to/characters \\
        --output-dir /path/to/style_lora/ \\
        --style-name "cinematic_volumetric" \\
        --target-size 400

Author: AI Pipeline
Date: 2025-01-17
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import imagehash

# ML libraries
import torch
import torch.nn.functional as F
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import umap

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.core.utils.logger import setup_logger
from scripts.core.utils.path_utils import ensure_dir
from scripts.core.utils.checkpoint_manager import CheckpointManager


class StyleFeatureExtractor:
    """
    Extracts comprehensive style features from images:
    - Color features (hue, saturation, temperature, palette)
    - Lighting features (contrast, dynamic range, key direction)
    - Texture features (edge density, detail, smoothness)
    - Rendering features (PBR indicators, anti-aliasing)
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def extract_features(self, image_path: str) -> Dict[str, float]:
        """Extract all style features from an image."""
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.warning(f"Failed to read {image_path}")
            return {}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = {}

        # Color features
        features.update(self._extract_color_features(img_rgb, img_hsv, img_lab))

        # Lighting features
        features.update(self._extract_lighting_features(gray, img_rgb))

        # Texture features
        features.update(self._extract_texture_features(gray))

        # Rendering features
        features.update(self._extract_rendering_features(img_rgb, gray))

        return features

    def _extract_color_features(self, img_rgb: np.ndarray,
                                img_hsv: np.ndarray,
                                img_lab: np.ndarray) -> Dict[str, float]:
        """Extract color-related style features."""
        h, s, v = cv2.split(img_hsv)
        l, a, b = cv2.split(img_lab)
        r, g, b_ch = cv2.split(img_rgb)

        features = {
            # Hue (dominant color)
            'dominant_hue': float(np.median(h)),  # 0-179
            'hue_variance': float(np.var(h)),

            # Saturation (color intensity)
            'saturation_mean': float(np.mean(s)),  # 0-255
            'saturation_std': float(np.std(s)),

            # Value (brightness)
            'value_mean': float(np.mean(v)),
            'value_std': float(np.std(v)),

            # Color temperature (warm vs cool)
            'color_temperature': float((np.mean(r) - np.mean(b_ch)) / 255.0),  # -1 to 1

            # Color palette diversity
            'color_diversity': float(np.std(h)),

            # L*a*b* features (perceptual color)
            'lightness_mean': float(np.mean(l)),
            'a_channel_mean': float(np.mean(a) - 128),  # Red-green axis
            'b_channel_mean': float(np.mean(b) - 128),  # Blue-yellow axis
        }

        return features

    def _extract_lighting_features(self, gray: np.ndarray,
                                   img_rgb: np.ndarray) -> Dict[str, float]:
        """Extract lighting-related style features."""
        # Luminance
        luminance = 0.299 * img_rgb[:,:,0] + 0.587 * img_rgb[:,:,1] + 0.114 * img_rgb[:,:,2]

        features = {
            # Contrast
            'contrast': float(np.std(luminance)),
            'contrast_ratio': float(luminance.max() / (luminance.min() + 1e-6)),

            # Dynamic range
            'dynamic_range': float(luminance.max() - luminance.min()),

            # Histogram distribution (lighting mood)
            'histogram_mean': float(np.mean(gray)),
            'histogram_std': float(np.std(gray)),
            'histogram_skewness': float(stats.skew(gray.flatten())),

            # Shadow/highlight ratio
            'shadow_ratio': float(np.sum(gray < 64) / gray.size),
            'highlight_ratio': float(np.sum(gray > 192) / gray.size),
            'midtone_ratio': float(np.sum((gray >= 64) & (gray <= 192)) / gray.size),
        }

        # Key light direction (approximate from gradient)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        angle = np.arctan2(gy.mean(), gx.mean())

        features['key_light_angle'] = float(angle)  # -π to π
        features['gradient_magnitude'] = float(np.sqrt(gx.mean()**2 + gy.mean()**2))

        return features

    def _extract_texture_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract texture and detail features."""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Laplacian (detail/sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()

        features = {
            # Edge density
            'edge_density': float(edges.sum() / edges.size),

            # Texture detail (higher = more detailed)
            'texture_detail': float(laplacian_var),

            # Smoothness (inverse of detail)
            'smoothness': float(1.0 / (laplacian_var + 1e-6)),
        }

        return features

    def _extract_rendering_features(self, img_rgb: np.ndarray,
                                    gray: np.ndarray) -> Dict[str, float]:
        """Extract rendering quality indicators."""
        # Anti-aliasing indicator (edge smoothness)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.argwhere(edges > 0)

        if len(edge_pixels) > 0:
            # Sample edge neighborhoods
            smoothness_scores = []
            for y, x in edge_pixels[::max(1, len(edge_pixels)//100)]:  # Sample 100 points
                if 2 <= y < gray.shape[0]-2 and 2 <= x < gray.shape[1]-2:
                    neighborhood = gray[y-2:y+3, x-2:x+3]
                    smoothness_scores.append(neighborhood.std())

            aa_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0
        else:
            aa_smoothness = 0

        features = {
            'antialiasing_smoothness': float(aa_smoothness),

            # PBR indicator (color-lighting correlation)
            # PBR materials show consistent color under varying lighting
            'pbr_consistency': float(np.corrcoef(
                img_rgb.reshape(-1, 3).T
            ).mean()),
        }

        return features


class StyleClusterer:
    """
    Clusters images by style consistency using extracted features.
    """

    def __init__(self,
                 method: str = 'hdbscan',
                 min_cluster_size: int = 15,
                 min_samples: int = 3,
                 n_clusters: int = None,
                 logger=None):
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_clusters = n_clusters
        self.logger = logger or logging.getLogger(__name__)

        self.scaler = StandardScaler()
        self.umap_reducer = None
        self.clusterer = None

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Cluster images by style features.

        Args:
            features: (N, D) feature matrix

        Returns:
            labels: (N,) cluster labels (-1 for noise)
        """
        self.logger.info(f"Clustering {len(features)} images by style...")

        # Normalize features
        features_normalized = self.scaler.fit_transform(features)

        # Dimensionality reduction with UMAP
        self.logger.info("Reducing dimensions with UMAP...")
        self.umap_reducer = umap.UMAP(
            n_components=min(20, features.shape[1]),
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        features_reduced = self.umap_reducer.fit_transform(features_normalized)

        # Cluster
        if self.method == 'hdbscan':
            self.logger.info(f"Clustering with HDBSCAN (min_cluster_size={self.min_cluster_size})...")
            self.clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels = self.clusterer.fit_predict(features_reduced)

        elif self.method == 'kmeans':
            k = self.n_clusters or 3
            self.logger.info(f"Clustering with KMeans (k={k})...")
            self.clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = self.clusterer.fit_predict(features_reduced)

        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        # Report
        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})
        n_noise = np.sum(labels == -1)

        self.logger.info(f"Found {n_clusters} style clusters")
        self.logger.info(f"Noise samples: {n_noise}")

        for label in sorted(unique_labels - {-1}):
            count = np.sum(labels == label)
            self.logger.info(f"  Cluster {label}: {count} images")

        return labels


class StyleOutlierFilter:
    """
    Filters style outliers using statistical methods.
    """

    def __init__(self, z_threshold: float = 2.0, logger=None):
        self.z_threshold = z_threshold
        self.logger = logger or logging.getLogger(__name__)

    def filter_outliers(self, features: np.ndarray) -> np.ndarray:
        """
        Filter outliers using Z-score method.

        Args:
            features: (N, D) feature matrix

        Returns:
            mask: (N,) boolean mask (True = keep, False = outlier)
        """
        self.logger.info(f"Filtering style outliers (Z-score threshold: {self.z_threshold})...")

        # Compute Z-scores for each feature
        z_scores = np.abs(stats.zscore(features, axis=0))

        # Mark as outlier if ANY feature exceeds threshold
        is_outlier = (z_scores > self.z_threshold).any(axis=1)

        n_outliers = is_outlier.sum()
        n_kept = (~is_outlier).sum()

        self.logger.info(f"Outliers detected: {n_outliers}")
        self.logger.info(f"Images kept: {n_kept}")

        return ~is_outlier


class StyleQualityFilter:
    """
    Quality filtering for style images (blur detection, size checks).
    """

    def __init__(self,
                 min_blur_score: float = 100.0,
                 min_width: int = 256,
                 min_height: int = 256,
                 logger=None):
        """
        Initialize quality filter.

        Args:
            min_blur_score: Minimum Laplacian variance (higher = sharper)
            min_width: Minimum image width
            min_height: Minimum image height
            logger: Logger instance
        """
        self.min_blur_score = min_blur_score
        self.min_width = min_width
        self.min_height = min_height
        self.logger = logger or logging.getLogger(__name__)

    def check_blur(self, image_path: Path) -> Tuple[bool, float]:
        """Check if image is too blurry using Laplacian variance."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, 0.0

        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        blur_score = laplacian.var()
        is_acceptable = blur_score >= self.min_blur_score

        return is_acceptable, blur_score

    def check_size(self, image_path: Path) -> Tuple[bool, int, int]:
        """Check if image meets size requirements."""
        img = cv2.imread(str(image_path))
        if img is None:
            return False, 0, 0

        height, width = img.shape[:2]
        is_acceptable = width >= self.min_width and height >= self.min_height

        return is_acceptable, width, height

    def filter_image(self, image_path: Path) -> Tuple[bool, Dict]:
        """
        Run all quality checks on an image.

        Returns:
            (passes, results_dict)
        """
        results = {}

        # Check size
        size_ok, width, height = self.check_size(image_path)
        results['size_ok'] = size_ok
        results['width'] = width
        results['height'] = height

        # Check blur
        blur_ok, blur_score = self.check_blur(image_path)
        results['blur_ok'] = blur_ok
        results['blur_score'] = blur_score

        passes = size_ok and blur_ok
        return passes, results


class PerceptualHashDeduplicator:
    """
    Deduplication using perceptual hashing.

    Uses imagehash library (pHash/aHash) to detect near-duplicate images.
    """

    def __init__(self,
                 hash_size: int = 16,
                 threshold: int = 8,
                 logger=None):
        """
        Initialize deduplicator.

        Args:
            hash_size: Hash size (higher = more precise)
            threshold: Hamming distance threshold (lower = stricter)
            logger: Logger instance
        """
        self.hash_size = hash_size
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        self.seen_hashes = {}  # hash -> image_path

    def compute_hash(self, image_path: Path) -> imagehash.ImageHash:
        """Compute perceptual hash for an image."""
        img = Image.open(image_path)
        phash = imagehash.average_hash(img, hash_size=self.hash_size)
        return phash

    def is_duplicate(self, image_path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Check if image is duplicate of previously seen image.

        Returns:
            (is_duplicate, original_path)
        """
        phash = self.compute_hash(image_path)

        # Check against all seen hashes
        for seen_hash, seen_path in self.seen_hashes.items():
            distance = phash - seen_hash  # Hamming distance

            if distance <= self.threshold:
                return True, seen_path

        # Not a duplicate, add to seen
        self.seen_hashes[phash] = image_path
        return False, None

    def reset(self):
        """Clear seen hashes."""
        self.seen_hashes.clear()


class StyleCaptionGenerator:
    """
    Generates style-aware captions using VLM or template-based approach.
    """

    def __init__(self,
                 method: str = 'template',
                 style_name: str = "pixar style",
                 device: str = 'cuda',
                 logger=None):
        self.method = method
        self.style_name = style_name
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        if method == 'qwen2_vl':
            self._init_qwen2_vl()
        elif method == 'internvl2':
            self._init_internvl2()
        # 'template' requires no initialization

    def _init_qwen2_vl(self):
        """Initialize Qwen2-VL model."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            self.logger.info("Loading Qwen2-VL model...")
            model_name = "Qwen/Qwen2-VL-7B-Instruct"

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained(model_name)

            self.logger.info("Qwen2-VL loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load Qwen2-VL: {e}")
            self.logger.info("Falling back to template-based captions")
            self.method = 'template'

    def _init_internvl2(self):
        """Initialize InternVL2 model."""
        try:
            from transformers import AutoModel, AutoTokenizer

            self.logger.info("Loading InternVL2 model...")
            model_name = "OpenGVLab/InternVL2-8B"

            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            self.logger.info("InternVL2 loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load InternVL2: {e}")
            self.logger.info("Falling back to template-based captions")
            self.method = 'template'

    def generate_caption(self,
                        image_path: str,
                        style_features: Dict[str, float] = None) -> str:
        """Generate style-aware caption for an image."""

        if self.method == 'template':
            return self._generate_template_caption(style_features)
        elif self.method == 'qwen2_vl':
            return self._generate_qwen2_caption(image_path, style_features)
        elif self.method == 'internvl2':
            return self._generate_internvl2_caption(image_path, style_features)
        else:
            return f"{self.style_name}"

    def _generate_template_caption(self, style_features: Dict[str, float]) -> str:
        """Generate caption from style features using templates."""
        if not style_features:
            return f"a 3d animated scene, {self.style_name}, high quality rendering"

        caption_parts = [f"a 3d animated scene, {self.style_name}"]

        # Lighting description
        contrast = style_features.get('contrast', 0)
        if contrast > 60:
            caption_parts.append("high contrast lighting")
        elif contrast < 30:
            caption_parts.append("soft ambient lighting")
        else:
            caption_parts.append("balanced studio lighting")

        # Color temperature
        temp = style_features.get('color_temperature', 0)
        if temp > 0.1:
            caption_parts.append("warm color palette")
        elif temp < -0.1:
            caption_parts.append("cool color palette")

        # Saturation
        sat = style_features.get('saturation_mean', 128)
        if sat > 160:
            caption_parts.append("vibrant colors")
        elif sat < 80:
            caption_parts.append("muted colors")

        # Detail level
        detail = style_features.get('texture_detail', 0)
        if detail > 500:
            caption_parts.append("high detail")

        # Rendering quality
        caption_parts.append("smooth shading, PBR materials")

        return ", ".join(caption_parts)

    def _generate_qwen2_caption(self,
                               image_path: str,
                               style_features: Dict[str, float]) -> str:
        """Generate caption using Qwen2-VL."""
        try:
            image = Image.open(image_path).convert('RGB')

            # Style-focused prompt
            prompt = f"""Describe the visual style of this image focusing on:
1. Lighting style (type, direction, mood)
2. Color grading (palette, temperature, saturation)
3. Rendering quality (materials, shading, effects)
4. Post-processing (DoF, bloom, filters)

Keep it concise (40-50 tokens). Start with "{self.style_name}". """

            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False
                )

            caption = self.processor.decode(output[0], skip_special_tokens=True)

            # Clean up
            caption = caption.replace(prompt, "").strip()

            return caption

        except Exception as e:
            self.logger.warning(f"VLM caption failed: {e}, using template")
            return self._generate_template_caption(style_features)

    def _generate_internvl2_caption(self,
                                   image_path: str,
                                   style_features: Dict[str, float]) -> str:
        """Generate caption using InternVL2."""
        try:
            image = Image.open(image_path).convert('RGB')

            prompt = f"<image>\nDescribe the visual style: lighting, color grading, rendering quality. Start with '{self.style_name}'. Be concise."

            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    images=image,
                    max_new_tokens=60,
                    do_sample=False
                )

            caption = self.tokenizer.decode(output[0], skip_special_tokens=True)
            caption = caption.replace(prompt, "").strip()

            return caption

        except Exception as e:
            self.logger.warning(f"VLM caption failed: {e}, using template")
            return self._generate_template_caption(style_features)


class StyleLoRADataPreparer:
    """
    Main orchestrator for Style LoRA data preparation pipeline.
    """

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        # Components
        self.feature_extractor = StyleFeatureExtractor(logger)
        self.clusterer = StyleClusterer(
            method=args.cluster_method,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            n_clusters=args.n_clusters,
            logger=logger
        )
        self.outlier_filter = StyleOutlierFilter(
            z_threshold=args.z_threshold,
            logger=logger
        )
        self.quality_filter = StyleQualityFilter(
            min_blur_score=100.0,
            min_width=256,
            min_height=256,
            logger=logger
        )
        self.deduplicator = PerceptualHashDeduplicator(
            hash_size=16,
            threshold=8,
            logger=logger
        )
        self.caption_generator = StyleCaptionGenerator(
            method=args.caption_method,
            style_name=args.style_name,
            device=args.device,
            logger=logger
        )

        # Initialize checkpoint manager for resume capability
        self.logger.info("Initializing checkpoint manager...")
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_path=Path(args.output_dir) / ".style_feature_extraction_checkpoint.json",
            save_interval=100,  # Save every 100 images (feature extraction is fast)
            logger=logger
        )

        # Prepare output directories
        self.output_dir = Path(args.output_dir)
        ensure_dir(self.output_dir)
        ensure_dir(self.output_dir / "images")
        ensure_dir(self.output_dir / "captions")
        ensure_dir(self.output_dir / "features")
        ensure_dir(self.output_dir / "visualizations")

    def collect_images(self) -> List[Path]:
        """Collect all images from input sources."""
        self.logger.info("Collecting images from input sources...")

        image_paths = []

        # From character instances
        if self.args.character_instances:
            instance_dir = Path(self.args.character_instances)
            if instance_dir.is_dir():
                patterns = ['*.png', '*.jpg', '*.jpeg']
                for pattern in patterns:
                    image_paths.extend(instance_dir.rglob(pattern))

        # From mixed sources
        if self.args.mixed_sources:
            for source_dir in self.args.mixed_sources:
                source_path = Path(source_dir)
                if source_path.is_dir():
                    patterns = ['*.png', '*.jpg', '*.jpeg']
                    for pattern in patterns:
                        image_paths.extend(source_path.rglob(pattern))

        self.logger.info(f"Collected {len(image_paths)} images")

        return image_paths

    def prepare_training_dataset(self):
        """
        Full pipeline:
        1. Collect images
        2. Extract style features
        3. Cluster by style consistency
        4. Filter outliers
        5. Generate captions
        6. Assemble dataset
        """

        # Step 1: Collect images
        image_paths = self.collect_images()
        if len(image_paths) == 0:
            self.logger.error("No images found!")
            return

        # Step 1.5: Quality filtering
        self.logger.info("=" * 60)
        self.logger.info("Step 1.5: Quality filtering (blur, size)...")
        self.logger.info("=" * 60)

        filtered_images = []
        quality_stats = defaultdict(int)

        for img_path in tqdm(image_paths, desc="Quality filtering"):
            passes, results = self.quality_filter.filter_image(img_path)

            if passes:
                filtered_images.append(img_path)
                quality_stats['passed'] += 1
            else:
                if not results['size_ok']:
                    quality_stats['size_failed'] += 1
                if not results['blur_ok']:
                    quality_stats['blur_failed'] += 1

        self.logger.info(f"Quality filtering results:")
        self.logger.info(f"  Passed: {quality_stats['passed']}")
        self.logger.info(f"  Failed (size): {quality_stats['size_failed']}")
        self.logger.info(f"  Failed (blur): {quality_stats['blur_failed']}")

        if len(filtered_images) == 0:
            self.logger.error("No images passed quality filtering!")
            return

        image_paths = filtered_images

        # Step 2: Extract features
        self.logger.info("=" * 60)
        self.logger.info("Step 2: Extracting style features...")
        self.logger.info("=" * 60)

        # Load checkpoint if exists
        if self.checkpoint_mgr.exists():
            self.checkpoint_mgr.load()
            self.logger.info(f"Resuming feature extraction...")

        # Get unprocessed images
        unprocessed = self.checkpoint_mgr.get_unprocessed_items(image_paths)
        self.logger.info(f"Processing {len(unprocessed)} remaining images (already processed: {len(self.checkpoint_mgr)})...")

        features_list = []
        valid_images = []

        for img_path in tqdm(unprocessed, desc="Extracting features"):
            features = self.feature_extractor.extract_features(img_path)
            if features:
                features_list.append(features)
                valid_images.append(img_path)

            # Mark as processed (auto-saves every 100 items)
            self.checkpoint_mgr.mark_processed(img_path)

        # Force save final checkpoint
        self.checkpoint_mgr.save(force=True)

        if len(features_list) == 0:
            self.logger.error("No valid features extracted!")
            return

        # Convert to numpy array
        feature_names = sorted(features_list[0].keys())
        feature_matrix = np.array([
            [f[name] for name in feature_names]
            for f in features_list
        ])

        self.logger.info(f"Extracted {feature_matrix.shape[1]} features from {len(valid_images)} images")

        # Save features
        features_json = {
            'feature_names': feature_names,
            'image_paths': [str(p) for p in valid_images],
            'features': feature_matrix.tolist()
        }
        with open(self.output_dir / "features" / "style_features.json", 'w') as f:
            json.dump(features_json, f, indent=2)

        # Step 3: Cluster by style
        self.logger.info("=" * 60)
        self.logger.info("Step 3: Clustering by style consistency...")
        self.logger.info("=" * 60)

        cluster_labels = self.clusterer.fit_predict(feature_matrix)

        # Step 4: Filter outliers
        if self.args.filter_outliers:
            self.logger.info("=" * 60)
            self.logger.info("Step 4: Filtering style outliers...")
            self.logger.info("=" * 60)

            keep_mask = self.outlier_filter.filter_outliers(feature_matrix)

            # Remove outliers
            valid_images = [img for img, keep in zip(valid_images, keep_mask) if keep]
            feature_matrix = feature_matrix[keep_mask]
            cluster_labels = cluster_labels[keep_mask]
            features_list = [f for f, keep in zip(features_list, keep_mask) if keep]

        # Step 5: Select target cluster (largest non-noise cluster)
        self.logger.info("=" * 60)
        self.logger.info("Step 5: Selecting style-consistent cluster...")
        self.logger.info("=" * 60)

        # Find largest cluster (excluding noise)
        unique_labels = set(cluster_labels) - {-1}
        if len(unique_labels) == 0:
            self.logger.error("No valid clusters found! All images marked as noise.")
            return

        cluster_sizes = {label: np.sum(cluster_labels == label) for label in unique_labels}
        target_cluster = max(cluster_sizes, key=cluster_sizes.get)

        self.logger.info(f"Selected cluster {target_cluster} with {cluster_sizes[target_cluster]} images")

        # Filter to target cluster
        cluster_mask = cluster_labels == target_cluster
        final_images = [img for img, keep in zip(valid_images, cluster_mask) if keep]
        final_features = [f for f, keep in zip(features_list, cluster_mask) if keep]

        # Step 5.5: Deduplication
        self.logger.info("=" * 60)
        self.logger.info("Step 5.5: Deduplicating images...")
        self.logger.info("=" * 60)

        deduplicated_images = []
        deduplicated_features = []
        duplicate_count = 0

        self.deduplicator.reset()

        for img_path, features in tqdm(zip(final_images, final_features), desc="Deduplicating", total=len(final_images)):
            is_dup, original_path = self.deduplicator.is_duplicate(img_path)

            if not is_dup:
                deduplicated_images.append(img_path)
                deduplicated_features.append(features)
            else:
                duplicate_count += 1

        self.logger.info(f"Deduplication complete:")
        self.logger.info(f"  Kept: {len(deduplicated_images)}")
        self.logger.info(f"  Removed: {duplicate_count} duplicates")

        final_images = deduplicated_images
        final_features = deduplicated_features

        # Limit to target size
        if len(final_images) > self.args.target_size:
            self.logger.info(f"Sampling {self.args.target_size} images from {len(final_images)}")
            indices = np.random.choice(len(final_images), self.args.target_size, replace=False)
            final_images = [final_images[i] for i in indices]
            final_features = [final_features[i] for i in indices]

        # Step 6: Generate captions
        self.logger.info("=" * 60)
        self.logger.info("Step 6: Generating style-aware captions...")
        self.logger.info("=" * 60)

        for img_path, features in tqdm(zip(final_images, final_features),
                                       desc="Generating captions",
                                       total=len(final_images)):
            # Generate caption
            caption = self.caption_generator.generate_caption(img_path, features)

            # Copy image
            img_name = img_path.name
            dest_path = self.output_dir / "images" / img_name

            if not dest_path.exists():
                img = cv2.imread(str(img_path))
                cv2.imwrite(str(dest_path), img)

            # Save caption
            caption_path = self.output_dir / "captions" / f"{img_path.stem}.txt"
            with open(caption_path, 'w') as f:
                f.write(caption)

        # Step 7: Save metadata
        self.logger.info("=" * 60)
        self.logger.info("Step 7: Saving dataset metadata...")
        self.logger.info("=" * 60)

        metadata = {
            'style_name': self.args.style_name,
            'dataset_size': len(final_images),
            'target_size': self.args.target_size,
            'cluster_method': self.args.cluster_method,
            'caption_method': self.args.caption_method,
            'feature_names': feature_names,
            'created_at': datetime.now().isoformat(),

            # Style statistics
            'style_statistics': {
                name: {
                    'mean': float(np.mean([f[name] for f in final_features])),
                    'std': float(np.std([f[name] for f in final_features])),
                    'min': float(np.min([f[name] for f in final_features])),
                    'max': float(np.max([f[name] for f in final_features]))
                }
                for name in feature_names
            }
        }

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Clean up checkpoint on successful completion
        self.checkpoint_mgr.cleanup()

        self.logger.info("=" * 60)
        self.logger.info("✅ Style LoRA dataset preparation complete!")
        self.logger.info("=" * 60)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Images: {len(final_images)}")
        self.logger.info(f"Captions: {len(final_images)}")
        self.logger.info("")
        self.logger.info("Next step: Train Style LoRA with this dataset")
        self.logger.info(f"  Recommended rank: 64-96")
        self.logger.info(f"  IMPORTANT: Disable color augmentation in training config!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Style LoRA training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From character instances
  python prepare_style_lora_data.py \\
      --character-instances /path/to/luca_instances/ \\
      --output-dir /path/to/style_lora/pixar_warm \\
      --style-name "pixar style warm indoor lighting" \\
      --target-size 300

  # From mixed sources (backgrounds + characters)
  python prepare_style_lora_data.py \\
      --mixed-sources /path/to/backgrounds/ /path/to/characters/ \\
      --output-dir /path/to/style_lora/cinematic \\
      --style-name "cinematic volumetric lighting" \\
      --target-size 400 \\
      --caption-method qwen2_vl
        """
    )

    # Input sources
    input_group = parser.add_argument_group('Input Sources')
    input_group.add_argument('--character-instances', type=str,
                           help='Directory with character instance images')
    input_group.add_argument('--mixed-sources', nargs='+',
                           help='Multiple source directories (backgrounds + characters)')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for training dataset')
    parser.add_argument('--style-name', type=str, required=True,
                       help='Style name/description (e.g., "pixar style warm indoor")')

    # Dataset parameters
    parser.add_argument('--target-size', type=int, default=300,
                       help='Target dataset size (default: 300)')

    # Clustering
    cluster_group = parser.add_argument_group('Clustering Options')
    cluster_group.add_argument('--cluster-method', type=str, default='hdbscan',
                              choices=['hdbscan', 'kmeans'],
                              help='Clustering method (default: hdbscan)')
    cluster_group.add_argument('--min-cluster-size', type=int, default=15,
                              help='Minimum cluster size for HDBSCAN (default: 15)')
    cluster_group.add_argument('--min-samples', type=int, default=3,
                              help='Minimum samples for HDBSCAN (default: 3)')
    cluster_group.add_argument('--n-clusters', type=int,
                              help='Number of clusters for KMeans (auto if not specified)')

    # Filtering
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument('--filter-outliers', action='store_true',
                             help='Filter style outliers using Z-score')
    filter_group.add_argument('--z-threshold', type=float, default=2.0,
                             help='Z-score threshold for outlier filtering (default: 2.0)')

    # Captioning
    caption_group = parser.add_argument_group('Caption Generation')
    caption_group.add_argument('--caption-method', type=str, default='template',
                              choices=['template', 'qwen2_vl', 'internvl2'],
                              help='Caption generation method (default: template)')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for processing (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Validate inputs
    if not args.character_instances and not args.mixed_sources:
        parser.error("Must provide either --character-instances or --mixed-sources")

    return args


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup logger
    log_dir = Path(args.output_dir) / "logs"
    ensure_dir(log_dir)

    logger = setup_logger(
        name="style_lora_prep",
        log_file=log_dir / f"prepare_style_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO
    )

    logger.info("=" * 60)
    logger.info("Style LoRA Training Data Preparation")
    logger.info("=" * 60)
    logger.info(f"Style name: {args.style_name}")
    logger.info(f"Target size: {args.target_size}")
    logger.info(f"Cluster method: {args.cluster_method}")
    logger.info(f"Caption method: {args.caption_method}")
    logger.info(f"Device: {args.device}")
    logger.info("")

    # Run pipeline
    preparer = StyleLoRADataPreparer(args, logger)
    preparer.prepare_training_dataset()


if __name__ == '__main__':
    main()
