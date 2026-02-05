#!/usr/bin/env python3
"""
Scene-based clustering for background images using DINOv2 + hierarchical HDBSCAN.

This script addresses the under-clustering issue in background scene classification
by using DINOv2 (optimized for scene understanding) instead of CLIP (optimized for
object recognition), and implements hierarchical clustering to achieve 50-100+
scene clusters instead of 2-4.

Key improvements:
- DINOv2-giant for scene-level semantic understanding
- Multi-scale features (CLS token + patch token averaging)
- Optional multi-feature fusion (color, layout, texture)
- Hierarchical clustering (coarse location types → fine specific locations)
- Higher-dimensional UMAP (16-32D instead of 2D)
- Adaptive parameters based on dataset size

Usage:
    python scene_clustering.py /path/to/backgrounds_dir \
        --output-dir /path/to/output \
        --model dinov2-giant \
        --hierarchical \
        --use-multi-features \
        --device cuda
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# Setup logging
def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    """Setup logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class DINOv2Embedder:
    """
    DINOv2 feature extractor for scene understanding.

    DINOv2 (Meta AI, 2023) is a self-supervised vision transformer optimized
    for scene-level understanding, unlike CLIP which is optimized for objects.
    """

    MODELS = {
        'dinov2-small': 'facebook/dinov2-small',
        'dinov2-base': 'facebook/dinov2-base',
        'dinov2-large': 'facebook/dinov2-large',
        'dinov2-giant': 'facebook/dinov2-giant',
    }

    EMBEDDING_DIMS = {
        'dinov2-small': 384,
        'dinov2-base': 768,
        'dinov2-large': 1024,
        'dinov2-giant': 1536,
    }

    def __init__(self, model_name: str = 'dinov2-giant', device: str = 'cuda'):
        """
        Initialize DINOv2 embedder.

        Args:
            model_name: One of dinov2-small/base/large/giant
            device: cuda or cpu
        """
        self.model_name = model_name
        self.device = device

        model_path = self.MODELS.get(model_name)
        if not model_path:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.MODELS.keys())}")

        logging.info(f"Loading {model_name} from {model_path}...")
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()

        self.embedding_dim = self.EMBEDDING_DIMS[model_name]
        logging.info(f"✅ {model_name} loaded (embedding dim: {self.embedding_dim}D)")

    @torch.no_grad()
    def extract_features(self, image_path: str, multi_scale: bool = True) -> np.ndarray:
        """
        Extract scene features from image.

        Args:
            image_path: Path to image
            multi_scale: If True, combine CLS token + patch tokens average
                        for richer representation (recommended for scenes)

        Returns:
            Feature vector (1536D for giant, 3072D if multi_scale=True)
        """
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            outputs = self.model(**inputs)

            # CLS token - global scene representation
            cls_token = outputs.last_hidden_state[:, 0, :]  # (1, embedding_dim)

            if multi_scale:
                # Average of patch tokens - spatial scene details
                patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (1, num_patches, embedding_dim)
                spatial_features = patch_tokens.mean(dim=1)  # (1, embedding_dim)

                # Concatenate for richer representation
                combined = torch.cat([cls_token, spatial_features], dim=-1)  # (1, 2*embedding_dim)
                features = combined / combined.norm(dim=-1, keepdim=True)
            else:
                features = cls_token / cls_token.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().flatten()

        except Exception as e:
            logging.warning(f"Failed to extract features from {image_path}: {e}")
            # Return zero vector as fallback
            dim = self.embedding_dim * 2 if multi_scale else self.embedding_dim
            return np.zeros(dim, dtype=np.float32)


class ColorHistogramExtractor:
    """Extract HSV color histogram features for scene understanding."""

    def __init__(self, h_bins: int = 32, s_bins: int = 32, v_bins: int = 32):
        """
        Initialize color histogram extractor.

        Args:
            h_bins: Number of bins for Hue channel
            s_bins: Number of bins for Saturation channel
            v_bins: Number of bins for Value channel
        """
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.feature_dim = h_bins + s_bins + v_bins

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract color histogram features.

        Returns:
            Normalized histogram features (96D by default)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return np.zeros(self.feature_dim, dtype=np.float32)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Compute histograms for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [self.h_bins], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [self.s_bins], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [self.v_bins], [0, 256])

            # Normalize
            hist_h = hist_h.flatten() / (hist_h.sum() + 1e-6)
            hist_s = hist_s.flatten() / (hist_s.sum() + 1e-6)
            hist_v = hist_v.flatten() / (hist_v.sum() + 1e-6)

            return np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)

        except Exception as e:
            logging.warning(f"Failed to extract color features from {image_path}: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)


class SpatialLayoutExtractor:
    """Extract spatial color layout features (grid-based HSV means)."""

    def __init__(self, grid_size: int = 4):
        """
        Initialize spatial layout extractor.

        Args:
            grid_size: Divide image into grid_size x grid_size cells
        """
        self.grid_size = grid_size
        self.feature_dim = grid_size * grid_size * 3  # HSV for each cell

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract spatial layout features.

        Returns:
            Grid-based HSV features (48D for 4x4 grid)
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return np.zeros(self.feature_dim, dtype=np.float32)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, w = hsv.shape[:2]

            features = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    y1 = i * h // self.grid_size
                    y2 = (i + 1) * h // self.grid_size
                    x1 = j * w // self.grid_size
                    x2 = (j + 1) * w // self.grid_size

                    patch = hsv[y1:y2, x1:x2]
                    mean_hsv = patch.mean(axis=(0, 1)) / 255.0  # Normalize to [0, 1]
                    features.extend(mean_hsv)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logging.warning(f"Failed to extract layout features from {image_path}: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)


class TextureDensityExtractor:
    """Extract edge/texture density features using multi-scale Canny edges."""

    def __init__(self, grid_size: int = 4):
        """
        Initialize texture extractor.

        Args:
            grid_size: Divide image into grid_size x grid_size cells
        """
        self.grid_size = grid_size
        self.feature_dim = grid_size * grid_size * 3  # 3 edge thresholds per cell

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract texture density features.

        Returns:
            Multi-scale edge density features (48D for 4x4 grid)
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return np.zeros(self.feature_dim, dtype=np.float32)

            # Multi-scale Canny edges
            edges_50 = cv2.Canny(img, 50, 150)
            edges_100 = cv2.Canny(img, 100, 200)
            edges_150 = cv2.Canny(img, 150, 250)

            h, w = img.shape
            features = []

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    y1 = i * h // self.grid_size
                    y2 = (i + 1) * h // self.grid_size
                    x1 = j * w // self.grid_size
                    x2 = (j + 1) * w // self.grid_size

                    patch_50 = edges_50[y1:y2, x1:x2]
                    patch_100 = edges_100[y1:y2, x1:x2]
                    patch_150 = edges_150[y1:y2, x1:x2]

                    density_50 = patch_50.sum() / (patch_50.size + 1e-6)
                    density_100 = patch_100.sum() / (patch_100.size + 1e-6)
                    density_150 = patch_150.sum() / (patch_150.size + 1e-6)

                    features.extend([density_50, density_100, density_150])

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logging.warning(f"Failed to extract texture features from {image_path}: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)


class HierarchicalSceneClusterer:
    """
    Hierarchical scene clustering with DINOv2 + HDBSCAN.

    Stage 1: Coarse clustering (location types: indoor/outdoor, urban/natural, etc.)
    Stage 2: Fine clustering within each coarse cluster (specific locations)
    """

    def __init__(
        self,
        dinov2_model: str = 'dinov2-giant',
        device: str = 'cuda',
        use_multi_features: bool = False,
        hierarchical: bool = True,
        method: str = 'kmeans',
        k_range: Tuple[int, int] = (30, 150),
    ):
        """
        Initialize hierarchical scene clusterer.

        Args:
            dinov2_model: DINOv2 model variant
            device: cuda or cpu
            use_multi_features: If True, fuse color/layout/texture features
            hierarchical: If True, use two-stage hierarchical clustering
            method: Clustering method ('hdbscan' or 'kmeans')
            k_range: K range for kmeans (min, max)
        """
        self.device = device
        self.use_multi_features = use_multi_features
        self.hierarchical = hierarchical
        self.method = method
        self.k_range = k_range

        # Initialize feature extractors
        self.dinov2 = DINOv2Embedder(dinov2_model, device)

        if use_multi_features:
            self.color_extractor = ColorHistogramExtractor()
            self.layout_extractor = SpatialLayoutExtractor()
            self.texture_extractor = TextureDensityExtractor()

    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract features from all images.

        Args:
            image_paths: List of image paths

        Returns:
            Feature matrix (N x D)
        """
        logging.info(f"Extracting features from {len(image_paths)} images...")

        # Extract DINOv2 features
        dinov2_features = []
        for img_path in tqdm(image_paths, desc="DINOv2 features"):
            feat = self.dinov2.extract_features(img_path, multi_scale=True)
            dinov2_features.append(feat)
        dinov2_features = np.array(dinov2_features)

        if not self.use_multi_features:
            return dinov2_features

        # Extract additional features
        logging.info("Extracting color/layout/texture features...")
        color_features = []
        layout_features = []
        texture_features = []

        for img_path in tqdm(image_paths, desc="Multi-features"):
            color_features.append(self.color_extractor.extract_features(img_path))
            layout_features.append(self.layout_extractor.extract_features(img_path))
            texture_features.append(self.texture_extractor.extract_features(img_path))

        color_features = np.array(color_features)
        layout_features = np.array(layout_features)
        texture_features = np.array(texture_features)

        # Fuse features with weighting
        # DINOv2 (primary): 1.0
        # Color (secondary): 0.5
        # Layout (tertiary): 0.3
        # Texture (tertiary): 0.2
        combined = np.concatenate([
            dinov2_features * 1.0,
            color_features * 0.5,
            layout_features * 0.3,
            texture_features * 0.2,
        ], axis=1)

        # L2 normalization
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        combined = combined / (norms + 1e-6)

        logging.info(f"✅ Feature extraction complete ({combined.shape[1]}D)")

        return combined

    def cluster(
        self,
        features: np.ndarray,
        image_paths: List[str],
        output_dir: str,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform hierarchical clustering.

        Args:
            features: Feature matrix (N x D)
            image_paths: List of image paths
            output_dir: Output directory

        Returns:
            (labels, metadata)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Choose clustering method and structure
        if self.method == 'kmeans':
            if self.hierarchical:
                labels, metadata = self._hierarchical_kmeans(features, image_paths)
            else:
                labels, metadata = self._flat_kmeans(features, image_paths)
        else:  # hdbscan
            if self.hierarchical:
                labels, metadata = self._hierarchical_cluster(features, image_paths)
            else:
                labels, metadata = self._flat_cluster(features, image_paths)

        # Save clustering report
        report_path = os.path.join(output_dir, 'scene_clustering_report.json')
        with open(report_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"✅ Clustering complete: {metadata['n_clusters']} scene clusters")
        logging.info(f"   Report saved to {report_path}")

        return labels, metadata

    def _flat_cluster(
        self,
        features: np.ndarray,
        image_paths: List[str],
    ) -> Tuple[np.ndarray, Dict]:
        """Single-stage flat clustering."""
        n_images = len(image_paths)

        # Adaptive parameters
        min_cluster_size = self._adaptive_min_cluster_size(n_images)
        min_samples = max(5, min_cluster_size // 5)

        logging.info(f"Flat clustering with min_cluster_size={min_cluster_size}, min_samples={min_samples}")

        # UMAP dimensionality reduction
        logging.info("Running UMAP dimensionality reduction...")
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=16,  # Keep 16D (not 2D!)
            metric='cosine',
            random_state=42,
            n_jobs=-1,
        )
        features_reduced = reducer.fit_transform(features)

        # HDBSCAN clustering
        logging.info("Running HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(features_reduced)

        # Compute metrics
        metadata = self._compute_metrics(labels, features_reduced)

        return labels, metadata

    def _hierarchical_cluster(
        self,
        features: np.ndarray,
        image_paths: List[str],
    ) -> Tuple[np.ndarray, Dict]:
        """Two-stage hierarchical clustering."""
        n_images = len(image_paths)

        logging.info("=" * 80)
        logging.info("STAGE 1: Coarse clustering (location types)")
        logging.info("=" * 80)

        # Stage 1: Coarse clustering
        reducer_coarse = umap.UMAP(
            n_neighbors=50,  # Large for global structure
            min_dist=0.0,
            n_components=16,
            metric='cosine',
            random_state=42,
            n_jobs=-1,
        )
        features_coarse = reducer_coarse.fit_transform(features)

        # OPTIMIZED v2: Much more granular - aim for 50-100 coarse clusters
        min_coarse_size = max(20, n_images // 100)  # More aggressive: //100 instead of //50
        clusterer_coarse = hdbscan.HDBSCAN(
            min_cluster_size=min_coarse_size,
            min_samples=3,  # Further reduced from 5 to 3
            metric='euclidean',
            cluster_selection_method='eom',
            core_dist_n_jobs=-1,
        )
        coarse_labels = clusterer_coarse.fit_predict(features_coarse)

        n_coarse = len(set(coarse_labels)) - (1 if -1 in coarse_labels else 0)
        logging.info(f"✅ Found {n_coarse} location types")

        # Stage 2: Fine clustering within each coarse cluster
        logging.info("=" * 80)
        logging.info("STAGE 2: Fine clustering (specific locations)")
        logging.info("=" * 80)

        fine_labels = np.full(len(features), -1, dtype=int)
        global_cluster_id = 0

        coarse_stats = []

        for coarse_id in sorted(set(coarse_labels)):
            if coarse_id == -1:
                continue

            # Get indices for this coarse cluster
            mask = coarse_labels == coarse_id
            cluster_features = features[mask]
            cluster_indices = np.where(mask)[0]

            logging.info(f"\n Location type {coarse_id}: {len(cluster_features)} images")

            # Too small, keep as single cluster (lowered threshold from 20 to 10)
            if len(cluster_features) < 10:
                fine_labels[cluster_indices] = global_cluster_id
                coarse_stats.append({
                    'coarse_id': int(coarse_id),
                    'n_images': len(cluster_features),
                    'n_fine_clusters': 1,
                })
                global_cluster_id += 1
                continue

            # Fine-grained UMAP (OPTIMIZED v2: was //20, now //30 for even finer clusters)
            min_fine_size = max(5, len(cluster_features) // 30)
            reducer_fine = umap.UMAP(
                n_neighbors=min(30, len(cluster_features) // 2),
                min_dist=0.0,
                n_components=10,
                metric='cosine',
                random_state=42,
                n_jobs=-1,
            )
            features_fine = reducer_fine.fit_transform(cluster_features)

            clusterer_fine = hdbscan.HDBSCAN(
                min_cluster_size=min_fine_size,
                min_samples=max(3, min_fine_size // 5),
                metric='euclidean',
                cluster_selection_method='eom',
                core_dist_n_jobs=-1,
            )
            sub_labels = clusterer_fine.fit_predict(features_fine)

            # Remap to global cluster IDs
            n_sub_clusters = 0
            for sub_id in sorted(set(sub_labels)):
                if sub_id == -1:
                    continue
                sub_mask = sub_labels == sub_id
                fine_labels[cluster_indices[sub_mask]] = global_cluster_id
                global_cluster_id += 1
                n_sub_clusters += 1

            coarse_stats.append({
                'coarse_id': int(coarse_id),
                'n_images': len(cluster_features),
                'n_fine_clusters': n_sub_clusters,
            })

            logging.info(f"   → {n_sub_clusters} specific locations")

        n_final = len(set(fine_labels)) - (1 if -1 in fine_labels else 0)
        logging.info("=" * 80)
        logging.info(f"✅ Hierarchical clustering complete: {n_final} total scene clusters")
        logging.info("=" * 80)

        # Compute metrics
        metadata = self._compute_metrics(fine_labels, features_coarse)
        metadata['hierarchical'] = True
        metadata['n_coarse_clusters'] = n_coarse
        metadata['coarse_cluster_stats'] = coarse_stats

        return fine_labels, metadata

    def _flat_kmeans(
        self,
        features: np.ndarray,
        image_paths: List[str],
    ) -> Tuple[np.ndarray, Dict]:
        """
        Single-stage k-means clustering with automatic k selection using Silhouette score.

        This method scans k values in the specified range and selects the optimal k
        that maximizes the Silhouette score, which measures cluster cohesion and separation.

        Args:
            features: Feature matrix (N x D)
            image_paths: List of image paths

        Returns:
            (labels, metadata) where labels are cluster assignments
        """
        n_images = len(image_paths)
        k_min, k_max = self.k_range

        logging.info(f"Flat k-means clustering with k range [{k_min}, {k_max}]")

        # UMAP dimensionality reduction
        logging.info("Running UMAP dimensionality reduction...")
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=16,  # Keep 16D for richer representation
            metric='cosine',
            random_state=42,
            n_jobs=-1,
        )
        features_reduced = reducer.fit_transform(features)

        # Scan k values to find optimal using Silhouette score
        logging.info(f"Scanning k values from {k_min} to {k_max}...")
        k_values = []
        silhouette_scores = []

        # Sample k values logarithmically to reduce computation
        k_candidates = np.unique(np.logspace(
            np.log10(k_min),
            np.log10(min(k_max, n_images - 1)),
            num=min(20, k_max - k_min + 1)
        ).astype(int))

        for k in tqdm(k_candidates, desc="Finding optimal k"):
            if k < 2 or k >= n_images:
                continue

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_temp = kmeans.fit_predict(features_reduced)

            # Compute Silhouette score
            try:
                score = silhouette_score(features_reduced, labels_temp)
                k_values.append(k)
                silhouette_scores.append(score)
            except Exception as e:
                logging.warning(f"Failed to compute Silhouette score for k={k}: {e}")
                continue

        if not k_values:
            logging.error("Failed to find valid k value!")
            # Fallback to middle of range
            optimal_k = (k_min + k_max) // 2
            logging.warning(f"Using fallback k={optimal_k}")
        else:
            # Select k with highest Silhouette score
            optimal_idx = np.argmax(silhouette_scores)
            optimal_k = k_values[optimal_idx]
            optimal_score = silhouette_scores[optimal_idx]

            logging.info(f"✅ Optimal k={optimal_k} (Silhouette score: {optimal_score:.4f})")

        # Run final k-means with optimal k
        logging.info(f"Running k-means with k={optimal_k}...")
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
        labels = kmeans_final.fit_predict(features_reduced)

        # Compute metrics
        metadata = self._compute_metrics(labels, features_reduced)
        metadata['method'] = 'kmeans'
        metadata['k_range'] = list(self.k_range)
        metadata['optimal_k'] = int(optimal_k)
        metadata['k_candidates_tested'] = len(k_values)

        return labels, metadata

    def _hierarchical_kmeans(
        self,
        features: np.ndarray,
        image_paths: List[str],
    ) -> Tuple[np.ndarray, Dict]:
        """
        Two-stage hierarchical k-means clustering.

        Stage 1: Coarse clustering to identify broad location types (indoor/outdoor, etc.)
        Stage 2: Fine-grained clustering within each coarse cluster for specific locations

        This provides better granularity than flat clustering and avoids merging
        distinct scenes that happen to be visually similar.

        Args:
            features: Feature matrix (N x D)
            image_paths: List of image paths

        Returns:
            (labels, metadata) with hierarchical structure preserved
        """
        n_images = len(image_paths)
        k_min, k_max = self.k_range

        logging.info("=" * 80)
        logging.info("STAGE 1: Coarse k-means clustering (location types)")
        logging.info("=" * 80)

        # Stage 1: Coarse clustering with smaller k
        reducer_coarse = umap.UMAP(
            n_neighbors=50,  # Large for global structure
            min_dist=0.0,
            n_components=16,
            metric='cosine',
            random_state=42,
            n_jobs=-1,
        )
        features_coarse = reducer_coarse.fit_transform(features)

        # Coarse k: use sqrt(n) or 10-20% of k_max as heuristic
        k_coarse_min = max(3, int(np.sqrt(n_images) * 0.5))
        k_coarse_max = max(k_coarse_min + 5, int(k_max * 0.2))

        logging.info(f"Coarse k range: [{k_coarse_min}, {k_coarse_max}]")

        # Find optimal coarse k
        k_coarse_candidates = np.unique(np.linspace(
            k_coarse_min,
            min(k_coarse_max, n_images - 1),
            num=min(10, k_coarse_max - k_coarse_min + 1)
        ).astype(int))

        best_coarse_k = k_coarse_min
        best_coarse_score = -1

        for k in k_coarse_candidates:
            if k < 2 or k >= n_images:
                continue
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_temp = kmeans.fit_predict(features_coarse)
            try:
                score = silhouette_score(features_coarse, labels_temp)
                if score > best_coarse_score:
                    best_coarse_score = score
                    best_coarse_k = k
            except:
                continue

        # Final coarse clustering
        logging.info(f"Running coarse k-means with k={best_coarse_k}...")
        kmeans_coarse = KMeans(n_clusters=best_coarse_k, random_state=42, n_init=20)
        coarse_labels = kmeans_coarse.fit_predict(features_coarse)

        n_coarse = len(set(coarse_labels))
        logging.info(f"✅ Found {n_coarse} location types")

        # Stage 2: Fine clustering within each coarse cluster
        logging.info("=" * 80)
        logging.info("STAGE 2: Fine k-means clustering (specific locations)")
        logging.info("=" * 80)

        fine_labels = np.full(len(features), -1, dtype=int)
        global_cluster_id = 0
        coarse_stats = []

        for coarse_id in sorted(set(coarse_labels)):
            # Get indices for this coarse cluster
            mask = coarse_labels == coarse_id
            cluster_features = features[mask]
            cluster_indices = np.where(mask)[0]
            n_cluster = len(cluster_features)

            logging.info(f"\n Location type {coarse_id}: {n_cluster} images")

            # Too small for subclustering
            if n_cluster < 15:
                fine_labels[cluster_indices] = global_cluster_id
                coarse_stats.append({
                    'coarse_id': int(coarse_id),
                    'n_images': n_cluster,
                    'n_fine_clusters': 1,
                })
                global_cluster_id += 1
                logging.info(f"   → Too small, kept as single cluster")
                continue

            # Fine-grained UMAP
            reducer_fine = umap.UMAP(
                n_neighbors=min(30, n_cluster // 2),
                min_dist=0.0,
                n_components=10,
                metric='cosine',
                random_state=42,
                n_jobs=-1,
            )
            features_fine = reducer_fine.fit_transform(cluster_features)

            # Fine k: aim for 3-10 subclusters depending on size
            k_fine_min = max(2, int(np.sqrt(n_cluster * 0.3)))
            k_fine_max = max(k_fine_min + 2, int(np.sqrt(n_cluster * 1.5)))
            k_fine_max = min(k_fine_max, n_cluster - 1)

            # Find optimal fine k
            k_fine_candidates = np.unique(np.linspace(
                k_fine_min,
                k_fine_max,
                num=min(5, k_fine_max - k_fine_min + 1)
            ).astype(int))

            best_fine_k = k_fine_min
            best_fine_score = -1

            for k in k_fine_candidates:
                if k < 2 or k >= n_cluster:
                    continue
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_temp = kmeans.fit_predict(features_fine)
                try:
                    score = silhouette_score(features_fine, labels_temp)
                    if score > best_fine_score:
                        best_fine_score = score
                        best_fine_k = k
                except:
                    continue

            # Final fine clustering
            kmeans_fine = KMeans(n_clusters=best_fine_k, random_state=42, n_init=15)
            sub_labels = kmeans_fine.fit_predict(features_fine)

            # Remap to global cluster IDs
            n_sub_clusters = 0
            for sub_id in sorted(set(sub_labels)):
                sub_mask = sub_labels == sub_id
                fine_labels[cluster_indices[sub_mask]] = global_cluster_id
                global_cluster_id += 1
                n_sub_clusters += 1

            coarse_stats.append({
                'coarse_id': int(coarse_id),
                'n_images': n_cluster,
                'n_fine_clusters': n_sub_clusters,
            })

            logging.info(f"   → {n_sub_clusters} specific locations (k={best_fine_k})")

        n_final = len(set(fine_labels))
        logging.info("=" * 80)
        logging.info(f"✅ Hierarchical k-means complete: {n_final} total scene clusters")
        logging.info("=" * 80)

        # Compute metrics
        metadata = self._compute_metrics(fine_labels, features_coarse)
        metadata['method'] = 'kmeans'
        metadata['hierarchical'] = True
        metadata['k_range'] = list(self.k_range)
        metadata['n_coarse_clusters'] = n_coarse
        metadata['coarse_cluster_stats'] = coarse_stats

        return fine_labels, metadata

    def _adaptive_min_cluster_size(self, n_images: int) -> int:
        """Automatically adjust min_cluster_size based on dataset size."""
        if n_images < 500:
            return 10
        elif n_images < 2000:
            return 20
        elif n_images < 5000:
            return 25
        else:
            return 30

    def _compute_metrics(self, labels: np.ndarray, features: np.ndarray) -> Dict:
        """Compute clustering quality metrics."""
        # Remove noise points for metric computation
        mask = labels != -1
        labels_clean = labels[mask]
        features_clean = features[mask]

        n_clusters = len(set(labels_clean))
        n_noise = (labels == -1).sum()

        metadata = {
            'n_clusters': n_clusters,
            'n_noise': int(n_noise),
            'noise_ratio': float(n_noise / len(labels)),
        }

        if n_clusters > 1:
            # Silhouette score (higher = better, range -1 to 1)
            try:
                sil_score = silhouette_score(features_clean, labels_clean)
                metadata['silhouette_score'] = float(sil_score)
            except:
                metadata['silhouette_score'] = None

            # Davies-Bouldin index (lower = better)
            try:
                db_score = davies_bouldin_score(features_clean, labels_clean)
                metadata['davies_bouldin_score'] = float(db_score)
            except:
                metadata['davies_bouldin_score'] = None

        # Cluster size statistics
        unique, counts = np.unique(labels_clean, return_counts=True)
        metadata['cluster_sizes'] = {
            'mean': float(counts.mean()),
            'std': float(counts.std()),
            'min': int(counts.min()),
            'max': int(counts.max()),
            'median': int(np.median(counts)),
        }

        return metadata

    def organize_clusters(
        self,
        labels: np.ndarray,
        image_paths: List[str],
        output_dir: str,
    ):
        """
        Organize images into cluster directories.

        Args:
            labels: Cluster labels (N,)
            image_paths: List of image paths
            output_dir: Output directory
        """
        logging.info(f"Organizing {len(image_paths)} images into clusters...")

        # Group by cluster
        cluster_groups = defaultdict(list)
        for img_path, label in zip(image_paths, labels):
            cluster_groups[label].append(img_path)

        # Create directories and copy images
        for label, paths in tqdm(cluster_groups.items(), desc="Creating cluster dirs"):
            if label == -1:
                cluster_dir = os.path.join(output_dir, 'noise')
            else:
                cluster_dir = os.path.join(output_dir, f'scene_{label:03d}')

            os.makedirs(cluster_dir, exist_ok=True)

            for img_path in paths:
                filename = os.path.basename(img_path)
                dst_path = os.path.join(cluster_dir, filename)
                shutil.copy2(img_path, dst_path)

        logging.info(f"✅ Clusters organized in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Scene-based clustering for background images using DINOv2 + hierarchical HDBSCAN"
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help="Directory containing background images"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help="Output directory for clustered scenes"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='dinov2-giant',
        choices=['dinov2-small', 'dinov2-base', 'dinov2-large', 'dinov2-giant'],
        help="DINOv2 model variant (default: dinov2-giant for best quality)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        '--method',
        type=str,
        default='kmeans',
        choices=['hdbscan', 'kmeans'],
        help="Clustering method: hdbscan (density-based, may merge similar scenes) or kmeans (fixed-k, better separation). Default: kmeans"
    )
    parser.add_argument(
        '--hierarchical',
        action='store_true',
        help="Use hierarchical clustering (coarse → fine)"
    )
    parser.add_argument(
        '--use-multi-features',
        action='store_true',
        help="Fuse DINOv2 with color/layout/texture features"
    )
    parser.add_argument(
        '--k-range',
        type=int,
        nargs=2,
        default=[30, 150],
        help="K range for kmeans (min max). Default: 30 150"
    )
    parser.add_argument(
        '--image-extensions',
        type=str,
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.webp'],
        help="Image file extensions to process"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(
        'scene_clustering',
        log_file=f'logs/scene_clustering_{Path(args.input_dir).name}.log'
    )

    # Check input directory
    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory not found: {args.input_dir}")
        return 1

    # Collect image paths
    logging.info(f"Scanning {args.input_dir}...")
    image_paths = []
    for ext in args.image_extensions:
        image_paths.extend(Path(args.input_dir).glob(f'**/*{ext}'))
    image_paths = sorted([str(p) for p in image_paths])

    if not image_paths:
        logging.error(f"No images found in {args.input_dir}")
        return 1

    logging.info(f"Found {len(image_paths)} images")

    # Initialize clusterer
    clusterer = HierarchicalSceneClusterer(
        dinov2_model=args.model,
        device=args.device,
        use_multi_features=args.use_multi_features,
        hierarchical=args.hierarchical,
        method=args.method,
        k_range=tuple(args.k_range),
    )

    # Extract features
    features = clusterer.extract_features(image_paths)

    # Cluster
    labels, metadata = clusterer.cluster(features, image_paths, args.output_dir)

    # Organize into directories
    clusterer.organize_clusters(labels, image_paths, args.output_dir)

    # Print summary
    logging.info("=" * 80)
    logging.info("CLUSTERING SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total images: {len(image_paths)}")
    logging.info(f"Scene clusters: {metadata['n_clusters']}")
    logging.info(f"Noise images: {metadata['n_noise']} ({metadata['noise_ratio']*100:.1f}%)")
    if metadata.get('silhouette_score'):
        logging.info(f"Silhouette score: {metadata['silhouette_score']:.3f} (higher = better)")
    if metadata.get('davies_bouldin_score'):
        logging.info(f"Davies-Bouldin index: {metadata['davies_bouldin_score']:.3f} (lower = better)")
    logging.info(f"Cluster sizes: mean={metadata['cluster_sizes']['mean']:.1f}, "
                f"median={metadata['cluster_sizes']['median']}, "
                f"min={metadata['cluster_sizes']['min']}, "
                f"max={metadata['cluster_sizes']['max']}")
    logging.info("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
