"""
Diversity Metrics Calculator

Computes multi-modal diversity metrics for dataset selection:
- Pose diversity (RTM-Pose keypoints)
- Face angle diversity (front/three-quarter/profile/back)
- Semantic diversity (CLIP embeddings)
- Background complexity
- Scale variety
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging


class DiversityMetrics:
    """Multi-modal diversity metrics calculator"""

    def __init__(self, device: str = 'cuda'):
        """
        Initialize diversity metrics calculator

        Args:
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Lazy load models
        self.clip_model = None
        self.clip_processor = None
        self.pose_model = None

    def _load_clip(self):
        """Lazy load CLIP model"""
        if self.clip_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
                model_name = "openai/clip-vit-base-patch32"
                self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(model_name)
                self.clip_model.eval()
                self.logger.info("Loaded CLIP model")
            except Exception as e:
                self.logger.error(f"Failed to load CLIP: {e}")
                raise

    def _load_pose_model(self):
        """Lazy load RTM-Pose model"""
        if self.pose_model is None:
            try:
                from mmpose.apis import init_model
                config = 'rtmpose-m_8xb256-420e_coco-256x192.py'
                checkpoint = 'rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth'
                # This is a placeholder - actual implementation needs proper model loading
                self.logger.info("RTM-Pose model loaded (placeholder)")
            except Exception as e:
                self.logger.warning(f"RTM-Pose not available: {e}")
                self.pose_model = None

    def extract_clip_embeddings(self, image_paths: List[Path]) -> np.ndarray:
        """
        Extract CLIP embeddings from images

        Args:
            image_paths: List of image paths

        Returns:
            Array of CLIP embeddings (N, embedding_dim)
        """
        self._load_clip()

        embeddings = []

        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                inputs = self.clip_processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                embeddings.append(image_features.cpu().numpy().squeeze())

            except Exception as e:
                self.logger.warning(f"Failed to extract CLIP embedding from {img_path}: {e}")
                # Use zero vector as fallback
                embeddings.append(np.zeros(512))

        return np.array(embeddings)

    def compute_pose_features(self, image_paths: List[Path]) -> np.ndarray:
        """
        Compute pose features from images

        Args:
            image_paths: List of image paths

        Returns:
            Array of pose features (N, feature_dim)
        """
        # Placeholder implementation
        # Real implementation would use RTM-Pose to extract keypoints
        # and compute normalized pose features

        self.logger.info("Computing pose features (simplified)")

        # Return random features for now
        # In production, this would be actual pose keypoint features
        return np.random.randn(len(image_paths), 34)  # 17 keypoints * 2 coords

    def classify_face_angle(self, image_path: Path) -> str:
        """
        Classify face viewing angle

        Args:
            image_path: Path to image

        Returns:
            Angle category: 'front', 'three_quarter', 'profile', or 'back'
        """
        # Placeholder implementation
        # Real implementation would use face landmarks or pose estimation

        # For now, return random category
        categories = ['front', 'three_quarter', 'profile', 'back']
        return np.random.choice(categories)

    def compute_background_complexity(self, image_path: Path) -> float:
        """
        Compute background complexity score

        Args:
            image_path: Path to image

        Returns:
            Complexity score (0-1)
        """
        try:
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)

            # Compute edge density as proxy for complexity
            from scipy import ndimage

            # Sobel edge detection
            dx = ndimage.sobel(img_array, axis=0, mode='constant')
            dy = ndimage.sobel(img_array, axis=1, mode='constant')
            mag = np.hypot(dx, dy)

            # Normalize to 0-1
            complexity = np.mean(mag) / 255.0

            return float(complexity)

        except Exception as e:
            self.logger.warning(f"Failed to compute background complexity for {image_path}: {e}")
            return 0.5

    def compute_scale_category(self, image_path: Path) -> str:
        """
        Estimate scale category (close-up, medium, full-body)

        Args:
            image_path: Path to image

        Returns:
            Scale category: 'close_up', 'medium', or 'full_body'
        """
        # Placeholder implementation
        # Real implementation would analyze detected person bbox size

        categories = ['close_up', 'medium', 'full_body']
        return np.random.choice(categories)

    def compute_diversity_matrix(
        self,
        image_paths: List[Path],
        use_clip: bool = True,
        use_pose: bool = True,
        use_angle: bool = True,
        use_background: bool = True,
        use_scale: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute multi-modal diversity feature matrix

        Args:
            image_paths: List of image paths
            use_clip: Include CLIP embeddings
            use_pose: Include pose features
            use_angle: Include face angle categories
            use_background: Include background complexity
            use_scale: Include scale categories

        Returns:
            Dictionary of feature arrays
        """
        self.logger.info(f"Computing diversity features for {len(image_paths)} images...")

        features = {}

        if use_clip:
            self.logger.info("Extracting CLIP embeddings...")
            features['clip'] = self.extract_clip_embeddings(image_paths)

        if use_pose:
            self.logger.info("Computing pose features...")
            features['pose'] = self.compute_pose_features(image_paths)

        if use_angle:
            self.logger.info("Classifying face angles...")
            features['angle'] = np.array([
                self.classify_face_angle(p) for p in image_paths
            ])

        if use_background:
            self.logger.info("Computing background complexity...")
            features['background'] = np.array([
                self.compute_background_complexity(p) for p in image_paths
            ])

        if use_scale:
            self.logger.info("Computing scale categories...")
            features['scale'] = np.array([
                self.compute_scale_category(p) for p in image_paths
            ])

        return features

    def stratified_sample(
        self,
        image_paths: List[Path],
        n_samples: int,
        n_clusters: int = 8,
        quality_scores: Optional[np.ndarray] = None,
        quality_weight: float = 0.3,
        diversity_weight: float = 0.7
    ) -> Tuple[List[Path], Dict]:
        """
        Perform stratified sampling for diversity

        Args:
            image_paths: List of all image paths
            n_samples: Target number of samples
            n_clusters: Number of diversity clusters
            quality_scores: Optional quality scores for each image
            quality_weight: Weight for quality in selection
            diversity_weight: Weight for diversity in selection

        Returns:
            Tuple of (selected_paths, selection_metadata)
        """
        self.logger.info(
            f"Performing stratified sampling: {len(image_paths)} -> {n_samples} images"
        )

        # Compute diversity features
        features = self.compute_diversity_matrix(image_paths)

        # Combine CLIP and pose features for clustering
        combined_features = []
        if 'clip' in features:
            combined_features.append(features['clip'])
        if 'pose' in features:
            combined_features.append(features['pose'])

        if not combined_features:
            raise ValueError("No features available for clustering")

        # Concatenate and normalize features
        feature_matrix = np.concatenate(combined_features, axis=1)
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)

        # Cluster images
        self.logger.info(f"Clustering into {n_clusters} diversity groups...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)

        # Initialize quality scores if not provided
        if quality_scores is None:
            quality_scores = np.ones(len(image_paths))

        # Sample from each cluster
        samples_per_cluster = n_samples // n_clusters
        selected_indices = []
        cluster_info = {}

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # Compute combined scores (quality + diversity)
            cluster_quality = quality_scores[cluster_indices]

            # Compute diversity within cluster (distance to cluster center)
            cluster_center = kmeans.cluster_centers_[cluster_id]
            cluster_features = feature_matrix[cluster_indices]
            distances = np.linalg.norm(cluster_features - cluster_center, axis=1)

            # Normalize distances (higher is more diverse within cluster)
            if len(distances) > 1:
                diversity_scores = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
            else:
                diversity_scores = np.ones_like(distances)

            # Combined score
            combined_scores = (
                quality_weight * cluster_quality +
                diversity_weight * diversity_scores
            )

            # Select top samples from this cluster
            n_select = min(samples_per_cluster, len(cluster_indices))
            top_indices = cluster_indices[np.argsort(combined_scores)[-n_select:]]
            selected_indices.extend(top_indices)

            cluster_info[f'cluster_{cluster_id}'] = {
                'total_size': int(len(cluster_indices)),
                'selected': int(n_select),
                'avg_quality': float(np.mean(cluster_quality)),
                'avg_diversity': float(np.mean(diversity_scores))
            }

        # Select remaining samples if needed
        remaining = n_samples - len(selected_indices)
        if remaining > 0:
            all_indices = set(range(len(image_paths)))
            unselected = list(all_indices - set(selected_indices))
            unselected_scores = quality_scores[unselected]
            additional = np.array(unselected)[np.argsort(unselected_scores)[-remaining:]]
            selected_indices.extend(additional)

        # Get selected paths
        selected_paths = [image_paths[i] for i in selected_indices]

        metadata = {
            'n_clusters': n_clusters,
            'n_selected': len(selected_paths),
            'quality_weight': quality_weight,
            'diversity_weight': diversity_weight,
            'cluster_info': cluster_info
        }

        self.logger.info(f"Selected {len(selected_paths)} diverse images")

        return selected_paths, metadata
