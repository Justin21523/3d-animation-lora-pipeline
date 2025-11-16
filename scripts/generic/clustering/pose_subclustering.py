#!/usr/bin/env python3
"""
Pose/View Subclustering for Multi-Character Identity Clusters

After identity-level clustering, this tool subdivides each character identity
into pose/view buckets for more granular control during LoRA training.

Purpose:
    - Separate same identity into pose/view buckets (front/three-quarter/profile/back)
    - Enable balanced angle/pose sampling for training
    - Improve caption consistency and LoRA generalization

Pipeline:
    1. Identity clusters (input) → Load character instances
    2. Pose estimation (RTM-Pose) → Extract keypoints
    3. View classification → Determine camera angle (front/3-4/profile/back)
    4. Pose feature extraction → Normalize and vectorize pose
    5. Subclustering → UMAP + HDBSCAN or KMeans by pose+view
    6. Output → identity_XXX/view_YYY/ folders

Usage:
    python scripts/generic/clustering/pose_subclustering.py \\
        /path/to/identity_clusters \\
        --output-dir /path/to/pose_subclustered \\
        --pose-model rtmpose \\
        --views front,three_quarter,profile,back \\
        --method umap_hdbscan \\
        --visualize
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import umap

# Conditional imports
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from mmpose.apis import init_model, inference_topdown
    from mmpose.structures import merge_data_samples
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False


@dataclass
class PoseInstance:
    """Container for pose estimation results"""
    image_path: Path
    keypoints: np.ndarray  # (N, 3) - x, y, confidence
    bbox: np.ndarray  # (4,) - x1, y1, x2, y2
    view_class: str  # front, three_quarter, profile, back
    pose_features: np.ndarray  # Normalized pose vector


class RTMPoseEstimator:
    """RTM-Pose based pose estimation"""

    def __init__(self, model_name: str = "rtmpose-m", device: str = "cuda"):
        """
        Initialize RTM-Pose model

        Args:
            model_name: Model variant (rtmpose-s/m/l)
            device: Device for inference
        """
        self.device = device
        self.model_name = model_name

        if not MMPOSE_AVAILABLE:
            raise ImportError(
                "MMPose not installed. Install with: "
                "pip install mmpose mmcv mmengine"
            )

        # Model configs
        config_map = {
            "rtmpose-s": "rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py",
            "rtmpose-m": "rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py",
            "rtmpose-l": "rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py",
        }

        checkpoint_map = {
            "rtmpose-s": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth",
            "rtmpose-m": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth",
            "rtmpose-l": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth",
        }

        config = config_map.get(model_name, config_map["rtmpose-m"])
        checkpoint = checkpoint_map.get(model_name, checkpoint_map["rtmpose-m"])

        print(f"Loading RTM-Pose model: {model_name}")
        self.model = init_model(config, checkpoint, device=device)
        print(f"✓ RTM-Pose loaded on {device}")

    def estimate_pose(self, image: Image.Image, bbox: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Estimate pose keypoints

        Args:
            image: Input image
            bbox: Optional bounding box [x1, y1, x2, y2]. If None, uses full image.

        Returns:
            Keypoints array (N, 3) with x, y, confidence or None if detection fails
        """
        image_np = np.array(image)

        # If no bbox provided, use full image
        if bbox is None:
            h, w = image_np.shape[:2]
            bbox = np.array([0, 0, w, h])

        # Prepare input
        bboxes = np.array([bbox])

        # Inference
        results = inference_topdown(self.model, image_np, bboxes)

        if len(results) == 0:
            return None

        # Extract keypoints (COCO format: 17 keypoints)
        keypoints = results[0].pred_instances.keypoints[0]  # (17, 2)
        scores = results[0].pred_instances.keypoint_scores[0]  # (17,)

        # Combine into (17, 3) format
        keypoints_with_conf = np.concatenate([keypoints, scores[:, None]], axis=1)

        return keypoints_with_conf


class ViewClassifier:
    """Classify character view angle based on pose keypoints"""

    # COCO keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize view classifier

        Args:
            confidence_threshold: Minimum confidence for keypoint to be valid
        """
        self.conf_thresh = confidence_threshold

    def classify_view(self, keypoints: np.ndarray) -> str:
        """
        Classify view angle based on keypoint geometry

        Args:
            keypoints: (17, 3) array of COCO keypoints

        Returns:
            View class: 'front', 'three_quarter', 'profile', 'back'
        """
        # Extract face keypoints
        nose = keypoints[self.NOSE]
        left_eye = keypoints[self.LEFT_EYE]
        right_eye = keypoints[self.RIGHT_EYE]
        left_ear = keypoints[self.LEFT_EAR]
        right_ear = keypoints[self.RIGHT_EAR]
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]

        # Check visibility
        face_visible = nose[2] > self.conf_thresh
        left_visible = left_eye[2] > self.conf_thresh and left_ear[2] > self.conf_thresh
        right_visible = right_eye[2] > self.conf_thresh and right_ear[2] > self.conf_thresh
        shoulders_visible = (left_shoulder[2] > self.conf_thresh and
                           right_shoulder[2] > self.conf_thresh)

        # Back view: no face visible, both shoulders visible
        if not face_visible and shoulders_visible:
            return "back"

        # Profile: only one side visible
        if left_visible and not right_visible:
            return "profile_left"
        if right_visible and not left_visible:
            return "profile_right"

        # Front vs three-quarter: check shoulder width ratio
        if shoulders_visible and left_visible and right_visible:
            shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
            eye_width = np.linalg.norm(left_eye[:2] - right_eye[:2])

            # Three-quarter: shoulder width >> eye width (perspective)
            if shoulder_width > eye_width * 1.5:
                return "three_quarter"
            else:
                return "front"

        # Default: front
        return "front"

    def normalize_view_label(self, view: str) -> str:
        """Normalize view labels (merge left/right profiles)"""
        if view.startswith("profile"):
            return "profile"
        return view


class PoseFeatureExtractor:
    """Extract normalized pose features for clustering"""

    def __init__(self, feature_dim: int = 34):
        """
        Initialize feature extractor

        Args:
            feature_dim: Dimension of output feature (17 keypoints * 2 = 34)
        """
        self.feature_dim = feature_dim

    def extract_features(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract normalized pose features

        Args:
            keypoints: (17, 3) COCO keypoints

        Returns:
            Normalized feature vector (34,)
        """
        # Use only x, y coordinates
        coords = keypoints[:, :2].copy()  # (17, 2)

        # Normalize: center at origin, scale to unit variance
        center = coords.mean(axis=0)
        coords_centered = coords - center

        scale = np.std(coords_centered)
        if scale > 0:
            coords_normalized = coords_centered / scale
        else:
            coords_normalized = coords_centered

        # Flatten to vector
        features = coords_normalized.flatten()  # (34,)

        return features


class PoseSubclusterer:
    """Main pose/view subclustering class"""

    def __init__(
        self,
        pose_model: str = "rtmpose-m",
        device: str = "cuda",
        method: str = "umap_hdbscan",
        min_cluster_size: int = 5,
        n_clusters: int = 4,
    ):
        """
        Initialize subclustering pipeline

        Args:
            pose_model: RTM-Pose model variant
            device: Device for inference
            method: Clustering method ('umap_hdbscan' or 'kmeans')
            min_cluster_size: Min samples for HDBSCAN
            n_clusters: Number of clusters for KMeans
        """
        self.device = device
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.n_clusters = n_clusters

        # Initialize components
        self.pose_estimator = RTMPoseEstimator(pose_model, device)
        self.view_classifier = ViewClassifier()
        self.feature_extractor = PoseFeatureExtractor()

    def process_identity_cluster(
        self,
        cluster_dir: Path,
        output_dir: Path,
        visualize: bool = False,
    ) -> Dict:
        """
        Process a single identity cluster

        Args:
            cluster_dir: Input identity cluster directory
            output_dir: Output directory for subclustered results
            visualize: Whether to save visualizations

        Returns:
            Processing statistics
        """
        print(f"\n{'='*60}")
        print(f"Processing: {cluster_dir.name}")
        print(f"{'='*60}")

        # Find all images
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(cluster_dir.glob(ext))

        if len(image_paths) == 0:
            print(f"⚠ No images found in {cluster_dir}")
            return {}

        print(f"Found {len(image_paths)} images")

        # Process each image
        pose_instances = []

        for img_path in tqdm(image_paths, desc="Pose estimation"):
            try:
                image = Image.open(img_path).convert('RGB')

                # Estimate pose
                keypoints = self.pose_estimator.estimate_pose(image)

                if keypoints is None:
                    print(f"⚠ No pose detected: {img_path.name}")
                    continue

                # Classify view
                view_class = self.view_classifier.classify_view(keypoints)
                view_class = self.view_classifier.normalize_view_label(view_class)

                # Extract features
                pose_features = self.feature_extractor.extract_features(keypoints)

                # Store result
                pose_instance = PoseInstance(
                    image_path=img_path,
                    keypoints=keypoints,
                    bbox=np.array([0, 0, image.width, image.height]),
                    view_class=view_class,
                    pose_features=pose_features,
                )
                pose_instances.append(pose_instance)

            except Exception as e:
                print(f"✗ Error processing {img_path.name}: {e}")
                continue

        print(f"✓ Successfully processed {len(pose_instances)}/{len(image_paths)} images")

        if len(pose_instances) < self.min_cluster_size:
            print(f"⚠ Too few instances ({len(pose_instances)}), skipping subclustering")
            return {}

        # Subcluster by pose + view
        subcluster_labels = self._subcluster(pose_instances)

        # Organize output
        stats = self._organize_output(
            pose_instances,
            subcluster_labels,
            output_dir,
            cluster_dir.name,
        )

        return stats

    def _subcluster(self, pose_instances: List[PoseInstance]) -> np.ndarray:
        """
        Subcluster instances by pose and view features

        Args:
            pose_instances: List of pose instances

        Returns:
            Cluster labels array
        """
        # Extract features
        pose_features = np.array([inst.pose_features for inst in pose_instances])

        # One-hot encode view classes
        view_classes = [inst.view_class for inst in pose_instances]
        unique_views = sorted(set(view_classes))
        view_to_idx = {v: i for i, v in enumerate(unique_views)}

        view_features = np.zeros((len(view_classes), len(unique_views)))
        for i, view in enumerate(view_classes):
            view_features[i, view_to_idx[view]] = 1.0

        # Combine pose + view features
        combined_features = np.concatenate([pose_features, view_features * 2.0], axis=1)

        # Normalize
        combined_features = normalize(combined_features, norm='l2')

        # Clustering
        if self.method == "umap_hdbscan":
            if not HDBSCAN_AVAILABLE:
                print("⚠ HDBSCAN not available, falling back to KMeans")
                return self._kmeans_cluster(combined_features)

            # UMAP dimensionality reduction
            reducer = umap.UMAP(
                n_neighbors=min(15, len(pose_instances) - 1),
                n_components=min(5, combined_features.shape[1]),
                metric='euclidean',
                random_state=42,
            )
            embedding = reducer.fit_transform(combined_features)

            # HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=2,
                metric='euclidean',
                cluster_selection_epsilon=0.3,
            )
            labels = clusterer.fit_predict(embedding)

        else:  # kmeans
            labels = self._kmeans_cluster(combined_features)

        return labels

    def _kmeans_cluster(self, features: np.ndarray) -> np.ndarray:
        """KMeans clustering"""
        n_clusters = min(self.n_clusters, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        return labels

    def _organize_output(
        self,
        pose_instances: List[PoseInstance],
        labels: np.ndarray,
        output_dir: Path,
        identity_name: str,
    ) -> Dict:
        """
        Organize subclustered results into output directories

        Args:
            pose_instances: List of pose instances
            labels: Cluster labels
            output_dir: Output base directory
            identity_name: Name of identity cluster

        Returns:
            Statistics dict
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group by subcluster
        subclusters = {}
        for instance, label in zip(pose_instances, labels):
            if label == -1:  # Noise
                subcluster_name = f"{identity_name}_noise"
            else:
                subcluster_name = f"{identity_name}_pose_{label:03d}"

            if subcluster_name not in subclusters:
                subclusters[subcluster_name] = []

            subclusters[subcluster_name].append(instance)

        # Copy images to subcluster directories
        stats = {"identity": identity_name, "subclusters": {}}

        for subcluster_name, instances in subclusters.items():
            subcluster_dir = output_dir / subcluster_name
            subcluster_dir.mkdir(parents=True, exist_ok=True)

            # Copy images
            for instance in instances:
                dest_path = subcluster_dir / instance.image_path.name
                shutil.copy2(instance.image_path, dest_path)

            # Collect view distribution
            view_counts = {}
            for inst in instances:
                view_counts[inst.view_class] = view_counts.get(inst.view_class, 0) + 1

            stats["subclusters"][subcluster_name] = {
                "count": len(instances),
                "views": view_counts,
            }

            print(f"  ✓ {subcluster_name}: {len(instances)} images, views={view_counts}")

        return stats

    def process_all_identities(
        self,
        input_dir: Path,
        output_dir: Path,
        visualize: bool = False,
    ) -> Dict:
        """
        Process all identity clusters

        Args:
            input_dir: Directory containing identity_XXX folders
            output_dir: Output directory for subclustered results
            visualize: Whether to save visualizations

        Returns:
            Overall statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all identity clusters
        identity_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

        print(f"\n{'='*60}")
        print(f"Pose/View Subclustering Pipeline")
        print(f"{'='*60}")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Found {len(identity_dirs)} identity clusters")
        print(f"Method: {self.method}")
        print(f"{'='*60}\n")

        all_stats = {}

        for identity_dir in identity_dirs:
            stats = self.process_identity_cluster(
                identity_dir,
                output_dir,
                visualize=visualize,
            )
            all_stats[identity_dir.name] = stats

        # Save overall statistics
        stats_file = output_dir / "pose_subclustering.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"✓ Pose subclustering complete!")
        print(f"✓ Statistics saved: {stats_file}")
        print(f"{'='*60}\n")

        return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Pose/View Subclustering for Identity Clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing identity_XXX folders",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for subclustered results",
    )

    parser.add_argument(
        "--pose-model",
        type=str,
        default="rtmpose-m",
        choices=["rtmpose-s", "rtmpose-m", "rtmpose-l"],
        help="RTM-Pose model variant (default: rtmpose-m)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="umap_hdbscan",
        choices=["umap_hdbscan", "kmeans"],
        help="Clustering method (default: umap_hdbscan)",
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum cluster size for HDBSCAN (default: 5)",
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of clusters for KMeans (default: 4)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save pose visualization images",
    )

    args = parser.parse_args()

    # Initialize subclustering pipeline
    subclusterer = PoseSubclusterer(
        pose_model=args.pose_model,
        device=args.device,
        method=args.method,
        min_cluster_size=args.min_cluster_size,
        n_clusters=args.n_clusters,
    )

    # Process all identities
    stats = subclusterer.process_all_identities(
        args.input_dir,
        args.output_dir,
        visualize=args.visualize,
    )

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for identity, identity_stats in stats.items():
        if not identity_stats:
            continue
        print(f"\n{identity}:")
        for subcluster_name, subcluster_stats in identity_stats.get("subclusters", {}).items():
            print(f"  {subcluster_name}: {subcluster_stats['count']} images")
            for view, count in subcluster_stats["views"].items():
                print(f"    - {view}: {count}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
