#!/usr/bin/env python3
"""
Action/Pose Clustering for Pose LoRA Training Data
Groups character instances by body pose/action using pose keypoints or visual features.

Usage:
    python scripts/generic/clustering/action_clustering.py \
        /path/to/instances \
        --output-dir /path/to/action_clusters \
        --method visual \
        --device cpu

Methods:
    - visual: CLIP visual embeddings + HDBSCAN (no pose estimation needed)
    - keypoints: RTM-Pose keypoints + geometric clustering (requires pose_estimation.py first)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm

try:
    from sklearn.cluster import HDBSCAN
    from sklearn.preprocessing import StandardScaler
    import umap
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    from PIL import Image
    import open_clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


class ActionClusterer:
    """Cluster character instances by action/pose."""

    def __init__(
        self,
        method: str = "visual",
        device: str = "cpu",
        min_cluster_size: int = 15,
        min_samples: int = 3
    ):
        """Initialize action clusterer.

        Args:
            method: 'visual' or 'keypoints'
            device: 'cpu' or 'cuda'
            min_cluster_size: Minimum instances per action cluster
            min_samples: HDBSCAN min_samples parameter
        """
        self.method = method
        self.device = device
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

        if method == "visual":
            if not HAS_CLIP:
                raise ImportError("visual method requires: pip install open_clip_torch pillow")
            self._load_clip_model()

    def _load_clip_model(self):
        """Load CLIP model for visual embeddings."""
        print("Loading CLIP model for visual feature extraction...")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai',
            device=self.device
        )
        self.clip_model.eval()
        print(f"✓ CLIP model loaded on {self.device}")

    def extract_visual_features(
        self,
        image_paths: List[str],
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 1000
    ) -> np.ndarray:
        """Extract CLIP visual embeddings with checkpointing support.

        Args:
            image_paths: List of image paths
            checkpoint_path: Path to save/load checkpoints
            checkpoint_interval: Save checkpoint every N images (default: 1000)

        Returns:
            Array of shape (N, feature_dim)
        """
        features = []
        start_idx = 0

        # Try to resume from checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📂 Found checkpoint: {checkpoint_path}")
            try:
                checkpoint_data = np.load(checkpoint_path, allow_pickle=True)
                features = checkpoint_data['features'].tolist()
                start_idx = checkpoint_data['processed_count'].item()
                print(f"✓ Resuming from checkpoint: {start_idx}/{len(image_paths)} images processed")

                # Verify checkpoint matches current image list
                if start_idx > len(image_paths):
                    print(f"⚠️  Checkpoint has more images ({start_idx}) than current list ({len(image_paths)})")
                    print(f"   Starting from scratch...")
                    features = []
                    start_idx = 0
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
                print(f"   Starting from scratch...")
                features = []
                start_idx = 0

        total_images = len(image_paths)
        remaining = total_images - start_idx

        print(f"🔧 Extracting CLIP features: {start_idx}/{total_images} → {total_images}")
        print(f"   Checkpoint interval: {checkpoint_interval} images")

        with torch.no_grad():
            for idx, img_path in enumerate(tqdm(
                image_paths[start_idx:],
                desc="Extracting visual features",
                initial=start_idx,
                total=total_images
            )):
                actual_idx = start_idx + idx

                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    image_features = self.clip_model.encode_image(image_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    features.append(image_features.cpu().numpy()[0])
                except Exception as e:
                    print(f"\n⚠️  Warning: Failed to process {img_path}: {e}")
                    # Use zero vector for failed images
                    features.append(np.zeros(512))

                # Save checkpoint periodically
                if checkpoint_path and (actual_idx + 1) % checkpoint_interval == 0:
                    try:
                        np.savez(
                            checkpoint_path,
                            features=np.array(features),
                            processed_count=actual_idx + 1
                        )
                        print(f"\n💾 Checkpoint saved: {actual_idx + 1}/{total_images} images")
                    except Exception as e:
                        print(f"\n⚠️  Failed to save checkpoint: {e}")

        # Save final checkpoint
        if checkpoint_path and len(features) == total_images:
            try:
                np.savez(
                    checkpoint_path,
                    features=np.array(features),
                    processed_count=total_images
                )
                print(f"💾 Final checkpoint saved: {total_images}/{total_images} images")
            except Exception as e:
                print(f"⚠️  Failed to save final checkpoint: {e}")

        return np.array(features)

    def cluster_actions(
        self,
        image_paths: List[str],
        output_dir: str,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 1000
    ) -> Dict:
        """Cluster images by action/pose with checkpointing support.

        Args:
            image_paths: List of image paths
            output_dir: Output directory
            checkpoint_dir: Directory to save checkpoints (default: output_dir/checkpoints)
            checkpoint_interval: Save checkpoint every N images (default: 1000)

        Returns:
            Clustering results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, 'features_checkpoint.npz')

        print(f"\n{'='*60}")
        print(f"ACTION CLUSTERING")
        print(f"{'='*60}")
        print(f"Method: {self.method}")
        print(f"Images: {len(image_paths)}")
        print(f"Min cluster size: {self.min_cluster_size}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"{'='*60}\n")

        # Extract features with checkpointing
        if self.method == "visual":
            features = self.extract_visual_features(
                image_paths,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=checkpoint_interval
            )
        else:
            raise NotImplementedError("keypoints method requires pose estimation first")

        print(f"\n✓ Extracted features: {features.shape}")

        # Dimensionality reduction with UMAP
        print("\n🔧 Reducing dimensions with UMAP...")
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=32,
            metric='cosine',
            random_state=42
        )
        features_reduced = reducer.fit_transform(features)
        print(f"✓ UMAP: {features.shape[1]}D → {features_reduced.shape[1]}D")

        # Clustering with HDBSCAN
        print(f"\n🔧 Clustering with HDBSCAN...")
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(features_reduced)

        # Count clusters
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)

        print(f"\n✓ Found {n_clusters} action clusters")
        print(f"  Noise samples: {n_noise}")

        # Organize images into clusters
        cluster_stats = self._organize_clusters(
            image_paths,
            labels,
            output_dir
        )

        # Save results
        results = {
            'method': self.method,
            'total_images': len(image_paths),
            'n_action_clusters': n_clusters,
            'n_noise': n_noise,
            'min_cluster_size': self.min_cluster_size,
            'cluster_stats': cluster_stats,
            'feature_dim': features.shape[1],
            'reduced_dim': features_reduced.shape[1]
        }

        results_path = os.path.join(output_dir, 'action_clustering.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Action clustering complete!")
        print(f"   Results: {results_path}")
        print(f"   Output: {output_dir}\n")

        return results

    def _organize_clusters(
        self,
        image_paths: List[str],
        labels: np.ndarray,
        output_dir: str
    ) -> Dict:
        """Organize images into cluster directories.

        Args:
            image_paths: List of image paths
            labels: Cluster labels
            output_dir: Output directory

        Returns:
            Cluster statistics
        """
        cluster_stats = {}

        # Group by label
        label_to_images = {}
        for img_path, label in zip(image_paths, labels):
            if label not in label_to_images:
                label_to_images[label] = []
            label_to_images[label].append(img_path)

        # Copy images to cluster directories
        for label, images in tqdm(label_to_images.items(), desc="Organizing clusters"):
            if label == -1:
                cluster_name = "noise"
            else:
                cluster_name = f"action_{label:03d}"

            cluster_dir = os.path.join(output_dir, cluster_name)
            os.makedirs(cluster_dir, exist_ok=True)

            # Copy images
            for img_path in images:
                basename = os.path.basename(img_path)
                dest_path = os.path.join(cluster_dir, basename)
                shutil.copy2(img_path, dest_path)

            cluster_stats[cluster_name] = len(images)

        return cluster_stats


def main():
    parser = argparse.ArgumentParser(description="Action/Pose clustering")
    parser.add_argument(
        "instances_dir",
        help="Directory with character instance images"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for action clusters"
    )
    parser.add_argument(
        "--method",
        default="visual",
        choices=["visual", "keypoints"],
        help="Clustering method"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for processing"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=15,
        help="Minimum instances per action cluster"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="HDBSCAN min_samples parameter"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: output_dir/checkpoints)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N images (default: 1000)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )

    args = parser.parse_args()

    if not HAS_SKLEARN:
        print("Error: Required packages not installed")
        print("Install: pip install scikit-learn umap-learn")
        return 1

    # Find images
    instances_dir = Path(args.instances_dir)
    if not instances_dir.exists():
        print(f"Error: {instances_dir} does not exist")
        return 1

    image_extensions = {'.png', '.jpg', '.jpeg'}
    image_paths = [
        str(p) for p in instances_dir.rglob('*')
        if p.suffix.lower() in image_extensions
    ]

    if len(image_paths) == 0:
        print(f"No images found in {instances_dir}")
        return 1

    # Initialize clusterer
    clusterer = ActionClusterer(
        method=args.method,
        device=args.device,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples
    )

    # Cluster actions with checkpointing
    results = clusterer.cluster_actions(
        image_paths,
        args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval
    )

    print(f"\n{'='*60}")
    print(f"📊 CLUSTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {results['total_images']}")
    print(f"Action clusters: {results['n_action_clusters']}")
    print(f"Noise samples: {results['n_noise']}")
    print(f"Feature dim: {results['feature_dim']}D → {results['reduced_dim']}D")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
