#!/usr/bin/env python3
"""
Multi-Encoder Character Instance Clustering (K-means + HDBSCAN)

Purpose: Cluster character instances by visual similarity using various vision encoders
Optimized for: 3D animated characters where face detection may fail
Supported Encoders: CLIP (ViT-L/14), DINOv2
Features:
  - GPU-accelerated batch processing
  - K-means with auto optimal k detection (Silhouette + Elbow + Davies-Bouldin)
  - HDBSCAN clustering (density-based, produces noise)
  - 100% instance coverage with K-means (no noise)

Usage (K-means with auto k):
    python clip_character_clustering.py \
      /path/to/instances \
      --output-dir /path/to/clustered \
      --method kmeans \
      --k-range 10 50 \
      --encoder clip \
      --batch-size 64 \
      --device cuda

Usage (HDBSCAN):
    python clip_character_clustering.py \
      /path/to/instances \
      --output-dir /path/to/clustered \
      --method hdbscan \
      --min-cluster-size 20 \
      --min-samples 3 \
      --device cuda
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform


@dataclass
class ClusteringConfig:
    """Configuration for CLIP clustering"""
    clip_model: str = "openai/clip-vit-large-patch14"
    min_cluster_size: int = 12
    min_samples: int = 2
    batch_size: int = 64
    use_pca: bool = True
    pca_components: int = 128
    use_umap: bool = True
    umap_components: int = 32
    device: str = "cuda"


class CLIPEmbedder:
    """Extract CLIP visual embeddings from images"""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        """Initialize CLIP model"""
        self.device = device
        self.model_name = model_name

        print(f"🔧 Loading CLIP model: {model_name}")

        from transformers import CLIPProcessor, CLIPModel

        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        print(f"✓ CLIP model loaded on {device}")

    def embed_images(self, image_paths: List[Path], batch_size: int = 64) -> np.ndarray:
        """
        Extract CLIP embeddings for a list of images

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing

        Returns:
            embeddings: (N, 768) array of CLIP embeddings
        """
        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP embeddings"):
                batch_paths = image_paths[i:i + batch_size]

                # Load images
                images = []
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        print(f"⚠️ Error loading {path.name}: {e}")
                        # Use blank image as fallback
                        images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

                # Process batch
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get image embeddings
                outputs = self.model.get_image_features(**inputs)
                batch_embeddings = outputs.cpu().numpy()

                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


class CharacterClusterer:
    """Cluster character instances using K-means or HDBSCAN"""

    def __init__(
        self,
        method: str = 'kmeans',
        min_cluster_size: int = 12,
        min_samples: int = 2,
        k_range: Tuple[int, int] = (10, 50)
    ):
        """
        Initialize clusterer

        Args:
            method: 'kmeans' or 'hdbscan'
            min_cluster_size: For HDBSCAN only
            min_samples: For HDBSCAN only
            k_range: For K-means auto k detection (min_k, max_k)
        """
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.k_range = k_range

    def find_optimal_k(
        self,
        embeddings: np.ndarray,
        k_range: Tuple[int, int],
        sample_size: Optional[int] = None
    ) -> Tuple[int, Dict]:
        """
        Find optimal k using multiple metrics.

        Methods:
        - Silhouette Score: 越高越好 ([-1, 1], cluster內緊密且cluster間分散)
        - Elbow Method: SSE曲線的肘點
        - Davies-Bouldin Index: 越低越好 (cluster內緊密且cluster間分離)
        - Calinski-Harabasz Score: 越高越好 (cluster間分散程度)

        Args:
            embeddings: Feature embeddings
            k_range: (min_k, max_k)
            sample_size: Subsample for faster computation (optional)

        Returns:
            optimal_k: Best k value
            metrics: All computed metrics
        """
        min_k, max_k = k_range

        # Subsample for very large datasets
        if sample_size and len(embeddings) > sample_size:
            print(f"   Subsampling {sample_size}/{len(embeddings)} for k optimization...")
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings

        print(f"\n🔍 Finding optimal k in range [{min_k}, {max_k}]...")

        metrics = {'k': [], 'silhouette': [], 'davies_bouldin': [], 'calinski_harabasz': [], 'inertia': []}

        # Test different k values
        k_values = list(range(min_k, max_k + 1, max(1, (max_k - min_k) // 10)))

        for k in tqdm(k_values, desc="Testing k values"):
            if k >= len(sample_embeddings):
                break

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(sample_embeddings)

            # Skip if only one cluster formed
            if len(set(labels)) < 2:
                continue

            metrics['k'].append(k)
            metrics['silhouette'].append(silhouette_score(sample_embeddings, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(sample_embeddings, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(sample_embeddings, labels))
            metrics['inertia'].append(kmeans.inertia_)

        if len(metrics['k']) == 0:
            print(f"   ⚠️ No valid k found, using min_k={min_k}")
            return min_k, metrics

        # Find optimal k using Silhouette score (primary metric)
        best_idx = np.argmax(metrics['silhouette'])
        optimal_k = metrics['k'][best_idx]

        print(f"   ✓ Optimal k = {optimal_k}")
        print(f"     Silhouette Score: {metrics['silhouette'][best_idx]:.4f}")
        print(f"     Davies-Bouldin Index: {metrics['davies_bouldin'][best_idx]:.4f}")
        print(f"     Calinski-Harabasz Score: {metrics['calinski_harabasz'][best_idx]:.1f}")

        return optimal_k, metrics

    def cluster(
        self,
        embeddings: np.ndarray,
        use_pca: bool = True,
        pca_components: int = 128,
        use_umap: bool = True,
        umap_components: int = 32,
        fixed_k: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """
        Cluster embeddings using dimensionality reduction + K-means/HDBSCAN

        Args:
            embeddings: (N, D) embedding matrix
            use_pca: Apply UMAP first (name kept for compatibility)
            pca_components: Initial UMAP dimensions
            use_umap: Apply final UMAP
            umap_components: Final UMAP dimensions
            fixed_k: Fixed k for K-means (overrides auto detection)

        Returns:
            labels: Cluster labels (-1 = noise for HDBSCAN only)
            info: Clustering statistics
            embeddings_final: Reduced embeddings for visualization
        """
        print(f"\n🔍 Clustering {len(embeddings)} instances using {self.method.upper()}...")

        # Normalize embeddings
        embeddings_norm = normalize(embeddings, norm='l2')

        # Optional initial UMAP reduction
        if use_pca and embeddings.shape[0] > pca_components:
            actual_components = min(pca_components, embeddings.shape[0] - 1, embeddings.shape[1])
            print(f"   Applying UMAP (initial): {embeddings.shape[1]}D → {actual_components}D")
            umap_initial = umap.UMAP(
                n_components=actual_components,
                n_neighbors=min(15, embeddings.shape[0] - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            embeddings_reduced = umap_initial.fit_transform(embeddings_norm)
        else:
            embeddings_reduced = embeddings_norm

        # Optional final UMAP (for visualization or further reduction)
        if use_umap:
            print(f"   Applying UMAP (final): {embeddings_reduced.shape[1]}D → {umap_components}D")
            umap_reducer = umap.UMAP(
                n_components=umap_components,
                n_neighbors=min(15, embeddings.shape[0] - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            embeddings_final = umap_reducer.fit_transform(embeddings_reduced)
        else:
            embeddings_final = embeddings_reduced

        # Clustering
        if self.method == 'kmeans':
            # K-means clustering
            if fixed_k:
                optimal_k = fixed_k
                print(f"   Using fixed k = {optimal_k}")
            else:
                # Auto find optimal k
                optimal_k, k_metrics = self.find_optimal_k(
                    embeddings_final,
                    self.k_range,
                    sample_size=10000  # Subsample if > 10k instances
                )

            print(f"   Running K-means with k={optimal_k}...")
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_final)

            # Statistics
            n_clusters = len(set(labels))
            n_noise = 0  # K-means has no noise

        else:  # HDBSCAN
            print(f"   Running HDBSCAN (min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples})")
            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(embeddings_final)

            # Statistics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()

        cluster_sizes = {}
        for label in set(labels):
            if label != -1:
                cluster_sizes[f"cluster_{label}"] = int((labels == label).sum())

        info = {
            "n_clusters": n_clusters,
            "n_noise": int(n_noise),
            "cluster_sizes": cluster_sizes,
            "noise_ratio": float(n_noise / len(labels))
        }

        print(f"\n✓ Clustering complete:")
        print(f"   Identities found: {n_clusters}")
        print(f"   Noise instances: {n_noise} ({100*n_noise/len(labels):.1f}%)")

        return labels, info, embeddings_final


def organize_clusters(
    image_paths: List[Path],
    labels: np.ndarray,
    output_dir: Path
):
    """Organize images into cluster folders"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Organizing {len(image_paths)} instances into clusters...")

    # Create cluster directories
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            cluster_dir = output_dir / "noise"
        else:
            cluster_dir = output_dir / f"character_{label}"
        cluster_dir.mkdir(exist_ok=True)

    # Copy images to cluster folders
    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Copying images"):
        if label == -1:
            dst_dir = output_dir / "noise"
        else:
            dst_dir = output_dir / f"character_{label}"

        dst_path = dst_dir / img_path.name
        shutil.copy2(img_path, dst_path)

    print(f"✓ Instances organized into {len(unique_labels)} folders")


def visualize_clusters(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str = "Character Clustering"
):
    """Create 2D visualization of clusters"""
    print(f"\n📊 Creating cluster visualization...")

    plt.figure(figsize=(16, 12))

    # Plot noise points
    noise_mask = labels == -1
    if noise_mask.any():
        plt.scatter(
            embeddings_2d[noise_mask, 0],
            embeddings_2d[noise_mask, 1],
            c='lightgray',
            s=20,
            alpha=0.3,
            label='Noise'
        )

    # Plot clusters
    unique_labels = sorted(set(labels) - {-1})
    colors = sns.color_palette("husl", len(unique_labels))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx]],
            s=50,
            alpha=0.7,
            label=f'Character {label} (n={mask.sum()})'
        )

    plt.title(title, fontsize=16)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-based character instance clustering (Film-Agnostic)"
    )
    parser.add_argument(
        "instances_dir",
        type=str,
        help="Directory with character instance images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for clustered instances"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (e.g., 'luca'). Auto-constructs output paths."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="kmeans",
        choices=["kmeans", "hdbscan"],
        help="Clustering method: 'kmeans' (100%% coverage, auto k) or 'hdbscan' (density-based, may produce noise)"
    )
    parser.add_argument(
        "--k-range",
        type=int,
        nargs=2,
        default=[10, 50],
        metavar=("MIN_K", "MAX_K"),
        help="K-means only: range for auto k detection (default: 10 50)"
    )
    parser.add_argument(
        "--fixed-k",
        type=int,
        default=None,
        help="K-means only: manually specify k (overrides auto detection)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=12,
        help="HDBSCAN only: Minimum instances per cluster"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="HDBSCAN only: Minimum samples for core points (higher = fewer clusters)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for CLIP encoding"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for processing"
    )

    args = parser.parse_args()

    # Determine output directory
    output_dir = args.output_dir
    if args.project:
        if not output_dir:
            base_dir = Path("/mnt/data/ai_data/datasets/3d-anime")
            output_dir = str(base_dir / args.project / "clustered")
            print(f"✓ Using project: {args.project}")
            print(f"   Auto output: {output_dir}")
    elif not output_dir:
        parser.error("Either --output-dir or --project must be specified")

    instances_dir = Path(args.instances_dir)
    output_dir = Path(output_dir)

    # Find all images (recursive search to handle nested directories)
    image_files = sorted(
        list(instances_dir.rglob("*.png")) +
        list(instances_dir.rglob("*.jpg")) +
        list(instances_dir.rglob("*.jpeg"))
    )

    print(f"\n{'='*60}")
    print(f"CLIP CHARACTER CLUSTERING ({args.method.upper()})")
    print(f"{'='*60}")
    print(f"Instances directory: {instances_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total instances: {len(image_files)}")
    print(f"Method: {args.method}")
    if args.method == "kmeans":
        if args.fixed_k:
            print(f"K-means k: {args.fixed_k} (fixed)")
        else:
            print(f"K-means k-range: {args.k_range[0]}-{args.k_range[1]} (auto)")
    else:
        print(f"Min cluster size: {args.min_cluster_size}")
        print(f"Min samples: {args.min_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    if len(image_files) == 0:
        print("❌ No images found in instances directory!")
        sys.exit(1)

    # Extract CLIP embeddings
    embedder = CLIPEmbedder(device=args.device)
    embeddings = embedder.embed_images(image_files, batch_size=args.batch_size)

    # Cluster
    clusterer = CharacterClusterer(
        method=args.method,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        k_range=tuple(args.k_range)
    )
    labels, info, embeddings_2d = clusterer.cluster(
        embeddings,
        use_pca=True,
        pca_components=128,
        use_umap=True,
        umap_components=2,  # For visualization
        fixed_k=args.fixed_k
    )

    # Organize into folders
    organize_clusters(image_files, labels, output_dir)

    # Visualize
    viz_path = output_dir / "cluster_visualization.png"
    visualize_clusters(embeddings_2d, labels, viz_path)

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "instances_dir": str(instances_dir),
        "output_dir": str(output_dir),
        "total_instances": len(image_files),
        "method": args.method,
        "batch_size": args.batch_size,
        "device": args.device,
        "clustering_info": info
    }

    # Add method-specific parameters to metadata
    if args.method == "kmeans":
        metadata["k_range"] = args.k_range
        if args.fixed_k:
            metadata["fixed_k"] = args.fixed_k
    else:
        metadata["min_cluster_size"] = args.min_cluster_size
        metadata["min_samples"] = args.min_samples

    metadata_path = output_dir / "clustering_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CLUSTERING COMPLETE")
    print(f"{'='*60}")
    print(f"Total instances: {len(image_files)}")
    print(f"Characters found: {info['n_clusters']}")
    print(f"Noise instances: {info['n_noise']} ({100*info['noise_ratio']:.1f}%)")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    if args.project:
        print(f"💡 Next steps for project '{args.project}':")
        print(f"   1. Review clusters and rename by character name")
        print(f"   2. (Optional) Generate captions for each cluster")
        print(f"   3. Prepare training dataset")


if __name__ == "__main__":
    main()
