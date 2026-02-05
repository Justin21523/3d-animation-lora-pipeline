#!/usr/bin/env python3
"""
Hierarchical Clustering for Noise Instances

Purpose: Re-cluster HDBSCAN noise instances using hierarchical clustering
to ensure ALL instances are assigned to a character group.

Strategy:
1. Load all instances from the 'noise/' folder
2. Extract CLIP embeddings
3. Apply hierarchical clustering (Agglomerative) with optimal k
4. Organize into character_noise_0, character_noise_1, etc.

Usage:
    python hierarchical_noise_clustering.py \
      --clustered-dir /path/to/clustered \
      --output-dir /path/to/clustered_final \
      --min-cluster-size 20 \
      --max-clusters 50 \
      --device cuda
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import umap


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


def find_optimal_clusters(
    embeddings: np.ndarray,
    min_k: int,
    max_k: int,
    sample_size: int = 5000,
    force_high_granularity: bool = True
) -> Tuple[int, Dict]:
    """
    Find optimal number of clusters using heuristic-based approach

    Args:
        embeddings: Feature embeddings
        min_k: Minimum number of clusters
        max_k: Maximum number of clusters
        sample_size: Subsample for faster computation
        force_high_granularity: Use heuristic k instead of Silhouette optimization

    Returns:
        optimal_k: Best k value
        metrics: All computed metrics
    """
    metrics = {'k': [], 'silhouette': [], 'davies_bouldin': []}

    if force_high_granularity:
        # Heuristic: For noise instances, use high granularity
        # Target: ~20-30 instances per cluster on average
        heuristic_k = max(min_k, min(max_k, len(embeddings) // 25))

        print(f"\n🔍 Using heuristic-based k selection...")
        print(f"   Total instances: {len(embeddings)}")
        print(f"   K range: [{min_k}, {max_k}]")
        print(f"   Selected k = {heuristic_k} (target: ~{len(embeddings) // heuristic_k} instances/cluster)")

        return heuristic_k, metrics

    # Original Silhouette-based optimization (fallback)
    # Subsample for very large datasets
    if len(embeddings) > sample_size:
        print(f"   Subsampling {sample_size}/{len(embeddings)} for k optimization...")
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings

    print(f"\n🔍 Finding optimal k in range [{min_k}, {max_k}]...")

    # Test different k values
    k_values = list(range(min_k, max_k + 1, max(1, (max_k - min_k) // 10)))

    for k in tqdm(k_values, desc="Testing k values"):
        if k >= len(sample_embeddings):
            break

        clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clusterer.fit_predict(sample_embeddings)

        # Skip if only one cluster formed
        if len(set(labels)) < 2:
            continue

        metrics['k'].append(k)
        metrics['silhouette'].append(silhouette_score(sample_embeddings, labels))
        metrics['davies_bouldin'].append(davies_bouldin_score(sample_embeddings, labels))

    if len(metrics['k']) == 0:
        print(f"   ⚠️ No valid k found, using min_k={min_k}")
        return min_k, metrics

    # Find optimal k using Silhouette score (primary metric)
    best_idx = np.argmax(metrics['silhouette'])
    optimal_k = metrics['k'][best_idx]

    print(f"   ✓ Optimal k = {optimal_k}")
    print(f"     Silhouette Score: {metrics['silhouette'][best_idx]:.4f}")
    print(f"     Davies-Bouldin Index: {metrics['davies_bouldin'][best_idx]:.4f}")

    return optimal_k, metrics


def hierarchical_cluster(
    embeddings: np.ndarray,
    n_clusters: int,
    use_umap: bool = True,
    umap_components: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical clustering with optional UMAP reduction

    Args:
        embeddings: (N, D) embedding matrix
        n_clusters: Number of clusters
        use_umap: Apply UMAP dimensionality reduction first
        umap_components: UMAP dimensions

    Returns:
        labels: Cluster labels
        embeddings_2d: 2D embeddings for visualization
    """
    print(f"\n🔍 Hierarchical clustering {len(embeddings)} noise instances...")

    # Normalize embeddings
    embeddings_norm = normalize(embeddings, norm='l2')

    # Optional UMAP reduction
    if use_umap and len(embeddings) > umap_components:
        print(f"   Applying UMAP: {embeddings.shape[1]}D → {umap_components}D")
        umap_reducer = umap.UMAP(
            n_components=umap_components,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        embeddings_reduced = umap_reducer.fit_transform(embeddings_norm)
    else:
        embeddings_reduced = embeddings_norm

    # Hierarchical clustering
    print(f"   Running Agglomerative Clustering with k={n_clusters}...")
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clusterer.fit_predict(embeddings_reduced)

    # Create 2D embeddings for visualization
    if embeddings_reduced.shape[1] > 2:
        print(f"   Creating 2D visualization...")
        umap_2d = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        embeddings_2d = umap_2d.fit_transform(embeddings_reduced)
    else:
        embeddings_2d = embeddings_reduced

    # Statistics
    unique_labels = set(labels)
    print(f"\n✓ Clustering complete:")
    print(f"   Clusters formed: {len(unique_labels)}")

    for label in sorted(unique_labels):
        count = (labels == label).sum()
        print(f"   character_noise_{label}: {count} instances")

    return labels, embeddings_2d


def organize_clusters(
    image_paths: List[Path],
    labels: np.ndarray,
    output_dir: Path
):
    """Organize images into cluster folders"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Organizing {len(image_paths)} noise instances into clusters...")

    # Create cluster directories
    unique_labels = set(labels)

    for label in unique_labels:
        cluster_dir = output_dir / f"character_noise_{label}"
        cluster_dir.mkdir(exist_ok=True)

    # Copy images to cluster folders
    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Copying images"):
        dst_dir = output_dir / f"character_noise_{label}"
        dst_path = dst_dir / img_path.name
        shutil.copy2(img_path, dst_path)

    print(f"✓ Noise instances organized into {len(unique_labels)} folders")


def visualize_clusters(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str = "Noise Hierarchical Clustering"
):
    """Create 2D visualization of clusters"""
    print(f"\n📊 Creating cluster visualization...")

    plt.figure(figsize=(16, 12))

    # Plot clusters
    unique_labels = sorted(set(labels))
    colors = sns.color_palette("husl", len(unique_labels))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx]],
            s=50,
            alpha=0.7,
            label=f'Noise Group {label} (n={mask.sum()})'
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
        description="Hierarchical clustering for HDBSCAN noise instances"
    )
    parser.add_argument(
        "--clustered-dir",
        type=str,
        required=True,
        help="Directory containing HDBSCAN clustered results (with noise/ folder)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for re-clustered noise instances (default: clustered-dir/noise_reclustered)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=20,
        help="Minimum instances per noise cluster (affects k range)"
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=50,
        help="Maximum number of noise clusters to create"
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

    clustered_dir = Path(args.clustered_dir)
    noise_dir = clustered_dir / "noise"

    # Check if noise directory exists
    if not noise_dir.exists():
        print(f"❌ Noise directory not found: {noise_dir}")
        print(f"   Make sure HDBSCAN clustering has been completed first.")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = clustered_dir / "noise_reclustered"

    # Find all noise images
    image_files = sorted(
        list(noise_dir.rglob("*.png")) +
        list(noise_dir.rglob("*.jpg")) +
        list(noise_dir.rglob("*.jpeg"))
    )

    print(f"\n{'='*60}")
    print(f"HIERARCHICAL NOISE CLUSTERING")
    print(f"{'='*60}")
    print(f"Noise directory: {noise_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total noise instances: {len(image_files)}")
    print(f"Min cluster size: {args.min_cluster_size}")
    print(f"Max clusters: {args.max_clusters}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    if len(image_files) == 0:
        print("✓ No noise instances found - all instances were successfully clustered!")
        sys.exit(0)

    # Calculate k range based on min cluster size
    min_k = max(2, len(image_files) // args.min_cluster_size // 2)
    max_k = min(args.max_clusters, len(image_files) // args.min_cluster_size)

    if min_k >= max_k:
        min_k = 2
        max_k = min(10, len(image_files) // 2)

    print(f"📊 K range: {min_k} to {max_k}")
    print()

    # Extract CLIP embeddings
    embedder = CLIPEmbedder(device=args.device)
    embeddings = embedder.embed_images(image_files, batch_size=args.batch_size)

    # Find optimal k
    optimal_k, metrics = find_optimal_clusters(
        embeddings,
        min_k=min_k,
        max_k=max_k
    )

    # Hierarchical clustering
    labels, embeddings_2d = hierarchical_cluster(
        embeddings,
        n_clusters=optimal_k,
        use_umap=True,
        umap_components=32
    )

    # Organize into folders
    organize_clusters(image_files, labels, output_dir)

    # Visualize
    viz_path = output_dir / "noise_clustering_visualization.png"
    visualize_clusters(embeddings_2d, labels, viz_path)

    # Save metadata
    cluster_sizes = {}
    for label in set(labels):
        cluster_sizes[f"character_noise_{label}"] = int((labels == label).sum())

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "noise_dir": str(noise_dir),
        "output_dir": str(output_dir),
        "total_noise_instances": len(image_files),
        "n_clusters": optimal_k,
        "cluster_sizes": cluster_sizes,
        "min_cluster_size": args.min_cluster_size,
        "max_clusters": args.max_clusters,
        "batch_size": args.batch_size,
        "device": args.device
    }

    metadata_path = output_dir / "noise_clustering_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"NOISE CLUSTERING COMPLETE")
    print(f"{'='*60}")
    print(f"Total noise instances: {len(image_files)}")
    print(f"Clusters created: {optimal_k}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    print(f"💡 Next steps:")
    print(f"   1. Review noise clusters in: {output_dir}")
    print(f"   2. Merge noise clusters with main character clusters if needed")
    print(f"   3. Rename clusters by character identity")


if __name__ == "__main__":
    main()
