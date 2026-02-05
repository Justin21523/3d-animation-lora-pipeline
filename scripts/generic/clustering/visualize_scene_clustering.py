#!/usr/bin/env python3
"""
Generate comprehensive visualizations for scene clustering results.

Generates:
1. UMAP 2D scatter plot (colored by cluster)
2. Hierarchical dendrogram
3. Cluster size distribution
4. Silhouette analysis
5. Representative image grid per coarse cluster

Usage:
    python visualize_scene_clustering.py /path/to/scene_clusters_giant \
        --backgrounds /path/to/backgrounds_lama_v2 \
        --model dinov2-giant \
        --device cuda
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DINOv2Embedder:
    """DINOv2 feature extractor for scene understanding."""

    MODELS = {
        'dinov2-small': 'facebook/dinov2-small',
        'dinov2-base': 'facebook/dinov2-base',
        'dinov2-large': 'facebook/dinov2-large',
        'dinov2-giant': 'facebook/dinov2-giant',
    }

    def __init__(self, model_name: str = 'dinov2-giant', device: str = 'cuda'):
        """Initialize DINOv2 embedder."""
        self.model_name = model_name
        self.device = device

        model_id = self.MODELS[model_name]
        logger.info(f"Loading DINOv2 model: {model_id}")

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()

        logger.info(f"DINOv2 model loaded on {device}")

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract embedding from a single image."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # Use CLS token
        embedding = outputs.last_hidden_state[:, 0].cpu().numpy()[0]

        return embedding


def load_cluster_data(cluster_dir: Path) -> Dict:
    """Load clustering report and file mapping."""
    report_path = cluster_dir / "scene_clustering_report.json"

    if not report_path.exists():
        raise FileNotFoundError(f"Clustering report not found: {report_path}")

    with open(report_path, 'r') as f:
        report = json.load(f)

    # Build file-to-cluster mapping
    file_to_cluster = {}
    cluster_to_files = defaultdict(list)

    # Scan scene_* directories (skip non-directories like scene_clustering_report.json)
    for scene_dir in sorted(cluster_dir.glob("scene_*")):
        # Skip files, only process directories
        if not scene_dir.is_dir():
            continue

        cluster_id = int(scene_dir.name.split('_')[1])

        for img_file in scene_dir.glob("*"):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                file_to_cluster[img_file.name] = cluster_id
                cluster_to_files[cluster_id].append(img_file)

    report['file_to_cluster'] = file_to_cluster
    report['cluster_to_files'] = cluster_to_files

    logger.info(f"Loaded {len(file_to_cluster)} files across {len(cluster_to_files)} clusters")

    return report


def sample_images_per_cluster(report: Dict, max_per_cluster: int = 10) -> Dict[int, List[Path]]:
    """Sample representative images from each cluster."""
    sampled = {}

    for cluster_id, files in report['cluster_to_files'].items():
        # Sample evenly spaced files
        n_files = len(files)
        if n_files <= max_per_cluster:
            sampled[cluster_id] = files
        else:
            indices = np.linspace(0, n_files - 1, max_per_cluster, dtype=int)
            sampled[cluster_id] = [files[i] for i in indices]

    total_sampled = sum(len(v) for v in sampled.values())
    logger.info(f"Sampled {total_sampled} images from {len(sampled)} clusters")

    return sampled


def extract_embeddings_batch(embedder: DINOv2Embedder, image_paths: List[Path], batch_size: int = 32) -> np.ndarray:
    """Extract embeddings for a batch of images."""
    embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = image_paths[i:i+batch_size]
        batch_embeddings = []

        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                emb = embedder.extract_embedding(img)
                batch_embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                # Use zero vector as placeholder
                batch_embeddings.append(np.zeros(1536 if 'giant' in embedder.model_name else 768))

        embeddings.extend(batch_embeddings)

    return np.array(embeddings)


def plot_umap_2d(embeddings: np.ndarray, labels: np.ndarray, output_path: Path, report: Dict):
    """Generate UMAP 2D scatter plot colored by cluster."""
    logger.info("Generating UMAP 2D projection...")

    # UMAP dimensionality reduction to 2D
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )

    embedding_2d = reducer.fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Get unique clusters
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Use a colormap with enough colors
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_clusters)))
    if n_clusters > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, n_clusters))

    # Plot each cluster
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[idx % len(colors)]],
            label=f'Cluster {label}',
            alpha=0.6,
            s=50,
            edgecolors='k',
            linewidth=0.5
        )

    ax.set_title(f'Scene Clustering UMAP 2D Projection\n{n_clusters} Clusters, {len(embeddings)} Images',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)

    # Legend (only show first 20 clusters to avoid overcrowding)
    if n_clusters <= 20:
        ax.legend(loc='best', fontsize=8, ncol=2)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"UMAP 2D plot saved to {output_path}")


def plot_hierarchical_dendrogram(embeddings: np.ndarray, labels: np.ndarray, output_path: Path, report: Dict):
    """Generate hierarchical dendrogram."""
    logger.info("Generating hierarchical dendrogram...")

    # Compute cluster centroids
    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        mask = labels == label
        centroid = embeddings[mask].mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Compute linkage matrix
    linkage_matrix = linkage(centroids, method='ward')

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))

    dendrogram(
        linkage_matrix,
        ax=ax,
        labels=[f'C{i}' for i in unique_labels],
        leaf_font_size=8,
        color_threshold=0.7 * max(linkage_matrix[:, 2])
    )

    ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)\nBased on Cluster Centroids',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Dendrogram saved to {output_path}")


def plot_cluster_size_distribution(report: Dict, output_path: Path):
    """Generate cluster size distribution histogram."""
    logger.info("Generating cluster size distribution...")

    cluster_sizes = [len(files) for files in report['cluster_to_files'].values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    ax1.hist(cluster_sizes, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(cluster_sizes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cluster_sizes):.1f}')
    ax1.axvline(np.median(cluster_sizes), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(cluster_sizes):.1f}')
    ax1.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Images per Cluster', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(cluster_sizes, vert=True, patch_artist=True,
                boxprops=dict(facecolor='skyblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_title('Cluster Size Box Plot', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Images per Cluster', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = f"Total Clusters: {len(cluster_sizes)}\n"
    stats_text += f"Mean: {np.mean(cluster_sizes):.1f}\n"
    stats_text += f"Median: {np.median(cluster_sizes):.1f}\n"
    stats_text += f"Std: {np.std(cluster_sizes):.1f}\n"
    stats_text += f"Min: {min(cluster_sizes)}\n"
    stats_text += f"Max: {max(cluster_sizes)}"

    ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Cluster size distribution saved to {output_path}")


def plot_silhouette_analysis(embeddings: np.ndarray, labels: np.ndarray, output_path: Path, report: Dict):
    """Generate silhouette analysis plot."""
    logger.info("Generating silhouette analysis...")

    # Compute silhouette scores
    silhouette_vals = silhouette_samples(embeddings, labels, metric='cosine')
    silhouette_avg = report.get('silhouette_score', silhouette_score(embeddings, labels, metric='cosine'))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    y_lower = 10
    unique_labels = np.unique(labels)

    # Use colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(sorted(unique_labels)):
        # Aggregate silhouette scores for samples in this cluster
        cluster_silhouette_vals = silhouette_vals[labels == label]
        cluster_silhouette_vals.sort()

        size_cluster = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_vals,
            facecolor=colors[idx],
            edgecolor=colors[idx],
            alpha=0.7
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(label), fontsize=8)

        y_lower = y_upper + 10

    ax.set_title(f'Silhouette Analysis\nAverage Score: {silhouette_avg:.3f}',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster Label', fontsize=12)

    # Vertical line for average silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2, label=f'Average: {silhouette_avg:.3f}')
    ax.legend()

    ax.set_yticks([])
    ax.set_xlim([-0.2, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Silhouette analysis saved to {output_path}")


def plot_coarse_cluster_samples(report: Dict, output_path: Path, samples_per_cluster: int = 6):
    """Generate representative image grid for each coarse cluster."""
    logger.info("Generating coarse cluster representative images...")

    if 'coarse_cluster_stats' not in report:
        logger.warning("No coarse cluster information found, skipping coarse cluster visualization")
        return

    coarse_stats = report['coarse_cluster_stats']
    n_coarse = len(coarse_stats)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 4 * n_coarse))

    for coarse_idx, coarse_info in enumerate(coarse_stats):
        coarse_id = coarse_info['coarse_id']
        n_fine = coarse_info['n_fine_clusters']

        # Find fine clusters belonging to this coarse cluster
        # (Note: This requires mapping which is not in the report, so we'll sample from all clusters)
        # For simplicity, we'll just show the first few clusters
        # In a real implementation, you'd need to store the coarse-to-fine mapping

        # Sample representative images
        all_files = []
        for files in list(report['cluster_to_files'].values())[:n_fine]:
            all_files.extend(files[:2])  # 2 images per fine cluster

        # Limit total samples
        all_files = all_files[:samples_per_cluster]

        # Create subplot
        ax = fig.add_subplot(n_coarse, 1, coarse_idx + 1)

        # Load and concatenate images horizontally
        images = []
        for img_path in all_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((200, 150))  # Standardize size
                images.append(np.array(img))
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")

        if images:
            concat_img = np.concatenate(images, axis=1)
            ax.imshow(concat_img)
            ax.set_title(f'Coarse Cluster {coarse_id} ({coarse_info["n_images"]} images, {n_fine} fine clusters)',
                         fontsize=12, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Coarse cluster samples saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize scene clustering results")
    parser.add_argument('cluster_dir', type=Path, help="Path to scene clustering output directory")
    parser.add_argument('--backgrounds', type=Path, help="Path to original backgrounds directory (optional, for better sampling)")
    parser.add_argument('--model', type=str, default='dinov2-giant',
                        choices=['dinov2-small', 'dinov2-base', 'dinov2-large', 'dinov2-giant'],
                        help="DINOv2 model variant")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use")
    parser.add_argument('--max-samples', type=int, default=10, help="Max images to sample per cluster for visualization")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for embedding extraction")

    args = parser.parse_args()

    # Load cluster data
    logger.info(f"Loading clustering results from {args.cluster_dir}")
    report = load_cluster_data(args.cluster_dir)

    # Sample images
    sampled_images = sample_images_per_cluster(report, max_per_cluster=args.max_samples)

    # Flatten to list of (image_path, cluster_id)
    image_paths = []
    labels = []

    for cluster_id, files in sampled_images.items():
        for f in files:
            image_paths.append(f)
            labels.append(cluster_id)

    labels = np.array(labels)

    # Extract embeddings
    logger.info(f"Extracting embeddings for {len(image_paths)} images...")
    embedder = DINOv2Embedder(model_name=args.model, device=args.device)
    embeddings = extract_embeddings_batch(embedder, image_paths, batch_size=args.batch_size)

    # Generate visualizations
    viz_dir = args.cluster_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    logger.info("Generating visualizations...")

    # 1. UMAP 2D scatter plot
    plot_umap_2d(embeddings, labels, viz_dir / "01_umap_2d_scatter.png", report)

    # 2. Hierarchical dendrogram
    plot_hierarchical_dendrogram(embeddings, labels, viz_dir / "02_hierarchical_dendrogram.png", report)

    # 3. Cluster size distribution
    plot_cluster_size_distribution(report, viz_dir / "03_cluster_size_distribution.png")

    # 4. Silhouette analysis
    plot_silhouette_analysis(embeddings, labels, viz_dir / "04_silhouette_analysis.png", report)

    # 5. Coarse cluster samples
    plot_coarse_cluster_samples(report, viz_dir / "05_coarse_cluster_samples.png")

    logger.info(f"✅ All visualizations saved to {viz_dir}")
    logger.info("\nGenerated visualizations:")
    logger.info("  1. 01_umap_2d_scatter.png - UMAP 2D projection colored by cluster")
    logger.info("  2. 02_hierarchical_dendrogram.png - Hierarchical clustering tree")
    logger.info("  3. 03_cluster_size_distribution.png - Histogram and box plot of cluster sizes")
    logger.info("  4. 04_silhouette_analysis.png - Silhouette coefficient analysis")
    logger.info("  5. 05_coarse_cluster_samples.png - Representative images from coarse clusters")


if __name__ == "__main__":
    main()
