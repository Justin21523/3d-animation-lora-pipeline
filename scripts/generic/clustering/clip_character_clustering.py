#!/usr/bin/env python3
"""
Multi-Encoder Character Instance Clustering

Purpose: Cluster character instances by visual similarity using various vision encoders
Optimized for: 3D animated characters where face detection may fail
Supported Encoders: CLIP, DINOv2, SigLIP
Features: GPU-accelerated batch processing, HDBSCAN clustering, visualization

Usage:
    python clip_character_clustering.py \
      /path/to/instances \
      --output-dir /path/to/clustered \
      --project luca \
      --encoder clip \
      --min-cluster-size 20 \
      --min-samples 3 \
      --cluster-selection-epsilon 0.5 \
      --batch-size 64 \
      --device cuda
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import umap


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
    """Cluster character instances using HDBSCAN"""

    def __init__(self, min_cluster_size: int = 12, min_samples: int = 2):
        """Initialize clusterer"""
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def cluster(
        self,
        embeddings: np.ndarray,
        use_pca: bool = True,
        pca_components: int = 128,
        use_umap: bool = True,
        umap_components: int = 32
    ) -> Tuple[np.ndarray, Dict]:
        """
        Cluster embeddings using dimensionality reduction + HDBSCAN

        Args:
            embeddings: (N, D) embedding matrix
            use_pca: Apply PCA first
            pca_components: PCA dimensions
            use_umap: Apply UMAP after PCA
            umap_components: UMAP dimensions

        Returns:
            labels: Cluster labels (-1 = noise)
            info: Clustering statistics
        """
        print(f"\n🔍 Clustering {len(embeddings)} instances...")

        # Normalize embeddings
        embeddings_norm = normalize(embeddings, norm='l2')

        # Optional PCA
        if use_pca and embeddings.shape[1] > pca_components:
            # Adjust PCA components to not exceed sample count
            actual_components = min(pca_components, len(embeddings) - 1, embeddings.shape[1])
            print(f"   Applying PCA: {embeddings.shape[1]}D → {actual_components}D")
            pca = PCA(n_components=actual_components, random_state=42)
            embeddings_reduced = pca.fit_transform(embeddings_norm)
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"   Explained variance: {explained_var:.1%}")
        else:
            embeddings_reduced = embeddings_norm

        # Optional UMAP
        if use_umap:
            print(f"   Applying UMAP: {embeddings_reduced.shape[1]}D → {umap_components}D")
            umap_reducer = umap.UMAP(
                n_components=umap_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            embeddings_final = umap_reducer.fit_transform(embeddings_reduced)
        else:
            embeddings_final = embeddings_reduced

        # HDBSCAN clustering
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
        "--min-cluster-size",
        type=int,
        default=12,
        help="Minimum instances per cluster"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum samples for HDBSCAN core points (higher = fewer clusters)"
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

    # Find all images
    image_files = sorted(
        list(instances_dir.glob("*.png")) +
        list(instances_dir.glob("*.jpg")) +
        list(instances_dir.glob("*.jpeg"))
    )

    print(f"\n{'='*60}")
    print(f"CLIP CHARACTER CLUSTERING")
    print(f"{'='*60}")
    print(f"Instances directory: {instances_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total instances: {len(image_files)}")
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
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples
    )
    labels, info, embeddings_2d = clusterer.cluster(
        embeddings,
        use_pca=True,
        pca_components=128,
        use_umap=True,
        umap_components=2  # For visualization
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
        "min_cluster_size": args.min_cluster_size,
        "batch_size": args.batch_size,
        "device": args.device,
        "clustering_info": info
    }

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
