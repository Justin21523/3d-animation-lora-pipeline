"""
Clustering module for 2D Animation LoRA Pipeline.

Provides identity-based clustering for multi-character scenes:
- FaceIdentityClusterer: Face detection + ArcFace embedding + HDBSCAN
- Cluster manipulation utilities (merge, split, rename)

Usage:
    from anime_pipeline.clustering import FaceIdentityClusterer

    clusterer = FaceIdentityClusterer(device="cuda")
    result = clusterer.cluster_images(images_dir, output_dir)
"""

from .face_identity_clusterer import (
    FaceIdentityClusterer,
    FaceDetection,
    FaceEmbedding,
    IdentityCluster,
    ClusteringResult,
    merge_clusters,
    split_cluster,
)

__all__ = [
    "FaceIdentityClusterer",
    "FaceDetection",
    "FaceEmbedding",
    "IdentityCluster",
    "ClusteringResult",
    "merge_clusters",
    "split_cluster",
]
