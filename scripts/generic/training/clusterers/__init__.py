"""
Clustering algorithm implementations.

Provides multiple clustering methods for organizing training data:
- HDBSCAN (density-based, auto-k)
- KMeans (partition-based, fixed-k)
- Spectral (graph-based)
- Agglomerative (hierarchical)
- DBSCAN (density-based)
"""

from .hdbscan_clusterer import HDBSCANClusterer, cluster_hdbscan
from .kmeans_clusterer import KMeansClusterer, cluster_kmeans
from .spectral_clusterer import SpectralClusterer
from .agglomerative_clusterer import AgglomerativeClusterer
from .dbscan_clusterer import DBSCANClusterer, cluster_dbscan

__all__ = [
    'HDBSCANClusterer',
    'KMeansClusterer',
    'SpectralClusterer',
    'AgglomerativeClusterer',
    'DBSCANClusterer',
    'cluster_hdbscan',
    'cluster_kmeans',
    'cluster_dbscan',
]
