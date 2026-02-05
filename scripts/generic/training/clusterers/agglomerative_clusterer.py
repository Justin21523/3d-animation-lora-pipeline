"""
Agglomerative (Hierarchical) clustering implementation.

Agglomerative clustering builds a hierarchy of clusters using a bottom-up
approach. Can produce a dendrogram for visualization.
"""

from typing import Optional, Dict, Any
import numpy as np
import logging

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("Agglomerative clustering requires: pip install scikit-learn")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base.clusterer import BaseClusterer


class AgglomerativeClusterer(BaseClusterer):
    """
    Clustering using Agglomerative (Hierarchical) algorithm.

    Agglomerative clustering merges samples into clusters hierarchically.
    Produces a tree structure that can be cut at different levels.

    Attributes:
        n_clusters: Number of clusters
        linkage: Linkage criterion
        distance_threshold: Distance threshold (if n_clusters is None)
        clusterer: AgglomerativeClustering instance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Agglomerative clusterer.

        Config parameters:
            n_clusters (int): Number of clusters, default 5
            linkage (str): Linkage criterion, default 'ward'
                Options: 'ward', 'complete', 'average', 'single'
            affinity (str): Distance metric, default 'euclidean'
            distance_threshold (float): Distance threshold, default None
            standardize (bool): Standardize features, default True
        """
        super().__init__(config)

    def configure(self):
        """Configure Agglomerative parameters."""
        self.n_clusters = self.config.get('n_clusters', 5)
        self.linkage = self.config.get('linkage', 'ward')
        self.affinity = self.config.get('affinity', 'euclidean')
        self.distance_threshold = self.config.get('distance_threshold', None)
        self.standardize = self.config.get('standardize', True)

        if self.standardize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        # If distance_threshold is set, n_clusters must be None
        if self.distance_threshold is not None:
            self.n_clusters = None

        self.logger.info(f"Agglomerative configured: n_clusters={self.n_clusters}, linkage={self.linkage}")

    def validate_config(self):
        """Validate configuration parameters."""
        if self.n_clusters is not None and self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2 or None, got {self.n_clusters}")

        valid_linkages = ['ward', 'complete', 'average', 'single']
        if self.linkage not in valid_linkages:
            raise ValueError(f"Invalid linkage '{self.linkage}'. Must be one of: {valid_linkages}")

    def fit(self, features: np.ndarray) -> 'AgglomerativeClusterer':
        """Train Agglomerative clustering on feature data."""
        if len(features.shape) != 2:
            raise ValueError(f"Expected 2D features array, got shape {features.shape}")

        self.logger.info(f"Fitting Agglomerative on {features.shape[0]} samples")

        if self.standardize:
            features = self.scaler.fit_transform(features)

        self.clusterer = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            affinity=self.affinity,
            distance_threshold=self.distance_threshold
        )

        self.labels_ = self.clusterer.fit_predict(features)
        self.n_clusters_ = len(np.unique(self.labels_))
        self.is_fitted_ = True

        self.logger.info(f"✓ Agglomerative completed with {self.n_clusters_} clusters")

        return self

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fit Agglomerative clustering and return cluster labels.

        Args:
            features: Feature matrix with shape (n_samples, n_features)

        Returns:
            Cluster labels with shape (n_samples,)
        """
        self.fit(features)
        return self.labels_

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Agglomerative clustering doesn't support out-of-sample prediction.
        Use fit_predict() on all data at once.
        """
        raise NotImplementedError(
            "Agglomerative clustering doesn't support predict(). "
            "Use fit_predict() on all data at once."
        )

    def __repr__(self) -> str:
        return f"AgglomerativeClusterer(n_clusters={self.n_clusters_}, linkage={self.linkage}, fitted={self.is_fitted_})"
