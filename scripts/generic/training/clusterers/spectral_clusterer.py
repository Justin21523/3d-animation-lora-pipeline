"""
Spectral clustering implementation.

Spectral clustering uses graph theory and eigenvalues to find clusters.
Works well for non-convex cluster shapes.
"""

from typing import Optional, Dict, Any
import numpy as np
import logging

try:
    from sklearn.cluster import SpectralClustering
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("Spectral clustering requires: pip install scikit-learn")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base.clusterer import BaseClusterer


class SpectralClusterer(BaseClusterer):
    """
    Clustering using Spectral Clustering algorithm.

    Spectral clustering performs dimensionality reduction before clustering,
    making it effective for complex, non-convex cluster shapes.

    Attributes:
        n_clusters: Number of clusters
        affinity: How to construct affinity matrix
        assign_labels: Strategy for assigning labels
        clusterer: SpectralClustering instance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Spectral clusterer.

        Config parameters:
            n_clusters (int): Number of clusters, default 5
            affinity (str): Affinity method, default 'rbf'
                Options: 'rbf', 'nearest_neighbors', 'precomputed'
            assign_labels (str): Label assignment, default 'kmeans'
                Options: 'kmeans', 'discretize'
            n_neighbors (int): Neighbors for knn affinity, default 10
            gamma (float): RBF kernel parameter, default 1.0
            random_state (int): Random seed, default 42
            standardize (bool): Standardize features, default True
        """
        super().__init__(config)

    def configure(self):
        """Configure Spectral parameters."""
        self.n_clusters = self.config.get('n_clusters', 5)
        self.affinity = self.config.get('affinity', 'rbf')
        self.assign_labels = self.config.get('assign_labels', 'kmeans')
        self.n_neighbors = self.config.get('n_neighbors', 10)
        self.gamma = self.config.get('gamma', 1.0)
        self.random_state = self.config.get('random_state', 42)
        self.standardize = self.config.get('standardize', True)

        if self.standardize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        self.logger.info(f"Spectral configured: n_clusters={self.n_clusters}")

    def validate_config(self):
        """Validate configuration parameters."""
        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters}")

    def fit(self, features: np.ndarray) -> 'SpectralClusterer':
        """Train Spectral clustering on feature data."""
        if len(features.shape) != 2:
            raise ValueError(f"Expected 2D features array, got shape {features.shape}")

        self.logger.info(f"Fitting Spectral on {features.shape[0]} samples")

        if self.standardize:
            features = self.scaler.fit_transform(features)

        self.clusterer = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            assign_labels=self.assign_labels,
            n_neighbors=self.n_neighbors,
            gamma=self.gamma,
            random_state=self.random_state
        )

        self.labels_ = self.clusterer.fit_predict(features)
        self.n_clusters_ = self.n_clusters
        self.is_fitted_ = True

        self.logger.info(f"✓ Spectral completed with {self.n_clusters_} clusters")

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Spectral clustering doesn't support out-of-sample prediction directly.
        This assigns new points to nearest cluster center (approximate).
        """
        if not self.is_fitted_:
            raise RuntimeError("Clusterer must be fitted before prediction")

        self.logger.warning("Spectral clustering uses approximate prediction")

        # Compute cluster centers from training data
        # This is an approximation - spectral clustering doesn't naturally support this
        # You would need to store training data and use a different approach

        raise NotImplementedError(
            "Spectral clustering doesn't support predict(). "
            "Use fit_predict() on all data at once."
        )

    def __repr__(self) -> str:
        return f"SpectralClusterer(n_clusters={self.n_clusters}, fitted={self.is_fitted_})"
