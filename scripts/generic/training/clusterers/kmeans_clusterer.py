"""
KMeans clustering implementation.

KMeans is a simple and fast clustering algorithm that partitions data into
k clusters. Requires specifying the number of clusters in advance.

Good for:
- Fixed-k scenarios (e.g., known number of characters)
- Fast prototyping
- Balanced cluster sizes
"""

from typing import Optional, Dict, Any
import numpy as np
import logging

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
except ImportError:
    raise ImportError(
        "KMeans requires: pip install scikit-learn"
    )

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base.clusterer import BaseClusterer


class KMeansClusterer(BaseClusterer):
    """
    Clustering using KMeans algorithm.

    KMeans partitions data into k clusters by minimizing within-cluster
    sum of squares. Simple and fast, but requires knowing k.

    Attributes:
        n_clusters: Number of clusters
        init: Initialization method
        max_iter: Maximum iterations
        n_init: Number of initializations
        clusterer: KMeans instance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize KMeans clusterer.

        Config parameters:
            n_clusters (int): Number of clusters, default 5
            init (str): Initialization method, default 'k-means++'
                Options: 'k-means++', 'random'
            max_iter (int): Maximum iterations, default 300
            n_init (int): Number of initializations, default 10
            random_state (int): Random seed, default 42
            standardize (bool): Standardize features, default True
        """
        super().__init__(config)

    def configure(self):
        """Configure KMeans parameters."""
        self.n_clusters = self.config.get('n_clusters', 5)
        self.init = self.config.get('init', 'k-means++')
        self.max_iter = self.config.get('max_iter', 300)
        self.n_init = self.config.get('n_init', 10)
        self.random_state = self.config.get('random_state', 42)
        self.standardize = self.config.get('standardize', True)

        # Initialize scaler if needed
        if self.standardize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        self.logger.info(
            f"KMeans configured: n_clusters={self.n_clusters}, "
            f"init={self.init}"
        )

    def validate_config(self):
        """Validate configuration parameters."""
        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters}")

        if self.init not in ['k-means++', 'random']:
            raise ValueError(
                f"Invalid init method '{self.init}'. "
                "Must be 'k-means++' or 'random'"
            )

        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")

    def fit(self, features: np.ndarray) -> 'KMeansClusterer':
        """
        Train KMeans on feature data.

        Args:
            features: Feature matrix with shape (n_samples, n_features)

        Returns:
            Self (for method chaining)
        """
        if len(features.shape) != 2:
            raise ValueError(f"Expected 2D features array, got shape {features.shape}")

        if features.shape[0] < self.n_clusters:
            raise ValueError(
                f"Number of samples ({features.shape[0]}) must be >= "
                f"n_clusters ({self.n_clusters})"
            )

        self.logger.info(
            f"Fitting KMeans on {features.shape[0]} samples "
            f"with k={self.n_clusters}"
        )

        # Standardize if configured
        if self.standardize:
            features = self.scaler.fit_transform(features)

        # Initialize KMeans
        self.clusterer = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state
        )

        # Fit
        self.labels_ = self.clusterer.fit_predict(features)
        self.n_clusters_ = self.n_clusters
        self.is_fitted_ = True

        # Compute silhouette score if enough samples
        if features.shape[0] > self.n_clusters:
            try:
                self.silhouette_score_ = silhouette_score(features, self.labels_)
                self.logger.info(
                    f"✓ KMeans completed with inertia={self.clusterer.inertia_:.2f}, "
                    f"silhouette={self.silhouette_score_:.3f}"
                )
            except Exception as e:
                self.logger.warning(f"Could not compute silhouette score: {e}")
                self.silhouette_score_ = None
        else:
            self.silhouette_score_ = None

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Assign new data points to nearest cluster centers.

        Args:
            features: Feature matrix with shape (n_samples, n_features)

        Returns:
            Cluster labels with shape (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Clusterer must be fitted before prediction")

        # Standardize if needed
        if self.standardize:
            features = self.scaler.transform(features)

        return self.clusterer.predict(features)

    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Return detailed clustering information.

        Returns:
            Dictionary with cluster statistics and quality metrics
        """
        info = super().get_cluster_info()

        if self.is_fitted_:
            info['inertia'] = float(self.clusterer.inertia_)
            info['n_iter'] = int(self.clusterer.n_iter_)

            if self.silhouette_score_ is not None:
                info['silhouette_score'] = float(self.silhouette_score_)

        return info

    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster center coordinates.

        Returns:
            Cluster centers with shape (n_clusters, n_features)
        """
        if not self.is_fitted_:
            raise RuntimeError("Clusterer must be fitted first")

        centers = self.clusterer.cluster_centers_

        # Inverse transform if standardized
        if self.standardize:
            centers = self.scaler.inverse_transform(centers)

        return centers

    def get_distances_to_centers(self, features: np.ndarray) -> np.ndarray:
        """
        Compute distances from samples to their assigned cluster centers.

        Args:
            features: Feature matrix with shape (n_samples, n_features)

        Returns:
            Distances array with shape (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Clusterer must be fitted first")

        # Standardize if needed
        if self.standardize:
            features = self.scaler.transform(features)

        # Get assigned cluster labels
        labels = self.clusterer.predict(features)

        # Compute distances to assigned centers
        distances = np.linalg.norm(
            features - self.clusterer.cluster_centers_[labels],
            axis=1
        )

        return distances

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"KMeansClusterer("
                f"n_clusters={self.n_clusters_}, "
                f"inertia={self.clusterer.inertia_:.2f}, "
                f"fitted=True)"
            )
        else:
            return (
                f"KMeansClusterer("
                f"n_clusters={self.n_clusters}, "
                f"fitted=False)"
            )


# Convenience function
def cluster_kmeans(
    features: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
    standardize: bool = True
) -> np.ndarray:
    """
    Cluster features using KMeans.

    Args:
        features: Feature matrix (n_samples, n_features)
        n_clusters: Number of clusters (default 5)
        random_state: Random seed (default 42)
        standardize: Standardize features (default True)

    Returns:
        Cluster labels array
    """
    config = {
        'n_clusters': n_clusters,
        'random_state': random_state,
        'standardize': standardize
    }

    clusterer = KMeansClusterer(config)
    labels = clusterer.fit_predict(features)

    return labels
