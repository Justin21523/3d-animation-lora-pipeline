"""
DBSCAN clustering implementation.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
finds clusters based on density and identifies outliers as noise.
"""

from typing import Optional, Dict, Any
import numpy as np
import logging

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("DBSCAN requires: pip install scikit-learn")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base.clusterer import BaseClusterer


class DBSCANClusterer(BaseClusterer):
    """
    Clustering using DBSCAN algorithm.

    DBSCAN groups together points that are closely packed and marks
    points in low-density regions as outliers.

    Attributes:
        eps: Maximum distance between two samples
        min_samples: Minimum samples in neighborhood
        metric: Distance metric
        clusterer: DBSCAN instance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DBSCAN clusterer.

        Config parameters:
            eps (float): Neighborhood radius, default 0.5
            min_samples (int): Minimum samples in neighborhood, default 5
            metric (str): Distance metric, default 'euclidean'
            algorithm (str): Algorithm for nearest neighbors, default 'auto'
            leaf_size (int): Leaf size for tree algorithms, default 30
            standardize (bool): Standardize features, default True
        """
        super().__init__(config)

    def configure(self):
        """Configure DBSCAN parameters."""
        self.eps = self.config.get('eps', 0.5)
        self.min_samples = self.config.get('min_samples', 5)
        self.metric = self.config.get('metric', 'euclidean')
        self.algorithm = self.config.get('algorithm', 'auto')
        self.leaf_size = self.config.get('leaf_size', 30)
        self.standardize = self.config.get('standardize', True)

        if self.standardize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        self.logger.info(f"DBSCAN configured: eps={self.eps}, min_samples={self.min_samples}")

    def validate_config(self):
        """Validate configuration parameters."""
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}")

    def fit(self, features: np.ndarray) -> 'DBSCANClusterer':
        """Train DBSCAN on feature data."""
        if len(features.shape) != 2:
            raise ValueError(f"Expected 2D features array, got shape {features.shape}")

        self.logger.info(f"Fitting DBSCAN on {features.shape[0]} samples")

        if self.standardize:
            features = self.scaler.fit_transform(features)

        self.clusterer = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=-1
        )

        self.labels_ = self.clusterer.fit_predict(features)

        # Count clusters (excluding noise: -1)
        unique_labels = np.unique(self.labels_)
        self.n_clusters_ = len(unique_labels[unique_labels != -1])

        self.is_fitted_ = True

        # Log results
        noise_count = np.sum(self.labels_ == -1)
        self.logger.info(
            f"✓ DBSCAN found {self.n_clusters_} clusters, "
            f"{noise_count} noise points ({noise_count/len(self.labels_)*100:.1f}%)"
        )

        return self

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fit DBSCAN and return cluster labels.

        DBSCAN performs fit and predict in one step.

        Args:
            features: Feature matrix with shape (n_samples, n_features)

        Returns:
            Cluster labels with shape (n_samples,)
        """
        self.fit(features)
        return self.labels_

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        DBSCAN doesn't support out-of-sample prediction.
        Use fit_predict() on all data at once.
        """
        raise NotImplementedError(
            "DBSCAN doesn't support predict(). "
            "Use fit_predict() on all data at once."
        )

    def __repr__(self) -> str:
        return f"DBSCANClusterer(n_clusters={self.n_clusters_}, eps={self.eps}, fitted={self.is_fitted_})"


# Convenience function
def cluster_dbscan(
    features: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = 'euclidean',
    standardize: bool = True
) -> np.ndarray:
    """
    Cluster features using DBSCAN.

    Args:
        features: Feature matrix (n_samples, n_features)
        eps: Neighborhood radius (default 0.5)
        min_samples: Minimum samples in neighborhood (default 5)
        metric: Distance metric (default 'euclidean')
        standardize: Standardize features (default True)

    Returns:
        Cluster labels array
    """
    config = {
        'eps': eps,
        'min_samples': min_samples,
        'metric': metric,
        'standardize': standardize
    }

    clusterer = DBSCANClusterer(config)
    labels = clusterer.fit_predict(features)

    return labels
