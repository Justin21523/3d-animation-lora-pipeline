"""
HDBSCAN-based clustering implementation.

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
is ideal for finding clusters of varying densities without specifying k.

Paper: https://arxiv.org/abs/1911.02282
"""

from typing import Optional, Dict, Any
import numpy as np
import logging

try:
    import hdbscan
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError(
        "HDBSCAN requires: pip install hdbscan scikit-learn"
    )

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base.clusterer import BaseClusterer


class HDBSCANClusterer(BaseClusterer):
    """
    Clustering using HDBSCAN algorithm.

    HDBSCAN automatically determines the number of clusters and identifies
    outliers/noise points. Well-suited for:
    - Character identity clustering (variable cluster sizes)
    - Scene/background clustering
    - Expression clustering

    Attributes:
        min_cluster_size: Minimum samples in a cluster
        min_samples: Minimum samples in neighborhood
        metric: Distance metric to use
        cluster_selection_method: Method for selecting clusters
        clusterer: HDBSCAN instance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize HDBSCAN clusterer.

        Config parameters:
            min_cluster_size (int): Minimum cluster size, default 10
            min_samples (int): Minimum samples for core point, default 2
            metric (str): Distance metric, default 'euclidean'
            cluster_selection_method (str): Selection method, default 'eom'
                Options: 'eom' (excess of mass), 'leaf'
            cluster_selection_epsilon (float): Distance threshold, default 0.0
            standardize (bool): Standardize features before clustering, default True
            alpha (float): RBF kernel parameter, default 1.0
        """
        super().__init__(config)

    def configure(self):
        """Configure HDBSCAN parameters."""
        self.min_cluster_size = self.config.get('min_cluster_size', 10)
        self.min_samples = self.config.get('min_samples', 2)
        self.metric = self.config.get('metric', 'euclidean')
        self.cluster_selection_method = self.config.get('cluster_selection_method', 'eom')
        self.cluster_selection_epsilon = self.config.get('cluster_selection_epsilon', 0.0)
        self.standardize = self.config.get('standardize', True)
        self.alpha = self.config.get('alpha', 1.0)

        # Initialize scaler if needed
        if self.standardize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        self.logger.info(
            f"HDBSCAN configured: min_cluster_size={self.min_cluster_size}, "
            f"min_samples={self.min_samples}, metric={self.metric}"
        )

    def validate_config(self):
        """Validate configuration parameters."""
        if self.min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {self.min_cluster_size}")

        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}")

        valid_metrics = ['euclidean', 'manhattan', 'cosine', 'l1', 'l2']
        if self.metric not in valid_metrics:
            self.logger.warning(
                f"Metric '{self.metric}' may not be supported. "
                f"Commonly used: {valid_metrics}"
            )

    def fit(self, features: np.ndarray) -> 'HDBSCANClusterer':
        """
        Train HDBSCAN on feature data.

        Args:
            features: Feature matrix with shape (n_samples, n_features)

        Returns:
            Self (for method chaining)
        """
        if len(features.shape) != 2:
            raise ValueError(f"Expected 2D features array, got shape {features.shape}")

        self.logger.info(f"Fitting HDBSCAN on {features.shape[0]} samples")

        # Standardize if configured
        if self.standardize:
            features = self.scaler.fit_transform(features)

        # Initialize HDBSCAN
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            alpha=self.alpha,
            core_dist_n_jobs=-1  # Use all CPU cores
        )

        # Fit and predict
        self.labels_ = self.clusterer.fit_predict(features)

        # Count clusters (excluding noise: -1)
        unique_labels = np.unique(self.labels_)
        self.n_clusters_ = len(unique_labels[unique_labels != -1])

        self.is_fitted_ = True

        # Log results
        noise_count = np.sum(self.labels_ == -1)
        self.logger.info(
            f"✓ HDBSCAN found {self.n_clusters_} clusters, "
            f"{noise_count} noise points ({noise_count/len(self.labels_)*100:.1f}%)"
        )

        return self

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fit HDBSCAN and return cluster labels.

        HDBSCAN performs fit and predict in one step, so this is the same as fit().

        Args:
            features: Feature matrix with shape (n_samples, n_features)

        Returns:
            Cluster labels with shape (n_samples,)
        """
        self.fit(features)
        return self.labels_

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        HDBSCAN doesn't support traditional predict() on new data.

        Use fit_predict() instead which fits and labels all data at once.

        Args:
            features: Feature matrix with shape (n_samples, n_features)

        Returns:
            Cluster labels with shape (n_samples,)

        Raises:
            NotImplementedError: HDBSCAN requires fit_predict() on all data
        """
        raise NotImplementedError(
            "HDBSCAN doesn't support predict() on new data. "
            "Use fit_predict() on all data at once."
        )

    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Return detailed clustering information.

        Returns:
            Dictionary with cluster statistics and quality metrics
        """
        info = super().get_cluster_info()

        if self.is_fitted_ and hasattr(self.clusterer, 'cluster_persistence_'):
            # Add HDBSCAN-specific metrics
            info['cluster_persistence'] = {
                int(label): float(persistence)
                for label, persistence in enumerate(self.clusterer.cluster_persistence_)
            }

            # Outlier scores
            if hasattr(self.clusterer, 'outlier_scores_'):
                info['mean_outlier_score'] = float(np.mean(self.clusterer.outlier_scores_))

        return info

    def get_probabilities(self) -> np.ndarray:
        """
        Get cluster membership probabilities.

        Returns:
            Probability array with shape (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Clusterer must be fitted first")

        if hasattr(self.clusterer, 'probabilities_'):
            return self.clusterer.probabilities_
        else:
            self.logger.warning("Probabilities not available for this clustering")
            return np.ones(len(self.labels_))

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"HDBSCANClusterer("
                f"n_clusters={self.n_clusters_}, "
                f"min_cluster_size={self.min_cluster_size}, "
                f"min_samples={self.min_samples}, "
                f"fitted=True)"
            )
        else:
            return (
                f"HDBSCANClusterer("
                f"min_cluster_size={self.min_cluster_size}, "
                f"min_samples={self.min_samples}, "
                f"fitted=False)"
            )


# Convenience function
def cluster_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 2,
    metric: str = 'euclidean',
    standardize: bool = True
) -> np.ndarray:
    """
    Cluster features using HDBSCAN.

    Args:
        features: Feature matrix (n_samples, n_features)
        min_cluster_size: Minimum cluster size (default 10)
        min_samples: Minimum samples for core point (default 2)
        metric: Distance metric (default 'euclidean')
        standardize: Standardize features (default True)

    Returns:
        Cluster labels array
    """
    config = {
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'metric': metric,
        'standardize': standardize
    }

    clusterer = HDBSCANClusterer(config)
    labels = clusterer.fit_predict(features)

    return labels
