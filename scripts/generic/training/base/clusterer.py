"""
Abstract base class for clustering algorithms.

Clusterers group similar items together based on their feature representations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
import logging


class BaseClusterer(ABC):
    """
    Abstract base class for all clustering implementations.

    Subclasses must implement:
    - fit(): Train the clustering algorithm on feature data
    - predict(): Assign new data points to clusters
    - fit_predict(): Convenience method combining fit() and predict()

    Optional methods to override:
    - configure(): Customize configuration loading
    - validate_config(): Validate configuration parameters
    - get_cluster_info(): Return detailed information about clusters
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the clusterer.

        Args:
            config: Configuration dictionary with algorithm-specific parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Clustering results
        self.labels_ = None
        self.n_clusters_ = None
        self.is_fitted_ = False

        # Allow subclasses to perform custom configuration first
        self.configure()

        # Then validate the configured parameters
        self.validate_config()

    @abstractmethod
    def fit(self, features: np.ndarray) -> 'BaseClusterer':
        """
        Train the clustering algorithm on feature data.

        Args:
            features: Feature matrix with shape (n_samples, n_features)

        Returns:
            Self (for method chaining)
        """
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Assign new data points to existing clusters.

        Args:
            features: Feature matrix with shape (n_samples, n_features)

        Returns:
            Cluster labels with shape (n_samples,)
            Label -1 indicates noise/outliers
        """
        pass

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Train the clustering algorithm and return cluster assignments.

        Default implementation calls fit() then predict().
        Some algorithms may provide optimized implementations.

        Args:
            features: Feature matrix with shape (n_samples, n_features)

        Returns:
            Cluster labels with shape (n_samples,)
        """
        self.fit(features)
        return self.predict(features)

    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Return detailed information about the clustering results.

        Returns:
            Dictionary with cluster statistics:
            - n_clusters: Number of clusters found
            - cluster_sizes: Size of each cluster
            - noise_count: Number of noise points (-1 label)
            - silhouette_score: Clustering quality metric (if available)
        """
        if not self.is_fitted_:
            return {"error": "Clusterer not fitted yet"}

        labels = self.labels_
        unique_labels = np.unique(labels)

        # Separate noise points (-1) from actual clusters
        cluster_labels = unique_labels[unique_labels != -1]
        noise_count = np.sum(labels == -1)

        cluster_sizes = {
            int(label): int(np.sum(labels == label))
            for label in cluster_labels
        }

        info = {
            "n_clusters": len(cluster_labels),
            "cluster_sizes": cluster_sizes,
            "noise_count": noise_count,
            "total_samples": len(labels),
        }

        return info

    def get_cluster_indices(self, label: int) -> np.ndarray:
        """
        Get indices of all samples belonging to a specific cluster.

        Args:
            label: Cluster label

        Returns:
            Array of indices
        """
        if not self.is_fitted_:
            raise RuntimeError("Clusterer must be fitted before getting cluster indices")

        return np.where(self.labels_ == label)[0]

    def get_noise_indices(self) -> np.ndarray:
        """
        Get indices of all noise/outlier samples (label -1).

        Returns:
            Array of indices for noise points
        """
        if not self.is_fitted_:
            raise RuntimeError("Clusterer must be fitted before getting noise indices")

        return np.where(self.labels_ == -1)[0]

    def configure(self):
        """
        Perform custom configuration setup.

        Subclasses can override this method to set algorithm-specific parameters.
        """
        pass

    def validate_config(self):
        """
        Validate configuration parameters.

        Subclasses can override this method to check for required config keys
        and valid parameter ranges.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def __repr__(self) -> str:
        if self.is_fitted_:
            return f"{self.__class__.__name__}(n_clusters={self.n_clusters_}, fitted=True)"
        else:
            return f"{self.__class__.__name__}(fitted=False)"
