"""
Abstract base class for feature extractors.

Feature extractors convert images into embedding vectors that can be used for
clustering, similarity search, and other downstream tasks.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import logging


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extraction implementations.

    Subclasses must implement:
    - extract_single(): Extract features from a single image
    - extract_batch(): Extract features from multiple images (can use default implementation)
    - get_feature_dim(): Return the dimensionality of extracted features

    Optional methods to override:
    - configure(): Customize configuration loading
    - validate_config(): Validate configuration parameters
    - cleanup(): Clean up resources on shutdown
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        Initialize the feature extractor.

        Args:
            config: Configuration dictionary with model-specific parameters
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.config = config or {}
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

        # Allow subclasses to perform custom configuration first
        self.configure()

        # Then validate the configured parameters
        self.validate_config()

    @abstractmethod
    def extract_single(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract features from a single image.

        Args:
            image: Input image (file path, PIL Image, or numpy array)

        Returns:
            Feature vector as numpy array with shape (feature_dim,)
        """
        pass

    def extract_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract features from multiple images.

        Default implementation processes images one by one. Subclasses can
        override for optimized batched processing.

        Args:
            images: List of input images
            batch_size: Number of images to process at once
            show_progress: Whether to show progress bar

        Returns:
            Feature matrix with shape (num_images, feature_dim)
        """
        features = []

        if show_progress:
            try:
                from tqdm import tqdm
                images = tqdm(images, desc=f"Extracting features ({self.__class__.__name__})")
            except ImportError:
                pass

        for image in images:
            try:
                feature = self.extract_single(image)
                features.append(feature)
            except Exception as e:
                self.logger.warning(f"Failed to extract features from {image}: {e}")
                # Return zero vector on failure
                features.append(np.zeros(self.get_feature_dim()))

        return np.array(features)

    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Return the dimensionality of extracted features.

        Returns:
            Integer dimension of feature vectors
        """
        pass

    def configure(self):
        """
        Perform custom configuration setup.

        Subclasses can override this method to initialize models, load weights,
        or perform other setup tasks.
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

    def cleanup(self):
        """
        Clean up resources (models, memory, etc.).

        Called when the extractor is no longer needed.
        Subclasses can override to release GPU memory or close file handles.
        """
        pass

    def _load_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Helper method to load and standardize image input.

        Args:
            image: Input image in various formats

        Returns:
            PIL Image object
        """
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, feature_dim={self.get_feature_dim()})"

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
