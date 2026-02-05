"""
Abstract base class for quality filtering.

Quality filters assess and filter training data based on various criteria
(blur, size, duplicates, etc.) to improve dataset quality.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Tuple
from PIL import Image
import numpy as np
import logging


class BaseQualityFilter(ABC):
    """
    Abstract base class for all quality filtering implementations.

    Subclasses must implement:
    - filter_single(): Assess a single image
    - filter_batch(): Assess multiple images (can use default implementation)

    Optional methods to override:
    - configure(): Customize configuration loading
    - validate_config(): Validate configuration parameters
    - get_filter_stats(): Return statistics about filtered data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the quality filter.

        Args:
            config: Configuration dictionary with filter-specific parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Filter statistics
        self.total_processed = 0
        self.total_passed = 0
        self.total_rejected = 0

        # Allow subclasses to perform custom configuration first
        self.configure()

        # Then validate the configured parameters
        self.validate_config()

    @abstractmethod
    def filter_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Assess whether a single image passes the quality filter.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            metadata: Optional metadata dict with additional information

        Returns:
            Tuple of (passed, reason):
            - passed: True if image passes filter, False otherwise
            - reason: Optional string explaining why image was rejected
        """
        pass

    def filter_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True
    ) -> List[Tuple[bool, Optional[str]]]:
        """
        Assess multiple images for quality filtering.

        Default implementation processes images one by one. Subclasses can
        override for optimized batched processing.

        Args:
            images: List of input images
            metadata: Optional list of metadata dicts (one per image)
            show_progress: Whether to show progress bar

        Returns:
            List of (passed, reason) tuples
        """
        if metadata is None:
            metadata = [None] * len(images)

        if len(metadata) != len(images):
            raise ValueError(f"Number of metadata entries ({len(metadata)}) must match number of images ({len(images)})")

        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                image_metadata_pairs = tqdm(
                    list(zip(images, metadata)),
                    desc=f"Filtering ({self.__class__.__name__})"
                )
            except ImportError:
                image_metadata_pairs = zip(images, metadata)
        else:
            image_metadata_pairs = zip(images, metadata)

        for image, meta in image_metadata_pairs:
            try:
                passed, reason = self.filter_single(image, metadata=meta)
                results.append((passed, reason))

                # Update statistics
                self.total_processed += 1
                if passed:
                    self.total_passed += 1
                else:
                    self.total_rejected += 1

            except Exception as e:
                self.logger.warning(f"Failed to filter {image}: {e}")
                results.append((False, f"Error: {str(e)}"))
                self.total_processed += 1
                self.total_rejected += 1

        return results

    def get_filter_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the filtering process.

        Returns:
            Dictionary with filter statistics:
            - total_processed: Total number of images processed
            - total_passed: Number of images that passed the filter
            - total_rejected: Number of images that were rejected
            - pass_rate: Percentage of images that passed
        """
        pass_rate = (self.total_passed / self.total_processed * 100) if self.total_processed > 0 else 0.0

        return {
            "total_processed": self.total_processed,
            "total_passed": self.total_passed,
            "total_rejected": self.total_rejected,
            "pass_rate": round(pass_rate, 2),
        }

    def reset_stats(self):
        """Reset filter statistics."""
        self.total_processed = 0
        self.total_passed = 0
        self.total_rejected = 0

    def configure(self):
        """
        Perform custom configuration setup.

        Subclasses can override this method to set filter-specific parameters.
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
        stats = self.get_filter_stats()
        return f"{self.__class__.__name__}(processed={stats['total_processed']}, pass_rate={stats['pass_rate']}%)"


class CompositeQualityFilter(BaseQualityFilter):
    """
    Composite filter that combines multiple quality filters.

    Useful for applying multiple filtering criteria in sequence.
    """

    def __init__(self, filters: List[BaseQualityFilter], mode: str = "all"):
        """
        Initialize the composite filter.

        Args:
            filters: List of quality filters to apply
            mode: Combination mode:
                - "all": Image must pass all filters (AND logic)
                - "any": Image must pass at least one filter (OR logic)
        """
        super().__init__()
        self.filters = filters
        self.mode = mode

        if mode not in ["all", "any"]:
            raise ValueError(f"Invalid mode '{mode}', must be 'all' or 'any'")

    def filter_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Apply all filters to a single image.

        Args:
            image: Input image
            metadata: Optional metadata

        Returns:
            Tuple of (passed, reason)
        """
        results = []
        reasons = []

        for filter_obj in self.filters:
            passed, reason = filter_obj.filter_single(image, metadata)
            results.append(passed)
            if reason:
                reasons.append(f"{filter_obj.__class__.__name__}: {reason}")

        if self.mode == "all":
            final_passed = all(results)
        else:  # mode == "any"
            final_passed = any(results)

        final_reason = "; ".join(reasons) if reasons else None

        return final_passed, final_reason

    def configure(self):
        """No additional configuration needed for composite filter."""
        pass

    def validate_config(self):
        """No additional validation needed for composite filter."""
        pass
