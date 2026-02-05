"""
Image size filter for quality assessment.

Filters images based on minimum/maximum dimensions or aspect ratio.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
from PIL import Image
import numpy as np
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.quality_filter import BaseQualityFilter


class SizeFilter(BaseQualityFilter):
    """
    Filter to reject images based on size constraints.

    Can filter by:
    - Minimum width/height
    - Maximum width/height
    - Minimum/maximum total pixels
    - Aspect ratio range

    Attributes:
        min_width: Minimum width in pixels
        min_height: Minimum height in pixels
        max_width: Maximum width in pixels (None = no limit)
        max_height: Maximum height in pixels (None = no limit)
        min_pixels: Minimum total pixels (None = no limit)
        max_pixels: Maximum total pixels (None = no limit)
        min_aspect_ratio: Minimum aspect ratio (width/height)
        max_aspect_ratio: Maximum aspect ratio (width/height)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the size filter.

        Config parameters:
            min_width (int): Minimum width, default 128
            min_height (int): Minimum height, default 128
            max_width (int): Maximum width, default None (no limit)
            max_height (int): Maximum height, default None (no limit)
            min_pixels (int): Minimum total pixels, default None
            max_pixels (int): Maximum total pixels, default None
            min_aspect_ratio (float): Minimum aspect ratio, default 0.5
            max_aspect_ratio (float): Maximum aspect ratio, default 2.0
        """
        super().__init__(config)

    def configure(self):
        """Configure size filter parameters."""
        self.min_width = self.config.get('min_width', 128)
        self.min_height = self.config.get('min_height', 128)
        self.max_width = self.config.get('max_width', None)
        self.max_height = self.config.get('max_height', None)
        self.min_pixels = self.config.get('min_pixels', None)
        self.max_pixels = self.config.get('max_pixels', None)
        self.min_aspect_ratio = self.config.get('min_aspect_ratio', 0.5)
        self.max_aspect_ratio = self.config.get('max_aspect_ratio', 2.0)

    def validate_config(self):
        """Validate configuration parameters."""
        if self.min_width < 0 or self.min_height < 0:
            raise ValueError("min_width and min_height must be non-negative")

        if self.max_width is not None and self.max_width < self.min_width:
            raise ValueError("max_width must be >= min_width")

        if self.max_height is not None and self.max_height < self.min_height:
            raise ValueError("max_height must be >= min_height")

        if self.min_aspect_ratio <= 0 or self.max_aspect_ratio <= 0:
            raise ValueError("aspect ratios must be positive")

        if self.max_aspect_ratio < self.min_aspect_ratio:
            raise ValueError("max_aspect_ratio must be >= min_aspect_ratio")

    def filter_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if image meets size constraints.

        Args:
            image: Input image
            metadata: Optional metadata (not used)

        Returns:
            Tuple of (passed, reason):
            - passed: True if meets constraints, False otherwise
            - reason: Explanation if rejected, None otherwise
        """
        try:
            # Load image
            pil_image = self._load_image(image)
            width, height = pil_image.size
            total_pixels = width * height
            aspect_ratio = width / height

            # Check minimum dimensions
            if width < self.min_width:
                return False, f"Width {width} < minimum {self.min_width}"

            if height < self.min_height:
                return False, f"Height {height} < minimum {self.min_height}"

            # Check maximum dimensions
            if self.max_width is not None and width > self.max_width:
                return False, f"Width {width} > maximum {self.max_width}"

            if self.max_height is not None and height > self.max_height:
                return False, f"Height {height} > maximum {self.max_height}"

            # Check pixel count
            if self.min_pixels is not None and total_pixels < self.min_pixels:
                return False, f"Total pixels {total_pixels} < minimum {self.min_pixels}"

            if self.max_pixels is not None and total_pixels > self.max_pixels:
                return False, f"Total pixels {total_pixels} > maximum {self.max_pixels}"

            # Check aspect ratio
            if aspect_ratio < self.min_aspect_ratio:
                return False, f"Aspect ratio {aspect_ratio:.2f} < minimum {self.min_aspect_ratio}"

            if aspect_ratio > self.max_aspect_ratio:
                return False, f"Aspect ratio {aspect_ratio:.2f} > maximum {self.max_aspect_ratio}"

            # All checks passed
            return True, None

        except Exception as e:
            self.logger.warning(f"Failed to check size for {image}: {e}")
            return False, f"Size check failed: {e}"

    def get_filter_stats(self) -> Dict[str, Any]:
        """
        Return detailed size filtering statistics.

        Returns:
            Dictionary with statistics including constraints
        """
        base_stats = super().get_filter_stats()
        base_stats['constraints'] = {
            'min_width': self.min_width,
            'min_height': self.min_height,
            'max_width': self.max_width,
            'max_height': self.max_height,
            'min_aspect_ratio': self.min_aspect_ratio,
            'max_aspect_ratio': self.max_aspect_ratio,
        }
        return base_stats


# Convenience function for backward compatibility
def filter_by_size(
    image_paths: list,
    min_width: int = 128,
    min_height: int = 128,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0
) -> list:
    """
    Filter images by size constraints.

    Args:
        image_paths: List of image file paths
        min_width: Minimum width (default 128)
        min_height: Minimum height (default 128)
        max_width: Maximum width (default None)
        max_height: Maximum height (default None)
        min_aspect_ratio: Minimum aspect ratio (default 0.5)
        max_aspect_ratio: Maximum aspect ratio (default 2.0)

    Returns:
        List of image paths that meet size constraints
    """
    config = {
        'min_width': min_width,
        'min_height': min_height,
        'max_width': max_width,
        'max_height': max_height,
        'min_aspect_ratio': min_aspect_ratio,
        'max_aspect_ratio': max_aspect_ratio,
    }

    size_filter = SizeFilter(config)
    valid_images = []

    for image_path in image_paths:
        passed, _ = size_filter.filter_single(image_path)
        if passed:
            valid_images.append(image_path)

    return valid_images
