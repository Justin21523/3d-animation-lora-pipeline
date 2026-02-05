"""
Blur detection filter for image quality assessment.

Uses Laplacian variance to detect blurry images that may reduce
training quality.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
from PIL import Image
import numpy as np
import cv2
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.quality_filter import BaseQualityFilter


class BlurFilter(BaseQualityFilter):
    """
    Filter to detect and reject blurry images.

    Uses Laplacian variance method: computes the variance of the Laplacian
    operator applied to the image. Low variance indicates blur.

    Attributes:
        threshold: Laplacian variance threshold (lower = more blurry)
        downscale_factor: Factor to downscale image before processing (speeds up computation)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the blur filter.

        Config parameters:
            threshold (float): Blur threshold, default 100.0
                Lower values = stricter (fewer images pass)
                Typical range: 50-150 for 3D animation, 80-120 recommended
            downscale_factor (int): Downscale factor for speed, default 2
            method (str): Detection method, default 'laplacian'
                Options: 'laplacian', 'fft', 'gradient'
        """
        super().__init__(config)

    def configure(self):
        """Configure blur filter parameters."""
        self.threshold = self.config.get('threshold', 100.0)
        self.downscale_factor = self.config.get('downscale_factor', 2)
        self.method = self.config.get('method', 'laplacian')

        if self.method not in ['laplacian', 'fft', 'gradient']:
            raise ValueError(
                f"Invalid method '{self.method}'. "
                "Must be one of: laplacian, fft, gradient"
            )

    def validate_config(self):
        """Validate configuration parameters."""
        threshold = self.config.get('threshold', 100.0)
        downscale_factor = self.config.get('downscale_factor', 2)

        if not isinstance(threshold, (int, float)) or threshold < 0:
            raise ValueError(f"threshold must be non-negative number, got {threshold}")

        if not isinstance(downscale_factor, int) or downscale_factor < 1:
            raise ValueError(f"downscale_factor must be positive integer, got {downscale_factor}")

    def compute_laplacian_variance(self, image: np.ndarray) -> float:
        """
        Compute Laplacian variance for blur detection.

        Args:
            image: Image as numpy array (grayscale or RGB)

        Returns:
            Laplacian variance (higher = sharper)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Downscale for speed
        if self.downscale_factor > 1:
            h, w = gray.shape
            new_h = h // self.downscale_factor
            new_w = w // self.downscale_factor
            gray = cv2.resize(gray, (new_w, new_h))

        # Compute Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return variance

    def compute_fft_blur(self, image: np.ndarray) -> float:
        """
        Compute blur metric using FFT (frequency domain analysis).

        High-frequency components indicate sharp edges.

        Args:
            image: Image as numpy array

        Returns:
            Blur score (higher = sharper)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Downscale
        if self.downscale_factor > 1:
            h, w = gray.shape
            new_h = h // self.downscale_factor
            new_w = w // self.downscale_factor
            gray = cv2.resize(gray, (new_w, new_h))

        # Compute FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Get high-frequency energy (outer regions)
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4

        # Create mask for high frequencies (exclude center)
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_w) ** 2 + (y - center_h) ** 2) > radius ** 2

        high_freq_energy = np.mean(magnitude[mask])

        return high_freq_energy

    def compute_gradient_magnitude(self, image: np.ndarray) -> float:
        """
        Compute gradient magnitude for blur detection.

        Args:
            image: Image as numpy array

        Returns:
            Mean gradient magnitude (higher = sharper)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Downscale
        if self.downscale_factor > 1:
            h, w = gray.shape
            new_h = h // self.downscale_factor
            new_w = w // self.downscale_factor
            gray = cv2.resize(gray, (new_w, new_h))

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        mean_magnitude = np.mean(magnitude)

        return mean_magnitude

    def compute_blur_score(self, image: np.ndarray) -> float:
        """
        Compute blur score using configured method.

        Args:
            image: Image as numpy array

        Returns:
            Blur score (interpretation depends on method)
        """
        if self.method == 'laplacian':
            return self.compute_laplacian_variance(image)
        elif self.method == 'fft':
            return self.compute_fft_blur(image)
        elif self.method == 'gradient':
            return self.compute_gradient_magnitude(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def filter_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if image is blurry.

        Args:
            image: Input image
            metadata: Optional metadata dict
                Can include 'blur_threshold' to override default

        Returns:
            Tuple of (passed, reason):
            - passed: True if NOT blurry (passes filter), False if blurry
            - reason: Explanation if blurry, None otherwise
        """
        try:
            # Load image
            pil_image = self._load_image(image)
            np_image = np.array(pil_image)

            # Use metadata threshold if provided
            threshold = metadata.get('blur_threshold', self.threshold) if metadata else self.threshold

            # Compute blur score
            blur_score = self.compute_blur_score(np_image)

            # Check threshold
            if blur_score < threshold:
                return False, f"Blurry (score={blur_score:.2f}, threshold={threshold})"
            else:
                return True, None

        except Exception as e:
            self.logger.warning(f"Failed to check blur for {image}: {e}")
            return False, f"Blur check failed: {e}"

    def get_filter_stats(self) -> Dict[str, Any]:
        """
        Return detailed blur filtering statistics.

        Returns:
            Dictionary with statistics including method and threshold
        """
        base_stats = super().get_filter_stats()
        base_stats['method'] = self.method
        base_stats['threshold'] = self.threshold
        return base_stats


# Convenience function for backward compatibility
def filter_blurry_images(
    image_paths: list,
    threshold: float = 100.0,
    method: str = 'laplacian'
) -> list:
    """
    Filter out blurry images from a list.

    Args:
        image_paths: List of image file paths
        threshold: Blur threshold (default 100.0)
        method: Detection method (default 'laplacian')

    Returns:
        List of sharp image paths (blurry images removed)
    """
    config = {
        'threshold': threshold,
        'method': method
    }

    blur_filter = BlurFilter(config)
    sharp_images = []

    for image_path in image_paths:
        passed, _ = blur_filter.filter_single(image_path)
        if passed:
            sharp_images.append(image_path)

    return sharp_images
