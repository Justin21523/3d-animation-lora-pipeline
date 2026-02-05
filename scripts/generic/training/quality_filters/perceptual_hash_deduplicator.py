"""
Perceptual hash-based image deduplication filter.

Uses imagehash library to detect near-duplicate images based on
perceptual hashing (pHash/aHash).
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
from PIL import Image
import numpy as np
import logging

try:
    import imagehash
except ImportError:
    raise ImportError(
        "imagehash is required for PerceptualHashDeduplicator. "
        "Install with: pip install imagehash"
    )

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.quality_filter import BaseQualityFilter


class PerceptualHashDeduplicator(BaseQualityFilter):
    """
    Deduplication filter using perceptual hashing.

    Detects near-duplicate images by computing perceptual hashes and comparing
    Hamming distances. Images with similar hashes (within threshold) are
    considered duplicates.

    Attributes:
        hash_size: Hash matrix size (larger = more sensitive)
        threshold: Hamming distance threshold (lower = stricter)
        hash_method: Hashing method ('average', 'perceptual', 'difference', 'wavelet')
        seen_hashes: Dictionary mapping hashes to image paths
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the deduplicator.

        Config parameters:
            hash_size (int): Hash size, default 16
            threshold (int): Hamming distance threshold, default 8
            hash_method (str): Hash method, default 'average'
                Options: 'average', 'perceptual', 'difference', 'wavelet'
        """
        super().__init__(config)

        # Track seen hashes
        self.seen_hashes = {}  # hash -> (image_path, index)
        self.next_index = 0

    def configure(self):
        """Configure deduplicator parameters."""
        self.hash_size = self.config.get('hash_size', 16)
        self.threshold = self.config.get('threshold', 8)
        self.hash_method = self.config.get('hash_method', 'average')

        # Select hash function
        hash_functions = {
            'average': imagehash.average_hash,
            'perceptual': imagehash.phash,
            'difference': imagehash.dhash,
            'wavelet': imagehash.whash,
        }

        if self.hash_method not in hash_functions:
            raise ValueError(
                f"Invalid hash_method '{self.hash_method}'. "
                f"Must be one of: {list(hash_functions.keys())}"
            )

        self.hash_func = hash_functions[self.hash_method]

    def validate_config(self):
        """Validate configuration parameters."""
        hash_size = self.config.get('hash_size', 16)
        threshold = self.config.get('threshold', 8)

        if not isinstance(hash_size, int) or hash_size < 4 or hash_size > 32:
            raise ValueError(f"hash_size must be integer in range [4, 32], got {hash_size}")

        if not isinstance(threshold, int) or threshold < 0:
            raise ValueError(f"threshold must be non-negative integer, got {threshold}")

    def compute_hash(self, image: Union[str, Path, Image.Image, np.ndarray]) -> imagehash.ImageHash:
        """
        Compute perceptual hash for an image.

        Args:
            image: Input image

        Returns:
            ImageHash object
        """
        pil_image = self._load_image(image)
        phash = self.hash_func(pil_image, hash_size=self.hash_size)
        return phash

    def filter_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if image is a duplicate of a previously seen image.

        Args:
            image: Input image
            metadata: Optional metadata (not used)

        Returns:
            Tuple of (passed, reason):
            - passed: True if NOT a duplicate (passes filter), False if duplicate
            - reason: Explanation if duplicate, None otherwise
        """
        try:
            phash = self.compute_hash(image)
        except Exception as e:
            self.logger.warning(f"Failed to compute hash for {image}: {e}")
            return False, f"Hash computation failed: {e}"

        # Check against seen hashes
        for seen_hash, (seen_path, seen_index) in self.seen_hashes.items():
            distance = phash - seen_hash  # Hamming distance

            if distance <= self.threshold:
                # Duplicate found
                image_name = Path(image).name if isinstance(image, (str, Path)) else f"image_{self.next_index}"
                return False, f"Duplicate of {seen_path} (distance={distance})"

        # Not a duplicate, add to seen hashes
        image_path = Path(image) if isinstance(image, (str, Path)) else f"image_{self.next_index}"
        self.seen_hashes[phash] = (image_path, self.next_index)
        self.next_index += 1

        return True, None

    def reset(self):
        """
        Clear seen hashes and reset statistics.

        Useful when starting a new filtering session.
        """
        self.seen_hashes.clear()
        self.next_index = 0
        self.reset_stats()
        self.logger.info("Reset deduplicator state")

    def get_filter_stats(self) -> Dict[str, Any]:
        """
        Return detailed deduplication statistics.

        Returns:
            Dictionary with statistics including unique images count
        """
        base_stats = super().get_filter_stats()
        base_stats['unique_images'] = len(self.seen_hashes)
        base_stats['hash_method'] = self.hash_method
        base_stats['threshold'] = self.threshold
        return base_stats

    def find_duplicates(
        self,
        images: list,
        return_groups: bool = False
    ) -> Union[Dict[str, list], list]:
        """
        Find all duplicate groups in a list of images.

        This method doesn't affect the internal state (seen_hashes).

        Args:
            images: List of image paths or Image objects
            return_groups: If True, return duplicate groups; if False, return list of duplicates to remove

        Returns:
            If return_groups=True: Dict mapping representative image to list of duplicates
            If return_groups=False: List of images to remove (keeping one from each group)
        """
        temp_hashes = {}
        duplicate_groups = {}

        for image in images:
            try:
                phash = self.compute_hash(image)
            except Exception as e:
                self.logger.warning(f"Failed to compute hash for {image}: {e}")
                continue

            # Find if this image is similar to any seen before
            found_duplicate = False
            for seen_hash, representative in temp_hashes.items():
                distance = phash - seen_hash

                if distance <= self.threshold:
                    # Add to existing group
                    if representative not in duplicate_groups:
                        duplicate_groups[representative] = []
                    duplicate_groups[representative].append(image)
                    found_duplicate = True
                    break

            if not found_duplicate:
                # New unique image, could be representative of a group
                temp_hashes[phash] = image

        if return_groups:
            return duplicate_groups
        else:
            # Return list of images to remove
            to_remove = []
            for duplicates in duplicate_groups.values():
                to_remove.extend(duplicates)
            return to_remove


# Convenience function for backward compatibility
def deduplicate_images(
    image_paths: list,
    hash_size: int = 16,
    threshold: int = 8,
    hash_method: str = 'average'
) -> list:
    """
    Deduplicate a list of images using perceptual hashing.

    Args:
        image_paths: List of image file paths
        hash_size: Hash size (default 16)
        threshold: Hamming distance threshold (default 8)
        hash_method: Hashing method (default 'average')

    Returns:
        List of unique image paths (duplicates removed)
    """
    config = {
        'hash_size': hash_size,
        'threshold': threshold,
        'hash_method': hash_method
    }

    deduplicator = PerceptualHashDeduplicator(config)
    unique_images = []

    for image_path in image_paths:
        passed, _ = deduplicator.filter_single(image_path)
        if passed:
            unique_images.append(image_path)

    return unique_images
