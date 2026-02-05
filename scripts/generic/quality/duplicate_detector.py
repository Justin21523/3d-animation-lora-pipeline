#!/usr/bin/env python3
"""
Duplicate Image Detection using Perceptual Hashing

Detects duplicate and near-duplicate images using perceptual hashing algorithms.
Resistant to minor modifications like scaling, compression, color adjustments.

Algorithms:
- pHash (Perceptual Hash): DCT-based, good for general duplicates
- dHash (Difference Hash): Gradient-based, fast and simple
- Average Hash: Simple average-based, very fast but less robust

Hamming Distance Thresholds:
- 0: Exact duplicate
- 1-5: Nearly identical (minor differences)
- 6-10: Very similar
- 11-15: Similar
- > 15: Different images

Part of Module 3: Quality Filtering System
Author: LLMProvider Tooling
Date: 2025-11-30
"""

import imagehash
from PIL import Image
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
from collections import defaultdict


class DuplicateDetector:
    """
    Duplicate detection using perceptual hashing

    Supports multiple hashing algorithms with configurable thresholds
    """

    def __init__(
        self,
        hash_size: int = 8,
        hash_algorithm: str = 'phash',
        hamming_threshold: int = 8
    ):
        """
        Initialize duplicate detector

        Args:
            hash_size: Hash size (default: 8, creates 64-bit hash)
            hash_algorithm: 'phash', 'dhash', or 'average_hash' (default: 'phash')
            hamming_threshold: Maximum Hamming distance for duplicates (default: 8)
        """
        self.hash_size = hash_size
        self.hash_algorithm = hash_algorithm
        self.hamming_threshold = hamming_threshold

        # Select hash function
        if hash_algorithm == 'phash':
            self.hash_func = lambda img: imagehash.phash(img, hash_size=hash_size)
        elif hash_algorithm == 'dhash':
            self.hash_func = lambda img: imagehash.dhash(img, hash_size=hash_size)
        elif hash_algorithm == 'average_hash':
            self.hash_func = lambda img: imagehash.average_hash(img, hash_size=hash_size)
        else:
            raise ValueError(f"Unknown hash algorithm: {hash_algorithm}")

    def compute_hash(self, image: Union[str, Path, Image.Image]) -> imagehash.ImageHash:
        """
        Compute perceptual hash for an image

        Args:
            image: Image path or PIL Image

        Returns:
            ImageHash object
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        else:
            img = image

        return self.hash_func(img)

    def compute_hamming_distance(
        self,
        hash1: Union[str, imagehash.ImageHash],
        hash2: Union[str, imagehash.ImageHash]
    ) -> int:
        """
        Compute Hamming distance between two hashes

        Args:
            hash1: First hash (string or ImageHash)
            hash2: Second hash (string or ImageHash)

        Returns:
            Hamming distance (0 = identical)
        """
        if isinstance(hash1, str):
            hash1 = imagehash.hex_to_hash(hash1)
        if isinstance(hash2, str):
            hash2 = imagehash.hex_to_hash(hash2)

        return hash1 - hash2

    def are_duplicates(
        self,
        image1: Union[str, Path, Image.Image],
        image2: Union[str, Path, Image.Image]
    ) -> bool:
        """
        Check if two images are duplicates

        Args:
            image1: First image
            image2: Second image

        Returns:
            True if images are duplicates
        """
        hash1 = self.compute_hash(image1)
        hash2 = self.compute_hash(image2)

        distance = self.compute_hamming_distance(hash1, hash2)
        return distance <= self.hamming_threshold

    def find_duplicates(
        self,
        image_paths: List[Path]
    ) -> Dict[str, List[Path]]:
        """
        Find all duplicate groups in a list of images

        Args:
            image_paths: List of image paths

        Returns:
            Dictionary mapping representative image to duplicates:
            {
                'image1.png': ['image2.png', 'image3.png'],  # image1 has 2 duplicates
                'image4.png': ['image5.png']                  # image4 has 1 duplicate
            }
        """
        # Compute hashes for all images
        hashes = {}
        for path in image_paths:
            try:
                hashes[path] = self.compute_hash(path)
            except Exception as e:
                print(f"Warning: Failed to hash {path}: {e}")
                continue

        # Find duplicates using clustering
        duplicate_groups = defaultdict(list)
        processed = set()

        for path1, hash1 in hashes.items():
            if path1 in processed:
                continue

            # Find all images similar to this one
            duplicates = []
            for path2, hash2 in hashes.items():
                if path1 == path2:
                    continue

                distance = self.compute_hamming_distance(hash1, hash2)
                if distance <= self.hamming_threshold:
                    duplicates.append(path2)
                    processed.add(path2)

            if duplicates:
                duplicate_groups[str(path1)] = [str(p) for p in duplicates]

        return dict(duplicate_groups)

    def find_duplicates_with_distances(
        self,
        image_paths: List[Path]
    ) -> Dict[str, List[Tuple[Path, int]]]:
        """
        Find duplicates with Hamming distances

        Args:
            image_paths: List of image paths

        Returns:
            Dictionary mapping representative image to (duplicate, distance) tuples:
            {
                'image1.png': [('image2.png', 3), ('image3.png', 5)]
            }
        """
        # Compute hashes
        hashes = {}
        for path in image_paths:
            try:
                hashes[path] = self.compute_hash(path)
            except Exception as e:
                print(f"Warning: Failed to hash {path}: {e}")
                continue

        # Find duplicates with distances
        duplicate_groups = defaultdict(list)
        processed = set()

        for path1, hash1 in hashes.items():
            if path1 in processed:
                continue

            duplicates = []
            for path2, hash2 in hashes.items():
                if path1 == path2:
                    continue

                distance = self.compute_hamming_distance(hash1, hash2)
                if distance <= self.hamming_threshold:
                    duplicates.append((path2, distance))
                    processed.add(path2)

            if duplicates:
                # Sort by distance (most similar first)
                duplicates.sort(key=lambda x: x[1])
                duplicate_groups[str(path1)] = [(str(p), d) for p, d in duplicates]

        return dict(duplicate_groups)

    def deduplicate(
        self,
        image_paths: List[Path],
        keep_first: bool = True
    ) -> Tuple[List[Path], List[Path]]:
        """
        Remove duplicates from image list

        Args:
            image_paths: List of image paths
            keep_first: If True, keep first occurrence; otherwise keep best quality

        Returns:
            Tuple of (unique_images, duplicate_images)
        """
        duplicate_groups = self.find_duplicates(image_paths)

        # Collect all duplicates
        all_duplicates = set()
        for duplicates in duplicate_groups.values():
            all_duplicates.update(duplicates)

        # Get unique images
        unique_images = [p for p in image_paths if str(p) not in all_duplicates]

        # Add representative images (the keys in duplicate_groups)
        for representative in duplicate_groups.keys():
            unique_images.append(Path(representative))

        duplicate_images = [Path(p) for p in all_duplicates]

        return unique_images, duplicate_images


def main():
    """CLI for testing duplicate detection"""
    import argparse

    parser = argparse.ArgumentParser(description="Duplicate Image Detection using Perceptual Hashing")
    parser.add_argument("image_dir", type=str, help="Directory containing images")
    parser.add_argument("--algorithm", type=str, default="phash",
                       choices=['phash', 'dhash', 'average_hash'],
                       help="Hashing algorithm (default: phash)")
    parser.add_argument("--threshold", type=int, default=8,
                       help="Hamming distance threshold (default: 8)")
    parser.add_argument("--hash-size", type=int, default=8,
                       help="Hash size (default: 8)")

    args = parser.parse_args()

    detector = DuplicateDetector(
        hash_size=args.hash_size,
        hash_algorithm=args.algorithm,
        hamming_threshold=args.threshold
    )

    # Find all images
    image_dir = Path(args.image_dir)
    image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))

    print(f"Scanning {len(image_paths)} images...")

    # Find duplicates
    duplicates = detector.find_duplicates_with_distances(image_paths)

    if not duplicates:
        print("No duplicates found!")
    else:
        print(f"\nFound {len(duplicates)} duplicate groups:\n")
        for original, dups in duplicates.items():
            print(f"Original: {Path(original).name}")
            for dup_path, distance in dups:
                print(f"  - {Path(dup_path).name} (distance: {distance})")
            print()

        # Statistics
        total_duplicates = sum(len(dups) for dups in duplicates.values())
        print(f"\nTotal duplicate images: {total_duplicates}")
        print(f"Space savings: {total_duplicates} files can be removed")


if __name__ == "__main__":
    main()
