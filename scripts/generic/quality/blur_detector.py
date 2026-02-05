#!/usr/bin/env python3
"""
Blur Detection using Laplacian Variance

Detects blurry images using the variance of the Laplacian operator.
Sharp images have high variance, blurry images have low variance.

Method:
- Convert to grayscale
- Apply Laplacian operator
- Compute variance of the result
- Threshold to classify sharp/blurry

Typical thresholds:
- < 100: Very blurry
- 100-200: Slightly blurry
- > 200: Sharp

Part of Module 3: Quality Filtering System
Author: LLMProvider Tooling
Date: 2025-11-30
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union
from PIL import Image


class BlurDetector:
    """
    Blur detection using Laplacian variance method

    Fast, lightweight, and effective for most use cases.
    No ML model required.
    """

    def __init__(self, threshold: float = 100.0):
        """
        Initialize blur detector

        Args:
            threshold: Laplacian variance threshold (default: 100.0)
                       Images below this are considered blurry
        """
        self.threshold = threshold

    def compute_blur_score(self, image: Union[str, Path, np.ndarray, Image.Image]) -> float:
        """
        Compute blur score using Laplacian variance

        Args:
            image: Image path, numpy array, or PIL Image

        Returns:
            Blur score (higher = sharper)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image

        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Apply Laplacian operator
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Compute variance
        variance = laplacian.var()

        return float(variance)

    def is_blurry(self, image: Union[str, Path, np.ndarray, Image.Image]) -> bool:
        """
        Check if image is blurry

        Args:
            image: Image path, numpy array, or PIL Image

        Returns:
            True if blurry, False if sharp
        """
        score = self.compute_blur_score(image)
        return score < self.threshold

    def classify_blur_level(self, image: Union[str, Path, np.ndarray, Image.Image]) -> str:
        """
        Classify blur level into categories

        Args:
            image: Image path, numpy array, or PIL Image

        Returns:
            Blur level: 'sharp', 'slightly_blurry', 'very_blurry'
        """
        score = self.compute_blur_score(image)

        if score >= 200:
            return 'sharp'
        elif score >= 100:
            return 'slightly_blurry'
        else:
            return 'very_blurry'

    def batch_detect(self, image_paths: list[Path]) -> dict[Path, dict]:
        """
        Batch process multiple images

        Args:
            image_paths: List of image paths

        Returns:
            Dictionary mapping paths to results:
            {
                path: {
                    'blur_score': float,
                    'is_blurry': bool,
                    'blur_level': str
                }
            }
        """
        results = {}

        for path in image_paths:
            try:
                score = self.compute_blur_score(path)
                results[path] = {
                    'blur_score': score,
                    'is_blurry': score < self.threshold,
                    'blur_level': self.classify_blur_level(path)
                }
            except Exception as e:
                results[path] = {
                    'error': str(e),
                    'blur_score': None,
                    'is_blurry': None,
                    'blur_level': None
                }

        return results


def main():
    """CLI for testing blur detection"""
    import argparse

    parser = argparse.ArgumentParser(description="Blur Detection using Laplacian Variance")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--threshold", type=float, default=100.0, help="Blur threshold (default: 100.0)")

    args = parser.parse_args()

    detector = BlurDetector(threshold=args.threshold)

    score = detector.compute_blur_score(args.image)
    is_blurry = detector.is_blurry(args.image)
    level = detector.classify_blur_level(args.image)

    print(f"Image: {args.image}")
    print(f"Blur Score: {score:.2f}")
    print(f"Is Blurry: {is_blurry}")
    print(f"Blur Level: {level}")
    print(f"\nInterpretation:")
    print(f"  < 100: Very blurry")
    print(f"  100-200: Slightly blurry")
    print(f"  > 200: Sharp")


if __name__ == "__main__":
    main()
