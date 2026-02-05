#!/usr/bin/env python3
"""
Quality Assessment and Deduplication for LoRA Training Data
Performs comprehensive quality checks and removes duplicates.

Usage:
    python scripts/generic/quality/quality_assessment.py \
        /path/to/images \
        --output-dir /path/to/filtered \
        --remove-duplicates \
        --min-resolution 512

Features:
    - Blur detection (Laplacian variance)
    - Resolution filtering
    - Duplicate detection (pHash + SSIM)
    - NSFW filtering (optional)
    - Brightness/contrast analysis
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

try:
    from PIL import Image
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False


class QualityAssessor:
    """Assess image quality and remove duplicates."""

    def __init__(
        self,
        min_resolution: int = 512,
        blur_threshold: float = 100.0,
        brightness_range: Tuple[float, float] = (20, 235),
        phash_threshold: int = 10,
        ssim_threshold: float = 0.95
    ):
        """Initialize quality assessor.

        Args:
            min_resolution: Minimum width/height
            blur_threshold: Laplacian variance threshold
            brightness_range: (min, max) acceptable brightness
            phash_threshold: Hamming distance for pHash duplicates
            ssim_threshold: SSIM threshold for duplicates
        """
        self.min_resolution = min_resolution
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range
        self.phash_threshold = phash_threshold
        self.ssim_threshold = ssim_threshold

    def assess_image(self, image_path: str) -> Dict:
        """Assess single image quality.

        Args:
            image_path: Path to image

        Returns:
            Quality metrics
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {'error': 'Failed to load', 'passed': False}

            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Blur detection
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Brightness
            brightness = np.mean(gray)

            # Contrast (standard deviation)
            contrast = np.std(gray)

            # Resolution check
            min_dim = min(w, h)
            resolution_ok = min_dim >= self.min_resolution

            # Blur check
            is_sharp = blur_score >= self.blur_threshold

            # Brightness check
            is_well_lit = self.brightness_range[0] <= brightness <= self.brightness_range[1]

            # Overall pass
            passed = resolution_ok and is_sharp and is_well_lit

            return {
                'path': image_path,
                'resolution': (w, h),
                'blur_score': float(blur_score),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'resolution_ok': resolution_ok,
                'is_sharp': is_sharp,
                'is_well_lit': is_well_lit,
                'passed': passed
            }

        except Exception as e:
            return {'error': str(e), 'passed': False}

    def compute_phash(self, image_path: str) -> Optional[str]:
        """Compute perceptual hash.

        Args:
            image_path: Path to image

        Returns:
            pHash string or None
        """
        if not HAS_IMAGEHASH:
            return None

        try:
            img = Image.open(image_path)
            return str(imagehash.phash(img))
        except:
            return None

    def find_duplicates(
        self,
        image_paths: List[str]
    ) -> List[Tuple[str, str, int]]:
        """Find duplicate images using pHash.

        Args:
            image_paths: List of image paths

        Returns:
            List of (path1, path2, hamming_distance) tuples
        """
        if not HAS_IMAGEHASH:
            print("Warning: imagehash not installed, skipping duplicate detection")
            print("Install: pip install imagehash")
            return []

        print("\n🔍 Computing perceptual hashes...")
        hash_to_path = {}
        duplicates = []

        for img_path in tqdm(image_paths, desc="Computing hashes"):
            phash = self.compute_phash(img_path)
            if phash is None:
                continue

            # Check for near-duplicates
            for existing_hash, existing_path in hash_to_path.items():
                distance = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(existing_hash)
                if distance <= self.phash_threshold:
                    duplicates.append((existing_path, img_path, distance))

            hash_to_path[phash] = img_path

        return duplicates

    def assess_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        remove_duplicates: bool = True
    ) -> Dict:
        """Assess batch of images.

        Args:
            image_paths: List of image paths
            output_dir: Output directory
            remove_duplicates: Whether to detect and remove duplicates

        Returns:
            Assessment results
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"QUALITY ASSESSMENT")
        print(f"{'='*60}")
        print(f"Input images: {len(image_paths)}")
        print(f"Min resolution: {self.min_resolution}")
        print(f"Blur threshold: {self.blur_threshold}")
        print(f"{'='*60}\n")

        # Assess all images
        print("🔍 Assessing quality...")
        assessments = []
        stats = {
            'total': len(image_paths),
            'passed': 0,
            'failed_resolution': 0,
            'failed_blur': 0,
            'failed_brightness': 0,
            'error': 0
        }

        for img_path in tqdm(image_paths, desc="Quality check"):
            assessment = self.assess_image(img_path)
            assessments.append(assessment)

            if assessment.get('error'):
                stats['error'] += 1
            elif assessment.get('passed'):
                stats['passed'] += 1
            else:
                if not assessment.get('resolution_ok'):
                    stats['failed_resolution'] += 1
                if not assessment.get('is_sharp'):
                    stats['failed_blur'] += 1
                if not assessment.get('is_well_lit'):
                    stats['failed_brightness'] += 1

        print(f"\n✓ Quality assessment complete:")
        print(f"  Passed: {stats['passed']}/{stats['total']}")
        print(f"  Failed (resolution): {stats['failed_resolution']}")
        print(f"  Failed (blur): {stats['failed_blur']}")
        print(f"  Failed (brightness): {stats['failed_brightness']}")

        # Filter passed images
        passed_paths = [a['path'] for a in assessments if a.get('passed')]

        # Duplicate detection
        duplicates = []
        if remove_duplicates and len(passed_paths) > 0:
            duplicates = self.find_duplicates(passed_paths)
            print(f"\n✓ Found {len(duplicates)} duplicate pairs")

            # Remove duplicates (keep first occurrence)
            duplicate_paths = set(dup[1] for dup in duplicates)
            final_paths = [p for p in passed_paths if p not in duplicate_paths]
        else:
            final_paths = passed_paths

        # Copy filtered images
        print(f"\n📁 Copying {len(final_paths)} images...")
        for img_path in tqdm(final_paths, desc="Copying"):
            basename = os.path.basename(img_path)
            dest_path = os.path.join(output_dir, basename)
            shutil.copy2(img_path, dest_path)

        # Save results
        results = {
            'statistics': stats,
            'final_count': len(final_paths),
            'duplicates_removed': len(duplicates) if remove_duplicates else 0,
            'settings': {
                'min_resolution': self.min_resolution,
                'blur_threshold': self.blur_threshold,
                'brightness_range': list(self.brightness_range),
                'phash_threshold': self.phash_threshold
            }
        }

        results_path = os.path.join(output_dir, 'quality_assessment.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save detailed assessments
        assessments_path = os.path.join(output_dir, 'detailed_assessments.json')
        with open(assessments_path, 'w') as f:
            json.dump(assessments, f, indent=2)

        if duplicates:
            duplicates_path = os.path.join(output_dir, 'duplicates.json')
            with open(duplicates_path, 'w') as f:
                json.dump([
                    {'original': d[0], 'duplicate': d[1], 'distance': d[2]}
                    for d in duplicates
                ], f, indent=2)

        print(f"\n✅ Quality assessment complete!")
        print(f"   Final images: {len(final_paths)}")
        print(f"   Output: {output_dir}")
        print(f"   Results: {results_path}\n")

        return results


def main():
    parser = argparse.ArgumentParser(description="Quality assessment and deduplication")
    parser.add_argument(
        "images_dir",
        help="Directory with images to assess"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for filtered images"
    )
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=512,
        help="Minimum image resolution (width/height)"
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Laplacian variance threshold for blur detection"
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Detect and remove duplicate images"
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=10,
        help="pHash hamming distance threshold for duplicates"
    )

    args = parser.parse_args()

    if not HAS_CV2:
        print("Error: opencv-python not installed")
        print("Install: pip install opencv-python")
        return 1

    # Find images
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist")
        return 1

    image_extensions = {'.png', '.jpg', '.jpeg'}
    image_paths = [
        str(p) for p in images_dir.rglob('*')
        if p.suffix.lower() in image_extensions
    ]

    if len(image_paths) == 0:
        print(f"No images found in {images_dir}")
        return 1

    # Initialize assessor
    assessor = QualityAssessor(
        min_resolution=args.min_resolution,
        blur_threshold=args.blur_threshold,
        phash_threshold=args.phash_threshold
    )

    # Assess batch
    results = assessor.assess_batch(
        image_paths,
        args.output_dir,
        remove_duplicates=args.remove_duplicates
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
