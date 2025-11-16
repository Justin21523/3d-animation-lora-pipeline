#!/usr/bin/env python3
"""
Test Single Reference Face Verification
========================================

Quick test using a single, highly representative reference image
to verify character identity via ArcFace.

Author: Claude Code
Date: 2025-11-14
"""

import argparse
from pathlib import Path
import sys
import logging
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image
import random

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging() -> logging.Logger:
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_arcface_with_single_reference(
    reference_image: Path,
    test_images: List[Path],
    threshold: float = 0.70,
    logger: logging.Logger = None
) -> Tuple[List[Tuple[Path, float]], List[Tuple[Path, float]]]:
    """
    Test ArcFace verification using single reference image

    Returns:
        (verified_list, rejected_list) with (path, similarity_score) tuples
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        if logger:
            logger.error("InsightFace not installed!")
        return [], []

    if logger:
        logger.info("Initializing ArcFace model (buffalo_l)...")

    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Extract reference embedding
    if logger:
        logger.info(f"Extracting reference embedding from: {reference_image.name}")

    ref_img = np.array(Image.open(reference_image).convert('RGB'))
    ref_faces = app.get(ref_img)

    if not ref_faces:
        if logger:
            logger.error("No face detected in reference image!")
        return [], []

    reference_embedding = ref_faces[0].embedding
    if logger:
        logger.info(f"✓ Reference embedding extracted (shape: {reference_embedding.shape})")

    # Test candidate images
    verified = []
    rejected = []

    for img_path in tqdm(test_images, desc="Testing candidates"):
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            faces = app.get(img)

            if not faces:
                rejected.append((img_path, 0.0))
                continue

            # Get embedding
            embedding = faces[0].embedding

            # Compute similarity
            similarity = float(np.dot(reference_embedding, embedding))

            if similarity >= threshold:
                verified.append((img_path, similarity))
            else:
                rejected.append((img_path, similarity))

        except Exception as e:
            if logger:
                logger.warning(f"Failed to process {img_path.name}: {e}")
            rejected.append((img_path, 0.0))

    return verified, rejected


def main():
    parser = argparse.ArgumentParser(description="Test single reference face verification")
    parser.add_argument('--reference', type=Path, required=True,
                        help='Single reference image (e.g., luca_face.png)')
    parser.add_argument('--test-dir', type=Path, required=True,
                        help='Directory with images to test')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of random samples to test (default: 100)')
    parser.add_argument('--threshold', type=float, default=0.70,
                        help='Similarity threshold (default: 0.70)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Optional: save verified images to this directory')

    args = parser.parse_args()
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("Single Reference Face Verification Test")
    logger.info("=" * 80)
    logger.info(f"Reference: {args.reference}")
    logger.info(f"Test directory: {args.test_dir}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Sample size: {args.num_samples}")

    # Verify reference exists
    if not args.reference.exists():
        logger.error(f"Reference image not found: {args.reference}")
        return 1

    # Find test images
    test_images = list(args.test_dir.glob("*.png")) + list(args.test_dir.glob("*.jpg"))
    logger.info(f"Found {len(test_images)} candidate images")

    # Random sample
    if len(test_images) > args.num_samples:
        test_images = random.sample(test_images, args.num_samples)
        logger.info(f"Randomly sampled {args.num_samples} images for testing")

    # Run test
    verified, rejected = test_arcface_with_single_reference(
        args.reference,
        test_images,
        args.threshold,
        logger
    )

    # Report results
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total tested: {len(test_images)}")
    logger.info(f"Verified: {len(verified)} ({len(verified)/len(test_images)*100:.1f}%)")
    logger.info(f"Rejected: {len(rejected)} ({len(rejected)/len(test_images)*100:.1f}%)")

    if verified:
        similarities = [s for _, s in verified]
        logger.info(f"Verified similarities: min={min(similarities):.3f}, max={max(similarities):.3f}, avg={np.mean(similarities):.3f}")

    if rejected and any(s > 0 for _, s in rejected):
        rejected_with_faces = [(p, s) for p, s in rejected if s > 0]
        if rejected_with_faces:
            similarities = [s for _, s in rejected_with_faces]
            logger.info(f"Rejected (with faces) similarities: min={min(similarities):.3f}, max={max(similarities):.3f}, avg={np.mean(similarities):.3f}")

    # Show top 10 verified
    if verified:
        logger.info("")
        logger.info("Top 10 verified images:")
        verified_sorted = sorted(verified, key=lambda x: x[1], reverse=True)
        for img_path, sim in verified_sorted[:10]:
            logger.info(f"  {img_path.name}: {sim:.3f}")

    # Show top 10 rejected (highest similarity among rejected)
    rejected_with_faces = [(p, s) for p, s in rejected if s > 0]
    if rejected_with_faces:
        logger.info("")
        logger.info("Top 10 rejected images (closest to threshold):")
        rejected_sorted = sorted(rejected_with_faces, key=lambda x: x[1], reverse=True)
        for img_path, sim in rejected_sorted[:10]:
            logger.info(f"  {img_path.name}: {sim:.3f}")

    # Optional: save verified images
    if args.output_dir and verified:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for img_path, sim in verified:
            shutil.copy2(img_path, args.output_dir / img_path.name)
        logger.info(f"✓ Saved {len(verified)} verified images to {args.output_dir}")

    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
