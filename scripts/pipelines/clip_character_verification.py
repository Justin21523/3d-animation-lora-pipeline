#!/usr/bin/env python3
"""
CLIP-Based Character Verification for 3D Animation
===================================================

Uses CLIP (Contrastive Language-Image Pre-training) to verify character identity
by comparing image embeddings with reference images AND text descriptions.

CLIP is superior to ArcFace for 3D animated characters because:
1. Understands semantic features (clothing, hair, body type)
2. Trained on diverse visual data including animation
3. Can leverage text descriptions for verification

Author: Claude Code
Date: 2025-11-14
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from PIL import Image
import random
import shutil
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def load_clip_model(model_name: str = "ViT-L/14", device: str = "cuda"):
    """
    Load CLIP model for character verification

    Args:
        model_name: CLIP model variant (ViT-B/32, ViT-L/14, etc.)
        device: Device to run on (cuda/cpu)

    Returns:
        model, preprocess function
    """
    import torch
    import clip

    logger = logging.getLogger(__name__)
    logger.info(f"Loading CLIP model: {model_name}")

    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    logger.info(f"✓ CLIP model loaded on {device}")
    return model, preprocess, device


def extract_clip_embedding(image_path: Path, model, preprocess, device) -> np.ndarray:
    """
    Extract CLIP embedding from image

    Returns:
        Normalized embedding vector
    """
    import torch

    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        # Normalize to unit vector for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy()[0]


def verify_with_clip(
    reference_image: Path,
    test_images: List[Path],
    character_description: Optional[str] = None,
    threshold: float = 0.85,
    model_name: str = "ViT-L/14",
    device: str = "cuda",
    logger: logging.Logger = None
) -> Tuple[List[Tuple[Path, float]], List[Tuple[Path, float]]]:
    """
    Verify character identity using CLIP

    Args:
        reference_image: Single reference image
        test_images: List of images to verify
        character_description: Optional text description (e.g., "a young boy with curly brown hair")
        threshold: Similarity threshold (0.0-1.0)
        model_name: CLIP model variant
        device: Device to use
        logger: Logger instance

    Returns:
        (verified_list, rejected_list) with (path, similarity_score) tuples
    """
    import torch
    import clip

    logger = logger or logging.getLogger(__name__)

    # Load CLIP
    model, preprocess, device = load_clip_model(model_name, device)

    # Extract reference embedding
    logger.info(f"Extracting reference embedding from: {reference_image.name}")
    reference_embedding = extract_clip_embedding(reference_image, model, preprocess, device)
    logger.info(f"✓ Reference embedding extracted (shape: {reference_embedding.shape})")

    # Optional: Get text embedding for character description
    text_embedding = None
    if character_description:
        logger.info(f"Encoding character description: '{character_description}'")
        text_tokens = clip.tokenize([character_description]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embedding = text_features.cpu().numpy()[0]
        logger.info(f"✓ Text embedding extracted")

    # Verify candidate images
    verified = []
    rejected = []

    for img_path in tqdm(test_images, desc="CLIP Verification"):
        try:
            # Extract image embedding
            img_embedding = extract_clip_embedding(img_path, model, preprocess, device)

            # Compute similarity with reference image
            image_similarity = float(np.dot(reference_embedding, img_embedding))

            # Optionally combine with text similarity
            if text_embedding is not None:
                text_similarity = float(np.dot(text_embedding, img_embedding))
                # Weighted combination: 70% image, 30% text
                combined_similarity = 0.7 * image_similarity + 0.3 * text_similarity
            else:
                combined_similarity = image_similarity

            # Verify
            if combined_similarity >= threshold:
                verified.append((img_path, combined_similarity))
            else:
                rejected.append((img_path, combined_similarity))

        except Exception as e:
            logger.warning(f"Failed to process {img_path.name}: {e}")
            rejected.append((img_path, 0.0))

    return verified, rejected


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-based character verification for 3D animation datasets"
    )
    parser.add_argument(
        '--reference', type=Path, required=True,
        help='Single reference image'
    )
    parser.add_argument(
        '--test-dir', type=Path, required=True,
        help='Directory with images to test'
    )
    parser.add_argument(
        '--character-description', type=str, default=None,
        help='Text description of character (e.g., "a young boy with curly brown hair in Pixar style")'
    )
    parser.add_argument(
        '--num-samples', type=int, default=None,
        help='Number of random samples to test (default: all)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.85,
        help='Similarity threshold (default: 0.85)'
    )
    parser.add_argument(
        '--model', type=str, default='ViT-L/14',
        choices=['ViT-B/32', 'ViT-L/14', 'RN50', 'RN101'],
        help='CLIP model variant (default: ViT-L/14)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=None,
        help='Optional: save verified images to this directory'
    )
    parser.add_argument(
        '--log-file', type=Path, default=None,
        help='Log file path'
    )

    args = parser.parse_args()
    logger = setup_logging(args.log_file)

    logger.info("=" * 80)
    logger.info("CLIP-Based Character Verification")
    logger.info("=" * 80)
    logger.info(f"Reference: {args.reference}")
    logger.info(f"Test directory: {args.test_dir}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"CLIP model: {args.model}")
    if args.character_description:
        logger.info(f"Character description: '{args.character_description}'")

    # Verify reference exists
    if not args.reference.exists():
        logger.error(f"Reference image not found: {args.reference}")
        return 1

    # Find test images
    test_images = list(args.test_dir.glob("*.png")) + list(args.test_dir.glob("*.jpg"))
    logger.info(f"Found {len(test_images)} candidate images")

    # Random sample if requested
    if args.num_samples and len(test_images) > args.num_samples:
        test_images = random.sample(test_images, args.num_samples)
        logger.info(f"Randomly sampled {args.num_samples} images for testing")

    # Run verification
    verified, rejected = verify_with_clip(
        args.reference,
        test_images,
        args.character_description,
        args.threshold,
        args.model,
        args.device,
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

    if rejected:
        rejected_nonzero = [(p, s) for p, s in rejected if s > 0]
        if rejected_nonzero:
            similarities = [s for _, s in rejected_nonzero]
            logger.info(f"Rejected similarities: min={min(similarities):.3f}, max={max(similarities):.3f}, avg={np.mean(similarities):.3f}")

    # Show top 10 verified
    if verified:
        logger.info("")
        logger.info("Top 10 verified images:")
        verified_sorted = sorted(verified, key=lambda x: x[1], reverse=True)
        for img_path, sim in verified_sorted[:10]:
            logger.info(f"  {img_path.name}: {sim:.3f}")

    # Show top 10 rejected (closest to threshold)
    rejected_nonzero = [(p, s) for p, s in rejected if s > 0]
    if rejected_nonzero:
        logger.info("")
        logger.info("Top 10 rejected images (closest to threshold):")
        rejected_sorted = sorted(rejected_nonzero, key=lambda x: x[1], reverse=True)
        for img_path, sim in rejected_sorted[:10]:
            logger.info(f"  {img_path.name}: {sim:.3f}")

    # Optional: save verified images
    if args.output_dir and verified:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for img_path, sim in verified:
            shutil.copy2(img_path, args.output_dir / img_path.name)
        logger.info(f"✓ Saved {len(verified)} verified images to {args.output_dir}")

    # Save results JSON
    if args.output_dir:
        results = {
            "reference_image": str(args.reference),
            "character_description": args.character_description,
            "threshold": args.threshold,
            "model": args.model,
            "total_tested": len(test_images),
            "verified_count": len(verified),
            "rejected_count": len(rejected),
            "verified_images": [{"path": str(p), "similarity": float(s)} for p, s in verified],
            "rejected_images": [{"path": str(p), "similarity": float(s)} for p, s in rejected]
        }
        with open(args.output_dir / "clip_verification_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Results saved to {args.output_dir / 'clip_verification_results.json'}")

    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
