#!/usr/bin/env python3
"""
CLIP Multi-Reference Verification
==================================

Uses multiple approved reference images to verify character identity via CLIP embeddings.

Author: Claude Code
Date: 2025-11-14
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(log_file: Path = None) -> logging.Logger:
    """Setup logging"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def extract_clip_embeddings(image_paths: List[Path], model, preprocess, device, logger):
    """Extract CLIP embeddings for multiple images"""
    import torch

    embeddings = []
    valid_paths = []

    for img_path in tqdm(image_paths, desc="Extracting embeddings"):
        try:
            image = Image.open(img_path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)

            embeddings.append(features.cpu().numpy()[0])
            valid_paths.append(img_path)
        except Exception as e:
            logger.warning(f"Failed to process {img_path.name}: {e}")

    return np.array(embeddings), valid_paths


def main():
    parser = argparse.ArgumentParser(description="CLIP multi-reference character verification")
    parser.add_argument('--approved-list', type=Path, required=True,
                        help='Text file with approved image filenames')
    parser.add_argument('--reference-dir', type=Path, required=True,
                        help='Directory containing reference images')
    parser.add_argument('--test-dir', type=Path, required=True,
                        help='Directory with images to verify')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for verified images')
    parser.add_argument('--threshold', type=float, default=0.85,
                        help='Similarity threshold (default: 0.85)')
    parser.add_argument('--model', type=str, default='ViT-L/14',
                        choices=['ViT-B/32', 'ViT-L/14'],
                        help='CLIP model (default: ViT-L/14)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--log-file', type=Path, default=None,
                        help='Log file path')

    args = parser.parse_args()
    logger = setup_logging(args.log_file)

    logger.info("=" * 80)
    logger.info("CLIP Multi-Reference Character Verification")
    logger.info("=" * 80)

    # Load CLIP
    import torch
    import clip

    logger.info(f"Loading CLIP model: {args.model}")
    model, preprocess = clip.load(args.model, device=args.device)
    model.eval()

    # Load approved reference images
    with open(args.approved_list, 'r') as f:
        approved_names = [line.strip() for line in f if line.strip()]

    reference_paths = [args.reference_dir / name for name in approved_names]
    reference_paths = [p for p in reference_paths if p.exists()]

    logger.info(f"Found {len(reference_paths)} approved reference images")

    # Extract reference embeddings
    logger.info("Extracting reference embeddings...")
    ref_embeddings, valid_ref_paths = extract_clip_embeddings(
        reference_paths, model, preprocess, args.device, logger
    )

    if len(ref_embeddings) == 0:
        logger.error("No reference embeddings extracted!")
        return 1

    logger.info(f"✓ Extracted {len(ref_embeddings)} reference embeddings")

    # Find test images
    test_images = list(args.test_dir.glob("*.png")) + list(args.test_dir.glob("*.jpg"))
    logger.info(f"Found {len(test_images)} test images")

    # Extract test embeddings
    logger.info("Extracting test image embeddings...")
    test_embeddings, valid_test_paths = extract_clip_embeddings(
        test_images, model, preprocess, args.device, logger
    )

    logger.info(f"✓ Extracted {len(test_embeddings)} test embeddings")

    # Compute similarities
    logger.info("Computing similarities with references...")
    verified = []
    rejected = []

    for i, test_emb in enumerate(tqdm(test_embeddings, desc="Verifying")):
        test_path = valid_test_paths[i]

        # Compute similarity with all references
        similarities = np.dot(ref_embeddings, test_emb)
        max_similarity = similarities.max()
        mean_similarity = similarities.mean()

        # Combined score: 70% max, 30% mean
        combined_score = 0.7 * max_similarity + 0.3 * mean_similarity

        if combined_score >= args.threshold:
            verified.append((test_path, float(combined_score)))
        else:
            rejected.append((test_path, float(combined_score)))

    # Report results
    logger.info("")
    logger.info("=" * 80)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total tested: {len(test_images)}")
    logger.info(f"Verified: {len(verified)} ({len(verified)/len(test_images)*100:.1f}%)")
    logger.info(f"Rejected: {len(rejected)} ({len(rejected)/len(test_images)*100:.1f}%)")

    if verified:
        scores = [s for _, s in verified]
        logger.info(f"Verified scores: min={min(scores):.3f}, max={max(scores):.3f}, avg={np.mean(scores):.3f}")

    # Copy verified images
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying {len(verified)} verified images...")
    for img_path, score in tqdm(verified, desc="Copying"):
        shutil.copy2(img_path, args.output_dir / img_path.name)

    # Save results
    results = {
        "reference_count": len(ref_embeddings),
        "threshold": args.threshold,
        "model": args.model,
        "total_tested": len(test_images),
        "verified_count": len(verified),
        "rejected_count": len(rejected),
        "verified_images": [{"path": str(p), "score": float(s)} for p, s in verified]
    }

    with open(args.output_dir / "clip_verification_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Results saved to {args.output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
