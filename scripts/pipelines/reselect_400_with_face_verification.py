#!/usr/bin/env python3
"""
Re-select 400 images from augmented dataset with strict face verification
=========================================================================

從已生成的 55,400 張增強資料集中，使用嚴格的臉部驗證重新選擇 400 張 Luca 圖片

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
import shutil

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


def verify_faces_arcface(
    image_paths: List[Path],
    reference_dir: Path,
    threshold: float = 0.55,
    logger: logging.Logger = None
) -> List[Tuple[Path, float]]:
    """
    Verify faces using ArcFace with strict threshold
    
    Returns:
        List of (image_path, similarity_score) for verified images
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        if logger:
            logger.error("InsightFace not installed!")
        return []
    
    if logger:
        logger.info("Initializing ArcFace model...")
    
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Load reference embeddings
    if logger:
        logger.info(f"Loading reference faces from {reference_dir}...")
    
    reference_embeddings = []
    ref_images = list(reference_dir.glob("*.png")) + list(reference_dir.glob("*.jpg"))
    
    for ref_path in tqdm(ref_images[:50], desc="Reference faces"):
        try:
            img = np.array(Image.open(ref_path).convert('RGB'))
            faces = app.get(img)
            if faces:
                reference_embeddings.append(faces[0].embedding)
        except:
            continue
    
    if not reference_embeddings:
        if logger:
            logger.error("No reference embeddings extracted!")
        return []
    
    reference_embeddings = np.array(reference_embeddings)
    if logger:
        logger.info(f"Loaded {len(reference_embeddings)} reference embeddings")
    
    # Verify candidate images
    verified = []
    
    for img_path in tqdm(image_paths, desc="Verifying faces"):
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            faces = app.get(img)
            
            if not faces:
                continue
            
            # Get embedding
            embedding = faces[0].embedding
            
            # Compute similarities
            similarities = np.dot(reference_embeddings, embedding)
            max_sim = float(np.max(similarities))
            
            if max_sim >= threshold:
                verified.append((img_path, max_sim))
                
        except Exception as e:
            continue
    
    if logger:
        logger.info(f"Verified {len(verified)}/{len(image_paths)} images (threshold={threshold})")
    
    return verified


def select_diverse_subset(
    verified_images: List[Tuple[Path, float]],
    target_count: int = 400,
    logger: logging.Logger = None
) -> List[Path]:
    """
    Select diverse subset using CLIP embeddings
    """
    try:
        import torch
        import clip
    except ImportError:
        if logger:
            logger.error("CLIP not installed!")
        return [img for img, _ in verified_images[:target_count]]
    
    if logger:
        logger.info("Loading CLIP model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    
    # Extract CLIP embeddings
    embeddings = []
    valid_paths = []
    
    for img_path, _ in tqdm(verified_images, desc="CLIP embeddings"):
        try:
            img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img)
            embeddings.append(emb.cpu().numpy())
            valid_paths.append(img_path)
        except:
            continue
    
    embeddings = np.vstack(embeddings)
    
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    if logger:
        logger.info(f"Extracted {len(embeddings)} CLIP embeddings")
    
    # K-means clustering for diversity
    from sklearn.cluster import KMeans
    
    n_clusters = min(8, target_count // 50)
    samples_per_cluster = target_count // n_clusters
    
    if logger:
        logger.info(f"Clustering into {n_clusters} groups, {samples_per_cluster} per cluster")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Select diverse samples
    selected = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Get embeddings for this cluster
        cluster_embs = embeddings[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_id]
        
        # Compute distances to centroid
        distances = np.linalg.norm(cluster_embs - centroid, axis=1)
        
        # Select closest to centroid (most representative)
        sorted_indices = np.argsort(distances)[:samples_per_cluster]
        
        for idx in sorted_indices:
            global_idx = cluster_indices[idx]
            selected.append(valid_paths[global_idx])
    
    if logger:
        logger.info(f"Selected {len(selected)} diverse images")
    
    return selected[:target_count]


def main():
    parser = argparse.ArgumentParser(description="Re-select 400 images with face verification")
    parser.add_argument('--augmented-dir', type=Path, required=True,
                        help='Directory with 55k augmented images')
    parser.add_argument('--reference-dir', type=Path, required=True,
                        help='Reference Luca faces directory')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for selected 400 images')
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='Face similarity threshold (default: 0.55)')
    parser.add_argument('--target-count', type=int, default=400,
                        help='Target number of images (default: 400)')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("Re-selecting 400 images with strict face verification")
    logger.info("=" * 80)
    
    # Find all images
    logger.info(f"Scanning {args.augmented_dir}...")
    image_paths = list(args.augmented_dir.glob("*.png")) + list(args.augmented_dir.glob("*.jpg"))
    logger.info(f"Found {len(image_paths)} images")
    
    # Step 1: Face verification
    verified = verify_faces_arcface(
        image_paths,
        args.reference_dir,
        threshold=args.threshold,
        logger=logger
    )
    
    if len(verified) < args.target_count:
        logger.warning(f"Only {len(verified)} verified images, less than target {args.target_count}")
        logger.warning(f"Consider lowering threshold (current: {args.threshold})")
    
    # Step 2: Diversity selection
    selected = select_diverse_subset(verified, args.target_count, logger)
    
    # Step 3: Copy to output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Copying {len(selected)} images to {args.output_dir}...")
    for img_path in tqdm(selected, desc="Copying"):
        shutil.copy2(img_path, args.output_dir / img_path.name)
    
    logger.info("✓ Done!")
    logger.info(f"Selected images: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
