#!/usr/bin/env python3
"""
Automated Quality Filtering for LoRA Training Data

Evaluates and selects high-quality, diverse character instances for training.

Quality Metrics:
- Sharpness (Laplacian variance)
- Completeness (alpha channel coverage)
- Face detection quality (optional)

Diversity Metrics:
- Pose/angle diversity (CLIP embeddings or pose estimation)
- Visual diversity to avoid near-duplicates

Selection Strategy:
- Per-character cluster filtering
- Balance quality vs diversity
- Stratified sampling by pose/angle

Usage:
    python quality_filter.py \
      --input-dir /path/to/clustered \
      --output-dir /path/to/filtered \
      --target-per-cluster 200 \
      --min-sharpness 100 \
      --min-completeness 0.85 \
      --diversity-method clip \
      --device cuda
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import json
import shutil
from dataclasses import dataclass, asdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import sys


@dataclass
class ImageQualityMetrics:
    """Quality metrics for a single image"""
    path: str
    sharpness: float
    completeness: float
    face_detected: bool
    face_confidence: float
    alpha_mean: float
    alpha_std: float
    overall_score: float


def compute_sharpness(image: np.ndarray) -> float:
    """
    Compute image sharpness using Laplacian variance

    Higher values = sharper image
    Typical threshold: 100-200 for acceptable sharpness
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def compute_completeness(image: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute alpha channel completeness

    Returns:
        completeness: Ratio of opaque pixels (alpha > 240)
        alpha_mean: Mean alpha value
        alpha_std: Std of alpha value
    """
    if image.shape[2] != 4:
        return 1.0, 255.0, 0.0

    alpha = image[:, :, 3]
    opaque_ratio = (alpha > 240).sum() / alpha.size
    alpha_mean = alpha.mean()
    alpha_std = alpha.std()

    return opaque_ratio, alpha_mean, alpha_std


def detect_face(image: np.ndarray, detector=None) -> Tuple[bool, float]:
    """
    Detect face in image (optional quality check)

    Returns:
        face_detected: Whether face was detected
        confidence: Detection confidence (0-1)
    """
    if detector is None:
        return False, 0.0

    try:
        from insightface.app import FaceAnalysis

        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        faces = detector.get(rgb)

        if len(faces) > 0:
            # Return highest confidence face
            confidence = max([face.det_score for face in faces])
            return True, float(confidence)
        else:
            return False, 0.0

    except Exception as e:
        return False, 0.0


def load_clip_model(device: str = "cuda"):
    """Load CLIP model for diversity analysis"""
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess
    except ImportError:
        print("⚠️  CLIP not installed, diversity analysis disabled")
        return None, None


def compute_clip_embedding(image: np.ndarray, model, preprocess, device: str = "cuda") -> np.ndarray:
    """Compute CLIP embedding for image"""
    from PIL import Image

    # Convert to PIL
    rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Preprocess and encode
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
        embedding = embedding.cpu().numpy().flatten()
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

    return embedding


def evaluate_image_quality(
    image_path: Path,
    face_detector=None,
    min_sharpness: float = 100.0,
    min_completeness: float = 0.85
) -> Optional[ImageQualityMetrics]:
    """
    Evaluate single image quality

    Returns None if image fails basic checks
    """
    # Read image
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    # Check if RGBA
    if image.shape[2] != 4:
        return None

    # Compute metrics
    sharpness = compute_sharpness(image)
    completeness, alpha_mean, alpha_std = compute_completeness(image)
    face_detected, face_confidence = detect_face(image, face_detector)

    # Overall score (weighted combination)
    # Sharpness normalized to 0-1 range (assume max ~500)
    norm_sharpness = min(sharpness / 500.0, 1.0)
    overall_score = (
        0.4 * norm_sharpness +
        0.4 * completeness +
        0.2 * face_confidence
    )

    metrics = ImageQualityMetrics(
        path=str(image_path),
        sharpness=sharpness,
        completeness=completeness,
        face_detected=face_detected,
        face_confidence=face_confidence,
        alpha_mean=alpha_mean,
        alpha_std=alpha_std,
        overall_score=overall_score
    )

    # Filter out low quality
    if sharpness < min_sharpness or completeness < min_completeness:
        return None

    return metrics


def cluster_by_diversity(
    embeddings: np.ndarray,
    n_clusters: int = 5
) -> np.ndarray:
    """
    Cluster images by visual diversity using K-Means

    Returns cluster labels
    """
    if len(embeddings) <= n_clusters:
        return np.arange(len(embeddings))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels


def stratified_sampling(
    metrics_list: List[ImageQualityMetrics],
    embeddings: np.ndarray,
    target_count: int,
    diversity_clusters: int = 5
) -> List[ImageQualityMetrics]:
    """
    Select images using stratified sampling for diversity

    Strategy:
    1. Cluster images by visual diversity (CLIP embeddings)
    2. Sample proportionally from each cluster
    3. Within each cluster, select by quality score
    """
    if len(metrics_list) <= target_count:
        return metrics_list

    # Cluster by diversity
    labels = cluster_by_diversity(embeddings, n_clusters=diversity_clusters)

    # Count per cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    selected = []

    for label in unique_labels:
        # Get images in this cluster
        cluster_indices = np.where(labels == label)[0]
        cluster_metrics = [metrics_list[i] for i in cluster_indices]

        # How many to select from this cluster
        cluster_target = int(target_count * (len(cluster_indices) / len(metrics_list)))
        cluster_target = max(1, cluster_target)  # At least 1 from each cluster

        # Sort by quality score
        cluster_metrics.sort(key=lambda x: x.overall_score, reverse=True)

        # Select top-k
        selected.extend(cluster_metrics[:cluster_target])

    # If we haven't reached target, add more from top quality
    if len(selected) < target_count:
        remaining = target_count - len(selected)
        # Sort all by quality
        all_metrics = sorted(metrics_list, key=lambda x: x.overall_score, reverse=True)
        # Add remaining
        for m in all_metrics:
            if m not in selected:
                selected.append(m)
                if len(selected) >= target_count:
                    break

    # If we exceeded, trim
    selected = selected[:target_count]

    return selected


def process_character_cluster(
    cluster_dir: Path,
    output_dir: Path,
    target_count: int,
    min_sharpness: float,
    min_completeness: float,
    diversity_method: str = "clip",
    device: str = "cuda",
    face_detector=None,
    clip_model=None,
    clip_preprocess=None
) -> Dict:
    """
    Process a single character cluster

    Returns statistics dict
    """
    cluster_name = cluster_dir.name
    output_cluster_dir = output_dir / cluster_name
    output_cluster_dir.mkdir(parents=True, exist_ok=True)

    # Get all instances
    instances = list(cluster_dir.glob("*.png"))

    if not instances:
        return {
            "cluster": cluster_name,
            "total": 0,
            "passed_quality": 0,
            "selected": 0,
            "rejected_sharpness": 0,
            "rejected_completeness": 0
        }

    print(f"\n📂 Processing {cluster_name} ({len(instances)} instances)")

    # Evaluate quality
    metrics_list = []
    embeddings_list = []

    stats = {
        "cluster": cluster_name,
        "total": len(instances),
        "passed_quality": 0,
        "selected": 0,
        "rejected_sharpness": 0,
        "rejected_completeness": 0
    }

    for img_path in tqdm(instances, desc=f"  Quality check", leave=False):
        metrics = evaluate_image_quality(
            img_path,
            face_detector=face_detector,
            min_sharpness=min_sharpness,
            min_completeness=min_completeness
        )

        if metrics is None:
            # Count rejection reasons
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                sharpness = compute_sharpness(image)
                completeness, _, _ = compute_completeness(image)
                if sharpness < min_sharpness:
                    stats["rejected_sharpness"] += 1
                if completeness < min_completeness:
                    stats["rejected_completeness"] += 1
            continue

        metrics_list.append(metrics)

        # Compute CLIP embedding if needed
        if diversity_method == "clip" and clip_model is not None:
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            embedding = compute_clip_embedding(image, clip_model, clip_preprocess, device)
            embeddings_list.append(embedding)

    stats["passed_quality"] = len(metrics_list)

    if not metrics_list:
        print(f"  ⚠️  No images passed quality check")
        return stats

    # Select diverse subset
    if diversity_method == "clip" and embeddings_list:
        embeddings = np.array(embeddings_list)
        selected_metrics = stratified_sampling(
            metrics_list,
            embeddings,
            target_count=min(target_count, len(metrics_list))
        )
    else:
        # Just sort by quality
        metrics_list.sort(key=lambda x: x.overall_score, reverse=True)
        selected_metrics = metrics_list[:target_count]

    stats["selected"] = len(selected_metrics)

    # Copy selected images
    for metrics in tqdm(selected_metrics, desc=f"  Copying", leave=False):
        src_path = Path(metrics.path)
        dst_path = output_cluster_dir / src_path.name
        shutil.copy2(src_path, dst_path)

    # Save metrics report
    metrics_report_path = output_cluster_dir / "quality_metrics.json"
    with open(metrics_report_path, 'w') as f:
        json.dump([asdict(m) for m in selected_metrics], f, indent=2)

    print(f"  ✓ Selected {len(selected_metrics)} / {len(metrics_list)} images")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Automated Quality Filtering")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with clustered character instances"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for filtered instances"
    )
    parser.add_argument(
        "--target-per-cluster",
        type=int,
        default=200,
        help="Target number of images per cluster (default: 200)"
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=100.0,
        help="Minimum sharpness threshold (Laplacian variance, default: 100)"
    )
    parser.add_argument(
        "--min-completeness",
        type=float,
        default=0.85,
        help="Minimum alpha completeness ratio (default: 0.85)"
    )
    parser.add_argument(
        "--diversity-method",
        type=str,
        choices=["clip", "none"],
        default="clip",
        help="Diversity analysis method (default: clip)"
    )
    parser.add_argument(
        "--diversity-clusters",
        type=int,
        default=5,
        help="Number of diversity clusters for stratified sampling (default: 5)"
    )
    parser.add_argument(
        "--use-face-detection",
        action="store_true",
        help="Use face detection for quality scoring"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for CLIP/face detection"
    )

    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"AUTOMATED QUALITY FILTERING")
    print(f"{'='*70}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target per cluster: {args.target_per_cluster}")
    print(f"Min sharpness: {args.min_sharpness}")
    print(f"Min completeness: {args.min_completeness}")
    print(f"Diversity method: {args.diversity_method}")
    print(f"Device: {args.device.upper()}")
    print(f"{'='*70}\n")

    # Load models
    face_detector = None
    if args.use_face_detection:
        try:
            from insightface.app import FaceAnalysis
            print("Loading face detection model...")
            face_detector = FaceAnalysis(providers=['CUDAExecutionProvider' if args.device == 'cuda' else 'CPUExecutionProvider'])
            face_detector.prepare(ctx_id=0 if args.device == 'cuda' else -1)
            print("✓ Face detector loaded\n")
        except Exception as e:
            print(f"⚠️  Failed to load face detector: {e}")
            print("Continuing without face detection\n")

    clip_model, clip_preprocess = None, None
    if args.diversity_method == "clip":
        print("Loading CLIP model...")
        clip_model, clip_preprocess = load_clip_model(device=args.device)
        if clip_model is not None:
            print("✓ CLIP model loaded\n")
        else:
            print("⚠️  CLIP not available, using quality-only selection\n")

    # Get all cluster directories (skip hidden dirs, noise, etc.)
    SKIP_DIRS = {'noise', '__pycache__', '.git', '.DS_Store'}
    cluster_dirs = [d for d in input_dir.iterdir()
                   if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith('.')]

    if not cluster_dirs:
        print("❌ No cluster directories found!")
        return

    # Sort by name for consistent processing
    cluster_dirs = sorted(cluster_dirs, key=lambda x: x.name)

    print(f"📂 Found {len(cluster_dirs)} character clusters:")
    for d in cluster_dirs:
        count = len(list(d.glob("*.png")))
        print(f"   - {d.name}: {count} instances")
    print()

    # Process each cluster
    all_stats = []

    for cluster_dir in cluster_dirs:
        stats = process_character_cluster(
            cluster_dir,
            output_dir,
            target_count=args.target_per_cluster,
            min_sharpness=args.min_sharpness,
            min_completeness=args.min_completeness,
            diversity_method=args.diversity_method,
            device=args.device,
            face_detector=face_detector,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess
        )
        all_stats.append(stats)

    # Save overall report
    report_path = output_dir / "quality_filter_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            "parameters": {
                "target_per_cluster": args.target_per_cluster,
                "min_sharpness": args.min_sharpness,
                "min_completeness": args.min_completeness,
                "diversity_method": args.diversity_method
            },
            "clusters": all_stats
        }, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"QUALITY FILTERING COMPLETE")
    print(f"{'='*70}")

    total_input = sum(s["total"] for s in all_stats)
    total_passed = sum(s["passed_quality"] for s in all_stats)
    total_selected = sum(s["selected"] for s in all_stats)
    total_rejected_sharpness = sum(s["rejected_sharpness"] for s in all_stats)
    total_rejected_completeness = sum(s["rejected_completeness"] for s in all_stats)

    print(f"Total input instances: {total_input}")
    print(f"Passed quality check: {total_passed} ({100*total_passed/total_input:.1f}%)")
    print(f"Final selection: {total_selected} ({100*total_selected/total_input:.1f}%)")
    print(f"\nRejection reasons:")
    print(f"  Low sharpness: {total_rejected_sharpness}")
    print(f"  Low completeness: {total_rejected_completeness}")
    print(f"{'='*70}")
    print(f"\n📁 Filtered instances saved to: {output_dir}")
    print(f"📊 Report saved to: {report_path}\n")


if __name__ == "__main__":
    main()
