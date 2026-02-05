#!/usr/bin/env python3
"""
Style Frame Selector for Style LoRA Training Data
Selects representative frames with consistent visual style, removing transitions and outliers.

Usage:
    python scripts/generic/quality/style_frame_selector.py \
        /path/to/frames \
        --output-dir /path/to/style_frames \
        --target-count 400 \
        --device cpu

Features:
    - CLIP-based style consistency analysis
    - Transition detection and removal
    - Quality filtering (blur, brightness)
    - Diversity sampling across scenes
"""

import argparse
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
    import torch
    import open_clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


class StyleFrameSelector:
    """Select representative frames for style LoRA training."""

    def __init__(
        self,
        device: str = "cpu",
        target_count: int = 400,
        blur_threshold: float = 100.0,
        brightness_range: Tuple[float, float] = (20, 235)
    ):
        """Initialize style frame selector.

        Args:
            device: 'cpu' or 'cuda'
            target_count: Target number of frames to select
            blur_threshold: Laplacian variance threshold (higher = less blurry)
            brightness_range: (min, max) acceptable brightness
        """
        self.device = device
        self.target_count = target_count
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range

        if not HAS_CLIP:
            raise ImportError("Requires: pip install open_clip_torch pillow opencv-python")

        self._load_clip_model()

    def _load_clip_model(self):
        """Load CLIP model for style analysis."""
        print("Loading CLIP model for style analysis...")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai',
            device=self.device
        )
        self.clip_model.eval()
        print(f"✓ CLIP model loaded on {self.device}")

    def analyze_quality(self, image_path: str) -> Dict:
        """Analyze image quality metrics.

        Args:
            image_path: Path to image

        Returns:
            Quality metrics dict
        """
        img = cv2.imread(image_path)
        if img is None:
            return {'is_valid': False}

        # Convert to grayscale for blur detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur detection (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brightness
        brightness = np.mean(gray)

        # Size
        h, w = img.shape[:2]

        return {
            'is_valid': True,
            'blur_score': laplacian_var,
            'brightness': brightness,
            'resolution': (w, h),
            'is_sharp': laplacian_var >= self.blur_threshold,
            'is_well_lit': self.brightness_range[0] <= brightness <= self.brightness_range[1]
        }

    def extract_style_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract CLIP features for style analysis.

        Args:
            image_paths: List of image paths

        Returns:
            Feature array (N, feature_dim)
        """
        features = []

        with torch.no_grad():
            for img_path in tqdm(image_paths, desc="Extracting style features"):
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    image_features = self.clip_model.encode_image(image_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    features.append(image_features.cpu().numpy()[0])
                except Exception as e:
                    print(f"Warning: Failed to process {img_path}: {e}")
                    features.append(np.zeros(512))

        return np.array(features)

    def detect_transitions(
        self,
        image_paths: List[str],
        features: np.ndarray,
        similarity_threshold: float = 0.85
    ) -> List[int]:
        """Detect transition frames (sudden style changes).

        Args:
            image_paths: List of image paths
            features: CLIP features
            similarity_threshold: Cosine similarity threshold

        Returns:
            Indices of non-transition frames
        """
        if len(features) < 2:
            return list(range(len(features)))

        # Compute consecutive frame similarities
        similarities = []
        for i in range(len(features) - 1):
            sim = np.dot(features[i], features[i + 1])
            similarities.append(sim)

        similarities = np.array(similarities)

        # Frames with low similarity to neighbors are likely transitions
        non_transition_indices = [0]  # Always keep first frame
        for i in range(1, len(image_paths) - 1):
            # Check similarity to previous and next frames
            prev_sim = similarities[i - 1]
            next_sim = similarities[i] if i < len(similarities) else 1.0

            if prev_sim >= similarity_threshold or next_sim >= similarity_threshold:
                non_transition_indices.append(i)

        non_transition_indices.append(len(image_paths) - 1)  # Always keep last frame

        return non_transition_indices

    def select_diverse_samples(
        self,
        image_paths: List[str],
        features: np.ndarray,
        target_count: int
    ) -> List[int]:
        """Select diverse samples using k-means-like selection.

        Args:
            image_paths: List of image paths
            features: CLIP features
            target_count: Number of samples to select

        Returns:
            Indices of selected frames
        """
        if len(features) <= target_count:
            return list(range(len(features)))

        from sklearn.cluster import KMeans

        # K-means clustering
        n_clusters = min(target_count, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        # Select one sample from each cluster (closest to centroid)
        selected_indices = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_features = features[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # Find closest to centroid
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)

        return sorted(selected_indices)

    def select_frames(
        self,
        image_paths: List[str],
        output_dir: str
    ) -> Dict:
        """Select style-consistent frames.

        Args:
            image_paths: List of image paths
            output_dir: Output directory

        Returns:
            Selection results
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"STYLE FRAME SELECTION")
        print(f"{'='*60}")
        print(f"Input frames: {len(image_paths)}")
        print(f"Target count: {self.target_count}")
        print(f"{'='*60}\n")

        # Step 1: Quality filtering
        print("🔍 Analyzing quality...")
        quality_filtered = []
        quality_stats = {'total': len(image_paths), 'passed': 0, 'failed': 0}

        for img_path in tqdm(image_paths, desc="Quality check"):
            metrics = self.analyze_quality(img_path)
            if metrics.get('is_valid') and metrics.get('is_sharp') and metrics.get('is_well_lit'):
                quality_filtered.append(img_path)
                quality_stats['passed'] += 1
            else:
                quality_stats['failed'] += 1

        print(f"✓ Quality filter: {len(quality_filtered)}/{len(image_paths)} passed")

        if len(quality_filtered) == 0:
            print("No frames passed quality check!")
            return {}

        # Step 2: Extract style features
        features = self.extract_style_features(quality_filtered)

        # Step 3: Remove transitions
        print("\n🔧 Detecting transitions...")
        non_transition_indices = self.detect_transitions(quality_filtered, features)
        transition_filtered = [quality_filtered[i] for i in non_transition_indices]
        transition_features = features[non_transition_indices]

        print(f"✓ Transition filter: {len(transition_filtered)}/{len(quality_filtered)} kept")

        # Step 4: Diverse sampling
        print(f"\n🔧 Selecting {self.target_count} diverse samples...")
        selected_indices = self.select_diverse_samples(
            transition_filtered,
            transition_features,
            self.target_count
        )
        selected_paths = [transition_filtered[i] for i in selected_indices]

        print(f"✓ Selected {len(selected_paths)} frames")

        # Step 5: Copy selected frames
        print("\n📁 Copying selected frames...")
        for img_path in tqdm(selected_paths, desc="Copying"):
            basename = os.path.basename(img_path)
            dest_path = os.path.join(output_dir, basename)
            shutil.copy2(img_path, dest_path)

        # Save results
        results = {
            'input_frames': len(image_paths),
            'quality_passed': len(quality_filtered),
            'transition_filtered': len(transition_filtered),
            'final_selected': len(selected_paths),
            'quality_stats': quality_stats,
            'target_count': self.target_count,
            'blur_threshold': self.blur_threshold,
            'brightness_range': list(self.brightness_range)
        }

        results_path = os.path.join(output_dir, 'style_selection.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Style frame selection complete!")
        print(f"   Selected: {len(selected_paths)} frames")
        print(f"   Output: {output_dir}")
        print(f"   Results: {results_path}\n")

        return results


def main():
    parser = argparse.ArgumentParser(description="Style frame selector")
    parser.add_argument(
        "frames_dir",
        help="Directory with source frames"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for selected frames"
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=400,
        help="Target number of frames to select"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for processing"
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Laplacian variance threshold for blur detection"
    )

    args = parser.parse_args()

    if not HAS_CV2:
        print("Error: opencv-python not installed")
        print("Install: pip install opencv-python")
        return 1

    # Find frames
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"Error: {frames_dir} does not exist")
        return 1

    image_extensions = {'.png', '.jpg', '.jpeg'}
    image_paths = sorted([
        str(p) for p in frames_dir.rglob('*')
        if p.suffix.lower() in image_extensions
    ])

    if len(image_paths) == 0:
        print(f"No frames found in {frames_dir}")
        return 1

    # Initialize selector
    selector = StyleFrameSelector(
        device=args.device,
        target_count=args.target_count,
        blur_threshold=args.blur_threshold
    )

    # Select frames
    results = selector.select_frames(image_paths, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
