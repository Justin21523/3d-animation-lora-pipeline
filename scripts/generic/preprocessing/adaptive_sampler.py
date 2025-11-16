#!/usr/bin/env python3
"""
Adaptive Frame Sampler

Purpose: Intelligent frame sampling to maintain diversity while reducing dataset size
Methods: Clustering-based, temporal distribution, quality-weighted, hybrid
Use Cases: Reduce redundancy while preserving visual diversity

Usage:
    python adaptive_sampler.py \
        --input-dir /path/to/frames \
        --output-dir /path/to/sampled \
        --method clustering \
        --target-count 500 \
        --preserve-diversity
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import shutil


@dataclass
class SamplingConfig:
    """Configuration for adaptive sampling"""
    method: str = "clustering"  # clustering, temporal, quality, hybrid
    target_count: int = 500  # Target number of frames to keep
    target_ratio: Optional[float] = None  # Alternative: keep this ratio (0-1)
    preserve_diversity: bool = True  # Ensure diverse samples
    quality_weighted: bool = True  # Prefer higher quality frames
    temporal_spread: bool = True  # Spread samples across time
    min_temporal_gap: int = 5  # Minimum frames between samples (temporal mode)
    num_clusters: Optional[int] = None  # Number of clusters (auto if None)
    feature_type: str = "combined"  # histogram, edge, combined
    create_hardlinks: bool = False  # Use hardlinks instead of copying
    save_metadata: bool = True  # Save sampling metadata


class AdaptiveFrameSampler:
    """Intelligent frame sampling using various strategies"""

    def __init__(self, config: SamplingConfig):
        """
        Initialize adaptive sampler

        Args:
            config: Sampling configuration
        """
        self.config = config

    def extract_histogram_features(self, image_path: Path) -> np.ndarray:
        """
        Extract color histogram features

        Args:
            image_path: Path to image

        Returns:
            Feature vector
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return np.zeros(512)  # 3 channels * 64 bins each + HSV

        # RGB histograms
        features = []
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [64], [0, 256])
            features.append(hist.flatten())

        # HSV histogram
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [64], [0, 256])
            features.append(hist.flatten())

        # Concatenate and normalize
        features = np.concatenate(features)
        features = features / (features.sum() + 1e-8)

        return features

    def extract_edge_features(self, image_path: Path) -> np.ndarray:
        """
        Extract edge-based features

        Args:
            image_path: Path to image

        Returns:
            Feature vector
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(128)

        # Canny edges
        edges = cv2.Canny(img, 50, 150)

        # Edge histogram (spatial bins)
        h, w = edges.shape
        grid_h, grid_w = 4, 4
        cell_h, cell_w = h // grid_h, w // grid_w

        features = []
        for i in range(grid_h):
            for j in range(grid_w):
                cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                features.append(cell.mean())

        # Sobel gradients
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # Gradient histogram
        grad_hist, _ = np.histogram(gradient_mag, bins=32, range=(0, 255))
        features.extend(grad_hist.tolist())

        # Normalize
        features = np.array(features)
        features = features / (features.sum() + 1e-8)

        return features

    def extract_combined_features(self, image_path: Path) -> np.ndarray:
        """
        Extract combined color and edge features

        Args:
            image_path: Path to image

        Returns:
            Feature vector
        """
        hist_feat = self.extract_histogram_features(image_path)
        edge_feat = self.extract_edge_features(image_path)
        return np.concatenate([hist_feat, edge_feat])

    def compute_quality_score(self, image_path: Path) -> float:
        """
        Compute quality score for prioritization

        Args:
            image_path: Path to image

        Returns:
            Quality score (higher is better)
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0

        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = laplacian.var()

        # Contrast (std dev)
        contrast = img.std()

        # Combine (weighted)
        quality = 0.7 * sharpness + 0.3 * contrast
        return float(quality)

    def sample_clustering(
        self,
        image_files: List[Path],
        target_count: int
    ) -> List[Path]:
        """
        Sample using clustering-based diversity

        Args:
            image_files: List of image paths
            target_count: Number of samples to select

        Returns:
            Sampled image paths
        """
        print(f"\n🔍 Extracting features from {len(image_files)} frames...")

        # Extract features
        features = []
        valid_files = []

        for img_path in tqdm(image_files, desc="Extracting features"):
            try:
                if self.config.feature_type == "histogram":
                    feat = self.extract_histogram_features(img_path)
                elif self.config.feature_type == "edge":
                    feat = self.extract_edge_features(img_path)
                else:  # combined
                    feat = self.extract_combined_features(img_path)

                features.append(feat)
                valid_files.append(img_path)
            except Exception as e:
                print(f"⚠️ Error processing {img_path.name}: {e}")
                continue

        features = np.array(features)

        # Determine number of clusters
        if self.config.num_clusters:
            n_clusters = min(self.config.num_clusters, target_count)
        else:
            n_clusters = target_count

        print(f"\n📊 Clustering into {n_clusters} groups...")

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Cluster
        if len(features) > 10000:
            # Use MiniBatch for large datasets
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=1024
            )
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        labels = kmeans.fit_predict(features_scaled)

        # Sample from each cluster
        print(f"\n🎯 Selecting samples from each cluster...")

        sampled = []
        samples_per_cluster = {}

        if self.config.quality_weighted:
            # Compute quality scores
            quality_scores = {}
            for img_path in tqdm(valid_files, desc="Computing quality"):
                quality_scores[img_path] = self.compute_quality_score(img_path)

        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_files = [f for f, m in zip(valid_files, cluster_mask) if m]

            if not cluster_files:
                continue

            # Select best from cluster
            if self.config.quality_weighted:
                # Sort by quality
                cluster_files.sort(
                    key=lambda x: quality_scores.get(x, 0),
                    reverse=True
                )

            # Take top samples from this cluster
            n_samples = max(1, len(cluster_files) * target_count // len(valid_files))
            sampled.extend(cluster_files[:n_samples])
            samples_per_cluster[cluster_id] = n_samples

        # Adjust to exact target count
        if len(sampled) > target_count:
            sampled = sampled[:target_count]
        elif len(sampled) < target_count:
            # Add more from largest clusters
            remaining = target_count - len(sampled)
            sampled_set = set(sampled)
            candidates = [f for f in valid_files if f not in sampled_set]

            if self.config.quality_weighted and candidates:
                candidates.sort(
                    key=lambda x: quality_scores.get(x, 0),
                    reverse=True
                )
                sampled.extend(candidates[:remaining])

        print(f"\n📊 Cluster distribution:")
        for cid, count in sorted(samples_per_cluster.items()):
            print(f"   Cluster {cid}: {count} samples")

        return sampled

    def sample_temporal(
        self,
        image_files: List[Path],
        target_count: int
    ) -> List[Path]:
        """
        Sample with temporal distribution

        Args:
            image_files: List of image paths (assumed sorted by time)
            target_count: Number of samples to select

        Returns:
            Sampled image paths
        """
        print(f"\n⏱️ Temporal sampling from {len(image_files)} frames...")

        # Sort by filename (assumes temporal ordering)
        sorted_files = sorted(image_files)

        if self.config.quality_weighted:
            # Compute quality scores
            print("Computing quality scores...")
            quality_scores = {}
            for img_path in tqdm(sorted_files, desc="Quality scoring"):
                quality_scores[img_path] = self.compute_quality_score(img_path)

        # Divide timeline into segments
        n_segments = target_count
        segment_size = len(sorted_files) / n_segments

        sampled = []

        for i in range(n_segments):
            start_idx = int(i * segment_size)
            end_idx = int((i + 1) * segment_size)
            segment_files = sorted_files[start_idx:end_idx]

            if not segment_files:
                continue

            # Select best from segment
            if self.config.quality_weighted:
                best = max(segment_files, key=lambda x: quality_scores.get(x, 0))
            else:
                # Take middle frame
                best = segment_files[len(segment_files) // 2]

            sampled.append(best)

        print(f"   Selected {len(sampled)} frames across timeline")
        return sampled

    def sample_quality_weighted(
        self,
        image_files: List[Path],
        target_count: int
    ) -> List[Path]:
        """
        Sample purely by quality score

        Args:
            image_files: List of image paths
            target_count: Number of samples to select

        Returns:
            Sampled image paths
        """
        print(f"\n⭐ Quality-weighted sampling from {len(image_files)} frames...")

        # Compute quality scores
        quality_scores = {}
        for img_path in tqdm(image_files, desc="Computing quality"):
            quality_scores[img_path] = self.compute_quality_score(img_path)

        # Sort by quality
        sorted_files = sorted(
            image_files,
            key=lambda x: quality_scores.get(x, 0),
            reverse=True
        )

        # Apply temporal spread if enabled
        if self.config.temporal_spread:
            print("   Applying temporal spread constraint...")
            sampled = []
            sampled_indices = set()

            # Create filename to index mapping
            file_to_idx = {f: i for i, f in enumerate(sorted(image_files))}

            for img_path in sorted_files:
                idx = file_to_idx[img_path]

                # Check if far enough from existing samples
                too_close = False
                for sampled_idx in sampled_indices:
                    if abs(idx - sampled_idx) < self.config.min_temporal_gap:
                        too_close = True
                        break

                if not too_close:
                    sampled.append(img_path)
                    sampled_indices.add(idx)

                if len(sampled) >= target_count:
                    break

            # If not enough, relax constraint
            if len(sampled) < target_count:
                remaining = target_count - len(sampled)
                sampled_set = set(sampled)
                candidates = [f for f in sorted_files if f not in sampled_set]
                sampled.extend(candidates[:remaining])

        else:
            sampled = sorted_files[:target_count]

        print(f"   Selected top {len(sampled)} quality frames")
        return sampled

    def sample_hybrid(
        self,
        image_files: List[Path],
        target_count: int
    ) -> List[Path]:
        """
        Hybrid sampling: clustering + quality + temporal

        Args:
            image_files: List of image paths
            target_count: Number of samples to select

        Returns:
            Sampled image paths
        """
        print(f"\n🔀 Hybrid sampling from {len(image_files)} frames...")

        # Phase 1: Cluster-based initial selection (oversample)
        cluster_target = int(target_count * 1.5)
        cluster_samples = self.sample_clustering(image_files, cluster_target)

        # Phase 2: Refine with temporal and quality
        if self.config.temporal_spread:
            # Sort cluster samples by quality
            quality_scores = {}
            for img_path in tqdm(cluster_samples, desc="Refining selection"):
                quality_scores[img_path] = self.compute_quality_score(img_path)

            sorted_samples = sorted(
                cluster_samples,
                key=lambda x: quality_scores.get(x, 0),
                reverse=True
            )

            # Apply temporal spread
            sampled = []
            sampled_indices = set()
            file_to_idx = {f: i for i, f in enumerate(sorted(image_files))}

            for img_path in sorted_samples:
                idx = file_to_idx[img_path]

                too_close = False
                for sampled_idx in sampled_indices:
                    if abs(idx - sampled_idx) < self.config.min_temporal_gap:
                        too_close = True
                        break

                if not too_close:
                    sampled.append(img_path)
                    sampled_indices.add(idx)

                if len(sampled) >= target_count:
                    break

            # Fill remaining if needed
            if len(sampled) < target_count:
                remaining = target_count - len(sampled)
                sampled_set = set(sampled)
                candidates = [f for f in sorted_samples if f not in sampled_set]
                sampled.extend(candidates[:remaining])

        else:
            sampled = cluster_samples[:target_count]

        print(f"   Final selection: {len(sampled)} frames")
        return sampled

    def sample_frames(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main adaptive sampling pipeline

        Args:
            input_dir: Directory with input frames
            output_dir: Directory to save sampled frames

        Returns:
            Statistics dictionary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all image files
        image_files = sorted(
            list(input_dir.glob("*.jpg")) +
            list(input_dir.glob("*.png"))
        )

        print(f"\n📊 Found {len(image_files)} frames in {input_dir}")

        # Determine target count
        if self.config.target_ratio:
            target_count = int(len(image_files) * self.config.target_ratio)
        else:
            target_count = min(self.config.target_count, len(image_files))

        print(f"🎯 Target: {target_count} frames ({target_count/len(image_files):.1%})")

        # Sample based on method
        if self.config.method == "clustering":
            sampled_files = self.sample_clustering(image_files, target_count)
        elif self.config.method == "temporal":
            sampled_files = self.sample_temporal(image_files, target_count)
        elif self.config.method == "quality":
            sampled_files = self.sample_quality_weighted(image_files, target_count)
        elif self.config.method == "hybrid":
            sampled_files = self.sample_hybrid(image_files, target_count)
        else:
            print(f"❌ Unknown method: {self.config.method}")
            return {}

        # Copy/link sampled files
        print(f"\n📁 Saving {len(sampled_files)} sampled frames...")

        for src_path in tqdm(sampled_files, desc="Saving frames"):
            dst_path = output_dir / src_path.name

            if self.config.create_hardlinks:
                try:
                    dst_path.hardlink_to(src_path)
                except:
                    shutil.copy2(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        # Save statistics
        stats = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "method": self.config.method,
            "config": {
                "target_count": self.config.target_count,
                "target_ratio": self.config.target_ratio,
                "preserve_diversity": self.config.preserve_diversity,
                "quality_weighted": self.config.quality_weighted,
                "temporal_spread": self.config.temporal_spread,
            },
            "total_input_frames": len(image_files),
            "sampled_frames": len(sampled_files),
            "sampling_rate": len(sampled_files) / len(image_files) if len(image_files) > 0 else 0,
            "reduction_rate": 1 - (len(sampled_files) / len(image_files)) if len(image_files) > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }

        if self.config.save_metadata:
            metadata_path = output_dir / "sampling_report.json"
            with open(metadata_path, 'w') as f:
                json.dump(stats, f, indent=2)

            print(f"\n📄 Report saved to: {metadata_path}")

        print(f"\n✅ Adaptive sampling complete!")
        print(f"   Input: {len(image_files)} frames")
        print(f"   Output: {len(sampled_files)} frames")
        print(f"   Reduction: {stats['reduction_rate']:.1%}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent frame sampling (Film-Agnostic)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with input frames"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save sampled frames"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hybrid",
        choices=["clustering", "temporal", "quality", "hybrid"],
        help="Sampling method (default: hybrid)"
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=500,
        help="Target number of frames to keep (default: 500)"
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        help="Alternative: target ratio of frames to keep (0-1)"
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        help="Number of clusters for clustering method (auto if not set)"
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="combined",
        choices=["histogram", "edge", "combined"],
        help="Feature type for clustering (default: combined)"
    )
    parser.add_argument(
        "--min-temporal-gap",
        type=int,
        default=5,
        help="Minimum frames between samples (default: 5)"
    )
    parser.add_argument(
        "--no-quality-weighting",
        action="store_true",
        help="Disable quality-based prioritization"
    )
    parser.add_argument(
        "--no-temporal-spread",
        action="store_true",
        help="Disable temporal spread constraint"
    )
    parser.add_argument(
        "--hardlinks",
        action="store_true",
        help="Use hardlinks instead of copying (saves space)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (for logging)"
    )

    args = parser.parse_args()

    # Create config
    config = SamplingConfig(
        method=args.method,
        target_count=args.target_count,
        target_ratio=args.target_ratio,
        preserve_diversity=True,
        quality_weighted=not args.no_quality_weighting,
        temporal_spread=not args.no_temporal_spread,
        min_temporal_gap=args.min_temporal_gap,
        num_clusters=args.num_clusters,
        feature_type=args.feature_type,
        create_hardlinks=args.hardlinks,
        save_metadata=True
    )

    # Run sampling
    sampler = AdaptiveFrameSampler(config)
    stats = sampler.sample_frames(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
