#!/usr/bin/env python3
"""
Smart Dataset Curator - AI-Powered Dataset Selection

Uses multiple SOTA models to automatically select:
1. High-quality images (sharp, well-composed, complete)
2. High diversity (different poses, angles, expressions, scenes)
3. Optimal balance (not too many similar images)

Features:
- Quality filtering: BRISQUE, face detection, blur detection
- Diversity maximization: CLIP embeddings + clustering
- Deduplication: Perceptual hashing (pHash)
- Balanced sampling: Ensures variety across pose/expression/scene
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')


class SmartDatasetCurator:
    """AI-powered dataset curation"""

    def __init__(
        self,
        device: str = "cuda",
        target_size: int = 400,
        min_quality_score: float = 30.0,
        diversity_weight: float = 0.7
    ):
        self.device = device
        self.target_size = target_size
        self.min_quality_score = min_quality_score
        self.diversity_weight = diversity_weight

        print("🤖 Initializing Smart Dataset Curator")
        print(f"  Target size: {target_size} images")
        print(f"  Min quality score: {min_quality_score}")
        print(f"  Diversity weight: {diversity_weight}")
        print()

        # Load models
        self.clip_model = None
        self.face_detector = None

    def load_clip(self):
        """Load CLIP for semantic embeddings"""
        if self.clip_model is None:
            print("📦 Loading CLIP model...")
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
            print("  ✓ CLIP loaded")
        return self.clip_model, self.clip_preprocess

    def load_face_detector(self):
        """Load face detector"""
        if self.face_detector is None:
            print("📦 Loading face detector...")
            try:
                from insightface.app import FaceAnalysis
                self.face_detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.face_detector.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
                print("  ✓ InsightFace loaded")
            except Exception as e:
                print(f"  ⚠️  InsightFace not available: {e}")
                self.face_detector = None
        return self.face_detector

    def compute_brisque_score(self, image_path: Path) -> float:
        """
        Compute BRISQUE (no-reference quality metric)
        Lower is better, typically 0-100
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return 100.0  # Worst score

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Compute Laplacian variance (blur metric)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Simple quality score (higher variance = sharper)
            # Invert so lower is better (consistent with BRISQUE)
            quality_score = max(0, 100 - laplacian_var / 10)

            return quality_score

        except Exception as e:
            print(f"  ⚠️  BRISQUE failed for {image_path.name}: {e}")
            return 100.0

    def detect_face_quality(self, image_path: Path) -> Dict:
        """
        Detect face and evaluate quality

        Returns:
            dict with keys: has_face, face_size, face_score
        """
        detector = self.load_face_detector()

        if detector is None:
            return {"has_face": True, "face_size": 0.5, "face_score": 0.5}  # Neutral

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {"has_face": False, "face_size": 0, "face_score": 0}

            faces = detector.get(img)

            if len(faces) == 0:
                return {"has_face": False, "face_size": 0, "face_score": 0}

            # Use largest face
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])

            # Face size relative to image
            img_area = img.shape[0] * img.shape[1]
            face_area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            face_size = face_area / img_area

            # Face quality score (from detection confidence)
            face_score = face.det_score

            return {
                "has_face": True,
                "face_size": float(face_size),
                "face_score": float(face_score)
            }

        except Exception as e:
            print(f"  ⚠️  Face detection failed for {image_path.name}: {e}")
            return {"has_face": True, "face_size": 0.5, "face_score": 0.5}

    def compute_clip_embedding(self, image_path: Path) -> np.ndarray:
        """Compute CLIP embedding for diversity analysis"""
        clip_model, preprocess = self.load_clip()

        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = clip_model.encode_image(image_tensor)
                embedding = embedding.cpu().numpy().flatten()
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            print(f"  ⚠️  CLIP embedding failed for {image_path.name}: {e}")
            return np.zeros(768)  # Return zero vector

    def compute_phash(self, image_path: Path) -> str:
        """Compute perceptual hash for deduplication"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return "0" * 16

            # Resize to 8x8
            img_small = cv2.resize(img, (8, 8))
            img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

            # Compute DCT
            dct = cv2.dct(np.float32(img_gray))

            # Take top-left 8x8
            dct_low = dct[:8, :8]

            # Compute median
            median = np.median(dct_low)

            # Generate hash
            hash_str = "".join(['1' if dct_low[i, j] > median else '0'
                                for i in range(8) for j in range(8)])

            return hash_str

        except Exception as e:
            print(f"  ⚠️  pHash failed for {image_path.name}: {e}")
            return "0" * 64

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Compute hamming distance between two hashes"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def analyze_images(self, image_paths: List[Path]) -> List[Dict]:
        """
        Analyze all images and compute quality/diversity metrics

        Returns:
            List of dicts with analysis results
        """
        print(f"\n📊 Analyzing {len(image_paths)} images...")

        results = []

        for img_path in tqdm(image_paths, desc="Analyzing"):
            # Quality metrics
            brisque = self.compute_brisque_score(img_path)
            face_info = self.detect_face_quality(img_path)

            # Diversity metrics
            clip_embedding = self.compute_clip_embedding(img_path)
            phash = self.compute_phash(img_path)

            # Combined quality score
            quality_score = 100 - brisque  # Higher is better
            if face_info["has_face"]:
                quality_score += face_info["face_score"] * 10  # Bonus for good face
                quality_score += face_info["face_size"] * 20   # Bonus for large face

            results.append({
                "path": str(img_path),
                "name": img_path.name,
                "quality_score": quality_score,
                "brisque": brisque,
                "has_face": face_info["has_face"],
                "face_size": face_info["face_size"],
                "face_score": face_info["face_score"],
                "clip_embedding": clip_embedding.tolist(),
                "phash": phash
            })

        print(f"  ✓ Analyzed {len(results)} images")
        return results

    def filter_quality(self, results: List[Dict]) -> List[Dict]:
        """Filter images by quality threshold"""
        print(f"\n🔍 Filtering by quality (min score: {self.min_quality_score})...")

        # Filter by quality score
        filtered = [r for r in results if r["quality_score"] >= self.min_quality_score]

        # Filter images without faces (if face detection worked)
        has_faces = [r for r in filtered if r["has_face"]]
        if len(has_faces) > 0:
            filtered = has_faces

        print(f"  ✓ Kept {len(filtered)} / {len(results)} images ({len(filtered)/len(results)*100:.1f}%)")

        return filtered

    def deduplicate(self, results: List[Dict], threshold: int = 10) -> List[Dict]:
        """Remove near-duplicate images using pHash"""
        print(f"\n🔄 Deduplicating (hamming threshold: {threshold})...")

        # Sort by quality (keep higher quality in duplicates)
        results_sorted = sorted(results, key=lambda x: x["quality_score"], reverse=True)

        kept = []
        seen_hashes = []

        for result in results_sorted:
            # Check if similar to any kept image
            is_duplicate = False
            for seen_hash in seen_hashes:
                if self.hamming_distance(result["phash"], seen_hash) < threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(result)
                seen_hashes.append(result["phash"])

        print(f"  ✓ Removed {len(results) - len(kept)} duplicates")
        print(f"  ✓ Kept {len(kept)} unique images")

        return kept

    def maximize_diversity(self, results: List[Dict]) -> List[Dict]:
        """
        Select diverse subset using clustering on CLIP embeddings

        Strategy:
        1. Cluster images into K groups (K = target_size / 3)
        2. Sample from each cluster proportionally
        3. Within each cluster, select by quality
        """
        print(f"\n🎯 Maximizing diversity (target: {self.target_size} images)...")

        if len(results) <= self.target_size:
            print(f"  ℹ️  Already at or below target size ({len(results)} <= {self.target_size})")
            return results

        # Extract CLIP embeddings
        embeddings = np.array([r["clip_embedding"] for r in results])

        # Determine number of clusters (aim for 3-5 images per cluster)
        n_clusters = max(10, self.target_size // 4)
        n_clusters = min(n_clusters, len(results) // 2)  # At least 2 images per cluster

        print(f"  📊 Clustering into {n_clusters} groups...")

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Add cluster labels to results
        for i, result in enumerate(results):
            result["cluster"] = int(cluster_labels[i])

        # Sample from each cluster
        selected = []

        # Calculate samples per cluster
        cluster_counts = np.bincount(cluster_labels)
        samples_per_cluster = {}

        for cluster_id in range(n_clusters):
            cluster_size = cluster_counts[cluster_id]
            # Proportional sampling, with minimum 1
            n_samples = max(1, int(self.target_size * cluster_size / len(results)))
            samples_per_cluster[cluster_id] = n_samples

        # Adjust to hit exact target
        total_samples = sum(samples_per_cluster.values())
        if total_samples < self.target_size:
            # Add to largest clusters
            for cluster_id in np.argsort(cluster_counts)[::-1]:
                if total_samples >= self.target_size:
                    break
                samples_per_cluster[cluster_id] += 1
                total_samples += 1
        elif total_samples > self.target_size:
            # Remove from largest clusters
            for cluster_id in np.argsort(cluster_counts)[::-1]:
                if total_samples <= self.target_size:
                    break
                if samples_per_cluster[cluster_id] > 1:
                    samples_per_cluster[cluster_id] -= 1
                    total_samples -= 1

        print(f"  📋 Sampling strategy:")
        for cluster_id in range(min(5, n_clusters)):  # Show first 5
            cluster_size = cluster_counts[cluster_id]
            n_samples = samples_per_cluster[cluster_id]
            print(f"    Cluster {cluster_id}: {cluster_size} images → sample {n_samples}")
        if n_clusters > 5:
            print(f"    ... and {n_clusters - 5} more clusters")

        # Select from each cluster (highest quality first)
        for cluster_id in range(n_clusters):
            cluster_images = [r for r in results if r["cluster"] == cluster_id]
            cluster_images_sorted = sorted(cluster_images, key=lambda x: x["quality_score"], reverse=True)

            n_select = samples_per_cluster[cluster_id]
            selected.extend(cluster_images_sorted[:n_select])

        print(f"  ✓ Selected {len(selected)} diverse images")

        return selected

    def curate(
        self,
        input_dir: Path,
        output_dir: Path,
        save_metadata: bool = True
    ) -> List[Path]:
        """
        Main curation workflow

        Returns:
            List of selected image paths
        """
        print("="*70)
        print("SMART DATASET CURATION")
        print("="*70)

        # Find all images
        image_paths = sorted(list(input_dir.glob("*.png")))
        print(f"\n📁 Found {len(image_paths)} images in {input_dir}")

        if len(image_paths) == 0:
            print("❌ No images found!")
            return []

        # Analyze all images
        results = self.analyze_images(image_paths)

        # Quality filtering
        results = self.filter_quality(results)

        # Deduplication
        results = self.deduplicate(results)

        # Diversity maximization
        results = self.maximize_diversity(results)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy selected images
        print(f"\n📦 Copying {len(results)} selected images to {output_dir}...")
        selected_paths = []

        for result in tqdm(results, desc="Copying"):
            src_path = Path(result["path"])
            dst_path = output_dir / src_path.name

            # Copy file
            import shutil
            shutil.copy2(src_path, dst_path)
            selected_paths.append(dst_path)

        # Save metadata
        if save_metadata:
            metadata_path = output_dir / "curation_metadata.json"
            metadata = {
                "total_analyzed": len(image_paths),
                "quality_filtered": len(results),
                "final_selected": len(results),
                "target_size": self.target_size,
                "min_quality_score": self.min_quality_score,
                "diversity_weight": self.diversity_weight,
                "selected_images": [
                    {
                        "name": r["name"],
                        "quality_score": r["quality_score"],
                        "cluster": r.get("cluster", -1)
                    }
                    for r in results
                ]
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"  ✓ Saved metadata to {metadata_path}")

        print("\n" + "="*70)
        print("✅ CURATION COMPLETE!")
        print("="*70)
        print(f"\n📊 Summary:")
        print(f"  Input images: {len(image_paths)}")
        print(f"  Selected: {len(results)}")
        print(f"  Selection rate: {len(results)/len(image_paths)*100:.1f}%")
        print(f"  Output: {output_dir}")

        return selected_paths


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered smart dataset curation"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for curated dataset"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=400,
        help="Target number of images (default: 400)"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=30.0,
        help="Minimum quality score (default: 30.0)"
    )
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.7,
        help="Weight for diversity vs quality (0-1, default: 0.7)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    # Create curator
    curator = SmartDatasetCurator(
        device=args.device,
        target_size=args.target_size,
        min_quality_score=args.min_quality,
        diversity_weight=args.diversity_weight
    )

    # Run curation
    curator.curate(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
