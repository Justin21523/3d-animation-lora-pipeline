"""
Face-based identity clustering for 2D Animation LoRA Pipeline.

Uses face detection (RetinaFace) and face recognition (ArcFace)
to cluster character images by identity, essential for multi-character
scene handling.

Workflow:
1. Detect faces in all images
2. Extract ArcFace embeddings for each face
3. Cluster embeddings with HDBSCAN
4. Export clusters to separate directories
"""

from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Single face detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5-point landmarks

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class FaceEmbedding:
    """Face embedding with metadata."""
    image_path: Path
    embedding: np.ndarray
    face_bbox: Tuple[int, int, int, int]
    confidence: float
    face_area: int


@dataclass
class IdentityCluster:
    """A cluster of faces belonging to the same identity."""
    cluster_id: int
    name: str
    image_paths: List[Path]
    embeddings: np.ndarray
    centroid: np.ndarray
    mean_confidence: float

    @property
    def size(self) -> int:
        return len(self.image_paths)


@dataclass
class ClusteringResult:
    """Complete clustering result."""
    clusters: Dict[int, IdentityCluster]
    noise_images: List[Path]
    total_images: int
    total_faces: int
    processing_time: float
    parameters: Dict

    def to_dict(self) -> Dict:
        return {
            "num_clusters": len(self.clusters),
            "cluster_sizes": {k: v.size for k, v in self.clusters.items()},
            "noise_count": len(self.noise_images),
            "total_images": self.total_images,
            "total_faces": self.total_faces,
            "processing_time": self.processing_time,
            "parameters": self.parameters,
        }


class FaceIdentityClusterer:
    """
    Face detection → ArcFace embedding → HDBSCAN clustering.

    Designed for 2D animation character identity clustering.
    Supports stub mode for testing without GPU/models.

    For 2D animated characters (cartoons, anime), standard face detection
    may fail due to stylized features. Use `fallback_to_clip=True` to
    automatically fall back to CLIP embeddings when no face is detected.

    Parameters:
        face_model: Face detection model ("retinaface" or "scrfd")
        embedding_model: Face recognition model ("arcface" or "adaface")
        min_cluster_size: Minimum cluster size for HDBSCAN (2D default: 20)
        min_samples: Core point threshold for HDBSCAN (2D default: 3)
        device: "cuda", "cpu", or "stub"
        min_face_size: Minimum face size in pixels
        face_confidence: Minimum detection confidence
        fallback_to_clip: If True, use CLIP embeddings when face detection fails
        animation_mode: "2d" or "3d" - adjusts thresholds for animation style
    """

    # Default paths from AI_WAREHOUSE 3.0
    DEFAULT_FACE_MODEL = "/mnt/c/ai_models/face/retinaface_r50.onnx"
    DEFAULT_EMBEDDING_MODEL = "/mnt/c/ai_models/face/arcface_r100.onnx"
    DEFAULT_CLIP_MODEL = "ViT-B/32"

    # 2D Animation specific thresholds (lower confidence for stylized faces)
    THRESHOLDS_2D = {
        "face_confidence": 0.3,  # Lower for cartoons
        "min_face_size": 32,  # Smaller faces in wide shots
        "min_cluster_size": 20,
        "min_samples": 3,
    }

    # 3D Animation thresholds (realistic renders)
    THRESHOLDS_3D = {
        "face_confidence": 0.5,
        "min_face_size": 48,
        "min_cluster_size": 12,
        "min_samples": 2,
    }

    def __init__(
        self,
        face_model: str = "retinaface",
        embedding_model: str = "arcface",
        min_cluster_size: int = 20,
        min_samples: int = 3,
        device: str = "cuda",
        min_face_size: int = 48,
        face_confidence: float = 0.5,
        fallback_to_clip: bool = True,
        animation_mode: str = "2d",
    ):
        self.face_model_name = face_model
        self.embedding_model_name = embedding_model
        self.device = device
        self.fallback_to_clip = fallback_to_clip
        self.animation_mode = animation_mode

        # Apply animation mode defaults if not explicitly overridden
        thresholds = self.THRESHOLDS_2D if animation_mode == "2d" else self.THRESHOLDS_3D
        self.min_cluster_size = min_cluster_size or thresholds["min_cluster_size"]
        self.min_samples = min_samples or thresholds["min_samples"]
        self.min_face_size = min_face_size or thresholds["min_face_size"]
        self.face_confidence = face_confidence or thresholds["face_confidence"]

        self.stub_mode = device == "stub"

        # Track face detection success rate for adaptive fallback
        self._face_detection_attempts = 0
        self._face_detection_successes = 0

        # Will be initialized lazily
        self._face_detector = None
        self._face_embedder = None
        self._clusterer = None
        self._clip_model = None
        self._clip_preprocess = None

        if not self.stub_mode:
            self._init_models()
            if self.fallback_to_clip:
                self._init_clip_model()

    def _init_models(self) -> None:
        """Initialize face detection and embedding models."""
        try:
            # Try to import insightface for face detection/recognition
            import insightface
            from insightface.app import FaceAnalysis

            # Initialize face analysis app
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if self.device == "cpu":
                providers = ['CPUExecutionProvider']

            self._face_app = FaceAnalysis(
                name=self.face_model_name,
                providers=providers,
            )
            self._face_app.prepare(ctx_id=0 if self.device == "cuda" else -1)

            logger.info(f"Initialized InsightFace with {self.face_model_name}")

        except ImportError:
            logger.warning(
                "InsightFace not available, falling back to stub mode. "
                "Install with: pip install insightface onnxruntime-gpu"
            )
            self.stub_mode = True
        except Exception as e:
            logger.warning(f"Failed to initialize face models: {e}, using stub mode")
            self.stub_mode = True

    def _init_clip_model(self) -> None:
        """Initialize CLIP model for fallback embeddings."""
        try:
            import clip
            import torch

            device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
            self._clip_model, self._clip_preprocess = clip.load(
                self.DEFAULT_CLIP_MODEL, device=device
            )
            logger.info(f"Initialized CLIP ({self.DEFAULT_CLIP_MODEL}) for fallback embeddings")

        except ImportError:
            logger.warning(
                "CLIP not available for fallback. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            )
            self._clip_model = None
        except Exception as e:
            logger.warning(f"Failed to initialize CLIP: {e}")
            self._clip_model = None

    def extract_clip_embedding(
        self,
        image_path: Union[str, Path],
    ) -> Optional[np.ndarray]:
        """
        Extract CLIP embedding for an image.

        Used as fallback when face detection fails on 2D animated characters.

        Args:
            image_path: Path to image file

        Returns:
            CLIP embedding (512-dim) or None if failed
        """
        if self._clip_model is None:
            return None

        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            image_input = self._clip_preprocess(image).unsqueeze(0)

            device = next(self._clip_model.parameters()).device
            image_input = image_input.to(device)

            with torch.no_grad():
                embedding = self._clip_model.encode_image(image_input)
                embedding = embedding.cpu().numpy().squeeze()

            # Normalize to unit vector
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding.astype(np.float32)

        except Exception as e:
            logger.warning(f"CLIP embedding failed for {image_path}: {e}")
            return None

    def get_face_detection_rate(self) -> float:
        """Get the current face detection success rate."""
        if self._face_detection_attempts == 0:
            return 0.0
        return self._face_detection_successes / self._face_detection_attempts

    def _init_clusterer(self) -> None:
        """Initialize HDBSCAN clusterer."""
        try:
            import hdbscan
            self._clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
            )
            logger.info(
                f"Initialized HDBSCAN (min_cluster_size={self.min_cluster_size}, "
                f"min_samples={self.min_samples})"
            )
        except ImportError:
            logger.warning(
                "HDBSCAN not available. Install with: pip install hdbscan"
            )
            self._clusterer = None

    def detect_faces(
        self,
        image_path: Union[str, Path],
    ) -> List[FaceDetection]:
        """
        Detect faces in a single image.

        Args:
            image_path: Path to image file

        Returns:
            List of FaceDetection objects
        """
        image_path = Path(image_path)

        if self.stub_mode:
            # Return synthetic face detection for testing
            return [FaceDetection(
                bbox=(50, 50, 150, 150),
                confidence=0.95,
                landmarks=np.array([
                    [70, 80], [130, 80],  # eyes
                    [100, 110],  # nose
                    [75, 130], [125, 130],  # mouth corners
                ]),
            )]

        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return []

        # Run face detection
        faces = self._face_app.get(img)

        detections = []
        for face in faces:
            bbox = tuple(map(int, face.bbox))
            x1, y1, x2, y2 = bbox

            # Filter by size
            if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                continue

            # Filter by confidence
            if face.det_score < self.face_confidence:
                continue

            detections.append(FaceDetection(
                bbox=bbox,
                confidence=float(face.det_score),
                landmarks=face.kps if hasattr(face, 'kps') else None,
            ))

        return detections

    def extract_embedding(
        self,
        image_path: Union[str, Path],
        face: Optional[FaceDetection] = None,
    ) -> Optional[FaceEmbedding]:
        """
        Extract face embedding from image.

        For 2D animated characters where face detection may fail,
        automatically falls back to CLIP embeddings if `fallback_to_clip=True`.

        Args:
            image_path: Path to image file
            face: Optional pre-detected face. If None, detects the largest face.

        Returns:
            FaceEmbedding or None if no face/embedding found
        """
        image_path = Path(image_path)

        if self.stub_mode:
            # Return synthetic embedding for testing
            np.random.seed(hash(str(image_path)) % (2**32))
            return FaceEmbedding(
                image_path=image_path,
                embedding=np.random.randn(512).astype(np.float32),
                face_bbox=(50, 50, 150, 150),
                confidence=0.95,
                face_area=10000,
            )

        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            return None

        # Track face detection attempts
        self._face_detection_attempts += 1

        # Run face analysis (detection + embedding)
        faces = self._face_app.get(img)

        if not faces:
            # No face detected - try CLIP fallback for 2D characters
            if self.fallback_to_clip and self._clip_model is not None:
                logger.debug(f"No face detected in {image_path.name}, using CLIP fallback")
                clip_embedding = self.extract_clip_embedding(image_path)
                if clip_embedding is not None:
                    # Return CLIP embedding with full-image bbox and zero confidence
                    # to indicate this is a fallback (confidence=0 means CLIP-based)
                    h, w = img.shape[:2]
                    return FaceEmbedding(
                        image_path=image_path,
                        embedding=clip_embedding,
                        face_bbox=(0, 0, w, h),  # Full image bbox
                        confidence=0.0,  # Signal that this is CLIP-based
                        face_area=w * h,
                    )
            return None

        # Face detected - count as success
        self._face_detection_successes += 1

        # Get the largest face if not specified
        if face is None:
            target_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        else:
            # Find matching face by bbox overlap
            target_face = None
            max_iou = 0
            for f in faces:
                iou = self._compute_iou(face.bbox, tuple(map(int, f.bbox)))
                if iou > max_iou:
                    max_iou = iou
                    target_face = f
            if target_face is None:
                target_face = faces[0]

        bbox = tuple(map(int, target_face.bbox))
        x1, y1, x2, y2 = bbox

        return FaceEmbedding(
            image_path=image_path,
            embedding=target_face.embedding,
            face_bbox=bbox,
            confidence=float(target_face.det_score),
            face_area=(x2 - x1) * (y2 - y1),
        )

    def extract_face_embeddings(
        self,
        images_dir: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
        num_workers: int = 4,
    ) -> Tuple[np.ndarray, List[Path]]:
        """
        Extract face embeddings from all images in directory.

        Args:
            images_dir: Directory containing images
            output_path: Optional path to save embeddings (.npz)
            extensions: Image file extensions to process
            num_workers: Number of parallel workers (CPU I/O only)

        Returns:
            Tuple of (embeddings array [N, 512], list of image paths)
        """
        images_dir = Path(images_dir)

        # Collect all image paths
        image_paths = []
        for ext in extensions:
            image_paths.extend(images_dir.glob(f"*{ext}"))
            image_paths.extend(images_dir.glob(f"*{ext.upper()}"))

        image_paths = sorted(set(image_paths))
        logger.info(f"Found {len(image_paths)} images in {images_dir}")

        embeddings = []
        valid_paths = []

        # Process images (parallel file loading, sequential GPU)
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 100 == 0:
                logger.info(f"Processing {i + 1}/{len(image_paths)} images...")

            result = self.extract_embedding(img_path)
            if result is not None:
                embeddings.append(result.embedding)
                valid_paths.append(img_path)

        if not embeddings:
            logger.warning("No face embeddings extracted")
            return np.array([]), []

        embeddings_array = np.vstack(embeddings)
        logger.info(f"Extracted {len(embeddings)} face embeddings, shape: {embeddings_array.shape}")

        # Optionally save
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                output_path,
                embeddings=embeddings_array,
                paths=[str(p) for p in valid_paths],
            )
            logger.info(f"Saved embeddings to {output_path}")

        return embeddings_array, valid_paths

    def cluster_by_identity(
        self,
        embeddings: np.ndarray,
        image_paths: List[Path],
    ) -> ClusteringResult:
        """
        Cluster face embeddings by identity using HDBSCAN.

        Args:
            embeddings: Face embeddings array [N, 512]
            image_paths: Corresponding image paths

        Returns:
            ClusteringResult with clusters and noise images
        """
        import time
        start_time = time.time()

        if len(embeddings) == 0:
            return ClusteringResult(
                clusters={},
                noise_images=[],
                total_images=0,
                total_faces=0,
                processing_time=0,
                parameters={},
            )

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / (norms + 1e-8)

        if self.stub_mode:
            # Simple clustering for testing
            n_samples = len(embeddings)
            n_clusters = max(1, n_samples // self.min_cluster_size)
            labels = np.array([i % n_clusters for i in range(n_samples)])
            noise_mask = np.random.rand(n_samples) < 0.1
            labels[noise_mask] = -1
        else:
            # Initialize clusterer if needed
            if self._clusterer is None:
                self._init_clusterer()

            if self._clusterer is None:
                # Fallback to simple KMeans
                from sklearn.cluster import KMeans
                n_clusters = max(1, len(embeddings) // self.min_cluster_size)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(embeddings_normalized)
            else:
                labels = self._clusterer.fit_predict(embeddings_normalized)

        # Build clusters
        clusters = {}
        noise_images = []

        unique_labels = set(labels)
        for label in unique_labels:
            mask = labels == label
            cluster_paths = [image_paths[i] for i in range(len(image_paths)) if mask[i]]
            cluster_embeddings = embeddings[mask]

            if label == -1:
                noise_images = cluster_paths
            else:
                centroid = cluster_embeddings.mean(axis=0)
                clusters[label] = IdentityCluster(
                    cluster_id=label,
                    name=f"character_{label}",
                    image_paths=cluster_paths,
                    embeddings=cluster_embeddings,
                    centroid=centroid,
                    mean_confidence=0.9,  # Placeholder
                )

        processing_time = time.time() - start_time

        result = ClusteringResult(
            clusters=clusters,
            noise_images=noise_images,
            total_images=len(image_paths),
            total_faces=len(embeddings),
            processing_time=processing_time,
            parameters={
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "face_model": self.face_model_name,
                "embedding_model": self.embedding_model_name,
            },
        )

        logger.info(
            f"Clustering complete: {len(clusters)} clusters, "
            f"{len(noise_images)} noise images, {processing_time:.2f}s"
        )

        return result

    def export_clusters(
        self,
        result: ClusteringResult,
        output_dir: Union[str, Path],
        copy_images: bool = True,
        save_metadata: bool = True,
    ) -> Path:
        """
        Export clustering results to directory structure.

        Args:
            result: ClusteringResult from cluster_by_identity
            output_dir: Output directory
            copy_images: If True, copy images; if False, create symlinks
            save_metadata: If True, save cluster_report.json

        Returns:
            Path to output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export each cluster
        for cluster_id, cluster in result.clusters.items():
            cluster_dir = output_dir / cluster.name
            cluster_dir.mkdir(exist_ok=True)

            for img_path in cluster.image_paths:
                dest = cluster_dir / img_path.name
                if dest.exists():
                    # Add suffix for duplicates
                    stem = img_path.stem
                    suffix = img_path.suffix
                    i = 1
                    while dest.exists():
                        dest = cluster_dir / f"{stem}_{i}{suffix}"
                        i += 1

                if copy_images:
                    shutil.copy2(img_path, dest)
                else:
                    dest.symlink_to(img_path.resolve())

        # Export noise
        if result.noise_images:
            noise_dir = output_dir / "noise"
            noise_dir.mkdir(exist_ok=True)
            for img_path in result.noise_images:
                dest = noise_dir / img_path.name
                if copy_images:
                    shutil.copy2(img_path, dest)
                else:
                    dest.symlink_to(img_path.resolve())

        # Save metadata
        if save_metadata:
            metadata = result.to_dict()
            metadata["clusters_detail"] = {
                k: {
                    "name": v.name,
                    "size": v.size,
                    "images": [str(p) for p in v.image_paths],
                }
                for k, v in result.clusters.items()
            }
            metadata["noise_images"] = [str(p) for p in result.noise_images]

            report_path = output_dir / "cluster_report.json"
            with open(report_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved cluster report to {report_path}")

        logger.info(f"Exported {len(result.clusters)} clusters to {output_dir}")
        return output_dir

    def cluster_images(
        self,
        images_dir: Union[str, Path],
        output_dir: Union[str, Path],
        copy_images: bool = True,
    ) -> ClusteringResult:
        """
        Complete pipeline: extract embeddings, cluster, and export.

        Args:
            images_dir: Directory containing images
            output_dir: Output directory for clustered images
            copy_images: If True, copy images; if False, create symlinks

        Returns:
            ClusteringResult
        """
        # Extract embeddings
        embeddings, paths = self.extract_face_embeddings(images_dir)

        if len(embeddings) == 0:
            logger.warning("No faces found, cannot cluster")
            return ClusteringResult(
                clusters={},
                noise_images=[],
                total_images=0,
                total_faces=0,
                processing_time=0,
                parameters={},
            )

        # Cluster
        result = self.cluster_by_identity(embeddings, paths)

        # Export
        self.export_clusters(result, output_dir, copy_images=copy_images)

        return result

    @staticmethod
    def _compute_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Compute intersection over union of two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


def merge_clusters(
    result: ClusteringResult,
    cluster_ids: List[int],
    new_name: str,
) -> ClusteringResult:
    """
    Merge multiple clusters into one.

    Args:
        result: Original clustering result
        cluster_ids: List of cluster IDs to merge
        new_name: Name for the merged cluster

    Returns:
        New ClusteringResult with merged cluster
    """
    # Collect all images from clusters to merge
    merged_paths = []
    merged_embeddings = []

    new_clusters = {}
    new_id = max(result.clusters.keys()) + 1 if result.clusters else 0

    for cid, cluster in result.clusters.items():
        if cid in cluster_ids:
            merged_paths.extend(cluster.image_paths)
            merged_embeddings.append(cluster.embeddings)
        else:
            new_clusters[cid] = cluster

    if merged_paths:
        merged_embeddings = np.vstack(merged_embeddings)
        centroid = merged_embeddings.mean(axis=0)

        new_clusters[new_id] = IdentityCluster(
            cluster_id=new_id,
            name=new_name,
            image_paths=merged_paths,
            embeddings=merged_embeddings,
            centroid=centroid,
            mean_confidence=0.9,
        )

    return ClusteringResult(
        clusters=new_clusters,
        noise_images=result.noise_images,
        total_images=result.total_images,
        total_faces=result.total_faces,
        processing_time=result.processing_time,
        parameters=result.parameters,
    )


def split_cluster(
    result: ClusteringResult,
    cluster_id: int,
    image_groups: List[List[Path]],
    new_names: List[str],
) -> ClusteringResult:
    """
    Split a cluster into multiple clusters.

    Args:
        result: Original clustering result
        cluster_id: Cluster to split
        image_groups: List of lists of image paths for each new cluster
        new_names: Names for each new cluster

    Returns:
        New ClusteringResult with split clusters
    """
    if cluster_id not in result.clusters:
        return result

    original = result.clusters[cluster_id]
    new_clusters = {k: v for k, v in result.clusters.items() if k != cluster_id}

    max_id = max(result.clusters.keys()) if result.clusters else -1

    for i, (paths, name) in enumerate(zip(image_groups, new_names)):
        new_id = max_id + 1 + i

        # Find embeddings for these paths
        path_set = set(paths)
        mask = [p in path_set for p in original.image_paths]
        cluster_embeddings = original.embeddings[mask]

        if len(cluster_embeddings) > 0:
            centroid = cluster_embeddings.mean(axis=0)
        else:
            centroid = np.zeros(512)

        new_clusters[new_id] = IdentityCluster(
            cluster_id=new_id,
            name=name,
            image_paths=list(paths),
            embeddings=cluster_embeddings,
            centroid=centroid,
            mean_confidence=original.mean_confidence,
        )

    return ClusteringResult(
        clusters=new_clusters,
        noise_images=result.noise_images,
        total_images=result.total_images,
        total_faces=result.total_faces,
        processing_time=result.processing_time,
        parameters=result.parameters,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster images by face identity"
    )
    parser.add_argument(
        "images_dir",
        help="Directory containing images to cluster"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for clusters"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=20,
        help="Minimum cluster size (default: 20 for 2D animation)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum samples for core points (default: 3)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "stub"],
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=48,
        help="Minimum face size in pixels"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying images"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    clusterer = FaceIdentityClusterer(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        device=args.device,
        min_face_size=args.min_face_size,
    )

    result = clusterer.cluster_images(
        args.images_dir,
        args.output_dir,
        copy_images=not args.symlink,
    )

    print(f"\nClustering Results:")
    print(f"  Total images: {result.total_images}")
    print(f"  Total faces: {result.total_faces}")
    print(f"  Clusters: {len(result.clusters)}")
    for cid, cluster in result.clusters.items():
        print(f"    {cluster.name}: {cluster.size} images")
    print(f"  Noise: {len(result.noise_images)} images")
    print(f"  Time: {result.processing_time:.2f}s")
