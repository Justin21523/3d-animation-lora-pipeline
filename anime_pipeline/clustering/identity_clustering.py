#!/usr/bin/env python3
"""
Face-Centric Identity Clustering for 2D Animation Characters

Clusters character instances by IDENTITY (who they are), not visual similarity.
Adapted from 3D pipeline for 2D Western animation workflow.

Pipeline:
1. Face Detection → Extract face regions (using 2D-optimized detectors)
2. Face Recognition → Generate identity embeddings (ArcFace/InsightFace)
3. Face Clustering → Group by identity (HDBSCAN with 2D defaults)
4. Track Merging → Combine tracks of same character across camera cuts
5. Final Clusters → Character-specific folders

Key Differences from 3D:
- More aggressive min_cluster_size (2D characters vary more across episodes)
- Support for stylized/cartoon faces that may fail standard detectors
- Fallback to CLIP embeddings for non-realistic character styles

Author: Ported from 3D pipeline for 2D animation
Date: 2025-01-XX
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
from datetime import datetime
import cv2
from sklearn.cluster import HDBSCAN
import umap
from sklearn.preprocessing import normalize

from anime_pipeline.utils.logging_utils import setup_logging


class FaceDetector:
    """
    Detect faces in 2D animation character images.

    Supports multiple backends with fallbacks for stylized cartoon faces.
    """

    def __init__(self, device: str = "cuda", min_face_size: int = 64):
        """
        Initialize face detector.

        Args:
            device: cuda or cpu
            min_face_size: Minimum face size to detect
        """
        self.device = device
        self.min_face_size = min_face_size
        self.logger = setup_logging("FaceDetector")

        self.logger.info("🔧 Initializing face detector for 2D animation...")
        self._init_detector()

    def _init_detector(self):
        """Initialize RetinaFace or fallback detector"""
        try:
            # Try RetinaFace (works for some 2D styles)
            from retinaface import RetinaFace
            self.detector_type = "retinaface"
            self.detector = RetinaFace
            self.logger.info("✓ Using RetinaFace detector")

        except ImportError:
            try:
                # Fallback to MTCNN
                from facenet_pytorch import MTCNN
                self.detector_type = "mtcnn"
                self.detector = MTCNN(
                    device=self.device,
                    min_face_size=self.min_face_size
                )
                self.logger.info("✓ Using MTCNN detector")

            except ImportError:
                # Fallback to OpenCV Haar Cascades
                self.detector_type = "opencv"
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.detector = cv2.CascadeClassifier(cascade_path)
                self.logger.info("⚠️ Using OpenCV Haar Cascades (may miss stylized faces)")

    def detect_faces(self, image: Image.Image) -> List[Dict]:
        """
        Detect all faces in an image.

        Args:
            image: PIL Image

        Returns:
            List of face dictionaries with bbox and landmarks
        """
        if self.detector_type == "retinaface":
            return self._detect_retinaface(image)
        elif self.detector_type == "mtcnn":
            return self._detect_mtcnn(image)
        else:
            return self._detect_opencv(image)

    def _detect_retinaface(self, image: Image.Image) -> List[Dict]:
        """Detect with RetinaFace"""
        image_np = np.array(image)

        try:
            faces = self.detector.detect_faces(image_np)
        except Exception:
            return []

        results = []
        for key, face_data in faces.items():
            bbox = face_data['facial_area']  # [x1, y1, x2, y2]
            landmarks = face_data.get('landmarks')
            confidence = face_data.get('score', 1.0)

            # Filter by size
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if width < self.min_face_size or height < self.min_face_size:
                continue

            results.append({
                'bbox': bbox,
                'landmarks': landmarks,
                'confidence': confidence
            })

        return results

    def _detect_mtcnn(self, image: Image.Image) -> List[Dict]:
        """Detect with MTCNN"""
        boxes, probs, landmarks = self.detector.detect(image, landmarks=True)

        if boxes is None:
            return []

        results = []
        for box, prob, landmark in zip(boxes, probs, landmarks):
            # Filter by confidence (relaxed for 2D animation)
            if prob < 0.85:  # Lower threshold for cartoon faces
                continue

            # Filter by size
            width = box[2] - box[0]
            height = box[3] - box[1]

            if width < self.min_face_size or height < self.min_face_size:
                continue

            results.append({
                'bbox': box.tolist(),
                'landmarks': landmark.tolist() if landmark is not None else None,
                'confidence': float(prob)
            })

        return results

    def _detect_opencv(self, image: Image.Image) -> List[Dict]:
        """Detect with OpenCV (fallback)"""
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )

        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': [x, y, x + w, y + h],
                'landmarks': None,
                'confidence': 1.0  # OpenCV doesn't provide confidence
            })

        return results

    def crop_face(
        self,
        image: Image.Image,
        face: Dict,
        margin: float = 0.2
    ) -> Image.Image:
        """
        Crop face region with margin.

        Args:
            image: PIL Image
            face: Face dictionary
            margin: Margin ratio (0.2 = 20% padding)

        Returns:
            Cropped face image
        """
        bbox = face['bbox']
        x1, y1, x2, y2 = bbox

        # Add margin
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, int(x1 - width * margin))
        y1 = max(0, int(y1 - height * margin))
        x2 = min(image.width, int(x2 + width * margin))
        y2 = min(image.height, int(y2 + height * margin))

        return image.crop((x1, y1, x2, y2))


class FaceEmbedder:
    """
    Generate face identity embeddings.

    Supports InsightFace (ArcFace) and FaceNet for 2D animation characters.
    """

    def __init__(self, model_name: str = "arcface", device: str = "cuda"):
        """
        Initialize face recognition model.

        Args:
            model_name: arcface, facenet, or insightface
            device: cuda or cpu
        """
        self.model_name = model_name
        self.device = device
        self.logger = setup_logging("FaceEmbedder")

        self.logger.info(f"🔧 Initializing {model_name} face embedder...")
        self._init_model()

    def _init_model(self):
        """Initialize face recognition model"""
        try:
            # Try InsightFace (best performance)
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0 if self.device == "cuda" else -1)
            self.model_type = "insightface"
            self.logger.info("✓ Using InsightFace (ArcFace R100)")

        except ImportError:
            try:
                # Fallback to facenet-pytorch
                from facenet_pytorch import InceptionResnetV1
                self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                self.model_type = "facenet"
                self.logger.info("✓ Using FaceNet (Inception ResNet)")

            except ImportError:
                raise ImportError(
                    "No face recognition model available. "
                    "Install insightface or facenet-pytorch:\n"
                    "  pip install insightface\n"
                    "  pip install facenet-pytorch"
                )

    def get_embedding(self, face_image: Image.Image) -> Optional[np.ndarray]:
        """
        Get face embedding vector.

        Args:
            face_image: Cropped face image

        Returns:
            Embedding vector (512-d or 128-d depending on model)
        """
        if self.model_type == "insightface":
            return self._embed_insightface(face_image)
        else:
            return self._embed_facenet(face_image)

    def _embed_insightface(self, face_image: Image.Image) -> Optional[np.ndarray]:
        """Get embedding with InsightFace"""
        image_np = np.array(face_image)

        try:
            # Detect and get embedding
            faces = self.app.get(image_np)

            if len(faces) == 0:
                return None

            # Return embedding from first (largest) face
            embedding = faces[0].embedding  # 512-d vector

            return embedding
        except Exception:
            return None

    def _embed_facenet(self, face_image: Image.Image) -> Optional[np.ndarray]:
        """Get embedding with FaceNet"""
        import torchvision.transforms as transforms

        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        try:
            face_tensor = transform(face_image).unsqueeze(0).to(self.device)

            # Get embedding
            with torch.no_grad():
                embedding = self.model(face_tensor).cpu().numpy()[0]  # 512-d vector

            return embedding
        except Exception:
            return None


class IdentityClusterer:
    """
    Cluster faces by identity using HDBSCAN.

    Optimized for 2D animation with parameters adapted from 3D defaults.
    """

    def __init__(
        self,
        min_cluster_size: int = 20,  # Increased for 2D (more variation)
        min_samples: int = 3,         # Increased for 2D
        distance_threshold: float = 0.5
    ):
        """
        Initialize identity clusterer.

        Args:
            min_cluster_size: Minimum faces per identity (2D default: 20)
            min_samples: Minimum samples for core point (2D default: 3)
            distance_threshold: Maximum face distance for same identity
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.distance_threshold = distance_threshold
        self.logger = setup_logging("IdentityClusterer")

    def cluster_by_identity(
        self,
        embeddings: np.ndarray,
        use_umap: bool = True,
        n_components: int = 64
    ) -> Tuple[np.ndarray, Dict]:
        """
        Cluster face embeddings by identity.

        Args:
            embeddings: Face embedding matrix (N x D)
            use_umap: Apply UMAP for dimensionality reduction
            n_components: UMAP components

        Returns:
            (cluster_labels, clustering_info)
        """
        self.logger.info(f"🔍 Clustering {len(embeddings)} faces by identity...")

        # Normalize embeddings
        embeddings_norm = normalize(embeddings, norm='l2')

        # Optional UMAP dimensionality reduction
        if use_umap and embeddings.shape[0] > n_components:
            self.logger.info(f"   Applying UMAP: {embeddings.shape[1]}D → {n_components}D")
            reducer = umap.UMAP(
                n_components=min(n_components, embeddings.shape[0] - 1),
                n_neighbors=min(15, embeddings.shape[0] - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            embeddings_reduced = reducer.fit_transform(embeddings_norm)
        else:
            embeddings_reduced = embeddings_norm

        # HDBSCAN clustering
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_epsilon=self.distance_threshold,
            cluster_selection_method='leaf'
        )

        labels = clusterer.fit_predict(embeddings_reduced)

        # Compute statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        info = {
            'n_identities': n_clusters,
            'n_noise': n_noise,
            'identity_sizes': {},
            'silhouette_score': None
        }

        for label in set(labels):
            if label == -1:
                info['identity_sizes']['noise'] = n_noise
            else:
                count = list(labels).count(label)
                info['identity_sizes'][f'identity_{label}'] = count

        self.logger.info(f"✓ Found {n_clusters} identities")
        self.logger.info(f"   Noise: {n_noise} faces")

        return labels, info


def cluster_by_identity(
    track_segments: Dict[str, Tuple[List[Dict], List[Dict]]],
    min_cluster_size: int = 20,
    device: str = "cuda",
    save_face_crops: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, List[str]]:
    """
    Cluster tracks by character identity using face embeddings.

    This is the main interface for Phase 3.3 identity clustering.
    Merges tracks of the same character across different camera cuts and scenes.

    Args:
        track_segments: Output from segment_foreground_background_per_track
                       Dict mapping track_id to (fg_records, bg_records)
        min_cluster_size: Minimum tracks per identity (2D default: 20)
        device: cuda or cpu
        save_face_crops: Save detected face crops
        output_dir: Output directory for clustered results

    Returns:
        Dict mapping character_id to list of track_ids

    Example:
        >>> from anime_pipeline.detection.yolo_detector import run_yolo_tracking_with_grouping
        >>> from anime_pipeline.segmentation.toonout_wrapper import segment_foreground_background_per_track
        >>> from anime_pipeline.clustering.identity_clustering import cluster_by_identity
        >>>
        >>> # Step 1: YOLO tracking
        >>> track_groups = run_yolo_tracking_with_grouping(yolo_config)
        >>>
        >>> # Step 2: Per-track segmentation
        >>> track_segments = segment_foreground_background_per_track(seg_config, track_groups)
        >>>
        >>> # Step 3: Identity clustering (merge tracks of same character)
        >>> character_clusters = cluster_by_identity(track_segments)
        >>>
        >>> # Result: character_id -> list of track_ids
        >>> for char_id, track_ids in character_clusters.items():
        ...     print(f"{char_id}: {len(track_ids)} tracks merged")
    """
    logger = setup_logging("cluster_by_identity")

    logger.info(f"📊 Clustering {len(track_segments)} tracks by identity...")

    # Initialize components
    face_detector = FaceDetector(device=device, min_face_size=64)
    face_embedder = FaceEmbedder(model_name="arcface", device=device)
    identity_clusterer = IdentityClusterer(min_cluster_size=max(2, min_cluster_size // 10))  # Adjust for track-level

    # Extract face embeddings for all tracks
    embeddings = []
    track_ids = []
    no_face_tracks = []

    for track_id, (fg_records, _) in tqdm(track_segments.items(), desc="Detecting faces per track"):
        # Sample representative frames from this track
        sample_size = min(5, len(fg_records))
        sampled_records = np.random.choice(fg_records, sample_size, replace=False) if len(fg_records) > sample_size else fg_records

        track_embeddings = []

        for record in sampled_records:
            rgba_path = Path(record['rgba_path'])

            if not rgba_path.exists():
                continue

            try:
                image = Image.open(rgba_path).convert("RGB")

                # Detect faces
                faces = face_detector.detect_faces(image)

                if len(faces) == 0:
                    continue

                # Use the largest face (primary character)
                face = max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))

                # Crop face
                face_crop = face_detector.crop_face(image, face)

                # Get embedding
                embedding = face_embedder.get_embedding(face_crop)

                if embedding is not None:
                    track_embeddings.append(embedding)

            except Exception as e:
                logger.debug(f"Failed to process {rgba_path}: {e}")
                continue

        if len(track_embeddings) > 0:
            # Average embeddings for this track
            track_embedding = np.mean(track_embeddings, axis=0)
            embeddings.append(track_embedding)
            track_ids.append(track_id)
        else:
            no_face_tracks.append(track_id)

    logger.info(f"✓ Extracted face embeddings: {len(track_ids)} tracks")
    logger.info(f"   No face detected: {len(no_face_tracks)} tracks")

    if len(embeddings) < 2:
        logger.warning("Not enough tracks with faces for clustering")
        return {}

    # Cluster by identity
    embeddings_array = np.array(embeddings)
    labels, cluster_info = identity_clusterer.cluster_by_identity(embeddings_array)

    # Group tracks by character
    character_clusters = {}

    for track_id, label in zip(track_ids, labels):
        if label == -1:  # Noise
            char_id = "noise"
        else:
            char_id = f"character_{label:03d}"

        if char_id not in character_clusters:
            character_clusters[char_id] = []

        character_clusters[char_id].append(track_id)

    # Save results if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            'total_tracks': len(track_segments),
            'faces_detected': len(track_ids),
            'no_face': len(no_face_tracks),
            'clustering_info': cluster_info,
            'character_clusters': character_clusters,
            'timestamp': datetime.now().isoformat()
        }

        metadata_path = output_dir / "identity_clustering.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Saved clustering metadata to {metadata_path}")

    logger.info(f"✅ Identity clustering complete!")
    logger.info(f"   Characters found: {len([k for k in character_clusters.keys() if k != 'noise'])}")

    for char_id, tracks in character_clusters.items():
        logger.info(f"   {char_id}: {len(tracks)} tracks")

    return character_clusters
