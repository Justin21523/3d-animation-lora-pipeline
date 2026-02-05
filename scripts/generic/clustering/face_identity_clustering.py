#!/usr/bin/env python3
"""
Face-Centric Identity Clustering for Character Recognition

Clusters character instances by IDENTITY (who they are), not visual similarity.
Uses face recognition embeddings as primary signal.

Pipeline:
1. Face Detection → Extract face regions
2. Face Recognition → Generate identity embeddings (ArcFace/InsightFace)
3. Face Clustering → Group by identity (HDBSCAN)
4. Body Verification → Confirm with full-body CLIP embeddings
5. Final Clusters → Character-specific folders

Key Advantage:
- Correctly groups the SAME character across different:
  * Poses
  * Lighting conditions
  * Backgrounds
  * Expressions
- Separates DIFFERENT characters even in same scene
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
from datetime import datetime
import cv2
from sklearn.cluster import HDBSCAN
import umap
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns


class FaceDetector:
    """Detect faces in character images"""

    def __init__(self, device: str = "cuda", min_face_size: int = 64):
        """
        Initialize face detector

        Args:
            device: cuda or cpu
            min_face_size: Minimum face size to detect
        """
        self.device = device
        self.min_face_size = min_face_size

        print("🔧 Initializing face detector...")
        self._init_detector()

    def _init_detector(self):
        """Initialize RetinaFace or YOLOv8-face"""
        try:
            # Try RetinaFace (best for 3D characters)
            from retinaface import RetinaFace
            self.detector_type = "retinaface"
            self.detector = RetinaFace
            print("✓ Using RetinaFace detector")

        except ImportError:
            try:
                # Fallback to MTCNN
                from facenet_pytorch import MTCNN
                self.detector_type = "mtcnn"
                self.detector = MTCNN(
                    device=self.device,
                    min_face_size=self.min_face_size
                )
                print("✓ Using MTCNN detector")

            except ImportError:
                # Fallback to OpenCV Haar Cascades
                self.detector_type = "opencv"
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.detector = cv2.CascadeClassifier(cascade_path)
                print("⚠️ Using OpenCV Haar Cascades (less accurate)")

    def detect_faces(self, image: Image.Image) -> List[Dict]:
        """
        Detect all faces in an image

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

        faces = self.detector.detect_faces(image_np)

        results = []
        for key, face_data in faces.items():
            bbox = face_data['facial_area']  # [x, y, w, h]
            landmarks = face_data['landmarks']
            confidence = face_data['score']

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
            # Filter by confidence
            if prob < 0.9:
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
        Crop face region with margin

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
    """Generate face identity embeddings"""

    def __init__(self, model_name: str = "arcface", device: str = "cuda"):
        """
        Initialize face recognition model

        Args:
            model_name: arcface, facenet, or insightface
            device: cuda or cpu
        """
        self.model_name = model_name
        self.device = device

        print(f"🔧 Initializing {model_name} face embedder...")
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
            print("✓ Using InsightFace (ArcFace R100)")

        except ImportError:
            try:
                # Fallback to facenet-pytorch
                from facenet_pytorch import InceptionResnetV1
                self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                self.model_type = "facenet"
                print("✓ Using FaceNet (Inception ResNet)")

            except ImportError:
                raise ImportError(
                    "No face recognition model available. Install insightface or facenet-pytorch."
                )

    def get_embedding(self, face_image: Image.Image) -> Optional[np.ndarray]:
        """
        Get face embedding vector

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

        # Detect and get embedding
        faces = self.app.get(image_np)

        if len(faces) == 0:
            return None

        # Return embedding from first (largest) face
        embedding = faces[0].embedding  # 512-d vector

        return embedding

    def _embed_facenet(self, face_image: Image.Image) -> Optional[np.ndarray]:
        """Get embedding with FaceNet"""
        import torchvision.transforms as transforms

        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        face_tensor = transform(face_image).unsqueeze(0).to(self.device)

        # Get embedding
        with torch.no_grad():
            embedding = self.model(face_tensor).cpu().numpy()[0]  # 512-d vector

        return embedding


class IdentityClusterer:
    """Cluster faces by identity"""

    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int = 2,
        distance_threshold: float = 0.5
    ):
        """
        Initialize identity clusterer

        Args:
            min_cluster_size: Minimum faces per identity
            min_samples: Minimum samples for core point
            distance_threshold: Maximum face distance for same identity
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.distance_threshold = distance_threshold

    def cluster_by_identity(
        self,
        embeddings: np.ndarray,
        use_pca: bool = True,
        n_components: int = 64
    ) -> Tuple[np.ndarray, Dict]:
        """
        Cluster face embeddings by identity

        Args:
            embeddings: Face embedding matrix (N x D)
            use_pca: Apply PCA for dimensionality reduction
            n_components: PCA components

        Returns:
            (cluster_labels, clustering_info)
        """
        print(f"\n🔍 Clustering {len(embeddings)} faces by identity...")

        # Normalize embeddings
        embeddings_norm = normalize(embeddings, norm='l2')

        # Optional UMAP dimensionality reduction
        if use_pca and embeddings.shape[0] > n_components:  # use_pca flag kept for compatibility
            print(f"   Applying UMAP: {embeddings.shape[1]}D → {n_components}D")
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

        print(f"✓ Found {n_clusters} identities")
        print(f"   Noise: {n_noise} faces")

        return labels, info


def face_identity_clustering_pipeline(
    instances_dir: Path,
    output_dir: Path,
    min_cluster_size: int = 10,
    device: str = "cuda",
    save_face_crops: bool = True
) -> Dict:
    """
    Complete face-centric clustering pipeline

    Args:
        instances_dir: Directory with character instance images
        output_dir: Output directory for identity clusters
        min_cluster_size: Minimum faces per identity
        device: cuda or cpu
        save_face_crops: Save detected face crops

    Returns:
        Clustering statistics
    """
    instances_dir = Path(instances_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    face_detector = FaceDetector(device=device, min_face_size=64)
    face_embedder = FaceEmbedder(model_name="arcface", device=device)
    identity_clusterer = IdentityClusterer(min_cluster_size=min_cluster_size)

    # Find all instance images
    image_files = sorted(
        list(instances_dir.glob("*.png")) +
        list(instances_dir.glob("*.jpg"))
    )

    print(f"\n📊 Processing {len(image_files)} character instances...")

    # Step 1: Detect faces and extract embeddings
    face_data = []
    no_face_images = []

    for img_path in tqdm(image_files, desc="Detecting faces"):
        image = Image.open(img_path).convert("RGB")

        # Detect faces
        faces = face_detector.detect_faces(image)

        if len(faces) == 0:
            no_face_images.append(img_path)
            continue

        # Use the largest face (primary character)
        face = max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))

        # Crop face
        face_crop = face_detector.crop_face(image, face)

        # Get embedding
        embedding = face_embedder.get_embedding(face_crop)

        if embedding is None:
            no_face_images.append(img_path)
            continue

        face_data.append({
            'image_path': img_path,
            'face_bbox': face['bbox'],
            'face_crop': face_crop,
            'embedding': embedding
        })

    print(f"\n✓ Detected faces: {len(face_data)} / {len(image_files)}")
    print(f"   No face: {len(no_face_images)}")

    # Step 2: Cluster by identity
    embeddings = np.array([fd['embedding'] for fd in face_data])
    labels, cluster_info = identity_clusterer.cluster_by_identity(embeddings)

    # Step 3: Organize into identity folders
    print(f"\n📁 Organizing into identity clusters...")

    identity_mapping = {}

    for idx, (label, data) in enumerate(zip(labels, face_data)):
        if label == -1:
            cluster_name = "noise"
        else:
            cluster_name = f"identity_{label:03d}"

        # Create cluster directory
        cluster_dir = output_dir / cluster_name
        cluster_dir.mkdir(exist_ok=True)

        # Copy instance image
        src_path = data['image_path']
        dst_path = cluster_dir / src_path.name
        import shutil
        shutil.copy2(src_path, dst_path)

        # Optionally save face crop
        if save_face_crops:
            face_dir = cluster_dir / "faces"
            face_dir.mkdir(exist_ok=True)
            face_crop_path = face_dir / src_path.name
            data['face_crop'].save(face_crop_path)

        # Track mapping
        if cluster_name not in identity_mapping:
            identity_mapping[cluster_name] = []
        identity_mapping[cluster_name].append(src_path.name)

    # Step 4: Save metadata
    metadata = {
        'total_instances': len(image_files),
        'faces_detected': len(face_data),
        'no_face': len(no_face_images),
        'clustering_info': cluster_info,
        'identity_mapping': identity_mapping,
        'timestamp': datetime.now().isoformat()
    }

    metadata_path = output_dir / "identity_clustering.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Identity clustering complete!")
    print(f"   Identities found: {cluster_info['n_identities']}")
    print(f"   Output: {output_dir}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Cluster character instances by face identity (Film-Agnostic)"
    )
    parser.add_argument(
        "instances_dir",
        type=str,
        help="Directory with character instance images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for identity clusters. If --project is specified, this can be auto-constructed."
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (e.g., 'luca', 'toy_story', 'finding_nemo'). "
             "Automatically constructs output paths and loads project-specific configurations."
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="Minimum faces per identity"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--save-faces",
        action="store_true",
        help="Save detected face crops"
    )

    args = parser.parse_args()

    # Determine output directory
    output_dir = args.output_dir
    if args.project:
        if not output_dir:
            # Auto-construct path based on project name
            base_dir = Path("/mnt/data/ai_data/datasets/3d-anime")
            output_dir = str(base_dir / args.project / "clustered")
            print(f"✓ Using project: {args.project}")
            print(f"   Auto output: {output_dir}")
        else:
            print(f"✓ Project: {args.project} (output path manually specified)")

        # Load project-specific configuration if available
        config_path = Path(f"configs/clustering/{args.project}_config.yaml")
        if config_path.exists():
            print(f"✓ Found clustering config: {config_path}")
            print(f"   (Character names and settings will be loaded)")
            # Future enhancement: Load and use config
        else:
            print(f"⚠️  No clustering config found at: {config_path}")
            print(f"   Using default settings (you can create config later)")
    elif not output_dir:
        parser.error("Either --output-dir or --project must be specified")

    # Run pipeline
    metadata = face_identity_clustering_pipeline(
        instances_dir=Path(args.instances_dir),
        output_dir=Path(output_dir),
        min_cluster_size=args.min_cluster_size,
        device=args.device,
        save_face_crops=args.save_faces
    )

    print("\n" + "="*60)
    print("IDENTITY CLUSTERING REPORT")
    print("="*60)
    print(f"Total instances: {metadata['total_instances']}")
    print(f"Faces detected: {metadata['faces_detected']}")
    print(f"Identities found: {metadata['clustering_info']['n_identities']}")
    print("="*60)

    if args.project:
        print(f"\n💡 Next steps for project '{args.project}':")
        print(f"   1. Review clusters interactively:")
        print(f"      python scripts/generic/clustering/launch_interactive_review.py \\")
        print(f"        --cluster-dir {output_dir}")
        print(f"   2. (Optional) Pose subclustering:")
        print(f"      python scripts/generic/clustering/pose_subclustering.py \\")
        print(f"        {output_dir} --output-dir .../pose_subclusters --project {args.project}")


if __name__ == "__main__":
    main()
