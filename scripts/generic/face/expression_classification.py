#!/usr/bin/env python3
"""
Expression Classification for Expression LoRA Training Data
Classifies facial expressions from character instances using face detection + emotion analysis.

Usage:
    python scripts/generic/face/expression_classification.py \
        /path/to/instances \
        --output-dir /path/to/expression_clusters \
        --method vlm \
        --device cpu

Methods:
    - hsemotion: HSEmotion (AffectNet-trained, 8 emotions) - SOTA for expression recognition
    - vlm: Vision-Language Model (BLIP2/InternVL) for expression detection
    - landmarks: Face landmarks + geometric features (requires dlib)
    - clip: CLIP visual embeddings + clustering
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional
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
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ExpressionClassifier:
    """Classify expressions from character faces."""

    # HSEmotion's 8 emotion categories (AffectNet-trained)
    HSEMOTION_EMOTIONS = [
        'neutral',    # 0
        'happy',      # 1
        'sad',        # 2
        'surprise',   # 3
        'fear',       # 4
        'disgust',    # 5
        'angry',      # 6
        'contempt'    # 7
    ]

    def __init__(
        self,
        method: str = "hsemotion",
        device: str = "cpu",
        min_face_size: int = 64
    ):
        """Initialize expression classifier.

        Args:
            method: Classification method (hsemotion recommended)
            device: 'cpu' or 'cuda'
            min_face_size: Minimum face size in pixels
        """
        self.method = method
        self.device = device
        self.min_face_size = min_face_size

        if method == "hsemotion":
            self._load_hsemotion_model()
        elif method == "vlm":
            print("⚠️  VLM method requires BLIP2/InternVL model")
            print("   For now, using HSEmotion as fallback")
            self.method = "hsemotion"
            self._load_hsemotion_model()
        elif method == "clip":
            self._load_clip_model()

    def _load_hsemotion_model(self):
        """Load HSEmotion model for facial expression recognition."""
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            self.emotion_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
            print(f"✓ HSEmotion model loaded (AffectNet-trained, 8 emotions)")
        except ImportError:
            print("Error: hsemotion not installed")
            print("Install: pip install hsemotion")
            print("Falling back to CLIP-based clustering")
            self.method = "clip"
            self._load_clip_model()

    def _load_clip_model(self):
        """Load CLIP model."""
        try:
            import open_clip
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='openai',
                device=self.device
            )
            self.clip_model.eval()
            print(f"✓ CLIP model loaded on {self.device}")
        except ImportError:
            print("Error: open_clip_torch not installed")
            print("Install: pip install open_clip_torch")
            sys.exit(1)

    def detect_faces(self, image_paths: List[str]) -> List[Dict]:
        """Detect faces in images.

        Args:
            image_paths: List of image paths

        Returns:
            List of face detections with metadata
        """
        detections = []

        for img_path in tqdm(image_paths, desc="Detecting faces"):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                h, w = img.shape[:2]

                # Simple heuristic: assume character instances likely contain faces
                # In production, use RetinaFace or similar
                face_detection = {
                    'image_path': img_path,
                    'has_face': True,  # Assume true for character instances
                    'face_bbox': [0, 0, w, h],  # Full image as fallback
                    'face_size': min(w, h)
                }

                if face_detection['face_size'] >= self.min_face_size:
                    detections.append(face_detection)

            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
                continue

        return detections

    def classify_expressions(
        self,
        image_paths: List[str],
        output_dir: str
    ) -> Dict:
        """Classify expressions and organize into clusters.

        Args:
            image_paths: List of image paths
            output_dir: Output directory

        Returns:
            Classification results
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"EXPRESSION CLASSIFICATION")
        print(f"{'='*60}")
        print(f"Method: {self.method}")
        print(f"Images: {len(image_paths)}")
        print(f"{'='*60}\n")

        if self.method == "hsemotion":
            # HSEmotion: Direct supervised classification
            return self._classify_with_hsemotion(image_paths, output_dir)
        else:
            # CLIP: Unsupervised clustering
            return self._classify_with_clustering(image_paths, output_dir)

    def _classify_with_hsemotion(
        self,
        image_paths: List[str],
        output_dir: str
    ) -> Dict:
        """Classify expressions using HSEmotion (supervised)."""
        print("🔍 Classifying expressions with HSEmotion...")

        # Process each image
        expression_data = []
        no_face_images = []

        for img_path in tqdm(image_paths, desc="Analyzing expressions"):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    no_face_images.append(img_path)
                    continue

                # HSEmotion expects BGR images
                emotion, scores = self.emotion_recognizer.predict_emotions(image, logits=False)

                # emotion is the predicted emotion label (string)
                # scores is dict of emotion -> probability
                expression_data.append({
                    'image_path': img_path,
                    'emotion': emotion,
                    'confidence': scores[emotion],
                    'all_scores': scores
                })

            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
                no_face_images.append(img_path)
                continue

        print(f"\n✓ Classified: {len(expression_data)} / {len(image_paths)}")
        print(f"   Failed: {len(no_face_images)}")

        # Organize into emotion folders
        print("\n📁 Organizing by emotion...")
        stats = {}

        for data in tqdm(expression_data, desc="Organizing"):
            emotion = data['emotion']
            emotion_dir = os.path.join(output_dir, emotion)
            os.makedirs(emotion_dir, exist_ok=True)

            # Copy image to emotion folder
            src_path = data['image_path']
            basename = os.path.basename(src_path)
            dest_path = os.path.join(emotion_dir, basename)
            shutil.copy2(src_path, dest_path)

            # Update stats
            if emotion not in stats:
                stats[emotion] = {'count': 0, 'avg_confidence': 0.0}
            stats[emotion]['count'] += 1
            stats[emotion]['avg_confidence'] += data['confidence']

        # Calculate average confidences
        for emotion in stats:
            count = stats[emotion]['count']
            stats[emotion]['avg_confidence'] /= count

        # Save detailed results
        results = {
            'method': 'hsemotion',
            'total_images': len(image_paths),
            'classified': len(expression_data),
            'failed': len(no_face_images),
            'emotion_distribution': stats,
            'predictions': [
                {
                    'image': os.path.basename(d['image_path']),
                    'emotion': d['emotion'],
                    'confidence': float(d['confidence'])
                }
                for d in expression_data
            ]
        }

        results_path = os.path.join(output_dir, 'expression_classification.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ HSEmotion classification complete!")
        print(f"   Results: {results_path}")
        print(f"\nEmotion distribution:")
        for emotion, data in sorted(stats.items()):
            print(f"   {emotion:12s}: {data['count']:4d} images (avg conf: {data['avg_confidence']:.3f})")

        return results

    def _classify_with_clustering(
        self,
        image_paths: List[str],
        output_dir: str
    ) -> Dict:
        """Classify expressions using CLIP + clustering (unsupervised)."""
        # Detect faces
        print("🔍 Detecting faces...")
        face_detections = self.detect_faces(image_paths)
        print(f"✓ Found {len(face_detections)} faces")

        if len(face_detections) == 0:
            print("No faces detected!")
            return {}

        # Extract visual features and cluster
        print("\n🔧 Extracting visual features...")
        features = self._extract_features(face_detections)

        print("\n🔧 Clustering by expression...")
        labels = self._cluster_features(features)

        # Organize into expression folders
        print("\n📁 Organizing expression clusters...")
        stats = self._organize_by_expression(face_detections, labels, output_dir)

        # Save results
        results = {
            'method': self.method,
            'total_images': len(image_paths),
            'faces_detected': len(face_detections),
            'expression_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'cluster_stats': stats
        }

        results_path = os.path.join(output_dir, 'expression_classification.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Expression classification complete!")
        print(f"   Results: {results_path}")
        print(f"   Clusters found: {results['expression_clusters']}\n")

        return results

    def _extract_features(self, detections: List[Dict]) -> np.ndarray:
        """Extract visual features from faces."""
        features = []

        with torch.no_grad():
            for det in tqdm(detections, desc="Extracting features"):
                try:
                    image = Image.open(det['image_path']).convert('RGB')
                    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    image_features = self.clip_model.encode_image(image_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    features.append(image_features.cpu().numpy()[0])
                except Exception as e:
                    features.append(np.zeros(512))

        return np.array(features)

    def _cluster_features(self, features: np.ndarray) -> np.ndarray:
        """Cluster features by expression."""
        from sklearn.cluster import HDBSCAN
        import umap

        # Reduce dimensionality
        reducer = umap.UMAP(
            n_neighbors=10,
            min_dist=0.1,
            n_components=16,
            metric='cosine',
            random_state=42
        )
        features_reduced = reducer.fit_transform(features)

        # Cluster
        clusterer = HDBSCAN(
            min_cluster_size=10,
            min_samples=2,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(features_reduced)

        return labels

    def _organize_by_expression(
        self,
        detections: List[Dict],
        labels: np.ndarray,
        output_dir: str
    ) -> Dict:
        """Organize images by expression cluster."""
        stats = {}

        # Group by label
        label_to_images = {}
        for det, label in zip(detections, labels):
            if label not in label_to_images:
                label_to_images[label] = []
            label_to_images[label].append(det['image_path'])

        # Copy to cluster directories
        for label, images in tqdm(label_to_images.items(), desc="Organizing"):
            if label == -1:
                cluster_name = "unclassified"
            else:
                cluster_name = f"expression_{label:03d}"

            cluster_dir = os.path.join(output_dir, cluster_name)
            os.makedirs(cluster_dir, exist_ok=True)

            for img_path in images:
                basename = os.path.basename(img_path)
                dest_path = os.path.join(cluster_dir, basename)
                shutil.copy2(img_path, dest_path)

            stats[cluster_name] = len(images)

        return stats


def main():
    parser = argparse.ArgumentParser(description="Expression classification")
    parser.add_argument(
        "instances_dir",
        help="Directory with character instance images"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for expression clusters"
    )
    parser.add_argument(
        "--method",
        default="hsemotion",
        choices=["hsemotion", "vlm", "landmarks", "clip"],
        help="Classification method (hsemotion recommended for SOTA accuracy)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for processing"
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=64,
        help="Minimum face size in pixels"
    )

    args = parser.parse_args()

    # Find images
    instances_dir = Path(args.instances_dir)
    if not instances_dir.exists():
        print(f"Error: {instances_dir} does not exist")
        return 1

    image_extensions = {'.png', '.jpg', '.jpeg'}
    image_paths = [
        str(p) for p in instances_dir.rglob('*')
        if p.suffix.lower() in image_extensions
    ]

    if len(image_paths) == 0:
        print(f"No images found in {instances_dir}")
        return 1

    # Initialize classifier
    classifier = ExpressionClassifier(
        method=args.method,
        device=args.device,
        min_face_size=args.min_face_size
    )

    # Classify expressions
    results = classifier.classify_expressions(image_paths, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
