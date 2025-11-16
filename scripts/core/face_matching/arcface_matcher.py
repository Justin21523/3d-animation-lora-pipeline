"""
ArcFace-based Face Matching Utility

Uses InsightFace ArcFace model for face recognition and matching
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import logging


class ArcFaceMatcher:
    """Face matching using InsightFace ArcFace embeddings"""

    def __init__(
        self,
        model_name: str = 'buffalo_l',
        device: str = 'cuda',
        similarity_threshold: float = 0.35
    ):
        """
        Initialize ArcFace matcher

        Args:
            model_name: InsightFace model name
            device: Device to run on ('cuda' or 'cpu')
            similarity_threshold: Cosine similarity threshold for matching
        """
        self.model_name = model_name
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)

        # Lazy load model
        self.app = None
        self.reference_embeddings = []
        self.reference_paths = []

    def _load_model(self):
        """Lazy load InsightFace model"""
        if self.app is None:
            try:
                from insightface.app import FaceAnalysis
                self.app = FaceAnalysis(
                    name=self.model_name,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.app.prepare(ctx_id=0 if self.device == 'cuda' else -1)
                self.logger.info(f"Loaded InsightFace model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load InsightFace model: {e}")
                raise

    def build_reference_embeddings(
        self,
        reference_dir: Path,
        max_references: Optional[int] = None
    ) -> int:
        """
        Build reference embeddings from directory of face images

        Args:
            reference_dir: Directory containing reference face images
            max_references: Maximum number of reference images to use

        Returns:
            Number of successfully processed reference images
        """
        self._load_model()

        reference_dir = Path(reference_dir)
        image_paths = sorted(
            list(reference_dir.glob('*.jpg')) +
            list(reference_dir.glob('*.png')) +
            list(reference_dir.glob('*.jpeg'))
        )

        if max_references:
            image_paths = image_paths[:max_references]

        self.logger.info(
            f"Building reference embeddings from {len(image_paths)} images..."
        )

        successful = 0
        for img_path in image_paths:
            try:
                embedding = self.extract_embedding(img_path)
                if embedding is not None:
                    self.reference_embeddings.append(embedding)
                    self.reference_paths.append(img_path)
                    successful += 1
            except Exception as e:
                self.logger.warning(f"Failed to process {img_path}: {e}")
                continue

        self.logger.info(
            f"Successfully built {successful}/{len(image_paths)} reference embeddings"
        )

        return successful

    def extract_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Extract face embedding from image

        Args:
            image_path: Path to image file

        Returns:
            Face embedding array or None if no face detected
        """
        self._load_model()

        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)

            # Detect faces
            faces = self.app.get(img_array)

            if len(faces) == 0:
                return None

            # Use largest face
            faces = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)
            face = faces[0]

            # Return normalized embedding
            embedding = face.normed_embedding
            return embedding

        except Exception as e:
            self.logger.warning(f"Failed to extract embedding from {image_path}: {e}")
            return None

    def match_face(
        self,
        image_path: Path,
        return_similarity: bool = False
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if face in image matches any reference face

        Args:
            image_path: Path to image to check
            return_similarity: If True, return best similarity score

        Returns:
            Tuple of (is_match, similarity_score)
        """
        if not self.reference_embeddings:
            raise ValueError("No reference embeddings loaded. Call build_reference_embeddings first.")

        embedding = self.extract_embedding(image_path)

        if embedding is None:
            return (False, 0.0) if return_similarity else (False, None)

        # Compute cosine similarity with all references
        similarities = [
            np.dot(embedding, ref_emb)
            for ref_emb in self.reference_embeddings
        ]

        max_similarity = max(similarities)
        is_match = max_similarity >= self.similarity_threshold

        if return_similarity:
            return is_match, max_similarity
        else:
            return is_match, None

    def batch_match_faces(
        self,
        image_paths: List[Path],
        return_similarities: bool = False
    ) -> List[Dict]:
        """
        Match multiple faces against reference embeddings

        Args:
            image_paths: List of image paths to check
            return_similarities: If True, include similarity scores

        Returns:
            List of match results with metadata
        """
        results = []

        for img_path in image_paths:
            is_match, similarity = self.match_face(img_path, return_similarity=True)

            result = {
                'image_path': str(img_path),
                'is_match': is_match,
            }

            if return_similarities:
                result['similarity'] = float(similarity) if similarity else 0.0

            results.append(result)

        return results

    def get_statistics(self) -> Dict:
        """Get matcher statistics"""
        return {
            'num_references': len(self.reference_embeddings),
            'similarity_threshold': self.similarity_threshold,
            'model_name': self.model_name,
            'device': self.device
        }
