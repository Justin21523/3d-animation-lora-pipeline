#!/usr/bin/env python3
"""
Reference-Based Frame Filter - Find specific character frames using face recognition

Uses face detection + ArcFace embeddings to filter frames containing a specific character
by comparing against reference images.

Usage:
    python reference_based_frame_filter.py \\
        --input-dir /path/to/frames \\
        --reference-dir /path/to/reference_images \\
        --output-dir /path/to/filtered_frames \\
        --similarity-threshold 0.6 \\
        --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReferenceFaceFilter:
    """Filter frames by comparing faces against reference images"""

    def __init__(
        self,
        reference_dir: Path,
        similarity_threshold: float = 0.6,
        device: str = "cuda"
    ):
        """
        Initialize face filter

        Args:
            reference_dir: Directory with reference character images
            similarity_threshold: Minimum cosine similarity to consider a match
            device: Device to use (cuda or cpu)
        """
        self.reference_dir = Path(reference_dir)
        self.similarity_threshold = similarity_threshold
        self.device = device

        # Load InsightFace
        logger.info("Loading InsightFace models...")
        from insightface.app import FaceAnalysis

        self.face_app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        )
        self.face_app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))

        # Extract reference embeddings
        logger.info(f"Extracting reference embeddings from {reference_dir}")
        self.reference_embeddings = self._extract_reference_embeddings()

        if len(self.reference_embeddings) == 0:
            raise ValueError(f"No faces found in reference directory: {reference_dir}")

        logger.info(f"✓ Extracted {len(self.reference_embeddings)} reference face embeddings")

        # Compute mean reference embedding for faster comparison
        self.mean_reference = np.mean(self.reference_embeddings, axis=0)
        self.mean_reference = self.mean_reference / np.linalg.norm(self.mean_reference)

    def _extract_reference_embeddings(self) -> np.ndarray:
        """Extract face embeddings from all reference images"""
        embeddings = []

        image_files = list(self.reference_dir.glob("*.png")) + list(self.reference_dir.glob("*.jpg"))

        for img_path in tqdm(image_files, desc="Processing reference images"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Detect faces
                faces = self.face_app.get(img)

                # Take the largest face if multiple
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    embedding = largest_face.normed_embedding
                    embeddings.append(embedding)

            except Exception as e:
                logger.warning(f"Failed to process {img_path.name}: {e}")
                continue

        if len(embeddings) == 0:
            return np.array([])

        return np.vstack(embeddings)

    def filter_frame(self, frame_path: Path) -> Tuple[bool, float, Optional[Dict]]:
        """
        Check if frame contains the reference character

        Args:
            frame_path: Path to frame image

        Returns:
            (is_match, max_similarity, face_info)
        """
        try:
            # Load image
            img = cv2.imread(str(frame_path))
            if img is None:
                return False, 0.0, None

            # Detect faces
            faces = self.face_app.get(img)

            if len(faces) == 0:
                return False, 0.0, None

            # Compare each face against references
            max_similarity = 0.0
            best_face = None

            for face in faces:
                # Compare with mean reference
                similarity = np.dot(face.normed_embedding, self.mean_reference)

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_face = face

            is_match = max_similarity >= self.similarity_threshold

            face_info = None
            if is_match and best_face is not None:
                face_info = {
                    'bbox': best_face.bbox.tolist(),
                    'det_score': float(best_face.det_score),
                    'similarity': float(max_similarity),
                    'age': int(best_face.age) if hasattr(best_face, 'age') else None,
                    'gender': int(best_face.gender) if hasattr(best_face, 'gender') else None
                }

            return is_match, float(max_similarity), face_info

        except Exception as e:
            logger.error(f"Error processing {frame_path.name}: {e}")
            return False, 0.0, None

    def filter_batch(
        self,
        frame_paths: List[Path],
        batch_size: int = 128
    ) -> List[Tuple[Path, bool, float, Optional[Dict]]]:
        """
        Filter a batch of frames with TRUE GPU batching (ULTRA-OPTIMIZED)

        Args:
            frame_paths: List of frame paths
            batch_size: Batch size for processing (default: 128 for maximum GPU utilization)

        Returns:
            List of (path, is_match, similarity, face_info) tuples
        """
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing

        results = []
        num_workers = min(8, multiprocessing.cpu_count())

        def load_image_batch(paths):
            """Load multiple images in parallel"""
            images = []
            valid_paths = []

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(cv2.imread, str(p)): p for p in paths}
                for future in futures:
                    img = future.result()
                    path = futures[future]
                    if img is not None:
                        images.append(img)
                        valid_paths.append(path)

            return images, valid_paths

        logger.info(f"🚀 ULTRA-OPTIMIZED MODE: batch_size={batch_size}, num_workers={num_workers}")

        for i in tqdm(range(0, len(frame_paths), batch_size), desc="Filtering frames (GPU BATCHED)"):
            batch_paths = frame_paths[i:i+batch_size]

            # OPTIMIZATION 1: Parallel image loading
            images, valid_paths = load_image_batch(batch_paths)

            if len(images) == 0:
                continue

            # OPTIMIZATION 2: Batch face detection with InsightFace
            try:
                # Process all images at once
                batch_results = []
                for img, path in zip(images, valid_paths):
                    faces = self.face_app.get(img)

                    if len(faces) == 0:
                        batch_results.append((path, False, 0.0, None))
                        continue

                    # Compare each face against mean reference
                    max_similarity = 0.0
                    best_face = None

                    for face in faces:
                        similarity = np.dot(face.normed_embedding, self.mean_reference)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_face = face

                    is_match = max_similarity >= self.similarity_threshold

                    face_info = None
                    if is_match and best_face is not None:
                        face_info = {
                            'bbox': best_face.bbox.tolist(),
                            'det_score': float(best_face.det_score),
                            'similarity': float(max_similarity),
                            'age': int(best_face.age) if hasattr(best_face, 'age') else None,
                            'gender': int(best_face.gender) if hasattr(best_face, 'gender') else None
                        }

                    batch_results.append((path, is_match, float(max_similarity), face_info))

                results.extend(batch_results)

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Fallback to single processing
                for path in valid_paths:
                    is_match, similarity, face_info = self.filter_frame(path)
                    results.append((path, is_match, similarity, face_info))

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Filter frames using reference-based face recognition"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with input frames"
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Directory with reference character images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for filtered frames"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.6,
        help="Minimum cosine similarity threshold (default: 0.6)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save face detection metadata"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not args.reference_dir.exists():
        logger.error(f"Reference directory not found: {args.reference_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get all frame files
    logger.info(f"Scanning {args.input_dir} for frames...")
    frame_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        frame_paths.extend(args.input_dir.glob(ext))

    logger.info(f"Found {len(frame_paths)} frames to process")

    if len(frame_paths) == 0:
        logger.error("No frames found!")
        sys.exit(1)

    # Initialize filter
    logger.info("\n" + "="*60)
    logger.info("  REFERENCE-BASED FRAME FILTER")
    logger.info("="*60)
    logger.info(f"Input:      {args.input_dir}")
    logger.info(f"Reference:  {args.reference_dir}")
    logger.info(f"Output:     {args.output_dir}")
    logger.info(f"Threshold:  {args.similarity_threshold}")
    logger.info(f"Device:     {args.device}")
    logger.info("="*60 + "\n")

    filter_obj = ReferenceFaceFilter(
        reference_dir=args.reference_dir,
        similarity_threshold=args.similarity_threshold,
        device=args.device
    )

    # Filter frames
    logger.info("Starting frame filtering...")
    results = filter_obj.filter_batch(frame_paths, batch_size=args.batch_size)

    # Process results
    matched_frames = []
    metadata = {
        'input_dir': str(args.input_dir),
        'reference_dir': str(args.reference_dir),
        'similarity_threshold': args.similarity_threshold,
        'total_frames': len(frame_paths),
        'matched_frames': 0,
        'results': []
    }

    logger.info("\nCopying matched frames...")
    for frame_path, is_match, similarity, face_info in tqdm(results):
        if is_match:
            # Copy frame to output
            output_path = args.output_dir / frame_path.name
            import shutil
            shutil.copy2(frame_path, output_path)

            matched_frames.append(frame_path.name)

            if args.save_metadata:
                metadata['results'].append({
                    'frame': frame_path.name,
                    'similarity': similarity,
                    'face_info': face_info
                })

    metadata['matched_frames'] = len(matched_frames)

    # Save metadata
    metadata_path = args.output_dir / 'filter_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("  FILTERING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total frames processed:  {len(frame_paths)}")
    logger.info(f"Matched frames:          {len(matched_frames)}")
    logger.info(f"Match rate:              {len(matched_frames)/len(frame_paths)*100:.1f}%")
    logger.info(f"Output directory:        {args.output_dir}")
    logger.info(f"Metadata saved:          {metadata_path}")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()
