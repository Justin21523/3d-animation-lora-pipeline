#!/usr/bin/env python3
"""
Character-Driven Frame Pre-Filtering for 3D Animation Pipeline

This tool filters frames based on visual similarity matching against reference images.
Supports two modes:
  - CLIP mode: Whole-image embedding similarity (robust to occlusion, pose, lighting)
  - Face mode: Face detection + ArcFace embedding (precise but requires visible faces)

Expected reduction: 60-80% of frames

Usage:
    # CLIP mode (recommended - more robust)
    python face_driven_prefilter.py \\
        --input-dir /path/to/frames \\
        --output-dir /path/to/filtered \\
        --project coco \\
        --mode clip \\
        --similarity-threshold 0.75

    # Face mode (precise when faces are clear)
    python face_driven_prefilter.py \\
        --input-dir /path/to/frames \\
        --output-dir /path/to/filtered \\
        --project coco \\
        --mode face \\
        --similarity-threshold 0.30

    # Batch processing with GPU
    python face_driven_prefilter.py \\
        --input-dir /path/to/frames \\
        --output-dir /path/to/filtered \\
        --project coco \\
        --mode clip \\
        --batch-size 32 \\
        --device cuda
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm
import logging
from PIL import Image

# Import InsightFace for face detection and recognition
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available - install with: pip install insightface")

# Import CLIP for whole-image embeddings
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available - install with: pip install transformers torch")


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG_DIR = Path("/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/projects")
DEFAULT_DOCS_DIR = Path("/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/docs/projects")

# CLIP Similarity Threshold Calibration (Cosine Similarity 0-1)
# ============================================================
# Based on empirical testing (Coco-Miguel, 50 frame sample):
#   - 0.52: 100% retention (50/50) - captures all similar frames
#   - 0.55: 98% retention (49/50) - ⭐ RECOMMENDED balance of precision/recall
#   - 0.60: 88% retention (44/50) - filters edge cases
#   - 0.70: 22% retention (11/50) - too strict, loses valid frames
#   - 0.75: 6% retention (3/50) - extremely strict
#
# Score Distribution (typical):
#   - Close-ups (front, good lighting): 0.70-0.78
#   - Mid-shots (side, varied lighting): 0.60-0.70
#   - Distant/occluded/back views: 0.52-0.60
#   - Unrelated scenes/characters: < 0.52
#
# Threshold Selection:
#   - 0.50-0.55: Permissive (95-100% retention) - first pass, high recall
#   - 0.55-0.65: Balanced (70-98% retention) - ⭐ RECOMMENDED for most cases
#   - 0.65-0.75: Strict (20-70% retention) - high precision, quality subset
#   - > 0.75: Very strict (< 20% retention) - near-duplicates only
#
# Adjustment Strategy:
#   - Too few kept (< 70% expected): Lower to 0.50-0.52
#   - Too many false positives (> 10%): Raise to 0.60-0.65
#   - Need quality subset: Use 0.70
DEFAULT_SIMILARITY_THRESHOLD_CLIP = 0.55  # ⭐ Balanced mode: 98% retention, filters obvious non-matches

DEFAULT_SIMILARITY_THRESHOLD_FACE = 0.30  # ArcFace cosine similarity (lower = more permissive)
DEFAULT_MIN_FACE_SIZE = 64  # Minimum face size in pixels
DEFAULT_BATCH_SIZE = 16  # Number of frames to process in parallel
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"  # CLIP model for embeddings


# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Face-Driven Pre-Filter Class
# ============================================================================

@dataclass
class PreFilterConfig:
    """Configuration for character-driven pre-filtering"""
    project: str
    mode: str = "clip"  # "clip" or "face"
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD_CLIP
    min_face_size: int = DEFAULT_MIN_FACE_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "cuda"  # cuda or cpu
    clip_model: str = DEFAULT_CLIP_MODEL  # CLIP model name
    save_rejected: bool = True  # Save rejected frames to separate directory
    save_report: bool = True  # Save detailed filtering report


class CharacterFilter:
    """
    Filter frames based on visual similarity matching against reference images.

    Supports two modes:
    - CLIP mode: Whole-image embedding similarity (robust, recommended)
    - Face mode: Face detection + ArcFace embedding (precise for clear faces)

    This is a critical optimization step that reduces dataset size by 60-80%
    by only keeping frames containing target characters.
    """

    def __init__(self, config: PreFilterConfig, config_dir: Path = DEFAULT_CONFIG_DIR, docs_dir: Path = DEFAULT_DOCS_DIR):
        """
        Initialize character filter.

        Args:
            config: Filter configuration
            config_dir: Path to project configs directory (for storing embeddings)
            docs_dir: Path to docs/projects directory (for reading reference images)
        """
        self.config = config
        self.config_dir = config_dir
        self.docs_dir = docs_dir
        self.reference_dir_docs = docs_dir / config.project / "reference-faces"
        self.reference_dir_config = config_dir / config.project / "reference_faces"

        self.face_app = None
        self.clip_model = None
        self.clip_processor = None

        # Initialize based on mode
        if config.mode == "face":
            if not INSIGHTFACE_AVAILABLE:
                raise RuntimeError("InsightFace is required for face mode - install with: pip install insightface")

            logger.info("Initializing InsightFace face detection and recognition...")
            self.face_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.device == 'cuda' else ['CPUExecutionProvider']
            )
            ctx_id = 0 if config.device == 'cuda' else -1
            self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

            # Load reference embeddings
            self.reference_embeddings = self._load_reference_embeddings_face()

        elif config.mode == "clip":
            if not CLIP_AVAILABLE:
                raise RuntimeError("CLIP is required for clip mode - install with: pip install transformers torch")

            logger.info(f"Initializing CLIP model: {config.clip_model}...")
            device = "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
            self.clip_model = CLIPModel.from_pretrained(config.clip_model).to(device)
            self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model)
            self.clip_model.eval()

            # Load reference embeddings
            self.reference_embeddings = self._load_reference_embeddings_clip()

        else:
            raise ValueError(f"Invalid mode '{config.mode}'. Must be 'clip' or 'face'")

        if not self.reference_embeddings:
            raise ValueError(f"No reference embeddings found for project '{config.project}'")

        logger.info(f"Loaded {len(self.reference_embeddings)} reference character(s)")
        for char_name, emb in self.reference_embeddings.items():
            logger.info(f"  • {char_name}: {len(emb)} reference images")

    def _load_reference_embeddings_face(self) -> Dict[str, np.ndarray]:
        """
        Load reference face embeddings (ArcFace) for all characters.

        Returns:
            Dictionary mapping character name to embedding array
        """
        if not self.reference_dir_docs.exists():
            logger.error(f"Reference faces directory not found: {self.reference_dir_docs}")
            return {}

        embeddings = {}

        for char_dir in self.reference_dir_docs.iterdir():
            if not char_dir.is_dir():
                continue

            # Get all image files
            image_files = list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpg"))
            if not image_files:
                logger.warning(f"No images found for character '{char_dir.name}'")
                continue

            # Extract face embeddings from each image
            char_embeddings = []
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue

                faces = self.face_app.get(img)
                if len(faces) == 0:
                    logger.warning(f"No face detected in {img_path.name}")
                    continue

                # Use largest face if multiple detected
                if len(faces) > 1:
                    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)

                char_embeddings.append(faces[0].embedding)

            if char_embeddings:
                embeddings[char_dir.name] = np.array(char_embeddings)

        return embeddings

    def _load_reference_embeddings_clip(self) -> Dict[str, np.ndarray]:
        """
        Load reference image embeddings (CLIP) for all characters.

        Returns:
            Dictionary mapping character name to embedding array
        """
        if not self.reference_dir_docs.exists():
            logger.error(f"Reference images directory not found: {self.reference_dir_docs}")
            return {}

        embeddings = {}
        device = self.clip_model.device

        for char_dir in self.reference_dir_docs.iterdir():
            if not char_dir.is_dir():
                continue

            # Get all image files
            image_files = list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpg"))
            if not image_files:
                logger.warning(f"No images found for character '{char_dir.name}'")
                continue

            # Extract CLIP embeddings from each image
            char_embeddings = []
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    inputs = self.clip_processor(images=img, return_tensors="pt").to(device)

                    with torch.no_grad():
                        image_features = self.clip_model.get_image_features(**inputs)
                        # Normalize embedding
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    char_embeddings.append(image_features.cpu().numpy().squeeze())

                except Exception as e:
                    logger.warning(f"Failed to process {img_path.name}: {e}")
                    continue

            if char_embeddings:
                embeddings[char_dir.name] = np.array(char_embeddings)

        return embeddings

    def detect_faces_batch(
        self,
        frames: List[Path]
    ) -> List[Tuple[Path, List[np.ndarray]]]:
        """
        Detect faces in a batch of frames.

        Args:
            frames: List of frame paths

        Returns:
            List of tuples (frame_path, [face_embeddings])
        """
        results = []

        for frame_path in frames:
            img = cv2.imread(str(frame_path))
            if img is None:
                logger.warning(f"Failed to load image: {frame_path}")
                results.append((frame_path, []))
                continue

            # Detect faces
            faces = self.face_app.get(img)

            # Filter by minimum size
            valid_faces = []
            for face in faces:
                bbox = face.bbox
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]

                if face_width >= self.config.min_face_size and face_height >= self.config.min_face_size:
                    valid_faces.append(face.embedding)

            results.append((frame_path, valid_faces))

        return results

    def match_face_to_references(
        self,
        face_embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Match a face embedding to reference characters.

        Args:
            face_embedding: Face embedding vector (512-d)

        Returns:
            Tuple of (matched_character_name or None, best_similarity_score)
        """
        best_match = None
        best_similarity = -1.0

        for char_name, ref_embeddings in self.reference_embeddings.items():
            # Compute cosine similarity with all reference faces for this character
            similarities = np.dot(ref_embeddings, face_embedding) / (
                np.linalg.norm(ref_embeddings, axis=1) * np.linalg.norm(face_embedding)
            )

            max_similarity = np.max(similarities)

            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match = char_name

        # Check if similarity exceeds threshold
        if best_similarity >= self.config.similarity_threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity

    def extract_clip_embedding_batch(
        self,
        frames: List[Path]
    ) -> List[Tuple[Path, Optional[np.ndarray]]]:
        """
        Extract CLIP embeddings for a batch of frames.

        Args:
            frames: List of frame paths

        Returns:
            List of tuples (frame_path, clip_embedding or None)
        """
        results = []
        device = self.clip_model.device

        for frame_path in frames:
            try:
                img = Image.open(frame_path).convert("RGB")
                inputs = self.clip_processor(images=img, return_tensors="pt").to(device)

                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    # Normalize embedding
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                results.append((frame_path, image_features.cpu().numpy().squeeze()))

            except Exception as e:
                logger.warning(f"Failed to process {frame_path.name}: {e}")
                results.append((frame_path, None))

        return results

    def match_clip_to_references(
        self,
        clip_embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Match a CLIP embedding to reference characters.

        Args:
            clip_embedding: CLIP embedding vector

        Returns:
            Tuple of (matched_character_name or None, best_similarity_score)
        """
        best_match = None
        best_similarity = -1.0

        for char_name, ref_embeddings in self.reference_embeddings.items():
            # Compute cosine similarity with all reference images for this character
            similarities = np.dot(ref_embeddings, clip_embedding)
            max_similarity = np.max(similarities)

            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match = char_name

        # Check if similarity exceeds threshold
        if best_similarity >= self.config.similarity_threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity

    def filter_frames(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main filtering pipeline.

        Args:
            input_dir: Directory with input frames
            output_dir: Directory to save filtered frames

        Returns:
            Statistics dictionary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_rejected:
            rejected_dir = output_dir.parent / f"{output_dir.name}_rejected"
            rejected_dir.mkdir(parents=True, exist_ok=True)
        else:
            rejected_dir = None

        # Find all frames
        frame_files = sorted(
            list(input_dir.glob("*.jpg")) +
            list(input_dir.glob("*.png"))
        )

        logger.info(f"\n📊 Found {len(frame_files)} frames in {input_dir}")
        logger.info(f"   Mode: {self.config.mode}")
        logger.info(f"   Similarity threshold: {self.config.similarity_threshold}")
        if self.config.mode == "face":
            logger.info(f"   Min face size: {self.config.min_face_size}x{self.config.min_face_size}")
        logger.info(f"   Batch size: {self.config.batch_size}")

        # Process frames in batches
        kept_frames: List[Path] = []
        rejected_frames: List[Path] = []
        frame_stats: Dict[str, Dict] = {}

        num_batches = (len(frame_files) + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(frame_files))
            batch_frames = frame_files[start_idx:end_idx]

            # Process based on mode
            if self.config.mode == "clip":
                # CLIP mode - whole image embedding
                batch_results = self.extract_clip_embedding_batch(batch_frames)

                for frame_path, clip_emb in tqdm(
                    batch_results,
                    desc=f"Batch {batch_idx + 1}/{num_batches}",
                    leave=False
                ):
                    if clip_emb is None:
                        # Failed to extract embedding
                        rejected_frames.append(frame_path)
                        if rejected_dir:
                            shutil.copy2(frame_path, rejected_dir / frame_path.name)
                        frame_stats[frame_path.name] = {
                            "status": "rejected",
                            "reason": "embedding_failed",
                            "matched_characters": [],
                            "best_similarity": 0.0
                        }
                        continue

                    char_name, similarity = self.match_clip_to_references(clip_emb)

                    if char_name:
                        # Match found
                        kept_frames.append(frame_path)
                        shutil.copy2(frame_path, output_dir / frame_path.name)
                        frame_stats[frame_path.name] = {
                            "status": "kept",
                            "matched_characters": [char_name],
                            "best_similarity": float(similarity)
                        }
                    else:
                        # No match
                        rejected_frames.append(frame_path)
                        if rejected_dir:
                            shutil.copy2(frame_path, rejected_dir / frame_path.name)
                        frame_stats[frame_path.name] = {
                            "status": "rejected",
                            "reason": "no_match",
                            "matched_characters": [],
                            "best_similarity": float(similarity)
                        }

            elif self.config.mode == "face":
                # Face mode - detect faces and match
                batch_results = self.detect_faces_batch(batch_frames)

                for frame_path, face_embeddings in tqdm(
                    batch_results,
                    desc=f"Batch {batch_idx + 1}/{num_batches}",
                    leave=False
                ):
                    matched_characters = set()
                    best_match_score = 0.0

                    for face_emb in face_embeddings:
                        char_name, similarity = self.match_face_to_references(face_emb)
                        if char_name:
                            matched_characters.add(char_name)
                            best_match_score = max(best_match_score, similarity)

                    # Keep frame if at least one target character detected
                    if matched_characters:
                        kept_frames.append(frame_path)
                        shutil.copy2(frame_path, output_dir / frame_path.name)
                        frame_stats[frame_path.name] = {
                            "status": "kept",
                            "num_faces": len(face_embeddings),
                            "matched_characters": list(matched_characters),
                            "best_similarity": float(best_match_score)
                        }
                    else:
                        rejected_frames.append(frame_path)
                        if rejected_dir:
                            shutil.copy2(frame_path, rejected_dir / frame_path.name)
                        frame_stats[frame_path.name] = {
                            "status": "rejected",
                            "num_faces": len(face_embeddings),
                            "matched_characters": [],
                            "best_similarity": float(best_match_score) if face_embeddings else 0.0
                        }

        # Generate statistics
        stats = {
            "project": self.config.project,
            "mode": self.config.mode,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "similarity_threshold": self.config.similarity_threshold,
            "total_input_frames": len(frame_files),
            "frames_kept": len(kept_frames),
            "frames_rejected": len(rejected_frames),
            "reduction_rate": len(rejected_frames) / len(frame_files) if len(frame_files) > 0 else 0,
            "characters_detected": {
                char: sum(1 for s in frame_stats.values() if char in s.get("matched_characters", []))
                for char in self.reference_embeddings.keys()
            },
            "timestamp": datetime.now().isoformat()
        }

        if self.config.mode == "face":
            stats["min_face_size"] = self.config.min_face_size
        elif self.config.mode == "clip":
            stats["clip_model"] = self.config.clip_model

        # Save report
        if self.config.save_report:
            report_path = output_dir / "prefilter_report.json"
            with open(report_path, 'w') as f:
                json.dump(stats, f, indent=2)

            detailed_path = output_dir / "prefilter_detailed.json"
            with open(detailed_path, 'w') as f:
                json.dump(frame_stats, f, indent=2)

        # Print summary
        logger.info(f"\n✅ Character pre-filtering complete!")
        logger.info(f"   Mode: {self.config.mode}")
        logger.info(f"   Total input frames: {len(frame_files)}")
        logger.info(f"   Frames kept: {len(kept_frames)} ({len(kept_frames)/len(frame_files)*100:.1f}%)")
        logger.info(f"   Frames rejected: {len(rejected_frames)} ({stats['reduction_rate']*100:.1f}%)")
        logger.info(f"\n   Characters detected:")
        for char, count in stats["characters_detected"].items():
            logger.info(f"     • {char}: {count} frames")

        if self.config.save_report:
            logger.info(f"\n📄 Report saved to: {report_path}")

        return stats


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter frames based on visual similarity to reference characters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # CLIP mode (recommended - robust to occlusion, lighting, pose)
    python face_driven_prefilter.py \\
        --input-dir /mnt/data/ai_data/datasets/3d-anime/coco/frames \\
        --output-dir /mnt/data/ai_data/datasets/3d-anime/coco/frames_filtered \\
        --project coco \\
        --mode clip \\
        --similarity-threshold 0.75

    # Face mode (precise when faces are clear and visible)
    python face_driven_prefilter.py \\
        --input-dir frames/ \\
        --output-dir frames_filtered/ \\
        --project coco \\
        --mode face \\
        --similarity-threshold 0.30

    # CLIP mode with custom model
    python face_driven_prefilter.py \\
        --input-dir frames/ \\
        --output-dir frames_filtered/ \\
        --project coco \\
        --mode clip \\
        --clip-model openai/clip-vit-base-patch32 \\
        --batch-size 32
        """
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory with input frames'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory to save filtered frames'
    )

    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='Project name (e.g., coco, elio, turning-red, up)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='clip',
        choices=['clip', 'face'],
        help='Matching mode: clip (whole-image, robust) or face (face-only, precise) (default: clip)'
    )

    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=None,
        help=f'Cosine similarity threshold (default: {DEFAULT_SIMILARITY_THRESHOLD_CLIP} for clip, {DEFAULT_SIMILARITY_THRESHOLD_FACE} for face)'
    )

    parser.add_argument(
        '--clip-model',
        type=str,
        default=DEFAULT_CLIP_MODEL,
        help=f'CLIP model name (default: {DEFAULT_CLIP_MODEL})'
    )

    parser.add_argument(
        '--min-face-size',
        type=int,
        default=DEFAULT_MIN_FACE_SIZE,
        help=f'Minimum face size in pixels (default: {DEFAULT_MIN_FACE_SIZE})'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Number of frames to process in parallel (default: {DEFAULT_BATCH_SIZE})'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for face detection (default: cuda)'
    )

    parser.add_argument(
        '--no-save-rejected',
        action='store_true',
        help='Do not save rejected frames (saves disk space)'
    )

    parser.add_argument(
        '--config-dir',
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f'Config directory (default: {DEFAULT_CONFIG_DIR})'
    )

    parser.add_argument(
        '--docs-dir',
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help=f'Docs directory with reference images (default: {DEFAULT_DOCS_DIR})'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set default similarity threshold based on mode
    if args.similarity_threshold is None:
        if args.mode == "clip":
            similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD_CLIP
        else:
            similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD_FACE
    else:
        similarity_threshold = args.similarity_threshold

    # Create configuration
    config = PreFilterConfig(
        project=args.project,
        mode=args.mode,
        similarity_threshold=similarity_threshold,
        min_face_size=args.min_face_size,
        batch_size=args.batch_size,
        device=args.device,
        clip_model=args.clip_model,
        save_rejected=not args.no_save_rejected,
        save_report=True
    )

    # Initialize filter
    filter_tool = CharacterFilter(config, config_dir=args.config_dir, docs_dir=args.docs_dir)

    # Run filtering
    stats = filter_tool.filter_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

    return 0


if __name__ == '__main__':
    exit(main())
