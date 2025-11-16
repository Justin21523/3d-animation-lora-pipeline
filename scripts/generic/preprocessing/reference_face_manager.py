#!/usr/bin/env python3
"""
Reference Face Manager for 3D Animation Pipeline

This tool manages reference face images for each character in a film project.
It extracts ArcFace embeddings from manually provided face images and stores
them for later use in face-driven pre-filtering and identity clustering.

Usage:
    # Add reference faces for a character
    python reference_face_manager.py --project coco --character miguel \\
        --add-references /path/to/miguel_*.jpg

    # Verify reference faces
    python reference_face_manager.py --project coco --verify

    # List all reference faces
    python reference_face_manager.py --project coco --list

    # Remove a character's references
    python reference_face_manager.py --project coco --character miguel --remove
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from tqdm import tqdm
import logging

# Try to import InsightFace for ArcFace embeddings
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: insightface not available. Install with: pip install insightface")


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_BASE_DIR = Path("/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline")
DEFAULT_CONFIG_DIR = DEFAULT_BASE_DIR / "configs" / "projects"
DEFAULT_EMBEDDING_MODEL = "buffalo_l"  # InsightFace model
MIN_FACE_SIZE = 64  # Minimum face size in pixels


# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Reference Face Manager Class
# ============================================================================

class ReferenceManager:
    """
    Manages reference face images and embeddings for character identification.

    Attributes:
        project (str): Project name (e.g., 'coco', 'elio')
        config_dir (Path): Path to project config directory
        reference_dir (Path): Path to reference faces directory
        face_app (FaceAnalysis): InsightFace face analysis app
    """

    def __init__(
        self,
        project: str,
        config_dir: Path = DEFAULT_CONFIG_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ):
        """
        Initialize reference face manager.

        Args:
            project: Project name
            config_dir: Base config directory path
            embedding_model: InsightFace model name
        """
        self.project = project
        self.config_dir = config_dir / project
        self.reference_dir = self.config_dir / "reference_faces"

        # Create directories if needed
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir.mkdir(parents=True, exist_ok=True)

        # Initialize face analysis
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_app = FaceAnalysis(
                    name=embedding_model,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info(f"Initialized InsightFace model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize InsightFace: {e}")
                self.face_app = None
        else:
            self.face_app = None
            logger.warning("InsightFace not available - embeddings cannot be extracted")

    def add_reference_face(
        self,
        character: str,
        image_path: Path,
        verify: bool = True
    ) -> Optional[np.ndarray]:
        """
        Add a reference face image for a character.

        Args:
            character: Character name
            image_path: Path to face image
            verify: Whether to verify face detection

        Returns:
            Face embedding if successful, None otherwise
        """
        # Create character directory
        char_dir = self.reference_dir / character
        char_dir.mkdir(parents=True, exist_ok=True)

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # Detect face and extract embedding
        if self.face_app is None:
            logger.warning("Face analysis not available - skipping embedding extraction")
            embedding = None
        else:
            faces = self.face_app.get(img)

            if len(faces) == 0:
                logger.error(f"No face detected in: {image_path}")
                if verify:
                    return None
                embedding = None
            elif len(faces) > 1:
                logger.warning(f"Multiple faces detected in: {image_path}, using largest")
                # Use largest face
                faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
                embedding = faces[0].embedding
            else:
                embedding = faces[0].embedding

            # Verify face size
            if len(faces) > 0:
                bbox = faces[0].bbox
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                    logger.warning(f"Face too small ({face_width}x{face_height}) in: {image_path}")
                    if verify:
                        return None

        # Copy image to reference directory
        target_path = char_dir / image_path.name
        shutil.copy2(image_path, target_path)
        logger.info(f"Added reference face: {target_path}")

        return embedding

    def add_references_batch(
        self,
        character: str,
        image_paths: List[Path],
        verify: bool = True
    ) -> Tuple[int, int]:
        """
        Add multiple reference faces for a character.

        Args:
            character: Character name
            image_paths: List of image paths
            verify: Whether to verify face detection

        Returns:
            Tuple of (successful_count, failed_count)
        """
        embeddings = []
        successful = 0
        failed = 0

        logger.info(f"Adding {len(image_paths)} reference faces for '{character}'...")

        for img_path in tqdm(image_paths, desc="Processing faces"):
            embedding = self.add_reference_face(character, img_path, verify)
            if embedding is not None:
                embeddings.append(embedding)
                successful += 1
            else:
                failed += 1

        # Save embeddings
        if embeddings:
            self.save_embeddings(character, embeddings)
            logger.info(f"Successfully added {successful} reference faces for '{character}'")

        if failed > 0:
            logger.warning(f"Failed to process {failed} images")

        return successful, failed

    def save_embeddings(
        self,
        character: str,
        embeddings: List[np.ndarray]
    ) -> Path:
        """
        Save face embeddings to disk.

        Args:
            character: Character name
            embeddings: List of embedding vectors

        Returns:
            Path to saved embeddings file
        """
        char_dir = self.reference_dir / character
        char_dir.mkdir(parents=True, exist_ok=True)

        embeddings_array = np.array(embeddings)
        embeddings_path = char_dir / f"{character}_embeddings.npy"
        np.save(embeddings_path, embeddings_array)

        logger.info(f"Saved {len(embeddings)} embeddings to: {embeddings_path}")

        # Also save metadata
        metadata = {
            "character": character,
            "project": self.project,
            "num_references": len(embeddings),
            "embedding_shape": list(embeddings_array.shape),
            "model": DEFAULT_EMBEDDING_MODEL
        }
        metadata_path = char_dir / f"{character}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return embeddings_path

    def load_embeddings(self, character: str) -> Optional[np.ndarray]:
        """
        Load face embeddings from disk.

        Args:
            character: Character name

        Returns:
            Embedding array if exists, None otherwise
        """
        embeddings_path = self.reference_dir / character / f"{character}_embeddings.npy"

        if not embeddings_path.exists():
            logger.error(f"Embeddings not found for '{character}': {embeddings_path}")
            return None

        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded {len(embeddings)} embeddings for '{character}'")

        return embeddings

    def list_characters(self) -> List[str]:
        """
        List all characters with reference faces.

        Returns:
            List of character names
        """
        if not self.reference_dir.exists():
            return []

        characters = [
            d.name for d in self.reference_dir.iterdir()
            if d.is_dir() and list(d.glob("*.jpg")) + list(d.glob("*.png"))
        ]

        return sorted(characters)

    def verify_references(self, character: Optional[str] = None) -> Dict:
        """
        Verify reference faces and display statistics.

        Args:
            character: Specific character to verify, or None for all

        Returns:
            Dictionary with verification results
        """
        characters = [character] if character else self.list_characters()

        results = {}

        for char in characters:
            char_dir = self.reference_dir / char
            if not char_dir.exists():
                logger.warning(f"No reference directory for '{char}'")
                continue

            # Count images
            images = list(char_dir.glob("*.jpg")) + list(char_dir.glob("*.png"))

            # Check embeddings
            embeddings_path = char_dir / f"{char}_embeddings.npy"
            has_embeddings = embeddings_path.exists()
            num_embeddings = 0

            if has_embeddings:
                embeddings = np.load(embeddings_path)
                num_embeddings = len(embeddings)

            results[char] = {
                "num_images": len(images),
                "has_embeddings": has_embeddings,
                "num_embeddings": num_embeddings,
                "images": [img.name for img in images],
                "embeddings_match": len(images) == num_embeddings
            }

            # Display results
            print(f"\n{'='*60}")
            print(f"Character: {char}")
            print(f"{'='*60}")
            print(f"Reference images: {len(images)}")
            print(f"Embeddings: {num_embeddings} {'✓' if has_embeddings else '✗'}")
            print(f"Match: {'✓' if results[char]['embeddings_match'] else '✗ MISMATCH'}")

            if images:
                print(f"\nImages:")
                for img in images:
                    print(f"  • {img.name}")

        return results

    def remove_character(self, character: str, confirm: bool = True) -> bool:
        """
        Remove all reference faces for a character.

        Args:
            character: Character name
            confirm: Whether to ask for confirmation

        Returns:
            True if successful, False otherwise
        """
        char_dir = self.reference_dir / character

        if not char_dir.exists():
            logger.error(f"No references found for '{character}'")
            return False

        if confirm:
            response = input(f"Remove all references for '{character}'? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                logger.info("Operation cancelled")
                return False

        shutil.rmtree(char_dir)
        logger.info(f"Removed all references for '{character}'")

        return True


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage reference face images for character identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Add reference faces
    python reference_face_manager.py --project coco --character miguel \\
        --add-references /path/to/miguel_001.jpg /path/to/miguel_002.jpg

    # Add from directory (glob pattern)
    python reference_face_manager.py --project coco --character miguel \\
        --add-references "/path/to/miguel_faces/*.jpg"

    # Verify references
    python reference_face_manager.py --project coco --verify

    # List characters
    python reference_face_manager.py --project coco --list

    # Remove character references
    python reference_face_manager.py --project coco --character miguel --remove
        """
    )

    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='Project name (e.g., coco, elio, turning-red, up)'
    )

    parser.add_argument(
        '--character',
        type=str,
        help='Character name'
    )

    parser.add_argument(
        '--add-references',
        nargs='+',
        type=str,
        help='Path(s) to reference face images (supports glob patterns)'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify reference faces and show statistics'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all characters with references'
    )

    parser.add_argument(
        '--remove',
        action='store_true',
        help='Remove all reference faces for a character'
    )

    parser.add_argument(
        '--config-dir',
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f'Config directory (default: {DEFAULT_CONFIG_DIR})'
    )

    parser.add_argument(
        '--no-verify-face',
        action='store_true',
        help='Skip face detection verification (faster but less reliable)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize manager
    manager = ReferenceManager(
        project=args.project,
        config_dir=args.config_dir
    )

    # Handle different operations
    if args.list:
        characters = manager.list_characters()
        print(f"\n{'='*60}")
        print(f"Project: {args.project}")
        print(f"Characters with references: {len(characters)}")
        print(f"{'='*60}")
        for char in characters:
            print(f"  • {char}")
        print()

    elif args.verify:
        manager.verify_references(args.character)

    elif args.remove:
        if not args.character:
            logger.error("--character required for remove operation")
            return 1
        manager.remove_character(args.character)

    elif args.add_references:
        if not args.character:
            logger.error("--character required for add-references operation")
            return 1

        # Expand glob patterns
        from glob import glob
        image_paths = []
        for pattern in args.add_references:
            matches = glob(pattern, recursive=True)
            if matches:
                image_paths.extend([Path(p) for p in matches])
            else:
                # Try as direct path
                p = Path(pattern)
                if p.exists():
                    image_paths.append(p)
                else:
                    logger.warning(f"No files found matching: {pattern}")

        if not image_paths:
            logger.error("No valid image paths provided")
            return 1

        # Add references
        verify = not args.no_verify_face
        successful, failed = manager.add_references_batch(
            args.character,
            image_paths,
            verify=verify
        )

        print(f"\n{'='*60}")
        print(f"Summary")
        print(f"{'='*60}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {len(image_paths)}")
        print()

    else:
        logger.error("No operation specified. Use --add-references, --verify, --list, or --remove")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
