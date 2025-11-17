#!/usr/bin/env python3
"""
Prepare Expression/Emotion LoRA Training Data

Prepares facial expression-specific images for emotion LoRA training by:
1. Extracting faces from character instances (face detection)
2. Classifying expressions using CLIP zero-shot + optional supervised models
3. Filtering by quality (detection confidence, expression confidence, blur, size)
4. Organizing/clustering by expression type
5. Balancing dataset across expression classes
6. Generating expression-focused captions
7. Assembling into kohya_ss training format

This is a HIGH-DIFFICULTY task due to domain gap between real-face emotion
models and 3D animated stylized faces.

Usage:
    python prepare_expression_lora_data.py \
        --character-instances /path/to/character_instances/ \
        --output-dir /path/to/training_data/expression_lora/ \
        --expressions happy sad angry surprised neutral fearful \
        --target-per-expression 150 \
        --device cuda

For detailed technical guide, see:
    docs/training/lora_types/02_EXPRESSION_LORA_DEEP_DIVE.md
"""

import sys
import shutil
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import numpy as np
from PIL import Image
import cv2
import torch
from tqdm import tqdm
import imagehash

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.utils.logger import setup_logger
from core.utils.checkpoint_manager import CheckpointManager
from core.expression.intensity_classifier import ExpressionIntensityClassifier


# Standard emotion categories (basic 6 + neutral)
STANDARD_EMOTIONS = {
    "happy": ["happy", "joyful", "smiling", "cheerful", "grinning"],
    "sad": ["sad", "unhappy", "crying", "melancholic", "downcast"],
    "angry": ["angry", "mad", "upset", "furious", "frowning"],
    "surprised": ["surprised", "shocked", "amazed", "astonished", "wide-eyed"],
    "neutral": ["neutral", "calm", "relaxed", "expressionless"],
    "fearful": ["fearful", "scared", "worried", "afraid", "nervous"],
    "disgusted": ["disgusted", "repulsed", "grossed out"],  # Optional
}


class FaceExtractor:
    """Extract faces from character instances using RetinaFace/InsightFace"""

    def __init__(self, det_thresh=0.3, det_size=(640, 640), min_face_size=80, padding_ratio=0.25, device='cuda'):
        """
        Initialize face extractor

        Args:
            det_thresh: Detection confidence threshold (lower for 3D faces)
            det_size: Detection input size
            min_face_size: Minimum face size in pixels
            padding_ratio: Padding around face bbox (0.2-0.3 for expressions)
            device: cuda or cpu
        """
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.min_face_size = min_face_size
        self.padding_ratio = padding_ratio
        self.device = device

        # Lazy import (only if needed)
        self.app = None

    def _init_detector(self):
        """Lazy initialization of face detector"""
        if self.app is None:
            try:
                from insightface.app import FaceAnalysis

                providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
                self.app = FaceAnalysis(name='buffalo_l', providers=providers)
                self.app.prepare(ctx_id=0, det_size=self.det_size, det_thresh=self.det_thresh)
            except ImportError:
                raise ImportError(
                    "insightface not installed. Install with: pip install insightface"
                )

    def extract_faces(self, img_path):
        """
        Extract all faces from an image

        Args:
            img_path: Path to image file

        Returns:
            List[Dict]: List of face dicts with bbox, landmarks, crop, etc.
        """
        self._init_detector()

        img = cv2.imread(str(img_path))
        if img is None:
            return []

        faces = self.app.get(img)

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Face size filter
            face_width = x2 - x1
            face_height = y2 - y1
            if face_width < self.min_face_size or face_height < self.min_face_size:
                continue

            # Add padding (important for expressions that extend beyond face box)
            pad_w = int(face_width * self.padding_ratio)
            pad_h = int(face_height * self.padding_ratio)

            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(img.shape[1], x2 + pad_w)
            y2 = min(img.shape[0], y2 + pad_h)

            face_crop = img[y1:y2, x1:x2]

            # Aspect ratio check (skip extreme ratios indicating partial faces)
            aspect_ratio = face_width / max(face_height, 1)
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                'det_score': float(face.det_score),
                'face_crop': face_crop,
                'face_width': face_width,
                'face_height': face_height,
            })

        return results


class CLIPExpressionClassifier:
    """
    CLIP-based zero-shot expression classifier

    Recommended for 3D animated faces due to domain gap from real-face models.
    """

    def __init__(self, emotions=None, confidence_threshold=0.5, device='cuda', logger=None):
        """
        Initialize CLIP expression classifier

        Args:
            emotions: List of emotion names (uses STANDARD_EMOTIONS if None)
            confidence_threshold: Minimum confidence for valid prediction
            device: cuda or cpu
            logger: Logger instance
        """
        self.emotions = emotions or list(STANDARD_EMOTIONS.keys())
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.logger = logger or setup_logger(self.__class__.__name__)

        # Lazy import
        self.model = None
        self.preprocess = None
        self.text_features = None

    def _init_model(self):
        """Lazy initialization of CLIP model"""
        if self.model is None:
            try:
                import clip

                self.logger.info(f"Loading CLIP model: ViT-L/14")
                self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

                # Generate prompts for 3D animated characters
                prompts = []
                for emotion in self.emotions:
                    # Use synonyms from STANDARD_EMOTIONS
                    descriptors = STANDARD_EMOTIONS.get(emotion, [emotion])
                    desc_str = ", ".join(descriptors[:3])  # Use top 3 descriptors
                    prompt = f"a 3d animated character with a {desc_str} expression, pixar style"
                    prompts.append(prompt)

                self.logger.info(f"Loaded {len(self.emotions)} emotion classes:")
                for emotion, prompt in zip(self.emotions, prompts):
                    self.logger.info(f"  - {emotion}: '{prompt}'")

                # Encode text prompts
                text_tokens = clip.tokenize(prompts).to(self.device)
                with torch.no_grad():
                    self.text_features = self.model.encode_text(text_tokens)
                    self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

            except ImportError:
                raise ImportError(
                    "CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git"
                )

    def classify(self, face_crop):
        """
        Classify expression using CLIP zero-shot

        Args:
            face_crop: np.array (BGR format from cv2)

        Returns:
            Dict with emotion, confidence, all_scores
        """
        self._init_model()

        import clip

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(face_rgb)

        # Preprocess
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Encode image
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            confidence, emotion_idx = similarity.max(dim=-1)

        emotion = self.emotions[emotion_idx.item()]
        confidence = confidence.item()

        # All scores
        all_scores = {
            label: float(similarity[0, i])
            for i, label in enumerate(self.emotions)
        }

        return {
            'emotion': emotion,
            'confidence': confidence,
            'all_scores': all_scores,
            'meets_threshold': confidence >= self.confidence_threshold,
        }


class PerceptualHashDeduplicator:
    """
    Deduplication using perceptual hashing.

    Uses imagehash library (pHash/aHash) to detect near-duplicate images.
    """

    def __init__(self,
                 hash_size: int = 16,
                 threshold: int = 8,
                 logger=None):
        """
        Initialize deduplicator.

        Args:
            hash_size: Hash size (larger = more sensitive)
            threshold: Hamming distance threshold (lower = stricter)
            logger: Logger instance
        """
        self.hash_size = hash_size
        self.threshold = threshold
        self.logger = logger

        # Track seen hashes
        self.seen_hashes = {}  # hash -> image_path

    def compute_hash(self, image_path: Path) -> imagehash.ImageHash:
        """
        Compute perceptual hash.

        Args:
            image_path: Path to image

        Returns:
            ImageHash object
        """
        img = Image.open(image_path)
        phash = imagehash.average_hash(img, hash_size=self.hash_size)
        return phash

    def is_duplicate(self, image_path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Check if image is duplicate of previously seen image.

        Args:
            image_path: Path to image

        Returns:
            (is_duplicate, original_image_path)
        """
        phash = self.compute_hash(image_path)

        # Check against seen hashes
        for seen_hash, seen_path in self.seen_hashes.items():
            distance = phash - seen_hash  # Hamming distance

            if distance <= self.threshold:
                # Duplicate found
                return True, seen_path

        # Not a duplicate, add to seen hashes
        self.seen_hashes[phash] = image_path

        return False, None

    def reset(self):
        """Clear seen hashes."""
        self.seen_hashes.clear()


class ExpressionLoRADataPreparer:
    """Main controller for expression LoRA data preparation"""

    def __init__(
        self,
        character_instances_dir: Path,
        output_dir: Path,
        emotions: Optional[List[str]] = None,
        target_per_expression: int = 150,
        device: str = 'cuda',
        logger=None
    ):
        """
        Initialize Expression LoRA data preparer

        Args:
            character_instances_dir: Directory containing character instances
            output_dir: Output directory for training data
            emotions: List of emotions to extract (default: standard 6)
            target_per_expression: Target number of images per expression
            device: cuda or cpu
            logger: Logger instance
        """
        self.character_instances_dir = Path(character_instances_dir)
        self.output_dir = Path(output_dir)
        self.emotions = emotions or ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fearful']
        self.target_per_expression = target_per_expression
        self.device = device
        self.logger = logger or setup_logger(__name__)

        # Create output directories
        self.faces_dir = self.output_dir / "faces_extracted"
        self.organized_dir = self.output_dir / "expressions_organized"
        self.final_dir = self.output_dir / "final_dataset"

        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.organized_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.face_extractor = None
        self.expression_classifier = None

        # Initialize deduplicator
        self.logger.info("Initializing deduplicator...")
        self.deduplicator = PerceptualHashDeduplicator(
            hash_size=16,
            threshold=8,
            logger=self.logger
        )

        # Initialize intensity classifier
        self.logger.info("Initializing intensity classifier...")
        self.intensity_classifier = ExpressionIntensityClassifier(
            device=self.device,
            logger=self.logger
        )

        # Initialize checkpoint manager for resume capability
        self.logger.info("Initializing checkpoint manager...")
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_path=self.output_dir / ".expression_classification_checkpoint.json",
            save_interval=50,  # Save every 50 images (expression classification is medium speed)
            logger=self.logger
        )

        self.logger.info(f"Expression LoRA Data Preparer initialized")
        self.logger.info(f"  Character Instances: {self.character_instances_dir}")
        self.logger.info(f"  Output: {self.output_dir}")
        self.logger.info(f"  Target Emotions: {self.emotions}")
        self.logger.info(f"  Target per expression: {self.target_per_expression}")

    def extract_all_faces(self, det_thresh=0.3, min_face_size=80):
        """
        Step 1: Extract faces from all character instances

        Args:
            det_thresh: Detection confidence threshold
            min_face_size: Minimum face size in pixels

        Returns:
            List of face metadata dicts
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 1: FACE EXTRACTION")
        self.logger.info("=" * 80)

        self.face_extractor = FaceExtractor(
            det_thresh=det_thresh,
            det_size=(640, 640),
            min_face_size=min_face_size,
            padding_ratio=0.25,
            device=self.device
        )

        # Scan character instances
        instance_files = sorted(list(self.character_instances_dir.glob("*.[jp][pn]g")))
        self.logger.info(f"Found {len(instance_files)} character instances")

        face_metadata = []
        face_count = 0

        for img_path in tqdm(instance_files, desc="Extracting faces"):
            faces = self.face_extractor.extract_faces(img_path)

            for i, face in enumerate(faces):
                # Save face crop
                face_filename = f"{img_path.stem}_face{i}.jpg"
                face_path = self.faces_dir / face_filename
                cv2.imwrite(str(face_path), face['face_crop'])

                # Save metadata
                face_metadata.append({
                    'source_image': str(img_path),
                    'face_id': i,
                    'face_path': str(face_path),
                    'bbox': face['bbox'],
                    'det_score': face['det_score'],
                    'face_width': face['face_width'],
                    'face_height': face['face_height'],
                })

                face_count += 1

        self.logger.info(f"✅ Extracted {face_count} faces from {len(instance_files)} images")

        # Save metadata
        metadata_path = self.faces_dir / "face_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(face_metadata, f, indent=2)

        return face_metadata

    def classify_expressions(self, face_metadata, confidence_threshold=0.5):
        """
        Step 2: Classify expressions using CLIP

        Args:
            face_metadata: List of face dicts from extract_all_faces
            confidence_threshold: Minimum confidence for valid prediction

        Returns:
            Updated face_metadata with expression predictions
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 2: EXPRESSION CLASSIFICATION")
        self.logger.info("=" * 80)

        self.expression_classifier = CLIPExpressionClassifier(
            emotions=self.emotions,
            confidence_threshold=confidence_threshold,
            device=self.device,
            logger=self.logger
        )

        # Load checkpoint if exists
        if self.checkpoint_mgr.exists():
            self.checkpoint_mgr.load()
            self.logger.info(f"Resuming expression classification...")

        # Filter to unprocessed faces
        unprocessed_faces = [
            face_meta for face_meta in face_metadata
            if not self.checkpoint_mgr.is_processed(face_meta['face_path'])
        ]
        self.logger.info(f"Processing {len(unprocessed_faces)} remaining faces (already processed: {len(self.checkpoint_mgr)})...")

        for face_meta in tqdm(unprocessed_faces, desc="Classifying expressions"):
            face_path = Path(face_meta['face_path'])
            face_crop = cv2.imread(str(face_path))

            # Classify expression
            result = self.expression_classifier.classify(face_crop)

            face_meta['expression'] = result['emotion']
            face_meta['expression_confidence'] = result['confidence']
            face_meta['expression_scores'] = result['all_scores']
            face_meta['meets_threshold'] = result['meets_threshold']

            # Estimate intensity
            intensity_level, intensity_score = self.intensity_classifier.classify_intensity(
                face_crop,
                emotion=result['emotion'],
                confidence=result['confidence']
            )

            face_meta['intensity_level'] = intensity_level
            face_meta['intensity_score'] = intensity_score

            # Mark as processed (auto-saves every 50 items)
            self.checkpoint_mgr.mark_processed(face_path)

        # Force save final checkpoint
        self.checkpoint_mgr.save(force=True)

        # Statistics
        total = len(face_metadata)
        meets_threshold = sum(1 for f in face_metadata if f['meets_threshold'])
        self.logger.info(f"✅ Classified {total} faces")
        self.logger.info(f"   Meets confidence threshold ({confidence_threshold}): {meets_threshold} / {total} ({meets_threshold/total:.1%})")

        # Expression distribution
        expression_counts = defaultdict(int)
        for face in face_metadata:
            expression_counts[face['expression']] += 1

        self.logger.info("Expression distribution (all):")
        for expr in sorted(expression_counts.keys()):
            count = expression_counts[expr]
            self.logger.info(f"  {expr}: {count} ({count/total:.1%})")

        return face_metadata

    def filter_quality(
        self,
        face_metadata,
        min_expression_confidence=0.5,
        min_detection_score=0.7,
        min_blur_score=100.0
    ):
        """
        Step 3: Filter faces by quality criteria

        Args:
            face_metadata: List of face dicts with expression predictions
            min_expression_confidence: Minimum expression classification confidence
            min_detection_score: Minimum face detection score
            min_blur_score: Minimum Laplacian variance (higher = sharper)

        Returns:
            Filtered face_metadata
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 3: QUALITY FILTERING")
        self.logger.info("=" * 80)

        filtered = []
        rejection_stats = defaultdict(int)

        for face in tqdm(face_metadata, desc="Quality filtering"):
            # 1. Detection confidence
            if face['det_score'] < min_detection_score:
                rejection_stats['low_detection_score'] += 1
                continue

            # 2. Expression confidence
            if face['expression_confidence'] < min_expression_confidence:
                rejection_stats['low_expression_confidence'] += 1
                continue

            # 3. Blur detection
            face_path = Path(face['face_path'])
            face_gray = cv2.imread(str(face_path), cv2.IMREAD_GRAYSCALE)
            blur_score = cv2.Laplacian(face_gray, cv2.CV_64F).var()

            if blur_score < min_blur_score:
                rejection_stats['too_blurry'] += 1
                continue

            face['blur_score'] = blur_score
            filtered.append(face)

        self.logger.info(f"✅ Quality filtering complete")
        self.logger.info(f"   Kept: {len(filtered)} / {len(face_metadata)} ({len(filtered)/len(face_metadata):.1%})")
        self.logger.info("❌ Rejections:")
        for reason, count in sorted(rejection_stats.items()):
            self.logger.info(f"  - {reason}: {count}")

        return filtered

    def deduplicate_faces(self, face_metadata):
        """
        Step 3.5: Deduplicate faces using perceptual hashing

        Args:
            face_metadata: Filtered face metadata

        Returns:
            Deduplicated face_metadata
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 3.5: DEDUPLICATION")
        self.logger.info("=" * 80)

        deduplicated = []
        duplicate_count = 0

        # Reset deduplicator
        self.deduplicator.reset()

        for face in tqdm(face_metadata, desc="Deduplicating faces"):
            face_path = Path(face['face_path'])

            # Check if duplicate
            is_dup, original_path = self.deduplicator.is_duplicate(face_path)

            if not is_dup:
                deduplicated.append(face)
            else:
                duplicate_count += 1

        self.logger.info(f"✅ Deduplication complete")
        self.logger.info(f"   Kept: {len(deduplicated)} / {len(face_metadata)}")
        self.logger.info(f"   Removed: {duplicate_count} duplicates ({duplicate_count/len(face_metadata):.1%})")

        return deduplicated

    def organize_by_expression(self, face_metadata):
        """
        Step 4: Organize faces into folders by expression

        Args:
            face_metadata: Filtered face metadata

        Returns:
            Dict mapping expression name to list of face dicts
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 4: ORGANIZE BY EXPRESSION")
        self.logger.info("=" * 80)

        # Group by expression
        expression_groups = defaultdict(list)
        for face in face_metadata:
            expression = face['expression']
            expression_groups[expression].append(face)

        # Create folders and copy files
        for expression, faces in expression_groups.items():
            expr_dir = self.organized_dir / expression
            expr_dir.mkdir(parents=True, exist_ok=True)

            for face in faces:
                src_path = Path(face['face_path'])
                dst_path = expr_dir / src_path.name
                shutil.copy(src_path, dst_path)
                face['organized_path'] = str(dst_path)

            self.logger.info(f"📁 {expression}: {len(faces)} faces → {expr_dir}")

        return expression_groups

    def balance_dataset(self, expression_groups, target_per_expression=None):
        """
        Step 5: Balance dataset across expressions

        Args:
            expression_groups: Dict from organize_by_expression
            target_per_expression: Target count (uses self.target_per_expression if None)

        Returns:
            Balanced expression_groups
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 5: BALANCE DATASET")
        self.logger.info("=" * 80)

        target = target_per_expression or self.target_per_expression
        balanced_groups = {}

        for expression, faces in expression_groups.items():
            n_faces = len(faces)

            if n_faces >= target:
                # Undersample
                sampled = list(np.random.choice(faces, size=target, replace=False))
                balanced_groups[expression] = sampled
                self.logger.info(f"📉 {expression}: {n_faces} → {target} (undersampled)")

            else:
                # Oversample (duplicate until target)
                repeats = target // n_faces
                remainder = target % n_faces

                balanced = faces * repeats
                balanced += list(np.random.choice(faces, size=remainder, replace=False))
                balanced_groups[expression] = balanced
                self.logger.info(f"📈 {expression}: {n_faces} → {target} (oversampled {repeats}x + {remainder})")

        return balanced_groups

    def generate_expression_caption(
        self,
        expression: str,
        intensity: str = "moderate",
        include_3d_style: bool = True
    ):
        """
        Generate caption for expression LoRA training

        Args:
            expression: Emotion name
            intensity: subtle, moderate, strong
            include_3d_style: Add 3D style tags

        Returns:
            Caption string
        """
        # Expression descriptors by intensity
        expression_descriptors = {
            'happy': {
                'subtle': "a slight smile, content expression",
                'moderate': "a happy smile, cheerful expression",
                'strong': "a wide smile, very happy and joyful expression",
            },
            'sad': {
                'subtle': "a slightly sad, downcast expression",
                'moderate': "a sad, unhappy expression with frown",
                'strong': "a very sad, crying expression",
            },
            'angry': {
                'subtle': "a slightly annoyed, frowning expression",
                'moderate': "an angry, upset expression with furrowed brows",
                'strong': "a very angry, furious expression",
            },
            'surprised': {
                'subtle': "a slightly surprised, curious expression",
                'moderate': "a surprised, shocked expression with wide eyes",
                'strong': "a very surprised, amazed expression with open mouth",
            },
            'neutral': {
                'subtle': "a calm, relaxed expression",
                'moderate': "a neutral, calm face",
                'strong': "a completely neutral expression, no emotion",
            },
            'fearful': {
                'subtle': "a slightly worried, nervous expression",
                'moderate': "a scared, fearful expression",
                'strong': "a very scared, terrified expression",
            },
            'disgusted': {
                'subtle': "a slightly disgusted expression",
                'moderate': "a disgusted, repulsed expression",
                'strong': "a very disgusted, grossed out expression",
            },
        }

        parts = []

        # Character description (generic, not identity-specific)
        parts.append("a 3d animated character")

        # Expression (primary focus)
        expr_desc = expression_descriptors.get(expression, {}).get(intensity, expression)
        parts.append(expr_desc)

        # 3D style tags
        if include_3d_style:
            parts.extend([
                "pixar style",
                "smooth shading",
                "detailed facial features",
                "expressive animation",
            ])

        caption = ", ".join(parts)
        return caption

    def assemble_final_dataset(self, balanced_groups, character_name=None):
        """
        Step 6: Generate captions and assemble final dataset

        Args:
            balanced_groups: Balanced expression groups
            character_name: Optional character name to prepend

        Returns:
            Dataset statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 6: ASSEMBLE FINAL DATASET")
        self.logger.info("=" * 80)

        images_dir = self.final_dir / "images"
        captions_dir = self.final_dir / "captions"
        images_dir.mkdir(parents=True, exist_ok=True)
        captions_dir.mkdir(parents=True, exist_ok=True)

        dataset_stats = {
            'total_images': 0,
            'expression_counts': {},
            'caption_method': 'template',
        }

        for expression, faces in balanced_groups.items():
            expr_count = 0

            for i, face in enumerate(tqdm(faces, desc=f"Processing {expression}")):
                # Copy face image
                src_path = Path(face['organized_path'])
                img_filename = f"{expression}_{i:04d}.jpg"
                dst_img_path = images_dir / img_filename
                shutil.copy(src_path, dst_img_path)

                # Generate caption with detected intensity
                intensity = face.get('intensity_level', 'moderate')  # Use detected intensity or default
                caption = self.generate_expression_caption(
                    expression,
                    intensity=intensity,
                    include_3d_style=True
                )

                # Optionally prepend character name
                if character_name:
                    caption = f"{character_name}, {caption}"

                # Save caption
                caption_path = captions_dir / f"{expression}_{i:04d}.txt"
                caption_path.write_text(caption, encoding='utf-8')

                expr_count += 1

            dataset_stats['expression_counts'][expression] = expr_count
            dataset_stats['total_images'] += expr_count

        # Save metadata
        metadata_path = self.final_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_stats, f, indent=2)

        self.logger.info(f"✅ Expression LoRA dataset assembled: {dataset_stats['total_images']} images")
        self.logger.info(f"   Expression distribution: {dataset_stats['expression_counts']}")
        self.logger.info(f"   Dataset location: {self.final_dir}")

        return dataset_stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Expression/Emotion LoRA training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python prepare_expression_lora_data.py \\
        --character-instances /path/to/character_instances/ \\
        --output-dir /path/to/training_data/expression_lora/ \\
        --expressions happy sad angry surprised neutral fearful \\
        --target-per-expression 150 \\
        --device cuda

For technical details:
    docs/training/lora_types/02_EXPRESSION_LORA_DEEP_DIVE.md
        """
    )

    # Input
    parser.add_argument(
        "--character-instances",
        type=str,
        required=True,
        help="Directory containing character instances (from SAM2 or clustering)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for training data"
    )

    # Expression selection
    parser.add_argument(
        "--expressions",
        type=str,
        nargs="+",
        default=['happy', 'sad', 'angry', 'surprised', 'neutral', 'fearful'],
        help="Emotions to extract (default: happy sad angry surprised neutral fearful)"
    )

    # Dataset size
    parser.add_argument(
        "--target-per-expression",
        type=int,
        default=150,
        help="Target number of images per expression (default: 150)"
    )

    # Quality filters
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.3,
        help="Face detection confidence threshold (lower for 3D faces, default: 0.3)"
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="Minimum face size in pixels (default: 80)"
    )
    parser.add_argument(
        "--min-expression-confidence",
        type=float,
        default=0.5,
        help="Minimum expression classification confidence (default: 0.5)"
    )
    parser.add_argument(
        "--min-blur-score",
        type=float,
        default=100.0,
        help="Minimum blur score (Laplacian variance, default: 100.0)"
    )

    # Caption options
    parser.add_argument(
        "--character-name",
        type=str,
        default=None,
        help="Character name to prepend to captions (optional, for character-specific expression LoRA)"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help="Device to use (default: cuda)"
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(__name__, level="INFO")

    logger.info("=" * 80)
    logger.info("EXPRESSION/EMOTION LORA DATA PREPARATION")
    logger.info("=" * 80)
    logger.info("⚠️  HIGH-DIFFICULTY TASK: Emotion detection on 3D animated faces")
    logger.info("    Using CLIP zero-shot approach to handle domain gap")
    logger.info("=" * 80)

    # Initialize preparer
    preparer = ExpressionLoRADataPreparer(
        character_instances_dir=args.character_instances,
        output_dir=args.output_dir,
        emotions=args.expressions,
        target_per_expression=args.target_per_expression,
        device=args.device,
        logger=logger
    )

    # Step 1: Extract faces
    face_metadata = preparer.extract_all_faces(
        det_thresh=args.det_thresh,
        min_face_size=args.min_face_size
    )

    if len(face_metadata) == 0:
        logger.error("❌ No faces detected! Check detection threshold and input images.")
        return

    # Step 2: Classify expressions
    face_metadata = preparer.classify_expressions(
        face_metadata,
        confidence_threshold=args.min_expression_confidence
    )

    # Step 3: Filter quality
    filtered_faces = preparer.filter_quality(
        face_metadata,
        min_expression_confidence=args.min_expression_confidence,
        min_detection_score=args.det_thresh,
        min_blur_score=args.min_blur_score
    )

    if len(filtered_faces) == 0:
        logger.error("❌ No faces passed quality filtering! Relax thresholds or check data quality.")
        return

    # Step 3.5: Deduplicate faces
    deduplicated_faces = preparer.deduplicate_faces(filtered_faces)

    if len(deduplicated_faces) == 0:
        logger.error("❌ No faces remaining after deduplication!")
        return

    # Step 4: Organize by expression
    expression_groups = preparer.organize_by_expression(deduplicated_faces)

    # Step 5: Balance dataset
    balanced_groups = preparer.balance_dataset(
        expression_groups,
        target_per_expression=args.target_per_expression
    )

    # Step 6: Assemble final dataset
    dataset_stats = preparer.assemble_final_dataset(
        balanced_groups,
        character_name=args.character_name
    )

    # Clean up checkpoint on successful completion
    preparer.checkpoint_mgr.cleanup()

    logger.info("=" * 80)
    logger.info("✅ EXPRESSION/EMOTION LORA DATA PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"  - final_dataset/images/: {dataset_stats['total_images']} expression images")
    logger.info(f"  - final_dataset/captions/: {dataset_stats['total_images']} caption files")
    logger.info(f"  - final_dataset/metadata.json: Dataset metadata")
    logger.info("")
    logger.info("Expression distribution:")
    for expr, count in sorted(dataset_stats['expression_counts'].items()):
        logger.info(f"  {expr}: {count}")
    logger.info("")
    logger.info("⚠️  IMPORTANT: Manual review recommended!")
    logger.info(f"   Review images in {args.output_dir}/expressions_organized/")
    logger.info("   Expression classification on 3D faces may have errors")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Review organized expressions in {args.output_dir}/expressions_organized/")
    logger.info(f"  2. (Optional) Use interactive selector to refine:")
    logger.info(f"     python scripts/generic/clustering/interactive_expression_selector.py \\")
    logger.info(f"         --expression-dir {args.output_dir}/expressions_organized \\")
    logger.info(f"         --output-dir {args.output_dir}/expressions_reviewed")
    logger.info(f"  3. Generate LoRA training config:")
    logger.info(f"     # Copy and modify configs/training/expression_lora_sdxl.toml")
    logger.info(f"  4. Train Expression LoRA using Kohya_ss sd-scripts")
    logger.info(f"  5. Test with:")
    logger.info(f"     python scripts/evaluation/test_lora_checkpoints.py ...")


if __name__ == "__main__":
    main()
