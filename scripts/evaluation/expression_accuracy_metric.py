#!/usr/bin/env python3
"""
Expression Accuracy Metric - FER Integration
============================================

Evaluates facial expression accuracy for Expression LoRA checkpoints using
Facial Expression Recognition (FER).

Metrics:
- Expression Classification Accuracy (does detected expression match expected?)
- Expression Confidence Score (how confident is the classification?)
- Face Detection Success Rate (percentage of images with detectable faces)
- Emotion Distribution (distribution across detected emotions)

Supported Expressions:
- happy, sad, angry, surprised, fearful, disgusted, neutral

Usage:
    python scripts/evaluation/expression_accuracy_metric.py \\
        --image-dir /path/to/generated_images \\
        --expected-expression "happy" \\
        --output-dir /path/to/metrics \\
        --device cuda

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import cv2
import numpy as np
from PIL import Image
import torch

# FER library (pip install fer)
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    warnings.warn("FER library not available. Install: pip install fer")

# DeepFace alternative (pip install deepface)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Expression Definitions
# ============================================================================

# Standard 7 basic emotions (Ekman model)
EXPRESSION_CATEGORIES = {
    'happy': {
        'description': 'Joyful, smiling, positive expression',
        'aliases': ['joy', 'smile', 'excited'],
        'deepface_label': 'happy'
    },
    'sad': {
        'description': 'Downcast, melancholy expression',
        'aliases': ['sadness', 'sorrow'],
        'deepface_label': 'sad'
    },
    'angry': {
        'description': 'Furrowed brow, intense, angry expression',
        'aliases': ['anger', 'mad'],
        'deepface_label': 'angry'
    },
    'surprised': {
        'description': 'Wide eyes, open mouth, surprised expression',
        'aliases': ['surprise', 'shock'],
        'deepface_label': 'surprise'
    },
    'fearful': {
        'description': 'Worried, tense, fearful expression',
        'aliases': ['fear', 'afraid', 'scared'],
        'deepface_label': 'fear'
    },
    'disgusted': {
        'description': 'Wrinkled nose, disgust expression',
        'aliases': ['disgust', 'revulsion'],
        'deepface_label': 'disgust'
    },
    'neutral': {
        'description': 'Calm, neutral, relaxed expression',
        'aliases': ['calm', 'relaxed'],
        'deepface_label': 'neutral'
    }
}

# Emotion label mapping (FER uses different labels)
FER_TO_STANDARD = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'surprise': 'surprised',
    'fear': 'fearful',
    'disgust': 'disgusted',
    'neutral': 'neutral'
}

STANDARD_TO_FER = {v: k for k, v in FER_TO_STANDARD.items()}


# ============================================================================
# FER Detector Wrapper
# ============================================================================

class FERDetector:
    """Facial Expression Recognition detector wrapper."""

    def __init__(
        self,
        backend: str = 'fer',
        device: str = 'cuda',
        confidence_threshold: float = 0.5
    ):
        """
        Initialize FER detector.

        Args:
            backend: FER backend ('fer' or 'deepface')
            device: Device for inference
            confidence_threshold: Minimum confidence for classification
        """
        self.backend = backend
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Initialize backend
        if backend == 'fer':
            if not FER_AVAILABLE:
                raise ImportError("FER library not available. Install: pip install fer")

            # Use MTCNN for face detection
            self.detector = FER(mtcnn=True)
            logger.info("Initialized FER detector (MTCNN backend)")

        elif backend == 'deepface':
            if not DEEPFACE_AVAILABLE:
                raise ImportError("DeepFace library not available. Install: pip install deepface")

            logger.info("Initialized DeepFace detector")

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def detect_expression(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect facial expression in an image.

        Args:
            image: Input image (RGB format, HWC)

        Returns:
            Dictionary with expression, confidence, and emotion scores
        """
        try:
            if self.backend == 'fer':
                return self._detect_fer(image)
            elif self.backend == 'deepface':
                return self._detect_deepface(image)
        except Exception as e:
            logger.warning(f"Expression detection failed: {e}")
            return None

    def _detect_fer(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect expression using FER library."""
        # FER expects BGR format (OpenCV convention)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detect emotions
        result = self.detector.detect_emotions(image_bgr)

        if not result or len(result) == 0:
            return None

        # Get first face (highest confidence)
        face = result[0]
        emotions = face['emotions']

        # Find dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]

        # Convert to standard labels
        standard_emotion = FER_TO_STANDARD.get(dominant_emotion, dominant_emotion)

        return {
            'expression': standard_emotion,
            'confidence': confidence,
            'emotions': {FER_TO_STANDARD.get(k, k): v for k, v in emotions.items()},
            'bbox': face['box']
        }

    def _detect_deepface(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect expression using DeepFace library."""
        # DeepFace expects RGB format
        try:
            # Analyze emotions
            result = DeepFace.analyze(
                image,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if not result:
                return None

            # Handle list or dict return
            if isinstance(result, list):
                result = result[0]

            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            confidence = emotions[dominant_emotion] / 100.0  # Convert to 0-1

            # Convert to standard labels
            standard_emotion = dominant_emotion.lower()

            return {
                'expression': standard_emotion,
                'confidence': confidence,
                'emotions': {k.lower(): v/100.0 for k, v in emotions.items()},
                'bbox': result.get('region', {})
            }

        except Exception as e:
            logger.debug(f"DeepFace analysis failed: {e}")
            return None


# ============================================================================
# Expression Analysis Functions
# ============================================================================

def normalize_expression_label(label: str) -> str:
    """
    Normalize expression label to standard format.

    Args:
        label: Input expression label

    Returns:
        Normalized label
    """
    label_lower = label.lower().strip()

    # Direct match
    if label_lower in EXPRESSION_CATEGORIES:
        return label_lower

    # Check aliases
    for standard_label, info in EXPRESSION_CATEGORIES.items():
        if label_lower in info.get('aliases', []):
            return standard_label

    return label_lower


def calculate_expression_similarity(
    detected: str,
    expected: str,
    emotion_scores: Dict[str, float]
) -> float:
    """
    Calculate similarity between detected and expected expression.

    Args:
        detected: Detected expression label
        expected: Expected expression label
        emotion_scores: Dictionary of all emotion scores

    Returns:
        Similarity score (0-1)
    """
    detected_norm = normalize_expression_label(detected)
    expected_norm = normalize_expression_label(expected)

    # Exact match
    if detected_norm == expected_norm:
        return 1.0

    # Partial credit for similar emotions
    similar_pairs = {
        ('happy', 'excited'): 0.7,
        ('sad', 'fearful'): 0.5,
        ('angry', 'disgusted'): 0.5,
        ('surprised', 'fearful'): 0.4,
    }

    pair = tuple(sorted([detected_norm, expected_norm]))
    if pair in similar_pairs:
        return similar_pairs[pair]

    # Use emotion score of expected emotion
    expected_score = emotion_scores.get(expected_norm, 0.0)
    return expected_score


# ============================================================================
# Main Metric Computation
# ============================================================================

class ExpressionAccuracyMetric:
    """Compute expression accuracy metrics for LoRA evaluation."""

    def __init__(
        self,
        backend: str = 'fer',
        device: str = 'cuda',
        confidence_threshold: float = 0.5
    ):
        """
        Initialize expression accuracy metric.

        Args:
            backend: FER backend ('fer' or 'deepface')
            device: Device for inference
            confidence_threshold: Minimum classification confidence
        """
        self.detector = FERDetector(
            backend=backend,
            device=device,
            confidence_threshold=confidence_threshold
        )
        self.confidence_threshold = confidence_threshold

    def evaluate_image(
        self,
        image_path: Path,
        expected_expression: str
    ) -> Dict[str, Any]:
        """
        Evaluate expression accuracy for a single image.

        Args:
            image_path: Path to generated image
            expected_expression: Expected expression category

        Returns:
            Metrics dictionary
        """
        # Load image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect expression
        detection = self.detector.detect_expression(image_rgb)

        if detection is None:
            return {
                'success': False,
                'error': 'no_face_detected',
                'expression_match': False,
                'confidence': 0.0,
                'similarity': 0.0
            }

        # Normalize labels
        detected = normalize_expression_label(detection['expression'])
        expected = normalize_expression_label(expected_expression)

        # Calculate match
        is_exact_match = (detected == expected)
        similarity = calculate_expression_similarity(
            detected,
            expected,
            detection['emotions']
        )

        return {
            'success': True,
            'detected_expression': detected,
            'expected_expression': expected,
            'expression_match': is_exact_match,
            'confidence': detection['confidence'],
            'similarity': similarity,
            'emotion_scores': detection['emotions']
        }

    def evaluate_batch(
        self,
        image_dir: Path,
        expected_expression: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate expression accuracy for a batch of images.

        Args:
            image_dir: Directory containing generated images
            expected_expression: Expected expression category
            output_dir: Optional directory to save results

        Returns:
            Aggregated metrics
        """
        image_paths = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))

        if not image_paths:
            logger.warning(f"No images found in {image_dir}")
            return {'error': 'no_images_found'}

        logger.info(f"Evaluating {len(image_paths)} images for expression: {expected_expression}")

        results = []
        for img_path in image_paths:
            result = self.evaluate_image(img_path, expected_expression)
            result['image_name'] = img_path.name
            results.append(result)

        # Aggregate metrics
        successful = [r for r in results if r['success']]

        if not successful:
            logger.error("No successful detections!")
            return {'error': 'all_detections_failed', 'total_images': len(image_paths)}

        # Calculate statistics
        exact_matches = sum(r['expression_match'] for r in successful)
        avg_similarity = np.mean([r['similarity'] for r in successful])

        # Emotion distribution
        detected_expressions = [r['detected_expression'] for r in successful]
        expression_counts = {}
        for expr in detected_expressions:
            expression_counts[expr] = expression_counts.get(expr, 0) + 1

        metrics = {
            'total_images': len(image_paths),
            'successful_detections': len(successful),
            'detection_rate': len(successful) / len(image_paths),
            'exact_match_rate': exact_matches / len(successful),
            'avg_confidence': np.mean([r['confidence'] for r in successful]),
            'avg_similarity': avg_similarity,
            'expected_expression': normalize_expression_label(expected_expression),
            'expression_distribution': expression_counts,
            'per_image_results': results
        }

        # Save results
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'expression_accuracy_{expected_expression}.json'

            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Saved metrics to {output_file}")

        return metrics


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate facial expression accuracy using FER',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--image-dir',
        type=Path,
        required=True,
        help='Directory containing generated images'
    )

    parser.add_argument(
        '--expected-expression',
        type=str,
        required=True,
        choices=list(EXPRESSION_CATEGORIES.keys()),
        help='Expected expression category'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for metrics (default: image_dir/expression_metrics)'
    )

    parser.add_argument(
        '--backend',
        type=str,
        default='fer',
        choices=['fer', 'deepface'],
        help='FER backend (default: fer)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for inference (default: cuda)'
    )

    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Minimum classification confidence (default: 0.5)'
    )

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = args.image_dir / 'expression_metrics'

    # Initialize metric
    metric = ExpressionAccuracyMetric(
        backend=args.backend,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )

    # Evaluate
    results = metric.evaluate_batch(
        args.image_dir,
        args.expected_expression,
        args.output_dir
    )

    # Print summary
    if 'error' not in results:
        print("\n" + "="*60)
        print("EXPRESSION ACCURACY EVALUATION RESULTS")
        print("="*60)
        print(f"Expected Expression:  {results['expected_expression']}")
        print(f"Total Images:         {results['total_images']}")
        print(f"Successful Detections: {results['successful_detections']}")
        print(f"Detection Rate:       {results['detection_rate']:.2%}")
        print(f"Exact Match Rate:     {results['exact_match_rate']:.2%}")
        print(f"Avg Confidence:       {results['avg_confidence']:.3f}")
        print(f"Avg Similarity:       {results['avg_similarity']:.3f}")
        print(f"\nExpression Distribution:")
        for expr, count in sorted(results['expression_distribution'].items(), key=lambda x: -x[1]):
            pct = count / results['successful_detections'] * 100
            print(f"  {expr:12s}: {count:3d} ({pct:5.1f}%)")
        print("="*60)
    else:
        print(f"\n❌ Error: {results['error']}")


if __name__ == '__main__':
    main()
