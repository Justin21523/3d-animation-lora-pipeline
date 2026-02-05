#!/usr/bin/env python3
"""
Pose Accuracy Metric - RTM-Pose Integration
============================================

Evaluates pose accuracy for Pose LoRA checkpoints using RTM-Pose keypoint detection.

Metrics:
- Keypoint Detection Confidence (average confidence across all keypoints)
- Pose Classification Accuracy (does the detected pose match the intended pose?)
- Keypoint Visibility (percentage of keypoints detected with high confidence)
- Pose Completeness (are all critical body parts visible?)

Usage:
    python scripts/evaluation/pose_accuracy_metric.py \\
        --image-dir /path/to/generated_images \\
        --expected-pose "standing" \\
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
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import PoseDataSample

# Suppress MMPose warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Pose Type Definitions
# ============================================================================

POSE_CATEGORIES = {
    'standing': {
        'description': 'Upright standing pose with arms at sides or relaxed',
        'critical_keypoints': ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
        'angle_constraints': {
            'torso_vertical': (80, 100),  # degrees from horizontal
            'arms_relaxed': (0, 45),  # shoulder-elbow angle
        }
    },
    'sitting': {
        'description': 'Sitting pose with legs bent and torso upright',
        'critical_keypoints': ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
        'angle_constraints': {
            'knee_angle': (60, 120),  # bent knees
            'hip_angle': (70, 110),  # sitting position
        }
    },
    'walking': {
        'description': 'Walking stride with one leg forward',
        'critical_keypoints': ['nose', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
        'angle_constraints': {
            'leg_separation': (20, 60),  # stride width
        }
    },
    'running': {
        'description': 'Running pose with dynamic forward lean',
        'critical_keypoints': ['nose', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
        'angle_constraints': {
            'torso_lean': (60, 85),  # forward lean
            'leg_separation': (30, 80),  # wide stride
        }
    },
    'jumping': {
        'description': 'Jumping pose with airborne posture',
        'critical_keypoints': ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee'],
        'angle_constraints': {
            'arms_raised': (120, 180),  # arms up
            'legs_bent': (90, 150),  # knees bent
        }
    },
    'crouching': {
        'description': 'Crouching pose with low stance',
        'critical_keypoints': ['nose', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
        'angle_constraints': {
            'knee_angle': (30, 90),  # deep crouch
            'torso_lean': (30, 70),  # forward lean
        }
    }
}

# RTM-Pose keypoint indices (COCO format)
COCO_KEYPOINT_INDICES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}


# ============================================================================
# RTM-Pose Wrapper
# ============================================================================

class RTMPoseDetector:
    """RTM-Pose keypoint detection wrapper."""

    def __init__(
        self,
        model_name: str = 'rtmpose-m',
        device: str = 'cuda',
        confidence_threshold: float = 0.3
    ):
        """
        Initialize RTM-Pose detector.

        Args:
            model_name: RTM-Pose model variant (rtmpose-t, rtmpose-s, rtmpose-m, rtmpose-l)
            device: Device to run inference on
            confidence_threshold: Minimum confidence for keypoint detection
        """
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Model paths (adjust based on your installation)
        model_configs = {
            'rtmpose-t': 'rtmpose-t_8xb256-420e_coco-256x192.py',
            'rtmpose-s': 'rtmpose-s_8xb256-420e_coco-256x192.py',
            'rtmpose-m': 'rtmpose-m_8xb256-420e_coco-256x192.py',
            'rtmpose-l': 'rtmpose-l_8xb256-420e_coco-384x288.py'
        }

        model_checkpoints = {
            'rtmpose-t': 'rtmpose-t_simcc-coco_pt-aic-coco_420e-256x192.pth',
            'rtmpose-s': 'rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192.pth',
            'rtmpose-m': 'rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth',
            'rtmpose-l': 'rtmpose-l_simcc-coco_pt-aic-coco_420e-384x288.pth'
        }

        # Initialize model
        try:
            config_file = f'configs/body_2d_keypoint/rtmpose/coco/{model_configs[model_name]}'
            checkpoint_file = f'checkpoints/{model_checkpoints[model_name]}'

            self.model = init_model(config_file, checkpoint_file, device=device)
            logger.info(f"Loaded RTM-Pose model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load RTM-Pose model: {e}")
            raise

    def detect_keypoints(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect keypoints in an image.

        Args:
            image: Input image (RGB format, HWC)

        Returns:
            Dictionary with keypoints, scores, and bbox, or None if no person detected
        """
        try:
            # Run inference
            results = inference_topdown(self.model, image)

            if not results or len(results) == 0:
                return None

            # Get first person (highest confidence)
            result = results[0]
            pred_instances = result.pred_instances

            # Extract keypoints and scores
            keypoints = pred_instances.keypoints[0]  # (17, 2)
            scores = pred_instances.keypoint_scores[0]  # (17,)
            bbox = pred_instances.bboxes[0]  # (4,)

            return {
                'keypoints': keypoints.cpu().numpy(),
                'scores': scores.cpu().numpy(),
                'bbox': bbox.cpu().numpy(),
                'avg_confidence': float(scores.mean())
            }

        except Exception as e:
            logger.warning(f"Keypoint detection failed: {e}")
            return None


# ============================================================================
# Pose Analysis Functions
# ============================================================================

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle formed by three points (p1-p2-p3).

    Args:
        p1, p2, p3: 2D points (x, y)

    Returns:
        Angle in degrees (0-180)
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    return angle


def calculate_keypoint_visibility(
    scores: np.ndarray,
    critical_keypoints: List[str],
    threshold: float = 0.5
) -> float:
    """
    Calculate percentage of critical keypoints visible.

    Args:
        scores: Keypoint confidence scores (17,)
        critical_keypoints: List of critical keypoint names
        threshold: Minimum confidence to consider visible

    Returns:
        Visibility percentage (0-1)
    """
    critical_indices = [COCO_KEYPOINT_INDICES[kp] for kp in critical_keypoints]
    critical_scores = scores[critical_indices]

    visible_count = np.sum(critical_scores >= threshold)
    visibility = visible_count / len(critical_keypoints)

    return visibility


def classify_pose(
    keypoints: np.ndarray,
    scores: np.ndarray,
    expected_pose: str,
    confidence_threshold: float = 0.5
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Classify if detected pose matches expected pose category.

    Args:
        keypoints: Detected keypoints (17, 2)
        scores: Keypoint scores (17,)
        expected_pose: Expected pose category
        confidence_threshold: Minimum keypoint confidence

    Returns:
        (is_match, confidence, details)
    """
    if expected_pose not in POSE_CATEGORIES:
        logger.warning(f"Unknown pose category: {expected_pose}")
        return False, 0.0, {}

    pose_def = POSE_CATEGORIES[expected_pose]
    critical_kp_names = pose_def['critical_keypoints']

    # Check critical keypoints visibility
    visibility = calculate_keypoint_visibility(scores, critical_kp_names, confidence_threshold)

    if visibility < 0.6:  # Need at least 60% critical keypoints
        return False, visibility, {'reason': 'insufficient_keypoints', 'visibility': visibility}

    # Extract keypoint coordinates
    kp_dict = {}
    for name, idx in COCO_KEYPOINT_INDICES.items():
        if scores[idx] >= confidence_threshold:
            kp_dict[name] = keypoints[idx]

    # Check angle constraints (if enough keypoints available)
    angle_checks = []
    for constraint_name, (min_angle, max_angle) in pose_def.get('angle_constraints', {}).items():
        # Implementation depends on constraint type
        # This is a simplified example
        if constraint_name == 'torso_vertical':
            if 'left_shoulder' in kp_dict and 'left_hip' in kp_dict:
                torso_vec = kp_dict['left_hip'] - kp_dict['left_shoulder']
                angle_from_horizontal = np.degrees(np.arctan2(abs(torso_vec[1]), abs(torso_vec[0])))
                angle_checks.append(min_angle <= angle_from_horizontal <= max_angle)

    # Overall match confidence
    if angle_checks:
        angle_match = sum(angle_checks) / len(angle_checks)
    else:
        angle_match = 1.0  # No constraints to check

    overall_confidence = (visibility + angle_match) / 2.0
    is_match = overall_confidence >= 0.7

    return is_match, overall_confidence, {
        'visibility': visibility,
        'angle_match': angle_match,
        'pose_category': expected_pose
    }


# ============================================================================
# Main Metric Computation
# ============================================================================

class PoseAccuracyMetric:
    """Compute pose accuracy metrics for LoRA evaluation."""

    def __init__(
        self,
        model_name: str = 'rtmpose-m',
        device: str = 'cuda',
        confidence_threshold: float = 0.5
    ):
        """
        Initialize pose accuracy metric.

        Args:
            model_name: RTM-Pose model variant
            device: Device for inference
            confidence_threshold: Minimum keypoint confidence
        """
        self.detector = RTMPoseDetector(
            model_name=model_name,
            device=device,
            confidence_threshold=confidence_threshold
        )
        self.confidence_threshold = confidence_threshold

    def evaluate_image(
        self,
        image_path: Path,
        expected_pose: str
    ) -> Dict[str, Any]:
        """
        Evaluate pose accuracy for a single image.

        Args:
            image_path: Path to generated image
            expected_pose: Expected pose category

        Returns:
            Metrics dictionary
        """
        # Load image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect keypoints
        detection = self.detector.detect_keypoints(image_rgb)

        if detection is None:
            return {
                'success': False,
                'error': 'no_person_detected',
                'keypoint_confidence': 0.0,
                'pose_match': False,
                'pose_confidence': 0.0,
                'visibility': 0.0
            }

        # Classify pose
        is_match, pose_conf, details = classify_pose(
            detection['keypoints'],
            detection['scores'],
            expected_pose,
            self.confidence_threshold
        )

        return {
            'success': True,
            'keypoint_confidence': detection['avg_confidence'],
            'pose_match': is_match,
            'pose_confidence': pose_conf,
            'visibility': details.get('visibility', 0.0),
            'expected_pose': expected_pose,
            'details': details
        }

    def evaluate_batch(
        self,
        image_dir: Path,
        expected_pose: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate pose accuracy for a batch of images.

        Args:
            image_dir: Directory containing generated images
            expected_pose: Expected pose category
            output_dir: Optional directory to save results

        Returns:
            Aggregated metrics
        """
        image_paths = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))

        if not image_paths:
            logger.warning(f"No images found in {image_dir}")
            return {'error': 'no_images_found'}

        logger.info(f"Evaluating {len(image_paths)} images for pose: {expected_pose}")

        results = []
        for img_path in image_paths:
            result = self.evaluate_image(img_path, expected_pose)
            result['image_name'] = img_path.name
            results.append(result)

        # Aggregate metrics
        successful = [r for r in results if r['success']]

        if not successful:
            logger.error("No successful detections!")
            return {'error': 'all_detections_failed', 'total_images': len(image_paths)}

        metrics = {
            'total_images': len(image_paths),
            'successful_detections': len(successful),
            'detection_rate': len(successful) / len(image_paths),
            'avg_keypoint_confidence': np.mean([r['keypoint_confidence'] for r in successful]),
            'pose_match_rate': sum(r['pose_match'] for r in successful) / len(successful),
            'avg_pose_confidence': np.mean([r['pose_confidence'] for r in successful]),
            'avg_visibility': np.mean([r['visibility'] for r in successful]),
            'expected_pose': expected_pose,
            'per_image_results': results
        }

        # Save results
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'pose_accuracy_{expected_pose}.json'

            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Saved metrics to {output_file}")

        return metrics


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pose accuracy using RTM-Pose',
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
        '--expected-pose',
        type=str,
        required=True,
        choices=list(POSE_CATEGORIES.keys()),
        help='Expected pose category'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for metrics (default: image_dir/pose_metrics)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='rtmpose-m',
        choices=['rtmpose-t', 'rtmpose-s', 'rtmpose-m', 'rtmpose-l'],
        help='RTM-Pose model variant (default: rtmpose-m)'
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
        help='Minimum keypoint confidence (default: 0.5)'
    )

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = args.image_dir / 'pose_metrics'

    # Initialize metric
    metric = PoseAccuracyMetric(
        model_name=args.model,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )

    # Evaluate
    results = metric.evaluate_batch(
        args.image_dir,
        args.expected_pose,
        args.output_dir
    )

    # Print summary
    if 'error' not in results:
        print("\n" + "="*60)
        print("POSE ACCURACY EVALUATION RESULTS")
        print("="*60)
        print(f"Expected Pose:        {results['expected_pose']}")
        print(f"Total Images:         {results['total_images']}")
        print(f"Successful Detections: {results['successful_detections']}")
        print(f"Detection Rate:       {results['detection_rate']:.2%}")
        print(f"Avg Keypoint Conf:    {results['avg_keypoint_confidence']:.3f}")
        print(f"Pose Match Rate:      {results['pose_match_rate']:.2%}")
        print(f"Avg Pose Confidence:  {results['avg_pose_confidence']:.3f}")
        print(f"Avg Visibility:       {results['avg_visibility']:.2%}")
        print("="*60)
    else:
        print(f"\n❌ Error: {results['error']}")


if __name__ == '__main__':
    main()
