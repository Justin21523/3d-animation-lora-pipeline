#!/usr/bin/env python3
"""
Action Clarity Metric - Optical Flow Analysis
=============================================

Evaluates action/motion clarity for Action LoRA checkpoints using optical flow
and motion analysis.

Metrics:
- Motion Magnitude (average motion strength across frames)
- Motion Coherence (how consistent is the motion direction?)
- Motion Focus (is motion concentrated on character vs background?)
- Action Recognition (does detected action match expected action?)

Supported Actions:
- running, jumping, waving, pointing, throwing, catching, climbing, dancing

Usage:
    python scripts/evaluation/action_clarity_metric.py \\
        --image-dir /path/to/generated_images \\
        --expected-action "running" \\
        --output-dir /path/to/metrics \\
        --device cuda

Note: This metric works best with image sequences (multi-frame generation).
      For single images, it analyzes motion cues from pose and blur patterns.

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

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Action Definitions
# ============================================================================

ACTION_CATEGORIES = {
    'running': {
        'description': 'Dynamic running action with forward motion',
        'expected_motion': 'high',
        'motion_pattern': 'directional',
        'blur_tolerance': 'high',
        'key_features': ['leg_motion', 'arm_swing', 'forward_lean']
    },
    'jumping': {
        'description': 'Jumping action pose, airborne leap',
        'expected_motion': 'high',
        'motion_pattern': 'vertical',
        'blur_tolerance': 'medium',
        'key_features': ['airborne', 'arms_raised', 'leg_extension']
    },
    'waving': {
        'description': 'Waving gesture with hand motion',
        'expected_motion': 'medium',
        'motion_pattern': 'localized',
        'blur_tolerance': 'low',
        'key_features': ['hand_raised', 'arm_motion']
    },
    'pointing': {
        'description': 'Pointing gesture with extended arm',
        'expected_motion': 'low',
        'motion_pattern': 'static',
        'blur_tolerance': 'low',
        'key_features': ['arm_extended', 'finger_direction']
    },
    'throwing': {
        'description': 'Throwing motion with athletic action',
        'expected_motion': 'high',
        'motion_pattern': 'directional',
        'blur_tolerance': 'high',
        'key_features': ['arm_extended', 'torso_rotation', 'follow_through']
    },
    'catching': {
        'description': 'Catching action with arms reaching',
        'expected_motion': 'medium',
        'motion_pattern': 'convergent',
        'blur_tolerance': 'medium',
        'key_features': ['arms_reaching', 'hands_open', 'focused_gaze']
    },
    'climbing': {
        'description': 'Climbing motion with ascending action',
        'expected_motion': 'medium',
        'motion_pattern': 'vertical',
        'blur_tolerance': 'medium',
        'key_features': ['hands_gripping', 'leg_lift', 'upward_motion']
    },
    'dancing': {
        'description': 'Dancing action with rhythmic movement',
        'expected_motion': 'high',
        'motion_pattern': 'complex',
        'blur_tolerance': 'high',
        'key_features': ['body_rotation', 'arm_motion', 'leg_motion']
    }
}

MOTION_LEVEL_THRESHOLDS = {
    'low': (0.0, 0.3),
    'medium': (0.3, 0.7),
    'high': (0.7, 1.0)
}


# ============================================================================
# Optical Flow Analysis
# ============================================================================

class OpticalFlowAnalyzer:
    """Analyze motion using optical flow (Farneback algorithm)."""

    def __init__(self, flow_method: str = 'farneback'):
        """
        Initialize optical flow analyzer.

        Args:
            flow_method: Flow estimation method ('farneback' or 'lucaskanade')
        """
        self.flow_method = flow_method
        logger.info(f"Initialized optical flow analyzer: {flow_method}")

    def compute_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> np.ndarray:
        """
        Compute optical flow between two frames.

        Args:
            frame1: First frame (grayscale)
            frame2: Second frame (grayscale)

        Returns:
            Flow field (H, W, 2) with (u, v) motion vectors
        """
        if self.flow_method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        else:
            # Lucas-Kanade (feature-based)
            # For simplicity, we'll use Farneback for dense flow
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

        return flow

    def analyze_flow(self, flow: np.ndarray) -> Dict[str, float]:
        """
        Analyze optical flow field.

        Args:
            flow: Flow field (H, W, 2)

        Returns:
            Dictionary with motion metrics
        """
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Motion metrics
        avg_magnitude = np.mean(magnitude)
        max_magnitude = np.max(magnitude)
        std_magnitude = np.std(magnitude)

        # Motion coherence (how consistent is the direction?)
        # Use circular statistics for angles
        mean_angle = np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle)))
        angle_variance = 1 - np.sqrt(np.mean(np.cos(angle - mean_angle))**2 + np.mean(np.sin(angle - mean_angle))**2)

        # Motion focus (ratio of high-motion pixels)
        high_motion_threshold = np.percentile(magnitude, 75)
        high_motion_ratio = np.sum(magnitude > high_motion_threshold) / magnitude.size

        return {
            'avg_magnitude': float(avg_magnitude),
            'max_magnitude': float(max_magnitude),
            'std_magnitude': float(std_magnitude),
            'coherence': float(1 - angle_variance),  # 1 = fully coherent
            'motion_focus': float(high_motion_ratio),
            'dominant_direction': float(mean_angle)
        }


# ============================================================================
# Motion Blur Analysis (for single images)
# ============================================================================

class MotionBlurAnalyzer:
    """Analyze motion blur patterns in static images."""

    def __init__(self):
        """Initialize motion blur analyzer."""
        logger.info("Initialized motion blur analyzer")

    def detect_motion_blur(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect and quantify motion blur in an image.

        Args:
            image: Input image (grayscale)

        Returns:
            Dictionary with blur metrics
        """
        # Compute Laplacian variance (overall sharpness)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian_var = laplacian.var()

        # Directional blur detection using Sobel filters
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude and direction
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        angle = np.arctan2(sobel_y, sobel_x)

        # Analyze gradient distribution
        avg_magnitude = np.mean(magnitude)
        std_magnitude = np.std(magnitude)

        # Directional consistency (motion blur creates consistent gradients)
        angle_variance = np.var(angle)

        # Blur score (lower Laplacian = more blur)
        # Normalize to 0-1 range (assuming typical range 0-1000)
        blur_score = 1.0 - min(laplacian_var / 1000.0, 1.0)

        return {
            'blur_score': float(blur_score),
            'gradient_magnitude': float(avg_magnitude),
            'gradient_std': float(std_magnitude),
            'directional_consistency': float(1.0 / (1.0 + angle_variance))
        }

    def classify_motion_level(
        self,
        blur_metrics: Dict[str, float]
    ) -> Tuple[str, float]:
        """
        Classify motion level based on blur metrics.

        Args:
            blur_metrics: Blur analysis results

        Returns:
            (motion_level, confidence)
        """
        blur_score = blur_metrics['blur_score']
        directional = blur_metrics['directional_consistency']

        # High motion: high blur + directional consistency
        if blur_score > 0.6 and directional > 0.5:
            return 'high', 0.8

        # Medium motion: moderate blur
        elif blur_score > 0.3:
            return 'medium', 0.6

        # Low motion: minimal blur
        else:
            return 'low', 0.7


# ============================================================================
# Action Classification (heuristic-based)
# ============================================================================

def classify_action_from_motion(
    motion_metrics: Dict[str, float],
    expected_action: str
) -> Tuple[bool, float]:
    """
    Classify if motion pattern matches expected action.

    Args:
        motion_metrics: Motion analysis results
        expected_action: Expected action category

    Returns:
        (is_match, confidence)
    """
    if expected_action not in ACTION_CATEGORIES:
        logger.warning(f"Unknown action category: {expected_action}")
        return False, 0.0

    action_def = ACTION_CATEGORIES[expected_action]
    expected_motion_level = action_def['expected_motion']

    # Get motion level thresholds
    min_motion, max_motion = MOTION_LEVEL_THRESHOLDS[expected_motion_level]

    # Check if motion magnitude matches expected level
    avg_magnitude = motion_metrics.get('avg_magnitude', 0.0)

    # Normalize magnitude to 0-1 (assuming typical range 0-50 pixels)
    normalized_magnitude = min(avg_magnitude / 50.0, 1.0)

    # Check if within expected range
    if min_motion <= normalized_magnitude <= max_motion:
        confidence = 0.7 + 0.3 * motion_metrics.get('coherence', 0.5)
        return True, confidence
    else:
        # Partial credit based on proximity
        if normalized_magnitude < min_motion:
            distance = min_motion - normalized_magnitude
        else:
            distance = normalized_magnitude - max_motion

        confidence = max(0.3, 0.7 - distance)
        return False, confidence


# ============================================================================
# Main Metric Computation
# ============================================================================

class ActionClarityMetric:
    """Compute action clarity metrics for LoRA evaluation."""

    def __init__(
        self,
        flow_method: str = 'farneback',
        use_sequences: bool = False
    ):
        """
        Initialize action clarity metric.

        Args:
            flow_method: Optical flow method
            use_sequences: If True, analyze image sequences; if False, analyze single images
        """
        self.flow_analyzer = OpticalFlowAnalyzer(flow_method)
        self.blur_analyzer = MotionBlurAnalyzer()
        self.use_sequences = use_sequences

    def evaluate_single_image(
        self,
        image_path: Path,
        expected_action: str
    ) -> Dict[str, Any]:
        """
        Evaluate action clarity for a single image (motion blur analysis).

        Args:
            image_path: Path to generated image
            expected_action: Expected action category

        Returns:
            Metrics dictionary
        """
        # Load image
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Analyze motion blur
        blur_metrics = self.blur_analyzer.detect_motion_blur(gray)

        # Classify motion level
        motion_level, motion_confidence = self.blur_analyzer.classify_motion_level(blur_metrics)

        # Check if matches expected action
        action_def = ACTION_CATEGORIES.get(expected_action, {})
        expected_motion = action_def.get('expected_motion', 'medium')

        is_match = (motion_level == expected_motion)

        return {
            'success': True,
            'method': 'motion_blur',
            'blur_score': blur_metrics['blur_score'],
            'motion_level': motion_level,
            'motion_confidence': motion_confidence,
            'expected_motion': expected_motion,
            'action_match': is_match,
            'blur_metrics': blur_metrics
        }

    def evaluate_image_sequence(
        self,
        image_paths: List[Path],
        expected_action: str
    ) -> Dict[str, Any]:
        """
        Evaluate action clarity for an image sequence (optical flow analysis).

        Args:
            image_paths: List of sequential image paths
            expected_action: Expected action category

        Returns:
            Metrics dictionary
        """
        if len(image_paths) < 2:
            logger.warning("Need at least 2 images for sequence analysis")
            return self.evaluate_single_image(image_paths[0], expected_action)

        # Load images
        frames = []
        for img_path in image_paths[:5]:  # Analyze first 5 frames
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

        # Compute optical flow between consecutive frames
        flow_results = []
        for i in range(len(frames) - 1):
            flow = self.flow_analyzer.compute_flow(frames[i], frames[i+1])
            flow_metrics = self.flow_analyzer.analyze_flow(flow)
            flow_results.append(flow_metrics)

        # Aggregate flow metrics
        avg_magnitude = np.mean([r['avg_magnitude'] for r in flow_results])
        avg_coherence = np.mean([r['coherence'] for r in flow_results])
        avg_focus = np.mean([r['motion_focus'] for r in flow_results])

        motion_metrics = {
            'avg_magnitude': avg_magnitude,
            'coherence': avg_coherence,
            'motion_focus': avg_focus
        }

        # Classify action
        is_match, confidence = classify_action_from_motion(motion_metrics, expected_action)

        return {
            'success': True,
            'method': 'optical_flow',
            'avg_motion_magnitude': avg_magnitude,
            'coherence': avg_coherence,
            'motion_focus': avg_focus,
            'action_match': is_match,
            'action_confidence': confidence,
            'expected_action': expected_action,
            'flow_results': flow_results
        }

    def evaluate_batch(
        self,
        image_dir: Path,
        expected_action: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate action clarity for a batch of images.

        Args:
            image_dir: Directory containing generated images
            expected_action: Expected action category
            output_dir: Optional directory to save results

        Returns:
            Aggregated metrics
        """
        image_paths = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))

        if not image_paths:
            logger.warning(f"No images found in {image_dir}")
            return {'error': 'no_images_found'}

        logger.info(f"Evaluating {len(image_paths)} images for action: {expected_action}")

        if self.use_sequences:
            # Treat all images as a single sequence
            result = self.evaluate_image_sequence(image_paths, expected_action)
            results = [result]
        else:
            # Evaluate each image independently
            results = []
            for img_path in image_paths:
                result = self.evaluate_single_image(img_path, expected_action)
                result['image_name'] = img_path.name
                results.append(result)

        # Aggregate metrics
        successful = [r for r in results if r.get('success', False)]

        if not successful:
            logger.error("No successful analyses!")
            return {'error': 'all_analyses_failed', 'total_images': len(image_paths)}

        # Calculate statistics
        if self.use_sequences:
            metrics = successful[0]  # Single sequence result
            metrics['total_images'] = len(image_paths)
        else:
            action_matches = sum(r.get('action_match', False) for r in successful)

            metrics = {
                'total_images': len(image_paths),
                'successful_analyses': len(successful),
                'action_match_rate': action_matches / len(successful),
                'avg_blur_score': np.mean([r['blur_score'] for r in successful]),
                'avg_motion_confidence': np.mean([r['motion_confidence'] for r in successful]),
                'expected_action': expected_action,
                'per_image_results': results
            }

        # Save results
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'action_clarity_{expected_action}.json'

            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Saved metrics to {output_file}")

        return metrics


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate action clarity using optical flow',
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
        '--expected-action',
        type=str,
        required=True,
        choices=list(ACTION_CATEGORIES.keys()),
        help='Expected action category'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for metrics (default: image_dir/action_metrics)'
    )

    parser.add_argument(
        '--flow-method',
        type=str,
        default='farneback',
        choices=['farneback', 'lucaskanade'],
        help='Optical flow method (default: farneback)'
    )

    parser.add_argument(
        '--use-sequences',
        action='store_true',
        help='Analyze as image sequence (requires sequential frames)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for inference (default: cuda) - currently unused'
    )

    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = args.image_dir / 'action_metrics'

    # Initialize metric
    metric = ActionClarityMetric(
        flow_method=args.flow_method,
        use_sequences=args.use_sequences
    )

    # Evaluate
    results = metric.evaluate_batch(
        args.image_dir,
        args.expected_action,
        args.output_dir
    )

    # Print summary
    if 'error' not in results:
        print("\n" + "="*60)
        print("ACTION CLARITY EVALUATION RESULTS")
        print("="*60)
        print(f"Expected Action:      {results.get('expected_action', 'N/A')}")
        print(f"Total Images:         {results['total_images']}")

        if args.use_sequences:
            print(f"Analysis Method:      Optical Flow (sequence)")
            print(f"Avg Motion Magnitude: {results.get('avg_motion_magnitude', 0):.3f}")
            print(f"Coherence:            {results.get('coherence', 0):.3f}")
            print(f"Motion Focus:         {results.get('motion_focus', 0):.2%}")
            print(f"Action Match:         {results.get('action_match', False)}")
            print(f"Action Confidence:    {results.get('action_confidence', 0):.3f}")
        else:
            print(f"Analysis Method:      Motion Blur (single images)")
            print(f"Successful Analyses:  {results['successful_analyses']}")
            print(f"Action Match Rate:    {results['action_match_rate']:.2%}")
            print(f"Avg Blur Score:       {results['avg_blur_score']:.3f}")
            print(f"Avg Motion Conf:      {results['avg_motion_confidence']:.3f}")

        print("="*60)
    else:
        print(f"\n❌ Error: {results['error']}")


if __name__ == '__main__':
    main()
