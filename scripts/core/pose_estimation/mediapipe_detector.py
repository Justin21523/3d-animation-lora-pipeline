#!/usr/bin/env python3
"""
MediaPipe Pose Detector - CPU-Friendly Alternative to RTM-Pose

Provides CPU-optimized pose detection using Google MediaPipe.
Suitable for running in parallel with GPU-intensive tasks like LoRA training.

Features:
- Pure CPU inference (60+ FPS on modern CPUs)
- 33 keypoints (more detailed than COCO 17)
- No GPU dependency
- Automatic conversion to COCO 17 format for compatibility

Usage:
    from scripts.core.pose_estimation.mediapipe_detector import MediaPipePoseDetector

    detector = MediaPipePoseDetector(model_complexity=2, confidence=0.5)
    result = detector.detect_with_person_bbox('/path/to/image.png')

Author: AI Pipeline
Date: 2025-01-21
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


class MediaPipePoseDetector:
    """
    CPU-friendly pose detector using MediaPipe.

    MediaPipe provides 33 keypoints vs COCO's 17, offering more detail
    for hands, face, and feet. We automatically convert to COCO 17 format
    for compatibility with existing pipeline components.
    """

    # MediaPipe keypoint names (33 total)
    MP_KEYPOINT_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]

    # Mapping from MediaPipe (33) to COCO (17) indices
    # Format: COCO index -> MediaPipe index
    MP_TO_COCO_MAPPING = {
        0: 0,   # nose -> nose
        1: 2,   # left_eye -> left_eye
        2: 5,   # right_eye -> right_eye
        3: 7,   # left_ear -> left_ear
        4: 8,   # right_ear -> right_ear
        5: 11,  # left_shoulder -> left_shoulder
        6: 12,  # right_shoulder -> right_shoulder
        7: 13,  # left_elbow -> left_elbow
        8: 14,  # right_elbow -> right_elbow
        9: 15,  # left_wrist -> left_wrist
        10: 16, # right_wrist -> right_wrist
        11: 23, # left_hip -> left_hip
        12: 24, # right_hip -> right_hip
        13: 25, # left_knee -> left_knee
        14: 26, # right_knee -> right_knee
        15: 27, # left_ankle -> left_ankle
        16: 28, # right_ankle -> right_ankle
    }

    def __init__(self,
                 model_complexity: int = 2,
                 confidence: float = 0.5,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize MediaPipe pose detector.

        Args:
            model_complexity: Model complexity (0=lite, 1=full, 2=heavy)
                            Higher = more accurate but slower
                            - 0 (Lite): ~60-80 FPS on CPU, less accurate
                            - 1 (Full): ~40-60 FPS on CPU, balanced
                            - 2 (Heavy): ~25-40 FPS on CPU, most accurate
            confidence: Minimum detection/tracking confidence (0.0-1.0)
            logger: Logger instance
        """
        if not HAS_MEDIAPIPE:
            raise ImportError(
                "MediaPipe not installed. Install with: pip install mediapipe"
            )

        self.model_complexity = model_complexity
        self.confidence = confidence
        self.logger = logger or logging.getLogger(__name__)

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # Process each image independently
            model_complexity=model_complexity,
            smooth_landmarks=False,  # No smoothing for static images
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )

        self.logger.info(f"MediaPipe Pose Detector initialized")
        self.logger.info(f"  Model complexity: {model_complexity}")
        self.logger.info(f"  Confidence threshold: {confidence}")
        self.logger.info(f"  Device: CPU (MediaPipe uses CPU by default)")

    def detect_with_person_bbox(self, image_path: str) -> Dict:
        """
        Detect pose from image (compatible with RTMPoseDetector interface).

        Args:
            image_path: Path to image file

        Returns:
            Dict with:
                - keypoints: (17, 2) array in COCO format
                - scores: (17,) confidence scores
                - success: bool, whether detection succeeded
                - num_valid: int, number of keypoints above threshold
                - raw_keypoints: (33, 2) original MediaPipe keypoints (optional)
                - raw_scores: (33,) original MediaPipe scores (optional)
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))

            if img is None:
                self.logger.warning(f"Failed to read image: {image_path}")
                return {
                    'keypoints': np.zeros((17, 2), dtype=np.float32),
                    'scores': np.zeros(17, dtype=np.float32),
                    'success': False,
                    'num_valid': 0
                }

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # Process image
            results = self.pose.process(img_rgb)

            if not results.pose_landmarks:
                self.logger.debug(f"No pose detected in: {image_path}")
                return {
                    'keypoints': np.zeros((17, 2), dtype=np.float32),
                    'scores': np.zeros(17, dtype=np.float32),
                    'success': False,
                    'num_valid': 0
                }

            # Extract MediaPipe keypoints (33 points)
            mp_keypoints = np.array([
                [lm.x * w, lm.y * h]
                for lm in results.pose_landmarks.landmark
            ], dtype=np.float32)

            mp_scores = np.array([
                lm.visibility
                for lm in results.pose_landmarks.landmark
            ], dtype=np.float32)

            # Convert to COCO 17 format
            coco_keypoints = np.zeros((17, 2), dtype=np.float32)
            coco_scores = np.zeros(17, dtype=np.float32)

            for coco_idx, mp_idx in self.MP_TO_COCO_MAPPING.items():
                coco_keypoints[coco_idx] = mp_keypoints[mp_idx]
                coco_scores[coco_idx] = mp_scores[mp_idx]

            # Count valid keypoints
            num_valid = np.sum(coco_scores > self.confidence)

            success = num_valid >= 5  # Require at least 5 visible keypoints

            return {
                'keypoints': coco_keypoints,
                'scores': coco_scores,
                'success': success,
                'num_valid': int(num_valid),
                # Optional: include raw MediaPipe data
                'raw_keypoints': mp_keypoints,
                'raw_scores': mp_scores
            }

        except Exception as e:
            self.logger.error(f"MediaPipe detection failed for {image_path}: {e}")
            return {
                'keypoints': np.zeros((17, 2), dtype=np.float32),
                'scores': np.zeros(17, dtype=np.float32),
                'success': False,
                'num_valid': 0
            }

    def visualize(self,
                  image_path: str,
                  keypoints: np.ndarray,
                  scores: np.ndarray,
                  output_path: str,
                  use_coco_skeleton: bool = True) -> bool:
        """
        Visualize pose keypoints on image.

        Args:
            image_path: Path to input image
            keypoints: Keypoints array (17, 2) in COCO format
            scores: Confidence scores (17,)
            output_path: Path to save visualization
            use_coco_skeleton: If True, draw COCO skeleton; else MediaPipe style

        Returns:
            True if visualization saved successfully
        """
        img = cv2.imread(str(image_path))

        if img is None:
            return False

        # Draw keypoints
        for i, (kp, score) in enumerate(zip(keypoints, scores)):
            if score > self.confidence:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(
                    img, str(i), (x + 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                )

        # Draw skeleton connections (COCO format)
        if use_coco_skeleton:
            # COCO skeleton definition
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Face
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
            ]

            for pt1_idx, pt2_idx in skeleton:
                if scores[pt1_idx] > self.confidence and scores[pt2_idx] > self.confidence:
                    pt1 = tuple(keypoints[pt1_idx].astype(int))
                    pt2 = tuple(keypoints[pt2_idx].astype(int))
                    cv2.line(img, pt1, pt2, (0, 255, 255), 2)

        # Save visualization
        cv2.imwrite(str(output_path), img)
        return True

    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


# Convenience factory function
def create_pose_detector(backend: str = 'mediapipe',
                        device: str = 'cpu',
                        **kwargs) -> MediaPipePoseDetector:
    """
    Factory function to create pose detector.

    Args:
        backend: 'mediapipe' or 'rtmpose'
        device: 'cpu' or 'cuda' (only for rtmpose)
        **kwargs: Additional arguments for detector

    Returns:
        Pose detector instance
    """
    if backend == 'mediapipe':
        return MediaPipePoseDetector(**kwargs)
    elif backend == 'rtmpose':
        # Import RTMPoseDetector if available
        try:
            from scripts.core.pose_estimation import RTMPoseDetector
            return RTMPoseDetector(device=device, **kwargs)
        except ImportError:
            raise ImportError("RTMPoseDetector not available. Use MediaPipe instead.")
    else:
        raise ValueError(f"Unknown backend: {backend}")
