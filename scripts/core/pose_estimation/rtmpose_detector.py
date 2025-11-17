#!/usr/bin/env python3
"""
RTM-Pose Detector

Wrapper for MMPose RTM-Pose model for keypoint detection.
Optimized for 3D animated characters.

Features:
- COCO 17-keypoint detection
- Batch processing support
- Confidence score filtering
- 3D character optimized thresholds

Author: AI Pipeline
Date: 2025-01-17
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# MMPose imports
try:
    from mmpose.apis import init_model, inference_topdown
    from mmpose.structures import merge_data_samples
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    logging.warning("MMPose not available. RTMPoseDetector will not work.")


class RTMPoseDetector:
    """
    RTM-Pose keypoint detector for 3D animated characters.

    Uses MMPose RTM-Pose model (COCO 17 keypoints).
    Optimized for 3D character pose estimation.

    COCO 17 Keypoints:
    0: nose, 1: left_eye, 2: right_eye,
    3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder,
    7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist,
    11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee,
    15: left_ankle, 16: right_ankle
    """

    # Model configurations (MMPose)
    MODEL_CONFIGS = {
        's': {
            'config': 'rtmpose-s_8xb256-420e_coco-256x192.py',
            'checkpoint': 'rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth'
        },
        'm': {
            'config': 'rtmpose-m_8xb256-420e_coco-256x192.py',
            'checkpoint': 'rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
        },
        'l': {
            'config': 'rtmpose-l_8xb256-420e_coco-256x192.py',
            'checkpoint': 'rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'
        }
    }

    def __init__(self,
                 model_size: str = 'm',
                 device: str = 'cuda',
                 conf_threshold: float = 0.3,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize RTM-Pose detector.

        Args:
            model_size: Model size ('s', 'm', 'l'). Default 'm'.
            device: Device ('cuda' or 'cpu'). Default 'cuda'.
            conf_threshold: Confidence threshold for keypoints (3D: 0.3 vs 2D: 0.5).
            logger: Logger instance.
        """
        if not MMPOSE_AVAILABLE:
            raise ImportError("MMPose is required for RTMPoseDetector. Install with: pip install mmpose")

        self.model_size = model_size
        self.device = device
        self.conf_threshold = conf_threshold
        self.logger = logger or logging.getLogger(__name__)

        # Initialize model
        self._init_model()

    def _init_model(self):
        """Initialize MMPose RTM-Pose model."""
        if self.model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid model_size: {self.model_size}. Choose from {list(self.MODEL_CONFIGS.keys())}")

        config_info = self.MODEL_CONFIGS[self.model_size]

        # Model paths (assumes MMPose model zoo or local cache)
        config_file = config_info['config']
        checkpoint_file = config_info['checkpoint']

        self.logger.info(f"Loading RTM-Pose-{self.model_size.upper()} model...")
        self.logger.info(f"Config: {config_file}")
        self.logger.info(f"Checkpoint: {checkpoint_file}")

        try:
            self.model = init_model(
                config_file,
                checkpoint_file,
                device=self.device
            )
            self.logger.info("RTM-Pose model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load RTM-Pose model: {e}")
            raise

    def detect(self, image_path: str, bbox: Optional[List[float]] = None) -> Dict:
        """
        Detect keypoints from a single image.

        Args:
            image_path: Path to image file.
            bbox: Optional bounding box [x1, y1, x2, y2]. If None, uses full image.

        Returns:
            {
                'keypoints': np.array (17, 2),  # (x, y) coordinates
                'scores': np.array (17,),       # Confidence scores [0, 1]
                'bbox': [x1, y1, x2, y2],       # Bounding box
                'success': bool                 # Detection successful
            }
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.warning(f"Failed to read image: {image_path}")
            return self._empty_result()

        # If no bbox provided, use full image
        if bbox is None:
            h, w = img.shape[:2]
            bbox = [0, 0, w, h]

        # Run inference
        try:
            results = inference_topdown(self.model, img, bboxes=[bbox])

            if len(results) == 0:
                return self._empty_result()

            # Extract keypoints and scores
            pred_instances = results[0].pred_instances
            keypoints = pred_instances.keypoints[0]  # (17, 2)
            scores = pred_instances.keypoint_scores[0]  # (17,)

            # Filter by confidence threshold
            valid_mask = scores >= self.conf_threshold
            num_valid = valid_mask.sum()

            # Require at least 5 valid keypoints
            if num_valid < 5:
                return self._empty_result()

            return {
                'keypoints': keypoints,
                'scores': scores,
                'bbox': bbox,
                'success': True,
                'num_valid': int(num_valid)
            }

        except Exception as e:
            self.logger.warning(f"Pose detection failed for {image_path}: {e}")
            return self._empty_result()

    def batch_detect(self,
                     image_paths: List[str],
                     bboxes: Optional[List[List[float]]] = None,
                     show_progress: bool = True) -> List[Dict]:
        """
        Detect keypoints from multiple images.

        Args:
            image_paths: List of image paths.
            bboxes: Optional list of bounding boxes. If None, uses full images.
            show_progress: Show progress bar.

        Returns:
            List of detection results (same format as detect()).
        """
        if bboxes is None:
            bboxes = [None] * len(image_paths)

        results = []
        iterator = zip(image_paths, bboxes)

        if show_progress:
            iterator = tqdm(list(iterator), desc="Detecting poses")

        for img_path, bbox in iterator:
            result = self.detect(img_path, bbox)
            results.append(result)

        return results

    def detect_with_person_bbox(self, image_path: str) -> Dict:
        """
        Detect pose with automatic person detection.

        Uses a simple bounding box heuristic based on image size.
        For better results, use external person detector (YOLO, etc.)

        Args:
            image_path: Path to image file.

        Returns:
            Detection result (same format as detect()).
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return self._empty_result()

        h, w = img.shape[:2]

        # Simple heuristic: assume person occupies central 80% of image
        # For character instances from segmentation, this is usually valid
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        bbox = [margin_x, margin_y, w - margin_x, h - margin_y]

        return self.detect(image_path, bbox)

    def _empty_result(self) -> Dict:
        """Return empty detection result."""
        return {
            'keypoints': np.zeros((17, 2), dtype=np.float32),
            'scores': np.zeros(17, dtype=np.float32),
            'bbox': [0, 0, 0, 0],
            'success': False,
            'num_valid': 0
        }

    def visualize(self,
                  image_path: str,
                  keypoints: np.ndarray,
                  scores: np.ndarray,
                  output_path: str,
                  conf_threshold: Optional[float] = None):
        """
        Visualize keypoints on image.

        Args:
            image_path: Path to source image.
            keypoints: Keypoints array (17, 2).
            scores: Confidence scores (17,).
            output_path: Path to save visualization.
            conf_threshold: Confidence threshold (uses self.conf_threshold if None).
        """
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.warning(f"Failed to read image: {image_path}")
            return

        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        # COCO skeleton connections
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6),  # Shoulders
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12),  # Torso
            (11, 12),  # Hips
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16)  # Right leg
        ]

        # Draw skeleton
        for i, j in skeleton:
            if scores[i] >= conf_threshold and scores[j] >= conf_threshold:
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)

        # Draw keypoints
        for idx, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score >= conf_threshold:
                x, y = int(kpt[0]), int(kpt[1])
                color = (0, 255, 0) if score > 0.7 else (0, 255, 255)
                cv2.circle(img, (x, y), 4, color, -1)
                cv2.putText(img, str(idx), (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Save
        cv2.imwrite(str(output_path), img)
        self.logger.info(f"Saved visualization to {output_path}")


def main():
    """Test RTM-Pose detector."""
    import argparse

    parser = argparse.ArgumentParser(description="Test RTM-Pose detector")
    parser.add_argument('--image', type=str, required=True, help='Test image path')
    parser.add_argument('--model-size', type=str, default='m', choices=['s', 'm', 'l'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--conf-threshold', type=float, default=0.3)
    parser.add_argument('--output', type=str, help='Output visualization path')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize detector
    detector = RTMPoseDetector(
        model_size=args.model_size,
        device=args.device,
        conf_threshold=args.conf_threshold
    )

    # Detect
    result = detector.detect_with_person_bbox(args.image)

    if result['success']:
        print(f"✅ Pose detected successfully")
        print(f"Valid keypoints: {result['num_valid']}/17")
        print(f"Keypoints shape: {result['keypoints'].shape}")
        print(f"Scores: {result['scores']}")

        # Visualize if output specified
        if args.output:
            detector.visualize(
                args.image,
                result['keypoints'],
                result['scores'],
                args.output
            )
    else:
        print(f"❌ Pose detection failed")


if __name__ == '__main__':
    main()
