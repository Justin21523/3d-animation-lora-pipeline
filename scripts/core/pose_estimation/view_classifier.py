#!/usr/bin/env python3
"""
View Classifier for Pose Estimation

Classifies character view angle based on pose keypoints.

View categories:
- front: Facing camera directly
- three_quarter_right: 45° right
- three_quarter_left: 45° left
- side_right: 90° right profile
- side_left: 90° left profile
- back: Facing away

Uses geometric analysis of shoulder-hip alignment and limb visibility.

Author: AI Pipeline
Date: 2025-01-17
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging


class ViewClassifier:
    """
    Classify character view angle from pose keypoints.

    Uses geometric heuristics based on:
    - Shoulder alignment (left/right visibility)
    - Hip alignment
    - Limb visibility (arms/legs)
    - Torso orientation
    """

    # COCO keypoint indices
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    # View angle definitions (degrees)
    VIEW_ANGLES = {
        'front': (0, 22.5),           # -22.5° to +22.5°
        'three_quarter_right': (22.5, 67.5),   # 22.5° to 67.5°
        'side_right': (67.5, 112.5),  # 67.5° to 112.5°
        'back_right': (112.5, 157.5), # 112.5° to 157.5°
        'back': (157.5, 180),         # 157.5° to 180° (and -180° to -157.5°)
        'back_left': (-157.5, -112.5),
        'side_left': (-112.5, -67.5),
        'three_quarter_left': (-67.5, -22.5),
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize view classifier.

        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def classify(self,
                 keypoints: np.ndarray,
                 scores: np.ndarray) -> Tuple[str, float]:
        """
        Classify view angle from keypoints.

        Args:
            keypoints: Keypoints array (17, 2)
            scores: Confidence scores (17,)

        Returns:
            (view_category, confidence)
            - view_category: front, three_quarter_left, etc.
            - confidence: Confidence score [0, 1]
        """
        # Check if required keypoints are valid
        required_joints = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_HIP, self.RIGHT_HIP
        ]

        if not all(scores[idx] > 0.3 for idx in required_joints):
            return "unknown", 0.0

        # Extract geometric features
        features = self._extract_view_features(keypoints, scores)

        # Classify based on features
        view, confidence = self._classify_view(features, scores)

        return view, confidence

    def _extract_view_features(self,
                               keypoints: np.ndarray,
                               scores: np.ndarray) -> Dict:
        """
        Extract geometric features for view classification.

        Args:
            keypoints: Keypoints array (17, 2)
            scores: Confidence scores (17,)

        Returns:
            Dictionary of view features
        """
        features = {}

        # Shoulder features
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0

        # Hip features
        left_hip = keypoints[self.LEFT_HIP]
        right_hip = keypoints[self.RIGHT_HIP]
        hip_mid = (left_hip + right_hip) / 2.0

        # Shoulder width (horizontal distance)
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        features['shoulder_width'] = float(shoulder_width)

        # Hip width
        hip_width = abs(right_hip[0] - left_hip[0])
        features['hip_width'] = float(hip_width)

        # Shoulder alignment (which shoulder is more visible)
        # Positive = right shoulder more visible, Negative = left more visible
        shoulder_offset = right_shoulder[0] - left_shoulder[0]
        features['shoulder_offset'] = float(shoulder_offset)

        # Hip alignment
        hip_offset = right_hip[0] - left_hip[0]
        features['hip_offset'] = float(hip_offset)

        # Torso orientation vector (shoulder to hip)
        torso_vector = hip_mid - shoulder_mid
        features['torso_angle'] = float(np.arctan2(torso_vector[0], torso_vector[1]) * 180 / np.pi)

        # Limb visibility (confidence scores)
        features['left_arm_visible'] = float(scores[self.LEFT_WRIST] > 0.3)
        features['right_arm_visible'] = float(scores[self.RIGHT_WRIST] > 0.3)
        features['left_leg_visible'] = float(scores[self.LEFT_ANKLE] > 0.3)
        features['right_leg_visible'] = float(scores[self.RIGHT_ANKLE] > 0.3)

        # Nose position relative to shoulder center (for front/back)
        if scores[self.NOSE] > 0.3:
            nose = keypoints[self.NOSE]
            nose_offset = nose[0] - shoulder_mid[0]
            features['nose_offset'] = float(nose_offset)
        else:
            features['nose_offset'] = 0.0

        # Shoulder-hip width ratio (compressed for side views)
        if hip_width > 0:
            features['torso_compression'] = float(shoulder_width / hip_width)
        else:
            features['torso_compression'] = 1.0

        return features

    def _classify_view(self, features: Dict, scores: np.ndarray) -> Tuple[str, float]:
        """
        Classify view based on geometric features.

        Strategy:
        1. Check shoulder/hip alignment for left/right orientation
        2. Check limb visibility for occlusion patterns
        3. Check torso compression for side views
        4. Check nose position for front/back

        Args:
            features: View features dict
            scores: Keypoint scores

        Returns:
            (view_category, confidence)
        """
        # Get key features
        shoulder_offset = features['shoulder_offset']
        shoulder_width = features['shoulder_width']
        torso_compression = features['torso_compression']
        left_arm_vis = features['left_arm_visible']
        right_arm_vis = features['right_arm_visible']

        # Determine left/right orientation from shoulder alignment
        if abs(shoulder_offset) < shoulder_width * 0.15:
            # Shoulders relatively aligned → front or back
            orientation = "center"
        elif shoulder_offset > 0:
            # Right shoulder more visible → turning right
            orientation = "right"
        else:
            # Left shoulder more visible → turning left
            orientation = "left"

        # Determine front/side/back from torso compression and limb visibility
        if torso_compression < 0.6:
            # Very compressed → side view
            angle_category = "side"
        elif torso_compression < 0.85:
            # Moderately compressed → three-quarter view
            angle_category = "three_quarter"
        else:
            # Not compressed → front or back
            # Use limb visibility to distinguish
            if left_arm_vis and right_arm_vis:
                # Both arms visible → likely front
                angle_category = "front"
            elif not left_arm_vis and not right_arm_vis:
                # No arms visible → likely back
                angle_category = "back"
            else:
                # One arm visible → three-quarter
                angle_category = "three_quarter"

        # Combine orientation and angle
        if angle_category == "front":
            if orientation == "center":
                view = "front"
                confidence = 0.80
            elif orientation == "right":
                view = "three_quarter_right"
                confidence = 0.70
            else:
                view = "three_quarter_left"
                confidence = 0.70

        elif angle_category == "three_quarter":
            if orientation == "right":
                view = "three_quarter_right"
                confidence = 0.75
            elif orientation == "left":
                view = "three_quarter_left"
                confidence = 0.75
            else:
                view = "front"  # Fallback
                confidence = 0.60

        elif angle_category == "side":
            if orientation == "right":
                view = "side_right"
                confidence = 0.80
            elif orientation == "left":
                view = "side_left"
                confidence = 0.80
            else:
                # Center but compressed → probably facing away at angle
                view = "back"
                confidence = 0.60

        elif angle_category == "back":
            if orientation == "center":
                view = "back"
                confidence = 0.75
            elif orientation == "right":
                view = "back_right"
                confidence = 0.70
            else:
                view = "back_left"
                confidence = 0.70

        else:
            view = "unknown"
            confidence = 0.0

        return view, confidence

    def classify_batch(self,
                      keypoints_batch: np.ndarray,
                      scores_batch: np.ndarray) -> list:
        """
        Classify batch of poses.

        Args:
            keypoints_batch: Keypoints array (N, 17, 2)
            scores_batch: Confidence scores (N, 17)

        Returns:
            List of (view, confidence) tuples
        """
        results = []
        for keypoints, scores in zip(keypoints_batch, scores_batch):
            view, conf = self.classify(keypoints, scores)
            results.append((view, conf))
        return results


def main():
    """Test view classifier."""
    import argparse

    parser = argparse.ArgumentParser(description="Test view classifier")
    parser.add_argument('--test-mode', type=str, default='demo',
                       choices=['demo', 'interactive'])
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize classifier
    classifier = ViewClassifier()

    if args.test_mode == 'demo':
        # Create sample poses for different views
        test_cases = [
            ("Front view", np.array([
                [100, 50],  # nose
                [95, 45], [105, 45],  # eyes
                [90, 50], [110, 50],  # ears
                [80, 80], [120, 80],  # shoulders (wide, aligned)
                [70, 120], [130, 120], [65, 160], [135, 160],  # arms
                [85, 150], [115, 150],  # hips
                [80, 200], [120, 200],  # knees
                [75, 250], [125, 250]   # ankles
            ], dtype=np.float32)),

            ("Three-quarter right", np.array([
                [105, 50],  # nose (offset right)
                [100, 45], [110, 45],  # eyes
                [95, 50], [115, 50],  # ears
                [70, 80], [110, 80],  # shoulders (right more visible)
                [60, 120], [120, 120], [55, 160], [125, 160],  # arms
                [75, 150], [105, 150],  # hips
                [70, 200], [110, 200],  # knees
                [65, 250], [115, 250]   # ankles
            ], dtype=np.float32)),

            ("Side right", np.array([
                [120, 50],  # nose (far right)
                [115, 45], [125, 45],  # eyes
                [110, 50], [130, 50],  # ears
                [90, 80], [95, 82],  # shoulders (compressed, overlapping)
                [80, 120], [100, 118], [75, 160], [105, 156],  # arms
                [92, 150], [94, 152],  # hips (compressed)
                [90, 200], [92, 202],  # knees
                [88, 250], [90, 252]   # ankles
            ], dtype=np.float32))
        ]

        scores = np.ones(17, dtype=np.float32)  # All high confidence

        print("=" * 60)
        print("View Classifier Demo")
        print("=" * 60)

        for name, keypoints in test_cases:
            view, confidence = classifier.classify(keypoints, scores)
            print(f"\n{name}:")
            print(f"  Detected view: {view}")
            print(f"  Confidence: {confidence:.2f}")

    print("\n✅ View classifier test complete")


if __name__ == '__main__':
    main()
