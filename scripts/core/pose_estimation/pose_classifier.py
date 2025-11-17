#!/usr/bin/env python3
"""
Rule-Based Pose Classifier

Classify poses using geometric rules from keypoints.

Supported actions:
- standing: Upright, neutral stance
- walking: One leg forward, moderate motion
- running: Wide leg spread, bent knees, dynamic
- sitting: Bent knees, low position
- jumping: Airborne, extended legs
- crouching: Bent knees, low crouch
- reaching: Extended arm(s)

Author: AI Pipeline
Date: 2025-01-17
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging


class RuleBasedPoseClassifier:
    """
    Classify poses using geometric rules.

    Analyzes keypoint geometry to determine action/pose category.
    Uses heuristic rules based on joint angles, distances, and positions.
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

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize pose classifier.

        Args:
            logger: Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)

    def classify(self,
                 keypoints: np.ndarray,
                 scores: np.ndarray) -> Tuple[str, float]:
        """
        Classify pose using geometric rules.

        Args:
            keypoints: Keypoints array (17, 2).
            scores: Confidence scores (17,).

        Returns:
            (action, confidence)
            - action: Detected action/pose
            - confidence: Confidence score [0, 1]
        """
        # Check validity of required keypoints
        required_joints = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_HIP, self.RIGHT_HIP,
            self.LEFT_KNEE, self.RIGHT_KNEE
        ]

        if not all(scores[idx] > 0.3 for idx in required_joints):
            return "unknown", 0.0

        # Extract geometric features
        features = self._extract_features(keypoints, scores)

        # Run classification rules (in priority order)
        # More specific actions first, then general ones

        # 1. Jumping (legs extended, high position)
        if self._is_jumping(features):
            return "jumping", 0.75

        # 2. Running (wide leg spread, diagonal torso, bent knees)
        if self._is_running(features):
            return "running", 0.75

        # 3. Walking (moderate leg spread, upright torso)
        if self._is_walking(features):
            return "walking", 0.70

        # 4. Sitting (bent knees, low knee position)
        if self._is_sitting(features):
            return "sitting", 0.75

        # 5. Crouching (bent knees, very low position)
        if self._is_crouching(features):
            return "crouching", 0.70

        # 6. Reaching (extended arm)
        if self._is_reaching(features):
            return "reaching", 0.65

        # 7. Standing (default, upright, legs together)
        if self._is_standing(features):
            return "standing", 0.70

        # If none match, return unknown
        return "unknown", 0.0

    def classify_batch(self,
                      keypoints_batch: np.ndarray,
                      scores_batch: np.ndarray) -> list:
        """
        Classify batch of poses.

        Args:
            keypoints_batch: Keypoints array (N, 17, 2).
            scores_batch: Confidence scores (N, 17).

        Returns:
            List of (action, confidence) tuples.
        """
        results = []
        for keypoints, scores in zip(keypoints_batch, scores_batch):
            action, conf = self.classify(keypoints, scores)
            results.append((action, conf))
        return results

    def _extract_features(self,
                         keypoints: np.ndarray,
                         scores: np.ndarray) -> Dict:
        """Extract geometric features from keypoints."""
        features = {}

        # Centers
        hip_center = (keypoints[self.LEFT_HIP] + keypoints[self.RIGHT_HIP]) / 2.0
        shoulder_center = (keypoints[self.LEFT_SHOULDER] + keypoints[self.RIGHT_SHOULDER]) / 2.0

        # Torso features
        torso_vector = shoulder_center - hip_center
        features['torso_height'] = float(np.linalg.norm(torso_vector))
        features['torso_angle'] = float(np.arctan2(torso_vector[0], torso_vector[1]) * 180 / np.pi)

        # Leg features
        left_knee = keypoints[self.LEFT_KNEE]
        right_knee = keypoints[self.RIGHT_KNEE]
        left_ankle = keypoints[self.LEFT_ANKLE]
        right_ankle = keypoints[self.RIGHT_ANKLE]

        # Horizontal leg spread
        features['leg_spread_horizontal'] = float(abs(left_knee[0] - right_knee[0]))

        # Vertical leg spread (different heights)
        features['leg_spread_vertical'] = float(abs(left_knee[1] - right_knee[1]))

        # Knee bend (hip-knee-ankle angle approximation)
        left_knee_bend = float(keypoints[self.LEFT_HIP][1] - left_knee[1])
        right_knee_bend = float(keypoints[self.RIGHT_HIP][1] - right_knee[1])
        features['left_knee_bend'] = left_knee_bend
        features['right_knee_bend'] = right_knee_bend
        features['avg_knee_bend'] = (left_knee_bend + right_knee_bend) / 2.0

        # Knee position relative to hip (for sitting/crouching)
        features['knee_below_hip'] = float(
            (left_knee[1] > keypoints[self.LEFT_HIP][1] + 20) or
            (right_knee[1] > keypoints[self.RIGHT_HIP][1] + 20)
        )

        # Ankle position (for jumping)
        avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2.0
        avg_hip_y = hip_center[1]
        features['ankle_hip_distance'] = float(avg_ankle_y - avg_hip_y)

        # Arm features (for reaching)
        if scores[self.LEFT_WRIST] > 0.3 and scores[self.LEFT_SHOULDER] > 0.3:
            left_arm_length = np.linalg.norm(
                keypoints[self.LEFT_WRIST] - keypoints[self.LEFT_SHOULDER]
            )
            features['left_arm_extended'] = float(left_arm_length)
        else:
            features['left_arm_extended'] = 0.0

        if scores[self.RIGHT_WRIST] > 0.3 and scores[self.RIGHT_SHOULDER] > 0.3:
            right_arm_length = np.linalg.norm(
                keypoints[self.RIGHT_WRIST] - keypoints[self.RIGHT_SHOULDER]
            )
            features['right_arm_extended'] = float(right_arm_length)
        else:
            features['right_arm_extended'] = 0.0

        return features

    def _is_standing(self, features: Dict) -> bool:
        """
        Check if pose is standing.

        Criteria:
        - Upright torso (angle near 0)
        - Legs relatively together (small horizontal spread)
        - Knees not heavily bent
        """
        torso_upright = abs(features['torso_angle']) < 15
        legs_together = features['leg_spread_horizontal'] < features['torso_height'] * 0.5
        knees_straight = features['avg_knee_bend'] > features['torso_height'] * 0.3

        return torso_upright and legs_together and knees_straight

    def _is_walking(self, features: Dict) -> bool:
        """
        Check if pose is walking.

        Criteria:
        - Upright torso
        - Moderate leg spread (horizontal OR vertical)
        - One leg forward
        """
        torso_upright = abs(features['torso_angle']) < 20
        moderate_spread = (
            features['leg_spread_horizontal'] > features['torso_height'] * 0.3 or
            features['leg_spread_vertical'] > features['torso_height'] * 0.2
        )
        not_too_wide = features['leg_spread_horizontal'] < features['torso_height'] * 0.8

        return torso_upright and moderate_spread and not_too_wide

    def _is_running(self, features: Dict) -> bool:
        """
        Check if pose is running.

        Criteria:
        - Wide leg spread (horizontal OR vertical)
        - Diagonal torso (leaning forward)
        - Bent knees
        """
        wide_spread = (
            features['leg_spread_horizontal'] > features['torso_height'] * 0.6 or
            features['leg_spread_vertical'] > features['torso_height'] * 0.3
        )
        torso_diagonal = abs(features['torso_angle']) > 10
        knees_bent = features['avg_knee_bend'] < features['torso_height'] * 0.5

        return wide_spread and knees_bent

    def _is_sitting(self, features: Dict) -> bool:
        """
        Check if pose is sitting.

        Criteria:
        - Bent knees (short knee-ankle distance)
        - Knees below or level with hips
        - Relatively upright torso
        """
        knees_bent = features['avg_knee_bend'] < features['torso_height'] * 0.3
        knees_low = features['knee_below_hip'] > 0.5
        torso_upright = abs(features['torso_angle']) < 30

        return knees_bent and knees_low and torso_upright

    def _is_crouching(self, features: Dict) -> bool:
        """
        Check if pose is crouching.

        Criteria:
        - Very bent knees
        - Very low position
        - Torso may lean forward
        """
        very_bent = features['avg_knee_bend'] < features['torso_height'] * 0.2
        very_low = features['ankle_hip_distance'] < features['torso_height'] * 0.8

        return very_bent and very_low

    def _is_jumping(self, features: Dict) -> bool:
        """
        Check if pose is jumping.

        Criteria:
        - Extended legs (ankles far from hips)
        - Less horizontal leg spread (legs together in air)
        - Upright or slightly forward torso
        """
        legs_extended = features['ankle_hip_distance'] > features['torso_height'] * 1.2
        legs_together = features['leg_spread_horizontal'] < features['torso_height'] * 0.4

        return legs_extended and legs_together

    def _is_reaching(self, features: Dict) -> bool:
        """
        Check if pose is reaching.

        Criteria:
        - One or both arms extended
        - Arm length > torso height
        """
        left_extended = features['left_arm_extended'] > features['torso_height'] * 1.0
        right_extended = features['right_arm_extended'] > features['torso_height'] * 1.0

        return left_extended or right_extended


def main():
    """Test pose classifier."""
    import argparse

    parser = argparse.ArgumentParser(description="Test pose classifier")
    parser.add_argument('--test-mode', type=str, default='demo',
                       choices=['demo', 'interactive'])
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize classifier
    classifier = RuleBasedPoseClassifier()

    if args.test_mode == 'demo':
        # Create sample poses
        test_cases = [
            ("Standing", np.array([
                [100, 50], [95, 45], [105, 45], [90, 50], [110, 50],  # Head
                [80, 80], [120, 80],  # Shoulders
                [70, 120], [130, 120], [65, 160], [135, 160],  # Arms
                [85, 150], [115, 150],  # Hips
                [80, 200], [120, 200],  # Knees
                [75, 250], [125, 250]   # Ankles
            ], dtype=np.float32)),

            ("Running", np.array([
                [100, 50], [95, 45], [105, 45], [90, 50], [110, 50],  # Head
                [75, 85], [125, 75],  # Shoulders (diagonal)
                [60, 125], [140, 115], [55, 165], [145, 155],  # Arms
                [80, 155], [120, 145],  # Hips
                [60, 180], [140, 210],  # Knees (wide spread, different heights)
                [55, 240], [145, 260]   # Ankles
            ], dtype=np.float32)),

            ("Sitting", np.array([
                [100, 50], [95, 45], [105, 45], [90, 50], [110, 50],  # Head
                [80, 80], [120, 80],  # Shoulders
                [70, 120], [130, 120], [65, 140], [135, 140],  # Arms
                [85, 140], [115, 140],  # Hips
                [80, 160], [120, 160],  # Knees (level with hips)
                [75, 180], [125, 180]   # Ankles (close to knees)
            ], dtype=np.float32))
        ]

        scores = np.ones(17, dtype=np.float32)  # All high confidence

        print("=" * 60)
        print("Rule-Based Pose Classifier Demo")
        print("=" * 60)

        for name, keypoints in test_cases:
            action, confidence = classifier.classify(keypoints, scores)
            print(f"\n{name} pose:")
            print(f"  Detected: {action}")
            print(f"  Confidence: {confidence:.2f}")

    print("\n✅ Pose classifier test complete")


if __name__ == '__main__':
    main()
