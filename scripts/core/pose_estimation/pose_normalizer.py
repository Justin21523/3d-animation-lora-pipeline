#!/usr/bin/env python3
"""
Pose Normalizer

Utilities for normalizing keypoints to consistent representation.
Removes position, scale, and rotation effects.

Operations:
- Center keypoints at hip midpoint
- Scale by torso height
- Rotate to vertical orientation

Author: AI Pipeline
Date: 2025-01-17
"""

import numpy as np
from typing import Tuple, Optional


class PoseNormalizer:
    """
    Normalize pose keypoints for consistent representation.

    Normalization steps:
    1. Center at hip midpoint
    2. Scale by torso height
    3. Rotate torso to vertical

    This allows comparing poses regardless of position, scale, or rotation.
    """

    # COCO keypoint indices
    KEYPOINT_NAMES = [
        'nose',          # 0
        'left_eye',      # 1
        'right_eye',     # 2
        'left_ear',      # 3
        'right_ear',     # 4
        'left_shoulder', # 5
        'right_shoulder',# 6
        'left_elbow',    # 7
        'right_elbow',   # 8
        'left_wrist',    # 9
        'right_wrist',   # 10
        'left_hip',      # 11
        'right_hip',     # 12
        'left_knee',     # 13
        'right_knee',    # 14
        'left_ankle',    # 15
        'right_ankle'    # 16
    ]

    # Key joint indices
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14

    def __init__(self, min_torso_height: float = 10.0):
        """
        Initialize pose normalizer.

        Args:
            min_torso_height: Minimum torso height (pixels) for valid pose.
        """
        self.min_torso_height = min_torso_height

    def normalize(self,
                  keypoints: np.ndarray,
                  scores: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Normalize keypoints.

        Args:
            keypoints: Keypoints array (17, 2).
            scores: Confidence scores (17,).

        Returns:
            (normalized_keypoints, success)
            - normalized_keypoints: Normalized keypoints (17, 2)
            - success: Whether normalization succeeded
        """
        # Check if required keypoints are valid
        required_joints = [
            self.LEFT_SHOULDER,
            self.RIGHT_SHOULDER,
            self.LEFT_HIP,
            self.RIGHT_HIP
        ]

        # All required joints must have score > 0.3
        if not all(scores[idx] > 0.3 for idx in required_joints):
            return keypoints.copy(), False

        # Step 1: Calculate hip center
        hip_center = self._get_hip_center(keypoints)

        # Step 2: Calculate torso height (shoulder-hip distance)
        torso_height = self._get_torso_height(keypoints)

        if torso_height < self.min_torso_height:
            return keypoints.copy(), False

        # Step 3: Center at hip
        centered = keypoints - hip_center

        # Step 4: Scale by torso height
        scaled = centered / torso_height

        # Step 5: Rotate to vertical (optional, based on torso angle)
        # For now, we'll skip rotation to preserve left/right orientation

        return scaled, True

    def normalize_batch(self,
                       keypoints_batch: np.ndarray,
                       scores_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize batch of keypoints.

        Args:
            keypoints_batch: Keypoints array (N, 17, 2).
            scores_batch: Confidence scores (N, 17).

        Returns:
            (normalized_batch, success_mask)
            - normalized_batch: Normalized keypoints (N, 17, 2)
            - success_mask: Success for each sample (N,)
        """
        N = len(keypoints_batch)
        normalized_batch = np.zeros_like(keypoints_batch)
        success_mask = np.zeros(N, dtype=bool)

        for i in range(N):
            normalized, success = self.normalize(
                keypoints_batch[i],
                scores_batch[i]
            )
            normalized_batch[i] = normalized
            success_mask[i] = success

        return normalized_batch, success_mask

    def _get_hip_center(self, keypoints: np.ndarray) -> np.ndarray:
        """Get hip center (midpoint between left and right hip)."""
        left_hip = keypoints[self.LEFT_HIP]
        right_hip = keypoints[self.RIGHT_HIP]
        return (left_hip + right_hip) / 2.0

    def _get_shoulder_center(self, keypoints: np.ndarray) -> np.ndarray:
        """Get shoulder center (midpoint between left and right shoulder)."""
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        return (left_shoulder + right_shoulder) / 2.0

    def _get_torso_height(self, keypoints: np.ndarray) -> float:
        """
        Get torso height (shoulder-hip distance).

        Returns:
            Torso height in pixels.
        """
        shoulder_center = self._get_shoulder_center(keypoints)
        hip_center = self._get_hip_center(keypoints)

        # Euclidean distance
        diff = shoulder_center - hip_center
        height = np.linalg.norm(diff)

        return float(height)

    def get_pose_features(self,
                         keypoints: np.ndarray,
                         scores: np.ndarray) -> Optional[dict]:
        """
        Extract geometric features from pose.

        Useful for rule-based pose classification.

        Args:
            keypoints: Keypoints array (17, 2).
            scores: Confidence scores (17,).

        Returns:
            Dictionary of pose features or None if invalid.
        """
        # Check validity
        required_joints = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_HIP, self.RIGHT_HIP,
            self.LEFT_KNEE, self.RIGHT_KNEE
        ]

        if not all(scores[idx] > 0.3 for idx in required_joints):
            return None

        features = {}

        # Centers
        hip_center = self._get_hip_center(keypoints)
        shoulder_center = self._get_shoulder_center(keypoints)

        # Torso features
        features['torso_height'] = self._get_torso_height(keypoints)

        # Torso angle (relative to vertical)
        torso_vector = shoulder_center - hip_center
        features['torso_angle'] = np.arctan2(torso_vector[0], torso_vector[1]) * 180 / np.pi

        # Shoulder width
        shoulder_width = np.linalg.norm(
            keypoints[self.RIGHT_SHOULDER] - keypoints[self.LEFT_SHOULDER]
        )
        features['shoulder_width'] = float(shoulder_width)

        # Hip width
        hip_width = np.linalg.norm(
            keypoints[self.RIGHT_HIP] - keypoints[self.LEFT_HIP]
        )
        features['hip_width'] = float(hip_width)

        # Leg spread (horizontal distance between knees)
        left_knee = keypoints[self.LEFT_KNEE]
        right_knee = keypoints[self.RIGHT_KNEE]
        horizontal_spread = abs(left_knee[0] - right_knee[0])
        features['leg_spread_horizontal'] = float(horizontal_spread)

        # Leg spread (vertical distance between knees)
        vertical_spread = abs(left_knee[1] - right_knee[1])
        features['leg_spread_vertical'] = float(vertical_spread)

        # Knee angles (simplified, just y-coordinate difference from hip)
        left_knee_bend = keypoints[self.LEFT_HIP][1] - left_knee[1]
        right_knee_bend = keypoints[self.RIGHT_HIP][1] - right_knee[1]
        features['left_knee_bend'] = float(left_knee_bend)
        features['right_knee_bend'] = float(right_knee_bend)

        # Aspect ratio (for standing vs crouching)
        features['aspect_ratio'] = float(horizontal_spread / (features['torso_height'] + 1e-6))

        return features


def main():
    """Test pose normalizer."""
    # Create sample keypoints (standing pose)
    keypoints = np.array([
        [100, 50],   # 0: nose
        [95, 45],    # 1: left_eye
        [105, 45],   # 2: right_eye
        [90, 50],    # 3: left_ear
        [110, 50],   # 4: right_ear
        [80, 80],    # 5: left_shoulder
        [120, 80],   # 6: right_shoulder
        [70, 120],   # 7: left_elbow
        [130, 120],  # 8: right_elbow
        [65, 160],   # 9: left_wrist
        [135, 160],  # 10: right_wrist
        [85, 150],   # 11: left_hip
        [115, 150],  # 12: right_hip
        [80, 200],   # 13: left_knee
        [120, 200],  # 14: right_knee
        [75, 250],   # 15: left_ankle
        [125, 250]   # 16: right_ankle
    ], dtype=np.float32)

    scores = np.ones(17, dtype=np.float32)  # All high confidence

    # Initialize normalizer
    normalizer = PoseNormalizer()

    # Normalize
    normalized, success = normalizer.normalize(keypoints, scores)

    print(f"Normalization success: {success}")
    print(f"Original keypoints shape: {keypoints.shape}")
    print(f"Normalized keypoints shape: {normalized.shape}")
    print(f"\nOriginal hip center: {normalizer._get_hip_center(keypoints)}")
    print(f"Normalized hip center: {normalizer._get_hip_center(normalized)}")
    print(f"\nOriginal torso height: {normalizer._get_torso_height(keypoints):.2f}")
    print(f"Normalized torso height: {normalizer._get_torso_height(normalized):.4f}")

    # Extract features
    features = normalizer.get_pose_features(keypoints, scores)
    print(f"\nPose features:")
    for key, value in features.items():
        print(f"  {key}: {value:.2f}")


if __name__ == '__main__':
    main()
