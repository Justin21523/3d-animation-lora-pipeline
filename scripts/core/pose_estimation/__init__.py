"""
Pose Estimation Module

Provides pose detection, classification, normalization, and view classification
utilities for 3D animated characters.

Components:
- RTMPoseDetector: RTM-Pose keypoint detection (MMPose, GPU-optimized)
- MediaPipePoseDetector: MediaPipe pose detection (CPU-friendly, 60+ FPS)
- RuleBasedPoseClassifier: Geometric pose classification
- PoseNormalizer: Keypoint normalization utilities
- ViewClassifier: View angle classification (front/side/back/three-quarter)
"""

from .rtmpose_detector import RTMPoseDetector
from .mediapipe_detector import MediaPipePoseDetector, create_pose_detector
from .pose_classifier import RuleBasedPoseClassifier
from .pose_normalizer import PoseNormalizer
from .view_classifier import ViewClassifier

__all__ = [
    'RTMPoseDetector',
    'MediaPipePoseDetector',
    'create_pose_detector',
    'RuleBasedPoseClassifier',
    'PoseNormalizer',
    'ViewClassifier'
]
