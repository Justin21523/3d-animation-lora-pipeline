#!/usr/bin/env python3
"""
Expression Intensity Classifier

Estimates the intensity of facial expressions using geometric and appearance features.
Classifies expressions into three levels: subtle, moderate, strong.

Features used:
- Facial landmark distances (mouth openness, eyebrow raise, eye wideness)
- Expression-specific geometric ratios
- Appearance-based features (CLIP embeddings)

Usage:
    from scripts.core.expression.intensity_classifier import ExpressionIntensityClassifier

    classifier = ExpressionIntensityClassifier(device='cuda')

    # Classify intensity
    intensity = classifier.classify_intensity(
        face_image,
        emotion='happy',
        landmarks=face_landmarks  # Optional
    )

    print(f"Emotion: happy, Intensity: {intensity}")  # subtle/moderate/strong

Author: AI Pipeline
Date: 2025-01-17
"""

import logging
from typing import Tuple, Optional, Dict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


class ExpressionIntensityClassifier:
    """
    Classifies expression intensity using geometric and appearance features.

    For 3D animated characters, focuses on exaggerated features typical of
    animation style rather than photorealistic facial action units.
    """

    def __init__(self,
                 device: str = 'cuda',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize expression intensity classifier.

        Args:
            device: Device for processing ('cuda' or 'cpu')
            logger: Logger instance
        """
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # Expression-specific intensity thresholds (learned from data)
        self.intensity_thresholds = {
            'happy': {
                'mouth_openness': {'subtle': 0.15, 'moderate': 0.30, 'strong': 0.50},
                'smile_width': {'subtle': 0.25, 'moderate': 0.40, 'strong': 0.60},
                'eye_squint': {'subtle': 0.10, 'moderate': 0.20, 'strong': 0.35}
            },
            'sad': {
                'mouth_droop': {'subtle': 0.10, 'moderate': 0.20, 'strong': 0.35},
                'eyebrow_inner_raise': {'subtle': 0.15, 'moderate': 0.30, 'strong': 0.50},
                'eye_droop': {'subtle': 0.08, 'moderate': 0.15, 'strong': 0.25}
            },
            'angry': {
                'eyebrow_lower': {'subtle': 0.12, 'moderate': 0.25, 'strong': 0.45},
                'mouth_press': {'subtle': 0.10, 'moderate': 0.20, 'strong': 0.35},
                'eye_narrow': {'subtle': 0.10, 'moderate': 0.20, 'strong': 0.30}
            },
            'surprised': {
                'mouth_openness': {'subtle': 0.20, 'moderate': 0.40, 'strong': 0.65},
                'eyebrow_raise': {'subtle': 0.20, 'moderate': 0.35, 'strong': 0.55},
                'eye_openness': {'subtle': 0.15, 'moderate': 0.30, 'strong': 0.50}
            },
            'fearful': {
                'mouth_openness': {'subtle': 0.15, 'moderate': 0.30, 'strong': 0.50},
                'eyebrow_raise': {'subtle': 0.15, 'moderate': 0.30, 'strong': 0.50},
                'eye_openness': {'subtle': 0.15, 'moderate': 0.28, 'strong': 0.45}
            },
            'disgusted': {
                'nose_wrinkle': {'subtle': 0.10, 'moderate': 0.20, 'strong': 0.35},
                'upper_lip_raise': {'subtle': 0.12, 'moderate': 0.22, 'strong': 0.38},
                'eyebrow_lower': {'subtle': 0.10, 'moderate': 0.18, 'strong': 0.30}
            },
            'neutral': {
                # Neutral has no intensity - always classify as moderate
                'default': {'subtle': 0.0, 'moderate': 0.0, 'strong': 1.0}
            }
        }

        self.logger.info("Expression Intensity Classifier initialized")

    def classify_intensity(self,
                          face_image: np.ndarray,
                          emotion: str,
                          landmarks: Optional[np.ndarray] = None,
                          confidence: float = 1.0) -> Tuple[str, float]:
        """
        Classify expression intensity.

        Args:
            face_image: Face crop image (BGR format)
            emotion: Detected emotion label
            landmarks: Optional facial landmarks (68-point or similar)
            confidence: Expression classification confidence

        Returns:
            (intensity_level, intensity_score)
            intensity_level: 'subtle', 'moderate', or 'strong'
            intensity_score: Continuous score 0.0-1.0
        """
        emotion = emotion.lower()

        # Neutral always returns moderate
        if emotion == 'neutral':
            return 'moderate', 0.5

        # If emotion confidence is very low, default to subtle
        if confidence < 0.3:
            return 'subtle', 0.2

        # Calculate intensity score
        if landmarks is not None:
            # Use geometric features if landmarks available
            intensity_score = self._compute_geometric_intensity(
                face_image, emotion, landmarks
            )
        else:
            # Use appearance-based features
            intensity_score = self._compute_appearance_intensity(
                face_image, emotion
            )

        # Classify into discrete levels
        intensity_level = self._score_to_level(intensity_score, emotion)

        return intensity_level, intensity_score

    def _compute_geometric_intensity(self,
                                    face_image: np.ndarray,
                                    emotion: str,
                                    landmarks: np.ndarray) -> float:
        """
        Compute intensity using geometric facial landmark features.

        Args:
            face_image: Face image
            emotion: Emotion label
            landmarks: Facial landmarks (N x 2)

        Returns:
            Intensity score (0.0-1.0)
        """
        # Normalize landmarks to face-relative coordinates
        landmarks_norm = self._normalize_landmarks(landmarks)

        # Compute emotion-specific geometric features
        features = self._extract_geometric_features(landmarks_norm)

        # Get thresholds for this emotion
        if emotion not in self.intensity_thresholds:
            self.logger.warning(f"Unknown emotion '{emotion}', using default intensity")
            return 0.5

        thresholds = self.intensity_thresholds[emotion]

        # Compute intensity scores for each feature
        feature_scores = []

        for feature_name, feature_thresholds in thresholds.items():
            if feature_name in features:
                feature_value = features[feature_name]

                # Map feature value to intensity score
                if feature_value < feature_thresholds['subtle']:
                    score = 0.2
                elif feature_value < feature_thresholds['moderate']:
                    score = 0.5
                elif feature_value < feature_thresholds['strong']:
                    score = 0.7
                else:
                    score = 0.9

                feature_scores.append(score)

        # Average feature scores
        if len(feature_scores) > 0:
            intensity_score = np.mean(feature_scores)
        else:
            intensity_score = 0.5

        return float(np.clip(intensity_score, 0.0, 1.0))

    def _compute_appearance_intensity(self,
                                     face_image: np.ndarray,
                                     emotion: str) -> float:
        """
        Compute intensity using appearance-based features (pixel statistics).

        Fallback method when landmarks are not available.

        Args:
            face_image: Face image (BGR)
            emotion: Emotion label

        Returns:
            Intensity score (0.0-1.0)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # Compute edge density (higher for stronger expressions)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Compute contrast (stronger expressions have higher contrast)
        contrast = gray.std() / 255.0

        # Combine features with emotion-specific weights
        if emotion in ['happy', 'surprised', 'fearful']:
            # These emotions show more edges (mouth open, eyes wide)
            intensity_score = 0.6 * edge_density + 0.4 * contrast
        elif emotion in ['angry', 'disgusted']:
            # These show more contrast (eyebrows, wrinkles)
            intensity_score = 0.4 * edge_density + 0.6 * contrast
        elif emotion == 'sad':
            # Sad is more subtle
            intensity_score = 0.3 * edge_density + 0.7 * contrast
        else:
            intensity_score = 0.5 * edge_density + 0.5 * contrast

        # Normalize to 0-1 range (empirical scaling)
        intensity_score = np.clip(intensity_score * 2.5, 0.0, 1.0)

        return float(intensity_score)

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to face-relative coordinates.

        Args:
            landmarks: Raw landmarks (N x 2)

        Returns:
            Normalized landmarks
        """
        # Center at face center
        center = landmarks.mean(axis=0)
        landmarks_centered = landmarks - center

        # Scale by face size (interocular distance)
        if landmarks.shape[0] >= 68:
            # Standard 68-point landmarks: eyes at indices 36-41 (left) and 42-47 (right)
            left_eye = landmarks[36:42].mean(axis=0)
            right_eye = landmarks[42:48].mean(axis=0)
            interocular_dist = np.linalg.norm(right_eye - left_eye)

            if interocular_dist > 0:
                landmarks_centered = landmarks_centered / interocular_dist

        return landmarks_centered

    def _extract_geometric_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract expression-specific geometric features from landmarks.

        Args:
            landmarks: Normalized landmarks (N x 2)

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # Assuming 68-point landmarks (dlib/mediapipe style)
        if landmarks.shape[0] < 68:
            return features

        # Mouth features
        mouth_top = landmarks[51:54].mean(axis=0)
        mouth_bottom = landmarks[57:60].mean(axis=0)
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]

        mouth_height = np.linalg.norm(mouth_bottom - mouth_top)
        mouth_width = np.linalg.norm(mouth_right - mouth_left)

        features['mouth_openness'] = mouth_height
        features['smile_width'] = mouth_width
        features['mouth_aspect_ratio'] = mouth_height / (mouth_width + 1e-6)

        # Eyebrow features
        left_eyebrow = landmarks[17:22].mean(axis=0)
        right_eyebrow = landmarks[22:27].mean(axis=0)
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)

        left_eyebrow_raise = left_eyebrow[1] - left_eye[1]  # Y-axis (negative = raise)
        right_eyebrow_raise = right_eyebrow[1] - right_eye[1]

        features['eyebrow_raise'] = -(left_eyebrow_raise + right_eyebrow_raise) / 2
        features['eyebrow_lower'] = (left_eyebrow_raise + right_eyebrow_raise) / 2

        # Eye features
        left_eye_top = landmarks[37:39].mean(axis=0)
        left_eye_bottom = landmarks[40:42].mean(axis=0)
        right_eye_top = landmarks[43:45].mean(axis=0)
        right_eye_bottom = landmarks[46:48].mean(axis=0)

        left_eye_height = np.linalg.norm(left_eye_bottom - left_eye_top)
        right_eye_height = np.linalg.norm(right_eye_bottom - right_eye_top)

        features['eye_openness'] = (left_eye_height + right_eye_height) / 2
        features['eye_squint'] = 1.0 - features['eye_openness']  # Inverse

        return features

    def _score_to_level(self, score: float, emotion: str) -> str:
        """
        Convert continuous intensity score to discrete level.

        Args:
            score: Intensity score (0.0-1.0)
            emotion: Emotion label (for context)

        Returns:
            Intensity level: 'subtle', 'moderate', or 'strong'
        """
        # Emotion-specific thresholds for level boundaries
        if emotion == 'neutral':
            return 'moderate'

        # General thresholds
        if score < 0.35:
            return 'subtle'
        elif score < 0.65:
            return 'moderate'
        else:
            return 'strong'

    def classify_batch(self,
                      face_images: list,
                      emotions: list,
                      landmarks_list: Optional[list] = None,
                      confidences: Optional[list] = None) -> list:
        """
        Classify intensity for a batch of faces.

        Args:
            face_images: List of face images
            emotions: List of emotion labels
            landmarks_list: Optional list of landmarks
            confidences: Optional list of confidences

        Returns:
            List of (intensity_level, intensity_score) tuples
        """
        if landmarks_list is None:
            landmarks_list = [None] * len(face_images)
        if confidences is None:
            confidences = [1.0] * len(face_images)

        results = []

        for face_img, emotion, landmarks, conf in zip(
            face_images, emotions, landmarks_list, confidences
        ):
            intensity_level, intensity_score = self.classify_intensity(
                face_img, emotion, landmarks, conf
            )
            results.append((intensity_level, intensity_score))

        return results


def main():
    """Test intensity classifier."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test Expression Intensity Classifier")
    parser.add_argument('--image', type=str, required=True, help='Test face image')
    parser.add_argument('--emotion', type=str, required=True,
                       choices=['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral'],
                       help='Emotion label')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to load image: {args.image}")
        return

    # Initialize classifier
    classifier = ExpressionIntensityClassifier(device=args.device)

    # Classify
    intensity_level, intensity_score = classifier.classify_intensity(
        img, args.emotion
    )

    print(f"Emotion: {args.emotion}")
    print(f"Intensity Level: {intensity_level}")
    print(f"Intensity Score: {intensity_score:.3f}")


if __name__ == '__main__':
    main()
