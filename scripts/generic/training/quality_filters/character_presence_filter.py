#!/usr/bin/env python3
"""
Character Presence Filter

Uses YOLOv8 person detection to verify exactly one character is present in the image.
This ensures synthetic training images contain the intended subject.

Filtering Criteria:
- Exactly 1 person detected (reject if 0 or >1)
- Confidence threshold (default 0.5)
- Bounding box size validation (not too small)

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import logging
from pathlib import Path
from typing import Union, Tuple, Optional
from PIL import Image
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "YOLOv8 required for character presence detection. Install:\n"
        "pip install ultralytics"
    )

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.quality_filter import BaseQualityFilter


class CharacterPresenceFilter(BaseQualityFilter):
    """
    Filter images based on character presence using YOLOv8 person detection.

    Ensures training images contain exactly one character with sufficient visibility.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        min_bbox_area: float = 0.05,  # 5% of image area
        device: str = "cuda"
    ):
        """
        Initialize character presence filter.

        Args:
            model_name: YOLOv8 model variant (n/s/m/l/x)
            confidence_threshold: Minimum detection confidence
            min_bbox_area: Minimum bbox area as fraction of image
            device: Device for inference (cuda/cpu)
        """
        super().__init__()

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.min_bbox_area = min_bbox_area
        self.device = device

        # Load YOLOv8 model
        self.logger.info(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        self.model.to(device)

        # COCO class ID for person
        self.person_class_id = 0

        self.logger.info(f"✓ Character Presence Filter initialized")
        self.logger.info(f"  Confidence threshold: {confidence_threshold}")
        self.logger.info(f"  Min bbox area: {min_bbox_area * 100:.1f}%")

    def filter_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if image contains exactly one character.

        Args:
            image: Input image

        Returns:
            (passed, rejection_reason)
        """
        try:
            # Load image if needed
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            elif isinstance(image, Image.Image):
                img = image
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            else:
                return False, "Invalid image format"

            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Get image dimensions
            img_width, img_height = img.size
            img_area = img_width * img_height

            # Run YOLOv8 detection
            results = self.model(
                img,
                conf=self.confidence_threshold,
                verbose=False
            )

            # Extract person detections
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                return False, "No person detected"

            # Filter for person class
            person_detections = []

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == self.person_class_id:
                    confidence = float(box.conf[0])

                    # Get bbox coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height

                    person_detections.append({
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'area': bbox_area,
                        'area_ratio': bbox_area / img_area
                    })

            # Check person count
            if len(person_detections) == 0:
                return False, "No person detected"

            if len(person_detections) > 1:
                return False, f"Multiple people detected ({len(person_detections)})"

            # Check single person validity
            person = person_detections[0]

            if person['area_ratio'] < self.min_bbox_area:
                return False, f"Person too small ({person['area_ratio']*100:.1f}% of image)"

            # Passed all checks
            return True, None

        except Exception as e:
            self.logger.error(f"Error in character presence detection: {e}")
            return False, f"Detection error: {str(e)}"

    def get_statistics(self) -> dict:
        """Get filter statistics."""
        stats = super().get_statistics()

        if self.total_processed > 0:
            stats['rejection_breakdown'] = {
                'no_person': 'tracked separately',
                'multiple_people': 'tracked separately',
                'too_small': 'tracked separately'
            }

        return stats


def main():
    """Test character presence filter."""
    import argparse

    parser = argparse.ArgumentParser(description="Test character presence filter")
    parser.add_argument('--image', type=Path, required=True, help='Test image path')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLOv8 model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Initialize filter
    filter_obj = CharacterPresenceFilter(
        model_name=args.model,
        confidence_threshold=args.confidence,
        device=args.device
    )

    # Test image
    passed, reason = filter_obj.filter_single(args.image)

    print(f"\nImage: {args.image}")
    print(f"Passed: {passed}")
    if reason:
        print(f"Reason: {reason}")

    print(f"\nStatistics:")
    stats = filter_obj.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
