"""
Detection and tracking utilities for 2D Animation LoRA Pipeline.

Includes:
- YOLOTracker: YOLO detection with ByteTrack multi-object tracking
- YoloTrackingConfig: Legacy tracking configuration
- run_yolo_tracking: Legacy tracking function

Supports stubbed CPU-friendly flows for testing.
"""

from .yolo_detector import YoloTrackingConfig, run_yolo_tracking
from .yolo_tracker import (
    YOLOTracker,
    Detection,
    Track,
    TrackingResult,
)

__all__ = [
    # New unified tracker
    "YOLOTracker",
    "Detection",
    "Track",
    "TrackingResult",
    # Legacy
    "YoloTrackingConfig",
    "run_yolo_tracking",
]
