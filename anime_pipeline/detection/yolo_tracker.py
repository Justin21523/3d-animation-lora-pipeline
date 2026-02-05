#!/usr/bin/env python3
"""
YOLO Detection with ByteTrack Multi-Object Tracking.

This module provides character detection and tracking for animation frames.
Supports both GPU (CUDA) and CPU inference, with stub mode for testing.

Features:
- YOLO v8/v11 person detection
- ByteTrack multi-object tracking
- Track filtering by length
- Export to Parquet/JSON formats

Usage:
    from anime_pipeline.detection import YOLOTracker

    tracker = YOLOTracker(
        model_path="/mnt/c/ai_models/detection/yolov11n.pt",
        device="cuda"
    )

    tracks = tracker.detect_and_track(
        frames_dir="/path/to/frames",
        output_dir="/path/to/output",
        min_track_length=10
    )

Author: Justin Lu
Date: 2025-12-02
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime

import numpy as np

# Conditional imports for GPU components
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import from core
try:
    from anime_pipeline.core.stub_framework import StubMode, StubConfig
    STUB_AVAILABLE = True
except ImportError:
    STUB_AVAILABLE = False


# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result."""
    frame_idx: int
    frame_path: str
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0
    class_name: str = "person"


@dataclass
class Track:
    """Track spanning multiple frames."""
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0

    @property
    def length(self) -> int:
        return len(self.detections)

    @property
    def avg_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)


@dataclass
class TrackingResult:
    """Complete tracking result for a video/frame sequence."""
    tracks: Dict[int, Track] = field(default_factory=dict)
    total_frames: int = 0
    total_detections: int = 0
    processing_time: float = 0.0


class YOLOTracker:
    """
    YOLO detection with ByteTrack multi-object tracking.

    Supports:
    - YOLOv8/v11 person detection
    - ByteTrack tracker (built into ultralytics)
    - BotSORT tracker (alternative)
    - Stub mode for testing without GPU
    """

    def __init__(
        self,
        model_path: str = "/mnt/c/ai_models/detection/yolov11n.pt",
        tracker_type: str = "bytetrack",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        use_stub: bool = False,
        verbose: bool = False
    ):
        """
        Initialize YOLO tracker.

        Args:
            model_path: Path to YOLO model weights
            tracker_type: Tracker type ("bytetrack" or "botsort")
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: Device to use ("cuda" or "cpu")
            use_stub: Use stub mode for testing
            verbose: Print detailed logging
        """
        self.model_path = model_path
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.use_stub = use_stub
        self.verbose = verbose

        self.model = None
        self._initialized = False

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # Check if we should use stub
        if use_stub or not ULTRALYTICS_AVAILABLE:
            self.use_stub = True
            logger.info("Using stub mode for YOLO tracking")
        else:
            self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize YOLO model."""
        if not ULTRALYTICS_AVAILABLE:
            logger.warning("Ultralytics not available, using stub mode")
            self.use_stub = True
            return

        if not Path(self.model_path).exists():
            logger.warning(f"Model not found: {self.model_path}, using stub mode")
            self.use_stub = True
            return

        try:
            self.model = YOLO(self.model_path)

            # Set device
            if self.device == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
                self.model.to("cuda")
                logger.info(f"YOLO model loaded on CUDA: {self.model_path}")
            else:
                self.model.to("cpu")
                logger.info(f"YOLO model loaded on CPU: {self.model_path}")

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.use_stub = True

    def _stub_detect(
        self,
        frame_paths: List[Path],
        num_fake_tracks: int = 3
    ) -> TrackingResult:
        """
        Generate stub detections for testing.

        Args:
            frame_paths: List of frame paths
            num_fake_tracks: Number of fake tracks to generate

        Returns:
            TrackingResult with stub data
        """
        import random

        result = TrackingResult(total_frames=len(frame_paths))

        # Generate fake tracks
        for track_id in range(num_fake_tracks):
            track = Track(track_id=track_id)

            # Random start/end within frame range
            start = random.randint(0, max(0, len(frame_paths) // 2))
            end = random.randint(start + 10, len(frame_paths))
            track.start_frame = start
            track.end_frame = end

            # Generate detections
            base_x = random.uniform(0.1, 0.7) * 1920  # Assume 1920 width
            base_y = random.uniform(0.1, 0.7) * 1080

            for frame_idx in range(start, end):
                if frame_idx >= len(frame_paths):
                    break

                # Add some movement
                x1 = base_x + random.gauss(0, 20)
                y1 = base_y + random.gauss(0, 20)
                x2 = x1 + random.uniform(100, 300)
                y2 = y1 + random.uniform(200, 400)

                detection = Detection(
                    frame_idx=frame_idx,
                    frame_path=str(frame_paths[frame_idx]),
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=random.uniform(0.7, 0.99)
                )
                track.detections.append(detection)
                result.total_detections += 1

            result.tracks[track_id] = track

        return result

    def detect_and_track(
        self,
        frames_dir: Path,
        output_dir: Optional[Path] = None,
        min_track_length: int = 10,
        batch_size: int = 16,
        save_visualization: bool = False
    ) -> TrackingResult:
        """
        Process frames and return tracks.

        Args:
            frames_dir: Directory containing frame images
            output_dir: Optional output directory for results
            min_track_length: Minimum track length to keep
            batch_size: Batch size for inference
            save_visualization: Save annotated frames

        Returns:
            TrackingResult with all tracks
        """
        import time

        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise ValueError(f"Frames directory does not exist: {frames_dir}")

        # Find frame images
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
        frame_paths = []
        for ext in image_extensions:
            frame_paths.extend(sorted(frames_dir.glob(f"*{ext}")))
            frame_paths.extend(sorted(frames_dir.glob(f"*{ext.upper()}")))

        frame_paths = sorted(set(frame_paths), key=lambda x: x.name)
        logger.info(f"Found {len(frame_paths)} frames in {frames_dir}")

        if not frame_paths:
            return TrackingResult()

        start_time = time.time()

        # Use stub or real detection
        if self.use_stub:
            result = self._stub_detect(frame_paths)
        else:
            result = self._real_detect_and_track(frame_paths, batch_size)

        # Filter short tracks
        original_count = len(result.tracks)
        result.tracks = {
            tid: track for tid, track in result.tracks.items()
            if track.length >= min_track_length
        }
        filtered_count = original_count - len(result.tracks)

        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} short tracks (< {min_track_length} frames)")

        result.processing_time = time.time() - start_time

        # Save results if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save tracks
            self.export_tracks(result, output_dir / "tracks.parquet")

            # Save metadata
            metadata = {
                "source_dir": str(frames_dir),
                "total_frames": result.total_frames,
                "total_tracks": len(result.tracks),
                "total_detections": result.total_detections,
                "processing_time": result.processing_time,
                "min_track_length": min_track_length,
                "stub_mode": self.use_stub,
                "timestamp": datetime.now().isoformat()
            }
            with open(output_dir / "tracking_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

        logger.info(
            f"Tracking complete: {len(result.tracks)} tracks, "
            f"{result.total_detections} detections, "
            f"{result.processing_time:.2f}s"
        )

        return result

    def _real_detect_and_track(
        self,
        frame_paths: List[Path],
        batch_size: int = 16
    ) -> TrackingResult:
        """
        Run actual YOLO detection and tracking.

        Args:
            frame_paths: List of frame paths
            batch_size: Batch size for inference

        Returns:
            TrackingResult
        """
        result = TrackingResult(total_frames=len(frame_paths))

        if not self._initialized or self.model is None:
            logger.error("Model not initialized")
            return result

        # Build tracker config path
        tracker_config = f"{self.tracker_type}.yaml"

        # Process with tracking
        tracks_dict: Dict[int, Track] = {}

        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]

            # Run tracking
            results = self.model.track(
                source=[str(p) for p in batch_paths],
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                tracker=tracker_config,
                persist=True,
                verbose=self.verbose,
                classes=[0]  # Only person class
            )

            # Process results
            for j, r in enumerate(results):
                frame_idx = i + j
                frame_path = str(batch_paths[j])

                if r.boxes is None or len(r.boxes) == 0:
                    continue

                boxes = r.boxes
                for box in boxes:
                    # Get track ID
                    if box.id is None:
                        continue

                    track_id = int(box.id.item())
                    conf = float(box.conf.item())
                    bbox = tuple(box.xyxy[0].tolist())
                    class_id = int(box.cls.item())

                    detection = Detection(
                        frame_idx=frame_idx,
                        frame_path=frame_path,
                        track_id=track_id,
                        bbox=bbox,
                        confidence=conf,
                        class_id=class_id
                    )

                    # Add to track
                    if track_id not in tracks_dict:
                        tracks_dict[track_id] = Track(track_id=track_id)

                    tracks_dict[track_id].detections.append(detection)
                    result.total_detections += 1

            if self.verbose and (i + batch_size) % 100 == 0:
                logger.debug(f"Processed {i + batch_size}/{len(frame_paths)} frames")

        # Update track metadata
        for track in tracks_dict.values():
            if track.detections:
                track.start_frame = min(d.frame_idx for d in track.detections)
                track.end_frame = max(d.frame_idx for d in track.detections)

        result.tracks = tracks_dict
        return result

    def export_tracks(
        self,
        result: TrackingResult,
        output_path: Path,
        format: str = "auto"
    ) -> None:
        """
        Export tracks to file.

        Args:
            result: TrackingResult to export
            output_path: Output file path
            format: Output format ("parquet", "json", or "auto")
        """
        output_path = Path(output_path)

        # Determine format
        if format == "auto":
            format = output_path.suffix.lower().lstrip('.')
            if format not in ["parquet", "json", "csv"]:
                format = "parquet" if PANDAS_AVAILABLE else "json"

        # Build records
        records = []
        for track_id, track in result.tracks.items():
            for det in track.detections:
                records.append({
                    "track_id": det.track_id,
                    "frame_idx": det.frame_idx,
                    "frame_path": det.frame_path,
                    "bbox_x1": det.bbox[0],
                    "bbox_y1": det.bbox[1],
                    "bbox_x2": det.bbox[2],
                    "bbox_y2": det.bbox[3],
                    "confidence": det.confidence,
                    "class_id": det.class_id,
                    "class_name": det.class_name
                })

        if format == "parquet" and PANDAS_AVAILABLE:
            df = pd.DataFrame(records)
            df.to_parquet(output_path, index=False)
            logger.info(f"Exported {len(records)} detections to {output_path}")

        elif format == "csv" and PANDAS_AVAILABLE:
            df = pd.DataFrame(records)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(records)} detections to {output_path}")

        else:
            # JSON fallback
            output_path = output_path.with_suffix('.json')
            with open(output_path, 'w') as f:
                json.dump(records, f, indent=2)
            logger.info(f"Exported {len(records)} detections to {output_path}")

    def get_track_summary(self, result: TrackingResult) -> Dict[str, Any]:
        """
        Get summary statistics for tracks.

        Returns:
            Summary dictionary
        """
        if not result.tracks:
            return {"track_count": 0}

        lengths = [t.length for t in result.tracks.values()]
        confidences = [t.avg_confidence for t in result.tracks.values()]

        return {
            "track_count": len(result.tracks),
            "total_detections": result.total_detections,
            "avg_track_length": sum(lengths) / len(lengths),
            "min_track_length": min(lengths),
            "max_track_length": max(lengths),
            "avg_confidence": sum(confidences) / len(confidences),
            "processing_time": result.processing_time
        }


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Detection with Tracking")
    parser.add_argument("--frames-dir", type=str, required=True, help="Input frames directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="/mnt/c/ai_models/detection/yolov11n.pt", help="YOLO model path")
    parser.add_argument("--tracker", type=str, default="bytetrack", choices=["bytetrack", "botsort"], help="Tracker type")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--min-track-length", type=int, default=10, help="Minimum track length")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--stub", action="store_true", help="Use stub mode (no GPU)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    tracker = YOLOTracker(
        model_path=args.model,
        tracker_type=args.tracker,
        conf_threshold=args.conf,
        device=args.device,
        use_stub=args.stub,
        verbose=args.verbose
    )

    result = tracker.detect_and_track(
        frames_dir=Path(args.frames_dir),
        output_dir=Path(args.output_dir),
        min_track_length=args.min_track_length
    )

    # Print summary
    summary = tracker.get_track_summary(result)
    print("\n" + "=" * 50)
    print("Tracking Summary")
    print("=" * 50)
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
