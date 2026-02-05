#!/usr/bin/env python3
"""
YOLO Character Cropper
=======================

Detects characters using YOLO and saves cropped images for manual review.
This is the first stage before fine segmentation (MobileSAM/SAM2).

Features:
- Reusable module with CLI interface
- Batch processing support
- Configurable padding and confidence
- Outputs detection metadata for downstream processing

Output Structure:
    output_dir/
    ├── {frame}_crop{N}.jpg        # Cropped character images
    └── detection_results.json     # Detection metadata

Usage as CLI:
    python yolo_character_cropper.py \\
        --input-dir /path/to/frames \\
        --output-dir /path/to/crops

Usage as Module:
    from yolo_character_cropper import YOLOCharacterCropper

    cropper = YOLOCharacterCropper(model_path="/path/to/yolo.pt")
    results = cropper.process_directory(input_dir, output_dir)

Author: LLMProvider Tooling
Date: 2025-12-07
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class CropResult:
    """Result of a single crop operation"""
    crop_id: int
    crop_file: str
    confidence: float
    original_bbox: Tuple[int, int, int, int]
    crop_region: Tuple[int, int, int, int]
    crop_size: Tuple[int, int]


@dataclass
class FrameResult:
    """Results for a single frame"""
    frame: str
    frame_size: Tuple[int, int]
    crops: List[CropResult]


class YOLOCharacterCropper:
    """
    YOLO-based character detection and cropping.

    Detects persons in frames and saves cropped regions for manual review
    before fine segmentation.
    """

    def __init__(
        self,
        model_path: str = "/mnt/c/ai_models/detection/yolov8x.pt",
        device: str = "cuda",
        confidence: float = 0.5,
        bbox_padding: float = 0.15,
        min_size: int = 64,
        classes: Optional[List[int]] = None
    ):
        """
        Initialize YOLO cropper.

        Args:
            model_path: Path to YOLO model weights
            device: Device to run on (cuda/cpu)
            confidence: Detection confidence threshold
            bbox_padding: Padding ratio around detected bbox
            min_size: Minimum crop size (width or height)
            classes: YOLO class IDs to detect (default: [0] for person)
        """
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self.bbox_padding = bbox_padding
        self.min_size = min_size
        self.classes = classes if classes is not None else [0]  # person

        self._model = None

    @property
    def model(self):
        """Lazy load YOLO model"""
        if self._model is None:
            from ultralytics import YOLO
            print(f"Loading YOLO model: {self.model_path}")
            self._model = YOLO(self.model_path)
            self._model.to(self.device)
            print(f"✓ YOLO loaded on {self.device}")
        return self._model

    def detect_and_crop(
        self,
        image: np.ndarray,
        frame_name: str = "frame"
    ) -> Tuple[List[np.ndarray], List[CropResult]]:
        """
        Detect characters and return cropped images.

        Args:
            image: Input image (BGR)
            frame_name: Name for output files

        Returns:
            Tuple of (list of crop images, list of CropResult)
        """
        h, w = image.shape[:2]

        # Run YOLO detection
        results = self.model(
            image,
            conf=self.confidence,
            classes=self.classes,
            verbose=False
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            return [], []

        crops = []
        crop_results = []

        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            # Calculate padding
            box_w = x2 - x1
            box_h = y2 - y1
            pad = int(self.bbox_padding * max(box_w, box_h))

            # Expand bbox with padding
            crop_x1 = max(0, int(x1) - pad)
            crop_y1 = max(0, int(y1) - pad)
            crop_x2 = min(w, int(x2) + pad)
            crop_y2 = min(h, int(y2) + pad)

            # Check minimum size
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1

            if crop_w < self.min_size or crop_h < self.min_size:
                continue

            # Crop image
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            crops.append(crop)

            # Record result
            crop_results.append(CropResult(
                crop_id=idx,
                crop_file=f"{frame_name}_crop{idx}.jpg",
                confidence=round(conf, 3),
                original_bbox=(int(x1), int(y1), int(x2), int(y2)),
                crop_region=(crop_x1, crop_y1, crop_x2, crop_y2),
                crop_size=(crop_w, crop_h)
            ))

        return crops, crop_results

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        save_crops: bool = True,
        show_progress: bool = True
    ) -> Dict:
        """
        Process all frames in a directory.

        Args:
            input_dir: Directory containing input frames
            output_dir: Output directory for crops
            save_crops: Whether to save cropped images
            show_progress: Show progress bar

        Returns:
            Statistics dictionary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if save_crops:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Get all frames
        frame_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        frame_paths = sorted([
            p for p in input_dir.iterdir()
            if p.suffix.lower() in frame_extensions
        ])

        # Statistics
        stats = {
            "total_frames": len(frame_paths),
            "frames_with_detections": 0,
            "total_crops": 0,
            "skipped_small": 0,
            "frame_results": []
        }

        # Process frames
        iterator = tqdm(frame_paths, desc="YOLO Detection") if show_progress else frame_paths

        for frame_path in iterator:
            image = cv2.imread(str(frame_path))
            if image is None:
                continue

            h, w = image.shape[:2]
            frame_basename = frame_path.stem

            # Detect and crop
            crops, crop_results = self.detect_and_crop(image, frame_basename)

            if not crop_results:
                continue

            stats["frames_with_detections"] += 1
            stats["total_crops"] += len(crop_results)

            # Save crops
            if save_crops:
                for crop, result in zip(crops, crop_results):
                    crop_path = output_dir / result.crop_file
                    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Record frame result
            stats["frame_results"].append(FrameResult(
                frame=frame_path.name,
                frame_size=(w, h),
                crops=crop_results
            ))

        # Save metadata
        if save_crops:
            self._save_metadata(output_dir, input_dir, stats)

        return stats

    def _save_metadata(self, output_dir: Path, input_dir: Path, stats: Dict):
        """Save detection metadata to JSON"""
        metadata = {
            "created": datetime.now().isoformat(),
            "source_dir": str(input_dir),
            "config": {
                "model_path": self.model_path,
                "confidence": self.confidence,
                "bbox_padding": self.bbox_padding,
                "min_size": self.min_size,
                "classes": self.classes
            },
            "stats": {
                "total_frames": stats["total_frames"],
                "frames_with_detections": stats["frames_with_detections"],
                "total_crops": stats["total_crops"]
            },
            "frames": [
                {
                    "frame": fr.frame,
                    "frame_size": list(fr.frame_size),
                    "crops": [asdict(c) for c in fr.crops]
                }
                for fr in stats["frame_results"]
            ]
        }

        metadata_path = output_dir / "detection_results.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Character Cropper - Detect and crop characters for manual review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python yolo_character_cropper.py \\
        --input-dir /path/to/frames \\
        --output-dir /path/to/crops

    # With custom settings
    python yolo_character_cropper.py \\
        --input-dir /path/to/frames \\
        --output-dir /path/to/crops \\
        --confidence 0.6 \\
        --padding 0.2

After running:
    1. Review cropped images in output directory
    2. DELETE unwanted crops (non-target characters)
    3. Run MobileSAM segmentation on remaining crops
        """
    )

    parser.add_argument("--input-dir", "-i", required=True, type=Path,
                        help="Directory containing input frames")
    parser.add_argument("--output-dir", "-o", required=True, type=Path,
                        help="Output directory for cropped images")
    parser.add_argument("--model", "-m",
                        default="/mnt/c/ai_models/detection/yolov8x.pt",
                        help="Path to YOLO model")
    parser.add_argument("--device", "-d", default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--padding", "-p", type=float, default=0.15,
                        help="Bbox padding ratio (default: 0.15)")
    parser.add_argument("--min-size", type=int, default=64,
                        help="Minimum crop size (default: 64)")

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    print("=" * 60)
    print("YOLO CHARACTER CROPPER")
    print("=" * 60)
    print(f"Input:      {args.input_dir}")
    print(f"Output:     {args.output_dir}")
    print(f"Confidence: {args.confidence}")
    print(f"Padding:    {args.padding}")
    print("=" * 60)

    cropper = YOLOCharacterCropper(
        model_path=args.model,
        device=args.device,
        confidence=args.confidence,
        bbox_padding=args.padding,
        min_size=args.min_size
    )

    stats = cropper.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Frames processed:        {stats['total_frames']}")
    print(f"Frames with detections:  {stats['frames_with_detections']}")
    print(f"Total crops saved:       {stats['total_crops']}")
    print(f"\nOutput: {args.output_dir}")
    print("=" * 60)
    print("\n📌 NEXT STEPS:")
    print("1. Review cropped images")
    print("2. DELETE unwanted crops")
    print("3. Run MobileSAM segmentation on remaining crops")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
