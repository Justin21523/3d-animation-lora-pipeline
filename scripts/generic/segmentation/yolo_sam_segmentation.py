#!/usr/bin/env python3
"""
Two-Stage Character Segmentation: YOLO Detection → Fine Segmentation

Strategy:
1. Stage 1: YOLO detects character bounding boxes
2. Stage 2: Fine segmentation within each bbox using:
   - ToonOut (for 2D animation characters)
   - MobileSAM / EfficientSAM (for 3D animation / semi-realistic)

Benefits:
- Much faster than full-frame SAM2
- Better results for specific domains (2D vs 3D)
- Scalable quality levels (fast mode vs quality mode)

Usage:
    # 2D animation (with ToonOut)
    python yolo_sam_segmentation.py \
      --input-dir /path/to/frames \
      --output-dir /path/to/segmented \
      --mode 2d \
      --yolo-model yolov11n.pt \
      --device cuda

    # 3D animation (with MobileSAM)
    python yolo_sam_segmentation.py \
      --input-dir /path/to/frames \
      --output-dir /path/to/segmented \
      --mode 3d \
      --yolo-model yolov11n.pt \
      --sam-variant mobile \
      --device cuda
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


@dataclass
class BBox:
    """Bounding box with metadata"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def expand(self, padding_ratio: float, img_width: int, img_height: int) -> 'BBox':
        """Expand bbox by padding ratio, clamped to image bounds"""
        pad = int(padding_ratio * max(self.width, self.height))
        return BBox(
            x1=max(0, self.x1 - pad),
            y1=max(0, self.y1 - pad),
            x2=min(img_width, self.x2 + pad),
            y2=min(img_height, self.y2 + pad),
            confidence=self.confidence,
            class_name=self.class_name
        )


class YOLODetector:
    """YOLO-based character detection"""

    def __init__(
        self,
        model_path: str = "yolov11n.pt",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        classes: Optional[List[int]] = None
    ):
        """
        Initialize YOLO detector

        Args:
            model_path: Path to YOLO model weights
            device: cuda or cpu
            confidence_threshold: Minimum confidence for detections
            classes: Filter specific classes (None = all classes)
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.classes = classes

        print(f"🔧 Loading YOLO model: {model_path}")

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(device)
            print(f"✓ YOLO model loaded on {device}")
        except ImportError:
            print("❌ ultralytics not installed!")
            print("   Install with: pip install ultralytics")
            sys.exit(1)

    def detect(self, image: np.ndarray) -> List[BBox]:
        """
        Detect characters in image

        Args:
            image: Image array (BGR)

        Returns:
            List of BBox objects
        """
        results = self.model(
            image,
            conf=self.confidence_threshold,
            classes=self.classes,
            verbose=False
        )

        bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]

                bboxes.append(BBox(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    confidence=conf,
                    class_name=class_name
                ))

        return bboxes


class MobileSAMSegmenter:
    """MobileSAM for fine segmentation within bbox"""

    def __init__(self, device: str = "cuda"):
        """Initialize MobileSAM"""
        self.device = device
        print("🔧 Loading MobileSAM...")

        try:
            from mobile_sam import sam_model_registry, SamPredictor

            # Model path (adjust as needed)
            checkpoint = "/mnt/c/ai_models/segmentation/mobile_sam/mobile_sam.pt"
            model_type = "vit_t"

            mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint)
            mobile_sam.to(device=device)
            mobile_sam.eval()

            self.predictor = SamPredictor(mobile_sam)
            print("✓ MobileSAM loaded")

        except ImportError:
            print("⚠️ MobileSAM not installed, falling back to alpha matting")
            self.predictor = None

    def segment(self, image: np.ndarray, bbox: BBox) -> np.ndarray:
        """
        Segment character within bbox

        Args:
            image: Full image (RGB)
            bbox: Character bounding box

        Returns:
            Binary mask (same size as image)
        """
        if self.predictor is None:
            # Fallback: simple thresholding
            return self._fallback_segment(image, bbox)

        # Set image for SAM
        self.predictor.set_image(image)

        # Convert bbox to SAM format
        box_prompt = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2])

        # Predict mask
        masks, scores, logits = self.predictor.predict(
            box=box_prompt,
            multimask_output=False
        )

        return masks[0].astype(np.uint8) * 255

    def _fallback_segment(self, image: np.ndarray, bbox: BBox) -> np.ndarray:
        """Fallback segmentation using simple methods"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Crop region
        crop = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]

        # Simple edge-based segmentation
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Fill contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crop_mask = np.zeros_like(gray)
        cv2.drawContours(crop_mask, contours, -1, 255, -1)

        # Place back into full mask
        mask[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = crop_mask

        return mask


class ToonOutSegmenter:
    """ToonOut for 2D animation character segmentation"""

    def __init__(self, device: str = "cuda"):
        """Initialize ToonOut"""
        self.device = device
        print("🔧 Loading ToonOut...")

        try:
            # ToonOut integration (adjust based on actual library)
            from toonout import ToonOutModel

            self.model = ToonOutModel(device=device)
            print("✓ ToonOut loaded")

        except ImportError:
            print("⚠️ ToonOut not installed, using alpha matting fallback")
            self.model = None

    def segment(self, image: np.ndarray, bbox: BBox) -> np.ndarray:
        """
        Segment 2D character within bbox

        Args:
            image: Full image (RGB)
            bbox: Character bounding box

        Returns:
            Binary mask (same size as image)
        """
        if self.model is None:
            return self._alpha_matting_fallback(image, bbox)

        # Crop to bbox
        crop = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]

        # Run ToonOut on crop
        fg_mask = self.model.segment(crop)

        # Create full-size mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = fg_mask

        return mask

    def _alpha_matting_fallback(self, image: np.ndarray, bbox: BBox) -> np.ndarray:
        """Fallback using simple background subtraction"""
        from rembg import remove

        # Crop region
        crop = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        crop_pil = Image.fromarray(crop)

        # Remove background
        output = remove(crop_pil)

        # Extract alpha channel
        alpha = np.array(output)[:, :, 3]

        # Create full mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = alpha

        return mask


class TwoStageSegmenter:
    """
    Two-stage segmentation pipeline:
    YOLO Detection → Fine Segmentation
    """

    def __init__(
        self,
        mode: str = "3d",  # "2d" or "3d"
        yolo_model: str = "yolov11n.pt",
        sam_variant: str = "mobile",  # "mobile", "efficient", "fast"
        device: str = "cuda",
        bbox_padding: float = 0.1,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize two-stage segmenter

        Args:
            mode: "2d" (use ToonOut) or "3d" (use SAM variants)
            yolo_model: YOLO model path
            sam_variant: Which SAM variant to use (if mode="3d")
            device: cuda or cpu
            bbox_padding: Expand bbox by this ratio
            confidence_threshold: YOLO confidence threshold
        """
        self.mode = mode
        self.bbox_padding = bbox_padding
        self.device = device

        # Initialize YOLO detector
        self.detector = YOLODetector(
            model_path=yolo_model,
            device=device,
            confidence_threshold=confidence_threshold
        )

        # Initialize fine segmenter based on mode
        if mode == "2d":
            print("📐 Mode: 2D Animation (ToonOut)")
            self.segmenter = ToonOutSegmenter(device=device)
        else:
            print("📐 Mode: 3D Animation (MobileSAM)")
            self.segmenter = MobileSAMSegmenter(device=device)

    def process_frame(
        self,
        image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[BBox]]:
        """
        Process single frame

        Args:
            image: Image array (BGR)

        Returns:
            (character_images, masks, bboxes)
        """
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Stage 1: YOLO detection
        bboxes = self.detector.detect(image)

        if not bboxes:
            return [], [], []

        # Expand bboxes
        expanded_bboxes = [
            bbox.expand(self.bbox_padding, w, h)
            for bbox in bboxes
        ]

        # Stage 2: Fine segmentation for each bbox
        character_images = []
        masks = []

        for bbox in expanded_bboxes:
            # Segment within bbox
            mask = self.segmenter.segment(image_rgb, bbox)

            # Extract character with alpha channel
            rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask

            character_images.append(rgba)
            masks.append(mask)

        return character_images, masks, expanded_bboxes


def process_directory(
    input_dir: Path,
    output_dir: Path,
    mode: str = "3d",
    yolo_model: str = "yolov11n.pt",
    sam_variant: str = "mobile",
    device: str = "cuda",
    bbox_padding: float = 0.1,
    confidence_threshold: float = 0.5
) -> Dict:
    """
    Process all frames in directory

    Args:
        input_dir: Directory with frames
        output_dir: Output directory
        mode: "2d" or "3d"
        yolo_model: YOLO model path
        sam_variant: SAM variant (if mode="3d")
        device: cuda or cpu
        bbox_padding: Bbox expansion ratio
        confidence_threshold: YOLO confidence threshold

    Returns:
        Processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output structure
    characters_dir = output_dir / "characters"
    masks_dir = output_dir / "masks"
    backgrounds_dir = output_dir / "backgrounds"

    for d in [characters_dir, masks_dir, backgrounds_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Initialize segmenter
    segmenter = TwoStageSegmenter(
        mode=mode,
        yolo_model=yolo_model,
        sam_variant=sam_variant,
        device=device,
        bbox_padding=bbox_padding,
        confidence_threshold=confidence_threshold
    )

    # Find all images
    image_files = sorted(
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg"))
    )

    print(f"\n📊 Processing {len(image_files)} frames...")

    stats = {
        "total_frames": len(image_files),
        "frames_with_characters": 0,
        "total_characters": 0,
        "failed": 0
    }

    instance_counter = 0

    for img_path in tqdm(image_files, desc="Segmenting"):
        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                stats["failed"] += 1
                continue

            # Process frame
            character_images, masks, bboxes = segmenter.process_frame(image)

            if not character_images:
                continue

            stats["frames_with_characters"] += 1
            stats["total_characters"] += len(character_images)

            # Save each character instance
            frame_name = img_path.stem

            for i, (char_img, mask, bbox) in enumerate(zip(character_images, masks, bboxes)):
                instance_name = f"{frame_name}_inst{instance_counter}"
                instance_counter += 1

                # Save character with alpha
                char_path = characters_dir / f"{instance_name}.png"
                cv2.imwrite(str(char_path), char_img)

                # Save mask
                mask_path = masks_dir / f"{instance_name}_mask.png"
                cv2.imwrite(str(mask_path), mask)

            # Save background (copy original for now)
            bg_path = backgrounds_dir / f"{frame_name}_background.jpg"
            cv2.imwrite(str(bg_path), image)

        except Exception as e:
            print(f"\n⚠️ Failed {img_path.name}: {e}")
            stats["failed"] += 1

    # Save metadata
    metadata = {
        "statistics": stats,
        "parameters": {
            "mode": mode,
            "yolo_model": yolo_model,
            "sam_variant": sam_variant if mode == "3d" else "toonout",
            "bbox_padding": bbox_padding,
            "confidence_threshold": confidence_threshold
        },
        "timestamp": datetime.now().isoformat()
    }

    metadata_path = output_dir / "segmentation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Segmentation complete!")
    print(f"   Frames with characters: {stats['frames_with_characters']}/{stats['total_frames']}")
    print(f"   Total character instances: {stats['total_characters']}")
    print(f"   Failed: {stats['failed']}")
    print(f"\n📁 Output: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage Character Segmentation (YOLO + Fine Segmentation)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory with frames"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["2d", "3d"],
        default="3d",
        help="Segmentation mode: 2d (ToonOut) or 3d (MobileSAM)"
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov11n.pt",
        help="YOLO model path"
    )
    parser.add_argument(
        "--sam-variant",
        type=str,
        choices=["mobile", "efficient", "fast"],
        default="mobile",
        help="SAM variant (if mode=3d)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device"
    )
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=0.1,
        help="Bbox expansion ratio (default: 0.1)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="YOLO confidence threshold (default: 0.5)"
    )

    args = parser.parse_args()

    # Process directory
    process_directory(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        mode=args.mode,
        yolo_model=args.yolo_model,
        sam_variant=args.sam_variant,
        device=args.device,
        bbox_padding=args.bbox_padding,
        confidence_threshold=args.confidence
    )


if __name__ == "__main__":
    main()
