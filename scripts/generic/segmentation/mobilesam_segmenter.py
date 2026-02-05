#!/usr/bin/env python3
"""
MobileSAM Character Segmenter
==============================

Fine segmentation of cropped character images using MobileSAM.
Takes pre-cropped images (from YOLO) and generates precise masks.

Features:
- Reusable module with CLI interface
- Automatic center-point prompting for cropped images
- Outputs RGBA images with transparent background
- Supports multiple background modes

Output Structure:
    output_dir/
    ├── characters/              # RGBA images with transparent bg
    ├── masks/                   # Binary masks
    └── segmentation_results.json

Usage as CLI:
    python mobilesam_segmenter.py \\
        --input-dir /path/to/yolo_crops \\
        --output-dir /path/to/segmented

Usage as Module:
    from mobilesam_segmenter import MobileSAMSegmenter

    segmenter = MobileSAMSegmenter()
    results = segmenter.process_directory(input_dir, output_dir)

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
from PIL import Image
from tqdm import tqdm


@dataclass
class SegmentResult:
    """Result of a single segmentation"""
    input_file: str
    output_file: str
    mask_file: str
    mask_area: int
    mask_ratio: float
    success: bool


class MobileSAMSegmenter:
    """
    MobileSAM-based character segmentation.

    Designed for cropped character images where the subject is centered.
    Uses center-point prompting for optimal segmentation.
    """

    def __init__(
        self,
        model_path: str = "/mnt/c/ai_models/segmentation/mobile_sam/mobile_sam.pt",
        device: str = "cuda",
        mask_threshold: float = 0.0,
        min_mask_ratio: float = 0.001,  # Very low threshold - almost never skip
        prompt_mode: str = "bbox"  # "bbox", "center", "multi"
    ):
        """
        Initialize MobileSAM segmenter.

        Args:
            model_path: Path to MobileSAM checkpoint
            device: Device to run on (cuda/cpu)
            mask_threshold: Mask binarization threshold
            min_mask_ratio: Minimum mask area ratio to consider valid (default: 0.001 = 0.1%)
            prompt_mode: Prompting strategy
                - "bbox": Use full image bbox as prompt (captures all foreground)
                - "center": Use center point (single subject)
                - "multi": Use multiple points (grid sampling)
        """
        self.model_path = model_path
        self.device = device
        self.mask_threshold = mask_threshold
        self.min_mask_ratio = min_mask_ratio
        self.prompt_mode = prompt_mode

        self._predictor = None

    @property
    def predictor(self):
        """Lazy load MobileSAM predictor"""
        if self._predictor is None:
            from mobile_sam import sam_model_registry, SamPredictor

            print(f"Loading MobileSAM: {self.model_path}")
            model = sam_model_registry["vit_t"](checkpoint=self.model_path)
            model.to(device=self.device)
            model.eval()

            self._predictor = SamPredictor(model)
            print(f"✓ MobileSAM loaded on {self.device}")

        return self._predictor

    def segment_image(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Segment a single image.

        Args:
            image: Input image (RGB, numpy array)
            point_coords: Optional point prompts [[x, y], ...]
            point_labels: Optional point labels [1=foreground, 0=background]
            box: Optional bbox prompt [x1, y1, x2, y2]

        Returns:
            Tuple of (binary mask, mask score)
        """
        h, w = image.shape[:2]

        # Set image
        self.predictor.set_image(image)

        # Choose prompting strategy based on mode
        if self.prompt_mode == "bbox":
            # Use full image as bbox - captures all foreground
            box = np.array([0, 0, w, h])
            point_coords = None
            point_labels = None
        elif self.prompt_mode == "multi":
            # Use multiple points in a grid pattern
            points = []
            for py in [h//4, h//2, 3*h//4]:
                for px in [w//4, w//2, 3*w//4]:
                    points.append([px, py])
            point_coords = np.array(points)
            point_labels = np.array([1] * len(points))  # all foreground
            box = None
        else:  # "center"
            # Single center point
            if point_coords is None:
                center_x, center_y = w // 2, h // 2
                point_coords = np.array([[center_x, center_y]])
                point_labels = np.array([1])  # foreground
            box = None

        # Predict mask
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True
        )

        # Select best mask (highest score)
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        score = scores[best_idx]

        # Convert to binary
        binary_mask = (mask > self.mask_threshold).astype(np.uint8) * 255

        return binary_mask, float(score)

    def apply_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background_mode: str = "transparent"
    ) -> np.ndarray:
        """
        Apply mask to image.

        Args:
            image: Input image (RGB)
            mask: Binary mask
            background_mode: transparent, white, black, gray

        Returns:
            Processed image (RGBA if transparent, else RGB)
        """
        if background_mode == "transparent":
            # Create RGBA image
            rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = image
            rgba[:, :, 3] = mask
            return rgba

        else:
            # Solid background
            if background_mode == "white":
                bg_color = (255, 255, 255)
            elif background_mode == "black":
                bg_color = (0, 0, 0)
            else:  # gray
                bg_color = (128, 128, 128)

            background = np.full_like(image, bg_color, dtype=np.uint8)
            mask_3ch = np.stack([mask / 255.0] * 3, axis=-1)
            result = (image * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)
            return result

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        background_mode: str = "transparent",
        show_progress: bool = True
    ) -> Dict:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing cropped images
            output_dir: Output directory
            background_mode: Background mode for output
            show_progress: Show progress bar

        Returns:
            Statistics dictionary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Create output directories
        characters_dir = output_dir / "characters"
        masks_dir = output_dir / "masks"
        characters_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Get all images
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths = sorted([
            p for p in input_dir.iterdir()
            if p.suffix.lower() in image_extensions and not p.name.startswith(".")
        ])

        # Filter out json files
        image_paths = [p for p in image_paths if p.suffix.lower() != ".json"]

        print(f"Found {len(image_paths)} images to process")

        # Statistics
        stats = {
            "total_images": len(image_paths),
            "successful": 0,
            "failed": 0,
            "skipped_small_mask": 0,
            "results": []
        }

        # Process images
        iterator = tqdm(image_paths, desc="MobileSAM Segmentation") if show_progress else image_paths

        for image_path in iterator:
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    stats["failed"] += 1
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]

                # Segment
                mask, score = self.segment_image(image_rgb)

                # Calculate mask ratio (for stats only, never skip)
                mask_area = np.sum(mask > 127)
                mask_ratio = mask_area / (h * w)

                # Track small masks but DON'T skip - process all images
                if mask_ratio < self.min_mask_ratio:
                    stats["skipped_small_mask"] += 1
                    # Continue processing instead of skipping

                # Apply mask
                result = self.apply_mask(image_rgb, mask, background_mode)

                # Save outputs
                basename = image_path.stem

                # Save character image
                if background_mode == "transparent":
                    char_filename = f"{basename}.png"
                    char_path = characters_dir / char_filename
                    Image.fromarray(result).save(char_path)
                else:
                    char_filename = f"{basename}.jpg"
                    char_path = characters_dir / char_filename
                    cv2.imwrite(str(char_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Save mask
                mask_filename = f"{basename}_mask.png"
                mask_path = masks_dir / mask_filename
                cv2.imwrite(str(mask_path), mask)

                # Record result
                stats["results"].append(SegmentResult(
                    input_file=image_path.name,
                    output_file=char_filename,
                    mask_file=mask_filename,
                    mask_area=int(mask_area),
                    mask_ratio=round(mask_ratio, 4),
                    success=True
                ))
                stats["successful"] += 1

            except Exception as e:
                print(f"\n⚠️ Error processing {image_path.name}: {e}")
                stats["failed"] += 1

        # Save metadata
        self._save_metadata(output_dir, input_dir, stats, background_mode)

        return stats

    def _save_metadata(self, output_dir: Path, input_dir: Path, stats: Dict, background_mode: str):
        """Save segmentation metadata"""
        metadata = {
            "created": datetime.now().isoformat(),
            "source_dir": str(input_dir),
            "config": {
                "model_path": self.model_path,
                "mask_threshold": self.mask_threshold,
                "min_mask_ratio": self.min_mask_ratio,
                "background_mode": background_mode
            },
            "stats": {
                "total_images": stats["total_images"],
                "successful": stats["successful"],
                "failed": stats["failed"],
                "skipped_small_mask": stats["skipped_small_mask"]
            },
            "results": [asdict(r) for r in stats["results"]]
        }

        metadata_path = output_dir / "segmentation_results.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="MobileSAM Character Segmenter - Fine segmentation for cropped images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (transparent background)
    python mobilesam_segmenter.py \\
        --input-dir /path/to/yolo_crops \\
        --output-dir /path/to/segmented

    # With gray background
    python mobilesam_segmenter.py \\
        --input-dir /path/to/yolo_crops \\
        --output-dir /path/to/segmented \\
        --background gray
        """
    )

    parser.add_argument("--input-dir", "-i", required=True, type=Path,
                        help="Directory containing cropped images")
    parser.add_argument("--output-dir", "-o", required=True, type=Path,
                        help="Output directory for segmented images")
    parser.add_argument("--model", "-m",
                        default="/mnt/c/ai_models/segmentation/mobile_sam/mobile_sam.pt",
                        help="Path to MobileSAM model")
    parser.add_argument("--device", "-d", default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--background", "-b", default="transparent",
                        choices=["transparent", "white", "black", "gray"],
                        help="Background mode (default: transparent)")
    parser.add_argument("--min-mask-ratio", type=float, default=0.001,
                        help="Minimum mask area ratio (default: 0.001)")
    parser.add_argument("--prompt-mode", default="bbox",
                        choices=["bbox", "center", "multi"],
                        help="Prompt mode: bbox (full image), center (single point), multi (grid)")

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    print("=" * 60)
    print("MOBILESAM CHARACTER SEGMENTER")
    print("=" * 60)
    print(f"Input:      {args.input_dir}")
    print(f"Output:     {args.output_dir}")
    print(f"Background: {args.background}")
    print("=" * 60)

    segmenter = MobileSAMSegmenter(
        model_path=args.model,
        device=args.device,
        min_mask_ratio=args.min_mask_ratio,
        prompt_mode=args.prompt_mode
    )

    stats = segmenter.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        background_mode=args.background
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total images:        {stats['total_images']}")
    print(f"Successful:          {stats['successful']}")
    print(f"Failed:              {stats['failed']}")
    print(f"Skipped (small):     {stats['skipped_small_mask']}")
    print(f"\nOutput: {args.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
