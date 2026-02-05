#!/usr/bin/env python3
"""
ToonOut Character Segmenter (BiRefNet)
======================================

Fine segmentation of cropped character images using BiRefNet/ToonOut model.
Optimized for 2D animation (anime, western cartoon) character segmentation.

Features:
- Reusable module with CLI interface
- Full-image segmentation (no prompts needed)
- Outputs RGBA images with transparent background
- High-quality anime/cartoon mask generation

Output Structure:
    output_dir/
    ├── characters/              # RGBA images with transparent bg
    ├── masks/                   # Binary masks
    └── segmentation_results.json

Usage as CLI:
    python toonout_segmenter.py \\
        --input-dir /path/to/yolo_crops \\
        --output-dir /path/to/segmented

Usage as Module:
    from toonout_segmenter import ToonOutSegmenter

    segmenter = ToonOutSegmenter()
    results = segmenter.process_directory(input_dir, output_dir)

Author: LLMProvider Tooling
Date: 2025-12-08
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

# Add BiRefNet to path
BIREFNET_PATH = Path("/mnt/c/ai_models/segmentation/BiRefNet")
sys.path.insert(0, str(BIREFNET_PATH))


@dataclass
class SegmentResult:
    """Result of a single segmentation"""
    input_file: str
    output_file: str
    mask_file: str
    mask_area: int
    mask_ratio: float
    success: bool


class ToonOutSegmenter:
    """
    BiRefNet/ToonOut-based character segmentation.

    Designed for 2D animation (anime, western cartoon) character images.
    Uses full-image segmentation without prompts.
    """

    def __init__(
        self,
        model_path: str = "/mnt/c/ai_models/segmentation/toonout/birefnet_finetuned_toonout.pth",
        device: str = "cuda",
        input_size: int = 1024,
        threshold: float = 0.5
    ):
        """
        Initialize ToonOut segmenter.

        Args:
            model_path: Path to ToonOut/BiRefNet checkpoint
            device: Device to run on (cuda/cpu)
            input_size: Input size for model (1024 recommended)
            threshold: Mask binarization threshold
        """
        self.model_path = model_path
        self.device = device
        self.input_size = input_size
        self.threshold = threshold

        self._model = None

    @property
    def model(self):
        """Lazy load BiRefNet model"""
        if self._model is None:
            import torch
            from models.birefnet import BiRefNet
            from utils import check_state_dict

            print(f"Loading ToonOut/BiRefNet: {self.model_path}")

            # Initialize model
            model = BiRefNet(bb_pretrained=False)

            # Load weights
            state_dict = torch.load(self.model_path, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)

            model.to(self.device)
            model.eval()

            self._model = model
            print(f"✓ ToonOut/BiRefNet loaded on {self.device}")

        return self._model

    def segment_image(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Segment a single image.

        Args:
            image: Input image (RGB, numpy array)

        Returns:
            Tuple of (binary mask 0-255, confidence score)
        """
        import torch
        from torch.cuda import amp

        h, w = image.shape[:2]

        # Preprocess
        # Resize to input_size while maintaining aspect ratio
        scale = self.input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2

        img_padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.float32)
        img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized

        # Normalize
        img_tensor = img_padded / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))  # HWC -> CHW
        img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            # Use autocast for mixed precision (compatible with different PyTorch versions)
            try:
                with amp.autocast(device_type='cuda'):
                    pred = self.model(img_tensor)[-1].sigmoid()
            except TypeError:
                # Fallback for older PyTorch versions
                with amp.autocast():
                    pred = self.model(img_tensor)[-1].sigmoid()

        # Postprocess
        pred = pred.squeeze().cpu().numpy().astype(np.float32)

        # Remove padding
        pred = pred[pad_h:pad_h+new_h, pad_w:pad_w+new_w]

        # Ensure contiguous array for OpenCV
        pred = np.ascontiguousarray(pred)

        # Resize back to original size
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_AREA)

        # Binarize
        confidence = float(pred.mean())
        binary_mask = (pred > self.threshold).astype(np.uint8) * 255

        return binary_mask, confidence

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

        print(f"Found {len(image_paths)} images to process")

        # Statistics
        stats = {
            "total_images": len(image_paths),
            "successful": 0,
            "failed": 0,
            "results": []
        }

        # Process images
        iterator = tqdm(image_paths, desc="ToonOut Segmentation") if show_progress else image_paths

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
                mask, confidence = self.segment_image(image_rgb)

                # Calculate mask ratio
                mask_area = np.sum(mask > 127)
                mask_ratio = mask_area / (h * w)

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
                "input_size": self.input_size,
                "threshold": self.threshold,
                "background_mode": background_mode
            },
            "stats": {
                "total_images": stats["total_images"],
                "successful": stats["successful"],
                "failed": stats["failed"]
            },
            "results": [asdict(r) for r in stats["results"]]
        }

        metadata_path = output_dir / "segmentation_results.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="ToonOut Character Segmenter - BiRefNet-based segmentation for 2D animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (transparent background)
    python toonout_segmenter.py \\
        --input-dir /path/to/yolo_crops \\
        --output-dir /path/to/segmented

    # With white background
    python toonout_segmenter.py \\
        --input-dir /path/to/yolo_crops \\
        --output-dir /path/to/segmented \\
        --background white
        """
    )

    parser.add_argument("--input-dir", "-i", required=True, type=Path,
                        help="Directory containing cropped images")
    parser.add_argument("--output-dir", "-o", required=True, type=Path,
                        help="Output directory for segmented images")
    parser.add_argument("--model", "-m",
                        default="/mnt/c/ai_models/segmentation/toonout/birefnet_finetuned_toonout.pth",
                        help="Path to ToonOut/BiRefNet model")
    parser.add_argument("--device", "-d", default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--background", "-b", default="transparent",
                        choices=["transparent", "white", "black", "gray"],
                        help="Background mode (default: transparent)")
    parser.add_argument("--input-size", type=int, default=1024,
                        help="Model input size (default: 1024)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Mask binarization threshold (default: 0.5)")

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    print("=" * 60)
    print("TOONOUT CHARACTER SEGMENTER (BiRefNet)")
    print("=" * 60)
    print(f"Input:      {args.input_dir}")
    print(f"Output:     {args.output_dir}")
    print(f"Background: {args.background}")
    print("=" * 60)

    segmenter = ToonOutSegmenter(
        model_path=args.model,
        device=args.device,
        input_size=args.input_size,
        threshold=args.threshold
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
    print(f"\nOutput: {args.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
