#!/usr/bin/env python3
"""
Background Inpainting for LoRA Training
Clean backgrounds by removing character remnants using LaMa inpainting.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.utils.logger import setup_logger


class BackgroundInpainter:
    """Clean backgrounds by inpainting character regions."""

    def __init__(self, method: str = "lama", device: str = "cuda"):
        """
        Initialize inpainter.

        Args:
            method: Inpainting method ('lama', 'telea', 'ns')
            device: Device for inference
        """
        self.method = method
        self.device = device

        if method == "lama":
            try:
                from lama_cleaner.model_manager import ModelManager
                from lama_cleaner.schema import Config

                self.model = ModelManager(
                    name="lama",
                    device=device
                )
                self.config = Config(
                    ldm_steps=25,
                    ldm_sampler="plms",
                    hd_strategy="Resize",
                    hd_strategy_resize_limit=1024
                )
                logging.info("LaMa model loaded successfully")
            except ImportError:
                logging.error("lama-cleaner not installed. Install with: pip install lama-cleaner")
                raise
        elif method in ["telea", "ns"]:
            # OpenCV inpainting (no model loading needed)
            logging.info(f"Using OpenCV {method} inpainting")
        else:
            raise ValueError(f"Unknown inpainting method: {method}")

    def create_mask_from_alpha(self, image: np.ndarray, threshold: int = 10) -> np.ndarray:
        """
        Create inpainting mask from alpha channel.

        Args:
            image: RGBA image
            threshold: Alpha threshold for mask creation

        Returns:
            Binary mask (255 = inpaint, 0 = keep)
        """
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            # Where alpha > threshold, we have content to remove
            mask = (alpha > threshold).astype(np.uint8) * 255
        else:
            # No alpha channel, try to detect edges
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Detect edges where background was removed
            mask = cv2.Canny(gray, 50, 150)
            # Dilate to fill gaps
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

        return mask

    def inpaint_opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint using OpenCV methods.

        Args:
            image: RGB image
            mask: Binary mask

        Returns:
            Inpainted image
        """
        if self.method == "telea":
            inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        elif self.method == "ns":
            inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
        else:
            raise ValueError(f"Unknown OpenCV method: {self.method}")

        return inpainted

    def inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint using LaMa model.

        Args:
            image: RGB image
            mask: Binary mask

        Returns:
            Inpainted image
        """
        # LaMa expects PIL Image
        pil_image = Image.fromarray(image)
        pil_mask = Image.fromarray(mask)

        # Inpaint
        result = self.model(pil_image, pil_mask, self.config)

        return np.array(result)

    def process_image(self, image_path: Path, output_path: Path,
                     mask_path: Optional[Path] = None) -> dict:
        """
        Process single image.

        Args:
            image_path: Path to input image
            output_path: Path to save result
            mask_path: Optional path to mask (if None, auto-generate)

        Returns:
            Metadata dict
        """
        # Read image
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Create or load mask
        if mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = self.create_mask_from_alpha(image)

        # Convert to RGB for inpainting
        if image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Inpaint
        if self.method == "lama":
            inpainted = self.inpaint_lama(image_rgb, mask)
        else:
            inpainted = self.inpaint_opencv(image_rgb, mask)

        # Convert back to BGR for saving
        inpainted_bgr = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)

        # Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), inpainted_bgr)

        # Calculate inpainted area percentage
        mask_area = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100

        return {
            "input": str(image_path),
            "output": str(output_path),
            "mask_area_percent": round(mask_area, 2),
            "method": self.method
        }

    def process_directory(self, input_dir: Path, output_dir: Path,
                         mask_dir: Optional[Path] = None) -> list:
        """
        Process all images in directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            mask_dir: Optional mask directory

        Returns:
            List of metadata dicts
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))

        logging.info(f"Found {len(image_files)} images to process")

        results = []
        for img_path in tqdm(image_files, desc="Inpainting backgrounds"):
            try:
                output_path = output_dir / img_path.name

                # Find corresponding mask if mask_dir provided
                mask_path = None
                if mask_dir:
                    mask_path = mask_dir / img_path.name

                metadata = self.process_image(img_path, output_path, mask_path)
                results.append(metadata)

            except Exception as e:
                logging.error(f"Failed to process {img_path}: {e}")
                continue

        logging.info(f"Successfully processed {len(results)}/{len(image_files)} images")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Clean backgrounds by inpainting character regions"
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                       help="Input directory with background images")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for clean backgrounds")
    parser.add_argument("--mask-dir", type=Path, default=None,
                       help="Optional directory with masks")
    parser.add_argument("--method", choices=["lama", "telea", "ns"], default="lama",
                       help="Inpainting method")
    parser.add_argument("--device", default="cuda",
                       help="Device for inference")
    parser.add_argument("--log-file", type=Path, default=None,
                       help="Log file path")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(
        "background_inpainting",
        log_file=args.log_file,
        console_level=logging.INFO
    )

    logger.info("=" * 80)
    logger.info("Background Inpainting for LoRA Training")
    logger.info("=" * 80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Device: {args.device}")

    # Initialize inpainter
    inpainter = BackgroundInpainter(method=args.method, device=args.device)

    # Process directory
    results = inpainter.process_directory(
        args.input_dir,
        args.output_dir,
        args.mask_dir
    )

    # Save metadata
    metadata_path = args.output_dir / "inpainting_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "method": args.method,
            "total_processed": len(results),
            "results": results
        }, f, indent=2)

    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info("✅ Background inpainting completed!")


if __name__ == "__main__":
    main()
