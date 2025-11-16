#!/usr/bin/env python3
"""
Inpaint transparent backgrounds in pure instance images using LaMa
Converts transparent PNG to natural-looking backgrounds
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

def simple_inpaint_background(image_path: str, output_path: str, bg_color="auto"):
    """
    Simple background replacement for transparent images

    Args:
        image_path: Path to PNG with transparency
        output_path: Output path for inpainted image
        bg_color: Background color ("auto", "white", "gray", or (R,G,B) tuple)
    """
    # Read image with alpha channel
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img.shape[2] != 4:
        # No alpha channel, just copy
        cv2.imwrite(str(output_path), img)
        return

    # Split channels
    bgr = img[:, :, :3]
    alpha = img[:, :, 3]

    # Create mask (0 = transparent, 255 = opaque)
    mask = (alpha < 128).astype(np.uint8) * 255

    # Determine background color
    if bg_color == "auto":
        # Use average color from edges as background hint
        edge_pixels = np.concatenate([
            bgr[0, :],  # Top row
            bgr[-1, :],  # Bottom row
            bgr[:, 0],  # Left column
            bgr[:, -1]  # Right column
        ])
        bg_mean = edge_pixels.mean(axis=0)
        bg_color = tuple(map(int, bg_mean))
    elif bg_color == "white":
        bg_color = (255, 255, 255)
    elif bg_color == "gray":
        bg_color = (200, 200, 200)

    # Use OpenCV inpainting (fast and reasonable quality)
    inpainted = cv2.inpaint(bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Blend with background color for smoother result
    bg = np.full_like(bgr, bg_color, dtype=np.uint8)
    alpha_normalized = alpha.astype(np.float32) / 255.0
    alpha_3ch = np.stack([alpha_normalized] * 3, axis=2)

    result = (inpainted * alpha_3ch + bg * (1 - alpha_3ch)).astype(np.uint8)

    # Save as regular RGB image
    cv2.imwrite(str(output_path), result)

def batch_inpaint_transparent(
    input_dir: str,
    output_dir: str,
    bg_color="auto"
):
    """
    Batch inpaint transparent backgrounds

    Args:
        input_dir: Directory with transparent PNG images
        output_dir: Output directory for inpainted images
        bg_color: Background color strategy
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all PNG images
    images = list(input_path.glob("*.png"))

    print(f"Found {len(images)} PNG images")
    print(f"Using background color strategy: {bg_color}")

    for img_path in tqdm(images, desc="Inpainting"):
        output_file = output_path / img_path.name
        simple_inpaint_background(img_path, output_file, bg_color)

        # Also copy caption if exists
        caption_file = img_path.with_suffix(".txt")
        if caption_file.exists():
            output_caption = output_path / caption_file.name
            output_caption.write_text(caption_file.read_text())

    print(f"\n✓ Inpainting complete! Output: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inpaint transparent backgrounds")
    parser.add_argument("--input-dir", required=True, help="Input directory with transparent PNGs")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--bg-color", default="auto", choices=["auto", "white", "gray"],
                       help="Background color strategy")

    args = parser.parse_args()

    batch_inpaint_transparent(
        args.input_dir,
        args.output_dir,
        args.bg_color
    )
