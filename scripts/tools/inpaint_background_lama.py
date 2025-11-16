#!/usr/bin/env python3
"""
Inpaint backgrounds in character images using SAM2 masks
Keeps character intact, replaces complex 3D backgrounds with simple inpainted backgrounds
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

def load_sam2_mask(image_path: str, masks_dir: str):
    """
    Load the corresponding SAM2 mask for an image
    Returns mask where 255 = background to inpaint, 0 = character to keep

    Args:
        image_path: Path to input image
        masks_dir: Directory containing SAM2 masks
    """
    # Construct mask filename from image filename
    image_stem = Path(image_path).stem

    # Remove _ctx suffix if present to find original mask
    if image_stem.endswith('_ctx'):
        mask_stem = image_stem[:-4]  # Remove '_ctx'
    else:
        mask_stem = image_stem

    # Mask filename pattern: scene0171_pos1_frame001711_t780.03s_inst12_mask.png
    mask_filename = f"{mask_stem}_mask.png"
    mask_path = Path(masks_dir) / mask_filename

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # Load mask (should be binary: 0 or 255)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Invert mask: SAM2 masks have 255=character, we need 255=background
    mask_inverted = 255 - mask

    return mask_inverted

def simple_inpaint_background(image_path: str, output_path: str, masks_dir: str, save_mask_path: str = None):
    """
    Inpaint background around character using SAM2 mask and OpenCV inpainting

    Args:
        image_path: Path to input image
        output_path: Path for output image
        masks_dir: Directory containing SAM2 masks
        save_mask_path: Optional path to save mask for debugging
    """
    img = cv2.imread(str(image_path))

    # Load SAM2 mask (255 = background to inpaint, 0 = character to keep)
    mask = load_sam2_mask(image_path, masks_dir)

    # Resize mask to match image dimensions if needed
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Save debug mask if requested
    if save_mask_path:
        cv2.imwrite(str(save_mask_path), mask)

    # Dilate mask slightly to avoid edge artifacts
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Use Telea inpainting (fast and natural looking)
    inpainted = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Smooth transition between character and background
    # Blur the mask for soft edges
    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0) / 255.0
    mask_3ch = np.stack([mask_blur] * 3, axis=2)

    # Blend original character with inpainted background
    result = (img * (1 - mask_3ch) + inpainted * mask_3ch).astype(np.uint8)

    cv2.imwrite(str(output_path), result)

def batch_inpaint_backgrounds(
    input_dir: str,
    output_dir: str,
    masks_dir: str,
    save_masks: bool = False
):
    """
    Batch inpaint backgrounds for all images in directory using SAM2 masks

    Args:
        input_dir: Directory with character images
        output_dir: Output directory for inpainted images
        masks_dir: Directory containing SAM2 masks
        save_masks: Whether to save processed masks for debugging
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if save_masks:
        mask_dir = output_path / "masks"
        mask_dir.mkdir(exist_ok=True)

    # Get all images
    images = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))

    print(f"Found {len(images)} images to process")
    print(f"Output directory: {output_path}")
    print(f"SAM2 masks directory: {masks_dir}")

    processed = 0
    skipped = 0

    for img_path in tqdm(images, desc="Inpainting backgrounds"):
        output_file = output_path / img_path.name

        mask_file = None
        if save_masks:
            mask_file = mask_dir / f"{img_path.stem}_mask.png"

        try:
            simple_inpaint_background(img_path, output_file, masks_dir, mask_file)

            # Copy caption if exists
            caption_file = img_path.with_suffix(".txt")
            if caption_file.exists():
                output_caption = output_path / caption_file.name
                output_caption.write_text(caption_file.read_text())

            processed += 1
        except FileNotFoundError as e:
            print(f"\n⚠️  Mask not found for {img_path.name}, skipping...")
            skipped += 1
        except Exception as e:
            print(f"\n⚠️  Error processing {img_path.name}: {e}")
            skipped += 1

    print(f"\n✓ Inpainting complete!")
    print(f"  Processed: {processed} images")
    print(f"  Skipped: {skipped} images")
    print(f"  Output: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inpaint backgrounds in character images using SAM2 masks")
    parser.add_argument("--input-dir", required=True, help="Input directory with images")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--masks-dir", required=True, help="Directory containing SAM2 masks")
    parser.add_argument("--save-masks", action="store_true", help="Save processed masks for debugging")

    args = parser.parse_args()

    batch_inpaint_backgrounds(
        args.input_dir,
        args.output_dir,
        args.masks_dir,
        args.save_masks
    )
