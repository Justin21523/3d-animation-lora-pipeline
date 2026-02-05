#!/usr/bin/env python3
"""
Complete Sea Monster SDXL Data Processing Pipeline

Steps:
1. Background removal (rembg with ISNet model)
2. Background inpainting (LaMa)
3. Letterbox resize to 1024x1024 (black borders)
4. Data augmentation (to reach 200+ images)

For Alberto and Luca sea monster forms.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import cv2
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def remove_background_rembg(image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """
    Remove background using rembg (ISNet model).

    Returns:
        character_img: RGBA image with transparent background
        mask: Binary mask (255 = character, 0 = background)
    """
    try:
        from rembg import remove, new_session

        # Use ISNet model (best for 3D characters)
        session = new_session("isnet-general-use")

        # Remove background
        output = remove(image, session=session)

        # Convert to RGBA
        if output.mode != 'RGBA':
            output = output.convert('RGBA')

        # Extract mask from alpha channel
        alpha = output.split()[3]
        mask = Image.new('L', alpha.size)
        mask.paste(alpha)

        return output, mask

    except ImportError:
        print("ERROR: rembg not installed. Install with: pip install rembg")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR removing background: {e}")
        raise


def inpaint_background_lama(
    image: Image.Image,
    mask: Image.Image
) -> Image.Image:
    """
    Inpaint background using LaMa.

    Args:
        image: Original RGB image
        mask: Binary mask (255 = inpaint this region, 0 = keep)

    Returns:
        Inpainted RGB image
    """
    try:
        from simple_lama_inpainting import SimpleLama

        # Initialize LaMa
        lama = SimpleLama()

        # Convert to numpy
        img_np = np.array(image.convert('RGB'))
        mask_np = np.array(mask)

        # Inpaint
        result = lama(img_np, mask_np)

        # Convert back to PIL
        return Image.fromarray(result)

    except ImportError:
        print("WARNING: simple-lama-inpainting not installed")
        print("  Install with: pip install simple-lama-inpainting")
        print("  Falling back to black background...")

        # Fallback: black background
        bg = Image.new('RGB', image.size, (0, 0, 0))
        return bg

    except Exception as e:
        print(f"WARNING: LaMa inpainting failed: {e}")
        print("  Falling back to black background...")
        bg = Image.new('RGB', image.size, (0, 0, 0))
        return bg


def letterbox_resize(image: Image.Image, target_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """
    Letterbox resize to target size with black borders.

    Args:
        image: PIL Image (RGB or RGBA)
        target_size: Target (width, height)

    Returns:
        Letterboxed PIL Image (RGB)
    """
    target_w, target_h = target_size

    # Convert RGBA to RGB on black background
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (0, 0, 0))
        bg.paste(image, mask=image.split()[3])  # Use alpha as mask
        image = bg
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    img_w, img_h = image.size

    # Calculate scale factor
    scale = min(target_w / img_w, target_h / img_h)

    # Calculate new dimensions
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    # Resize image
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create black canvas
    canvas = Image.new("RGB", target_size, (0, 0, 0))

    # Calculate position to paste (center)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2

    # Paste resized image onto canvas
    canvas.paste(resized, (paste_x, paste_y))

    return canvas


def augment_image(image: Image.Image, aug_id: int, seed: int = None) -> Image.Image:
    """
    Apply conservative augmentation for 3D sea monsters.

    Args:
        image: PIL Image (RGB)
        aug_id: Augmentation variant ID
        seed: Random seed

    Returns:
        Augmented PIL Image
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img = image.copy()

    if aug_id == 0:
        # Subtle brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))

    elif aug_id == 1:
        # Subtle contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))

    elif aug_id == 2:
        # Very light rotation
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))

    elif aug_id == 3:
        # Minimal saturation (preserve underwater colors)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.98, 1.02))

    elif aug_id == 4:
        # Light blur (DoF simulation)
        blur_radius = random.uniform(0.3, 0.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    else:
        # Combination
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))

    return img


def process_single_image(
    img_path: Path,
    output_dir: Path,
    character_name: str,
    augs_per_image: int,
    skip_rembg: bool = False,
    skip_lama: bool = False
) -> int:
    """
    Process single image through complete pipeline.

    Returns:
        Number of output images generated
    """
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')

        # Step 1: Background removal (optional)
        if not skip_rembg:
            char_rgba, mask = remove_background_rembg(img)

            # Step 2: Inpaint background (optional)
            if not skip_lama:
                # Invert mask for LaMa (255 = inpaint background)
                mask_inv = Image.eval(mask, lambda x: 255 - x)
                bg_inpainted = inpaint_background_lama(img, mask_inv)

                # Composite character over inpainted background
                processed = Image.new('RGB', img.size)
                processed.paste(bg_inpainted)
                processed.paste(char_rgba, mask=char_rgba.split()[3])
            else:
                # Just black background
                processed = Image.new('RGB', char_rgba.size, (0, 0, 0))
                processed.paste(char_rgba, mask=char_rgba.split()[3])
        else:
            processed = img

        # Step 3: Letterbox to 1024x1024
        letterboxed = letterbox_resize(processed, (1024, 1024))

        # Save original processed image
        orig_path = output_dir / f"{character_name}_{img_path.stem}_orig.png"
        letterboxed.save(orig_path, 'PNG', quality=95)
        count = 1

        # Step 4: Generate augmented versions
        for aug_idx in range(augs_per_image):
            seed_val = abs(hash(img_path.name) + aug_idx) % (2**32 - 1)
            aug_img = augment_image(letterboxed, aug_idx, seed=seed_val)

            aug_path = output_dir / f"{character_name}_{img_path.stem}_aug{aug_idx}.png"
            aug_img.save(aug_path, 'PNG', quality=95)
            count += 1

        return count

    except Exception as e:
        print(f"ERROR processing {img_path.name}: {e}")
        return 0


def process_character_dataset(
    input_dir: Path,
    output_dir: Path,
    character_name: str,
    target_count: int = 200,
    skip_rembg: bool = False,
    skip_lama: bool = False
) -> Dict:
    """
    Process complete character dataset.

    Returns:
        Processing statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    input_images = [
        f for f in input_dir.iterdir()
        if f.suffix in image_extensions
    ]

    base_count = len(input_images)

    print(f"\n{'='*70}")
    print(f"Processing {character_name} Sea Monster")
    print(f"{'='*70}")
    print(f"Pipeline:")
    print(f"  1. Background Removal (rembg): {'SKIP' if skip_rembg else 'YES'}")
    print(f"  2. Background Inpainting (LaMa): {'SKIP' if skip_lama else 'YES'}")
    print(f"  3. Letterbox to 1024x1024: YES")
    print(f"  4. Augmentation: YES")
    print(f"")
    print(f"Base images: {base_count}")
    print(f"Target count: {target_count}")

    # Calculate augmentations needed
    augs_per_image = max(1, int(np.ceil((target_count - base_count) / base_count)))
    total_expected = base_count * (1 + augs_per_image)

    print(f"Augmentations per image: {augs_per_image}")
    print(f"Expected total: {total_expected}")
    print(f"{'='*70}\n")

    # Process each image
    output_count = 0

    for img_path in tqdm(input_images, desc=f"Processing {character_name}"):
        count = process_single_image(
            img_path=img_path,
            output_dir=output_dir,
            character_name=character_name,
            augs_per_image=augs_per_image,
            skip_rembg=skip_rembg,
            skip_lama=skip_lama
        )
        output_count += count

    # Statistics
    stats = {
        "character": character_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "base_count": base_count,
        "target_count": target_count,
        "augs_per_image": augs_per_image,
        "total_output": output_count,
        "pipeline": {
            "background_removal": not skip_rembg,
            "inpainting": not skip_lama,
            "letterbox": True,
            "augmentation": True
        }
    }

    print(f"\n✅ {character_name} Complete:")
    print(f"   Base images: {base_count}")
    print(f"   Output images: {output_count}")
    print(f"   Saved to: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Complete Sea Monster SDXL Processing Pipeline")
    parser.add_argument("--input-dirs", nargs="+", required=True,
                        help="Input directories")
    parser.add_argument("--character-names", nargs="+", required=True,
                        help="Character names")
    parser.add_argument("--output-base", type=str, required=True,
                        help="Base output directory")
    parser.add_argument("--target-count", type=int, default=200,
                        help="Target images per character")
    parser.add_argument("--skip-rembg", action="store_true",
                        help="Skip background removal")
    parser.add_argument("--skip-lama", action="store_true",
                        help="Skip LaMa inpainting")

    args = parser.parse_args()

    if len(args.input_dirs) != len(args.character_names):
        print("ERROR: Number of input dirs must match character names")
        sys.exit(1)

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Process each character
    all_stats = []

    for input_dir_str, char_name in zip(args.input_dirs, args.character_names):
        input_dir = Path(input_dir_str)

        if not input_dir.exists():
            print(f"WARNING: {input_dir} does not exist, skipping...")
            continue

        output_dir = output_base / f"{char_name}_processed"

        stats = process_character_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            character_name=char_name,
            target_count=args.target_count,
            skip_rembg=args.skip_rembg,
            skip_lama=args.skip_lama
        )

        all_stats.append(stats)

    # Save statistics
    stats_file = output_base / "processing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'='*70}")
    print("ALL CHARACTERS PROCESSED")
    print(f"{'='*70}")
    for stats in all_stats:
        print(f"{stats['character']}: {stats['base_count']} → {stats['total_output']} images")
    print(f"\nStats saved to: {stats_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
