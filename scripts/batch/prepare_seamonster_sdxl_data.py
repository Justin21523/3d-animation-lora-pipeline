#!/usr/bin/env python3
"""
Prepare Sea Monster SDXL Training Data with Augmentation

For Alberto and Luca sea monster forms - augment small datasets to 200+ images
Uses conservative augmentation suitable for 3D animated characters.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import shutil
import random
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def get_augmentation_params(base_count: int, target_count: int = 200) -> int:
    """
    Calculate how many augmentations needed per image.

    Args:
        base_count: Original image count
        target_count: Target total count (default 200)

    Returns:
        Number of augmentations per image
    """
    # Calculate augmentations needed per image to reach target
    augs_per_image = max(1, int(np.ceil((target_count - base_count) / base_count)))

    return augs_per_image


def augment_image(image: Image.Image, aug_id: int, seed: int = None) -> Image.Image:
    """
    Apply conservative augmentation suitable for 3D animated sea monsters.

    Augmentations for 3D animation (sea monsters):
    - Subtle brightness/contrast (preserve materials)
    - Very light rotation (-5 to +5 degrees)
    - Minimal saturation adjustment (preserve underwater colors)
    - Light gaussian blur (simulate depth of field)

    NO horizontal flip (sea monsters have asymmetric features)
    NO color jitter (breaks underwater lighting consistency)

    Args:
        image: PIL Image
        aug_id: Augmentation variant ID (0-4)
        seed: Random seed for reproducibility

    Returns:
        Augmented PIL Image
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img = image.copy()

    # Augmentation variant based on aug_id
    if aug_id == 0:
        # Original + subtle brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))

    elif aug_id == 1:
        # Subtle contrast adjustment
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))

    elif aug_id == 2:
        # Very light rotation (-5 to +5 degrees)
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))

    elif aug_id == 3:
        # Minimal saturation (preserve underwater blue-green tones)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.98, 1.02))

    elif aug_id == 4:
        # Light gaussian blur (simulate DoF)
        blur_radius = random.uniform(0.3, 0.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    else:
        # For aug_id >= 5, combine multiple subtle augmentations
        # Brightness + Contrast
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))

    return img


def letterbox_resize(image: Image.Image, target_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """
    Letterbox resize to target size (preserves aspect ratio, adds black borders).

    Args:
        image: PIL Image
        target_size: Target (width, height)

    Returns:
        Letterboxed PIL Image
    """
    target_w, target_h = target_size
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


def process_character_dataset(
    input_dir: Path,
    output_dir: Path,
    character_name: str,
    target_count: int = 200,
    target_size: Tuple[int, int] = (1024, 1024)
) -> Dict:
    """
    Process single character sea monster dataset with augmentation.

    Args:
        input_dir: Source images directory
        output_dir: Output directory for augmented dataset
        character_name: Character name (for naming)
        target_count: Target number of images
        target_size: Target resolution (width, height)

    Returns:
        Processing statistics
    """
    # Create output directory
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
    print(f"Base images: {base_count}")
    print(f"Target count: {target_count}")

    # Calculate augmentations needed
    augs_per_image = get_augmentation_params(base_count, target_count)
    total_expected = base_count * (1 + augs_per_image)

    print(f"Augmentations per image: {augs_per_image}")
    print(f"Expected total: {total_expected}")
    print(f"{'='*70}\n")

    # Process each image
    output_count = 0

    for img_path in tqdm(input_images, desc=f"Processing {character_name}"):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')

            # Original image (letterboxed to 1024x1024)
            orig_letterboxed = letterbox_resize(img, target_size)
            orig_output_path = output_dir / f"{character_name}_{img_path.stem}_orig.png"
            orig_letterboxed.save(orig_output_path, 'PNG', quality=95)
            output_count += 1

            # Generate augmented versions
            for aug_idx in range(augs_per_image):
                # Apply augmentation (use abs and modulo to keep seed in valid range)
                seed_val = abs(hash(img_path.name) + aug_idx) % (2**32 - 1)
                aug_img = augment_image(img, aug_idx, seed=seed_val)

                # Letterbox resize
                aug_letterboxed = letterbox_resize(aug_img, target_size)

                # Save augmented image
                aug_output_path = output_dir / f"{character_name}_{img_path.stem}_aug{aug_idx}.png"
                aug_letterboxed.save(aug_output_path, 'PNG', quality=95)
                output_count += 1

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    # Statistics
    stats = {
        "character": character_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "base_count": base_count,
        "target_count": target_count,
        "augs_per_image": augs_per_image,
        "total_output": output_count,
        "target_size": target_size
    }

    print(f"\n✅ {character_name} Complete:")
    print(f"   Base images: {base_count}")
    print(f"   Augmented images: {output_count}")
    print(f"   Saved to: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare Sea Monster SDXL Training Data")
    parser.add_argument("--input-dirs", nargs="+", required=True,
                        help="Input directories (e.g., alberto_seamonster luca_seamonster)")
    parser.add_argument("--character-names", nargs="+", required=True,
                        help="Character names matching input dirs")
    parser.add_argument("--output-base", type=str, required=True,
                        help="Base output directory")
    parser.add_argument("--target-count", type=int, default=200,
                        help="Target number of images per character (default: 200)")
    parser.add_argument("--target-size", type=int, nargs=2, default=[1024, 1024],
                        help="Target image size (width height)")

    args = parser.parse_args()

    # Validate
    if len(args.input_dirs) != len(args.character_names):
        print("ERROR: Number of input dirs must match number of character names")
        sys.exit(1)

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    target_size = tuple(args.target_size)

    # Process each character
    all_stats = []

    for input_dir_str, char_name in zip(args.input_dirs, args.character_names):
        input_dir = Path(input_dir_str)

        if not input_dir.exists():
            print(f"WARNING: {input_dir} does not exist, skipping...")
            continue

        output_dir = output_base / f"{char_name}_sdxl"

        stats = process_character_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            character_name=char_name,
            target_count=args.target_count,
            target_size=target_size
        )

        all_stats.append(stats)

    # Save overall statistics
    stats_file = output_base / "augmentation_stats.json"
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
