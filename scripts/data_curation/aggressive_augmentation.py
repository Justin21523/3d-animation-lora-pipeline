#!/usr/bin/env python3
"""
Aggressive data augmentation for severe imbalances
Creates more synthetic variations when source data is limited
"""

import argparse
from pathlib import Path
import shutil
from PIL import Image, ImageFilter, ImageEnhance
import random
from typing import List, Tuple


def apply_aggressive_transforms(img: Image.Image, caption: str, base_name: str, output_dir: Path) -> List[Tuple[str, str]]:
    """
    Apply multiple transforms to create variations

    Returns list of (filename, caption) tuples
    """
    variations = []
    w, h = img.size

    # 1. Zoom variations (simulate different distances)
    for zoom_idx, zoom_factor in enumerate([0.7, 0.85, 1.15, 1.3]):
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)

        if zoom_factor < 1.0:
            # Zoom out - add background border
            bg_color = (130, 150, 170)  # Sky blue-ish
            canvas = Image.new("RGB", (w, h), bg_color)
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            paste_x = (w - new_w) // 2
            paste_y = (h - new_h) // 2
            canvas.paste(resized, (paste_x, paste_y))
            zoomed = canvas

            # Adjust caption for wider shot
            if "close-up" in caption or "close up" in caption:
                new_caption = caption.replace("close-up", "medium shot").replace("close up", "medium shot")
            else:
                new_caption = caption + ", wider shot, more environmental context"
        else:
            # Zoom in - crop center
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            crop_x = (new_w - w) // 2
            crop_y = (new_h - h) // 2
            zoomed = resized.crop((crop_x, crop_y, crop_x + w, crop_y + h))

            # Adjust caption for closer shot
            new_caption = caption + ", closer view, more detailed"

        filename = f"{base_name}_zoom{zoom_idx}.png"
        zoomed.save(output_dir / filename)
        variations.append((filename, new_caption))

    # 2. Lighting variations (simulate different conditions)
    for light_idx, (brightness, contrast) in enumerate([
        (0.8, 0.9),   # Darker/low contrast
        (1.2, 1.1),   # Brighter/higher contrast
    ]):
        adjusted = img.copy()

        # Adjust brightness
        enhancer = ImageEnhance.Brightness(adjusted)
        adjusted = enhancer.enhance(brightness)

        # Adjust contrast
        enhancer = ImageEnhance.Contrast(adjusted)
        adjusted = enhancer.enhance(contrast)

        filename = f"{base_name}_light{light_idx}.png"
        adjusted.save(output_dir / filename)

        light_desc = "darker lighting, muted colors" if brightness < 1 else "brighter lighting, vibrant colors"
        new_caption = caption + f", {light_desc}"
        variations.append((filename, new_caption))

    # 3. Simulated occlusion (add blur/overlay at edges)
    for occ_idx, edge in enumerate(["left", "right", "bottom"]):
        occluded = img.copy()
        mask = Image.new("L", (w, h), 255)

        if edge == "left":
            # Blur left edge
            for x in range(w // 4):
                alpha = int(255 * (x / (w // 4)))
                for y in range(h):
                    mask.putpixel((x, y), alpha)
            new_caption = caption + ", partially obscured on left side"

        elif edge == "right":
            # Blur right edge
            for x in range(w * 3 // 4, w):
                alpha = int(255 * ((w - x) / (w // 4)))
                for y in range(h):
                    mask.putpixel((x, y), alpha)
            new_caption = caption + ", partially obscured on right side"

        else:  # bottom
            # Blur bottom edge
            for y in range(h * 3 // 4, h):
                alpha = int(255 * ((h - y) / (h // 4)))
                for x in range(w):
                    mask.putpixel((x, y), alpha)
            new_caption = caption + ", partially obscured at bottom, foreground elements"

        # Apply blur with mask
        blurred = occluded.filter(ImageFilter.GaussianBlur(radius=8))
        occluded = Image.composite(occluded, blurred, mask)

        filename = f"{base_name}_occ{occ_idx}.png"
        occluded.save(output_dir / filename)
        variations.append((filename, new_caption))

    # 4. Environmental context (add color tints for different settings)
    for env_idx, (tint, desc) in enumerate([
        ((255, 235, 200), "warm indoor lighting, cozy atmosphere"),
        ((200, 220, 255), "cool outdoor lighting, italian riviera"),
        ((255, 245, 220), "golden hour lighting, sunset"),
    ]):
        tinted = img.copy().convert("RGB")
        overlay = Image.new("RGB", (w, h), tint)
        # Blend with 15% opacity
        tinted = Image.blend(tinted, overlay, 0.15)

        filename = f"{base_name}_env{env_idx}.png"
        tinted.save(output_dir / filename)
        new_caption = caption + f", {desc}"
        variations.append((filename, new_caption))

    return variations


def aggressive_augment(
    dataset_dir: Path,
    output_dir: Path,
    samples_per_image: int = 10,
    seed: int = 42
):
    """
    Apply aggressive augmentation to multiply dataset size

    Args:
        dataset_dir: Original dataset
        output_dir: Output directory
        samples_per_image: How many variations per source image
        seed: Random seed
    """
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy originals
    print(f"📋 Copying original dataset...")
    for img_file in dataset_dir.glob("*.png"):
        shutil.copy2(img_file, output_dir / img_file.name)
    for txt_file in dataset_dir.glob("*.txt"):
        shutil.copy2(txt_file, output_dir / txt_file.name)

    original_count = len(list(output_dir.glob("*.png")))
    print(f"✓ Copied {original_count} original images")

    # Get all source images
    source_images = list(dataset_dir.glob("*.png"))

    # Focus augmentation on underrepresented types
    print(f"\n🔄 Applying aggressive augmentation...")
    print(f"  Creating {samples_per_image} variations per image")

    aug_count = 0

    for img_path in source_images:
        caption_path = img_path.with_suffix(".txt")
        if not caption_path.exists():
            continue

        with open(caption_path, 'r', encoding='utf-8') as f:
            original_caption = f.read().strip()

        img = Image.open(img_path)
        base_name = img_path.stem

        # Generate variations
        variations = apply_aggressive_transforms(
            img, original_caption, base_name, output_dir
        )

        # Save captions
        for filename, caption in variations:
            caption_file = output_dir / f"{Path(filename).stem}.txt"
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(caption)
            aug_count += 1

    final_count = len(list(output_dir.glob("*.png")))

    print(f"\n✅ AGGRESSIVE AUGMENTATION COMPLETE")
    print(f"  Original:   {original_count}")
    print(f"  Augmented:  +{aug_count}")
    print(f"  Total:      {final_count}")
    print(f"  Multiplier: {final_count / original_count:.1f}x")
    print(f"\n📁 Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Aggressive dataset augmentation")
    parser.add_argument("dataset_dir", type=Path, help="Source dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--samples-per-image", type=int, default=10, help="Variations per image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    aggressive_augment(
        args.dataset_dir,
        args.output_dir,
        args.samples_per_image,
        args.seed
    )


if __name__ == "__main__":
    main()
