#!/usr/bin/env python3
"""
Convert _ctx images to pure instance segmentation images
Keeps the same captions but uses pure cutout versions without background
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_ctx_to_pure_instances(
    ctx_dir: str,
    instances_dir: str,
    output_dir: str
):
    """
    Convert _ctx images to pure instance images

    Args:
        ctx_dir: Directory containing _ctx.png images and .txt captions
        instances_dir: Directory containing pure instance segmentation images (without _ctx)
        output_dir: Output directory for pure instances + captions
    """
    ctx_path = Path(ctx_dir)
    instances_path = Path(instances_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all _ctx image files
    ctx_images = list(ctx_path.glob("*_ctx.png"))

    print(f"Found {len(ctx_images)} _ctx images")
    print(f"Looking for pure instances in: {instances_path}")

    found_count = 0
    missing_count = 0
    caption_count = 0

    for ctx_img in tqdm(ctx_images, desc="Converting"):
        # Get base filename without _ctx suffix
        base_name = ctx_img.stem.replace("_ctx", "")

        # Look for corresponding pure instance image
        pure_instance = instances_path / f"{base_name}.png"

        if pure_instance.exists():
            # Copy pure instance image
            shutil.copy2(pure_instance, output_path / f"{base_name}.png")
            found_count += 1

            # Copy caption if exists
            caption_file = ctx_img.with_suffix(".txt")
            if caption_file.exists():
                shutil.copy2(caption_file, output_path / f"{base_name}.txt")
                caption_count += 1
        else:
            print(f"\n⚠️  Missing pure instance: {base_name}.png")
            missing_count += 1

    print(f"\n✓ Conversion complete!")
    print(f"  Found: {found_count} images")
    print(f"  Missing: {missing_count} images")
    print(f"  Captions copied: {caption_count}")
    print(f"  Output directory: {output_path}")

    return found_count, missing_count

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert _ctx images to pure instances")
    parser.add_argument("--ctx-dir", required=True, help="Directory with _ctx images")
    parser.add_argument("--instances-dir", required=True, help="Directory with pure instances")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    convert_ctx_to_pure_instances(
        args.ctx_dir,
        args.instances_dir,
        args.output_dir
    )
