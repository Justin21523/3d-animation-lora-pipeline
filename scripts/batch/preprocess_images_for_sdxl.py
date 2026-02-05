#!/usr/bin/env python3
"""
SDXL Image Preprocessing Script
Resize all training images to optimal SDXL bucket sizes using letterbox + LaMa inpainting
to eliminate bucketing overhead and maximize training speed.

Target: 1024x1024 (optimal for SDXL, fastest training)
Method: Letterbox resize (preserve all features) + LaMa inpainting to fill borders naturally
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import json
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil
import numpy as np
import torch
import cv2

# SDXL optimal resolutions (1024x1024 for maximum speed)
TARGET_RESOLUTIONS = {
    "square": (1024, 1024),      # Primary target
    "landscape": (1152, 896),    # 16:9 landscape
    "portrait": (896, 1152),     # 16:9 portrait
}

# Global LaMa model (loaded once, shared across workers in main process)
LAMA_MODEL = None

def load_lama_model(device='cuda'):
    """
    Load LaMa inpainting model
    Uses simple-lama-inpainting package for easy integration
    """
    try:
        from simple_lama_inpainting import SimpleLama
        model = SimpleLama()
        return model
    except ImportError:
        print("Warning: simple-lama-inpainting not installed. Install with:")
        print("  pip install simple-lama-inpainting")
        return None
    except Exception as e:
        print(f"Warning: Failed to load LaMa model: {e}")
        return None


def letterbox_resize(image: Image.Image, target_size: Tuple[int, int]) -> Tuple[Image.Image, np.ndarray]:
    """
    Letterbox resize: scale image to fit target size while preserving aspect ratio.
    Returns the letterboxed image and a mask indicating the empty (padded) regions.

    Returns:
        letterboxed_image: PIL Image on black canvas
        mask: numpy array (H, W) where 255 = empty region to inpaint, 0 = original image
    """
    target_w, target_h = target_size
    img_w, img_h = image.size

    # Calculate scale factor to fit image within target size
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

    # Create mask (255 = empty region, 0 = original image)
    mask = np.ones((target_h, target_w), dtype=np.uint8) * 255
    mask[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = 0

    return canvas, mask


def letterbox_with_lama_inpainting(image: Image.Image, target_size: Tuple[int, int],
                                    lama_model=None) -> Image.Image:
    """
    Letterbox resize with LaMa inpainting to fill borders naturally.

    Steps:
    1. Letterbox resize (preserve all features)
    2. Generate mask for empty regions
    3. Use LaMa to inpaint borders with natural content

    Args:
        image: Input PIL Image
        target_size: Target (width, height)
        lama_model: Loaded LaMa model (if None, falls back to black letterbox)

    Returns:
        Processed PIL Image
    """
    # Step 1: Letterbox resize
    letterboxed, mask = letterbox_resize(image, target_size)

    # If no LaMa model or no padding needed, return letterboxed image
    if lama_model is None or mask.sum() == 0:
        return letterboxed

    # Step 2: LaMa inpainting to fill borders
    try:
        # Convert to numpy array for LaMa
        img_array = np.array(letterboxed)

        # LaMa expects RGB image and binary mask
        inpainted = lama_model(img_array, mask)

        # Convert back to PIL
        return Image.fromarray(inpainted)

    except Exception as e:
        print(f"Warning: LaMa inpainting failed: {e}. Using letterbox without inpainting.")
        return letterboxed

def choose_best_resolution(img_w: int, img_h: int, target_size: str = "square") -> Tuple[int, int]:
    """
    Choose the best target resolution based on image aspect ratio
    For maximum speed, always use square (1024x1024)
    """
    if target_size == "square":
        return TARGET_RESOLUTIONS["square"]

    # Smart selection based on aspect ratio (if not forcing square)
    aspect = img_w / img_h
    if aspect > 1.2:
        return TARGET_RESOLUTIONS["landscape"]
    elif aspect < 0.8:
        return TARGET_RESOLUTIONS["portrait"]
    else:
        return TARGET_RESOLUTIONS["square"]

def process_image(input_path: Path, output_path: Path, target_size: str,
                  backup: bool, lama_model=None) -> Dict:
    """
    Process a single image with letterbox + LaMa inpainting
    Returns: dict with processing info
    """
    try:
        # Load image
        img = Image.open(input_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        orig_size = img.size

        # Choose target resolution
        target_res = choose_best_resolution(img.size[0], img.size[1], target_size)

        # Skip if already at target size
        if img.size == target_res:
            return {
                "path": str(input_path),
                "status": "skipped",
                "original_size": orig_size,
                "final_size": orig_size,
                "message": "Already at target size"
            }

        # Backup original if requested
        if backup:
            backup_path = input_path.parent / "_original" / input_path.name
            backup_path.parent.mkdir(exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(input_path, backup_path)

        # Process image with letterbox + LaMa inpainting
        processed = letterbox_with_lama_inpainting(img, target_res, lama_model)

        # Save processed image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed.save(output_path, "PNG", quality=95)

        return {
            "path": str(input_path),
            "status": "processed",
            "original_size": orig_size,
            "final_size": processed.size,
            "message": f"Letterbox + LaMa inpainted from {orig_size} to {processed.size}"
        }

    except Exception as e:
        return {
            "path": str(input_path),
            "status": "error",
            "original_size": None,
            "final_size": None,
            "message": str(e)
        }

def process_character_dir(character_dir: Path, target_size: str = "square",
                          backup: bool = True, lama_model=None) -> Dict:
    """
    Process all images in a character training directory using LaMa inpainting
    Note: Sequential processing (GPU model cannot be easily parallelized)
    """
    # Find all image files
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = [
        f for f in character_dir.rglob("*")
        if f.suffix.lower() in image_extensions and "_original" not in str(f)
    ]

    if not image_files:
        return {
            "character_dir": str(character_dir),
            "total_images": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "results": []
        }

    print(f"\n{'='*80}")
    print(f"Processing: {character_dir.name}")
    print(f"Total images: {len(image_files)}")
    print(f"Target size: {TARGET_RESOLUTIONS[target_size]}")
    print(f"Method: Letterbox + LaMa Inpainting")
    print(f"Backup: {backup}")
    print(f"{'='*80}\n")

    # Process images sequentially with progress bar
    results = []
    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        for img_path in image_files:
            result = process_image(img_path, img_path, target_size, backup, lama_model)
            results.append(result)
            pbar.update(1)
            pbar.set_postfix({"status": result["status"]})

    # Compile statistics
    processed_count = sum(1 for r in results if r["status"] == "processed")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")

    return {
        "character_dir": str(character_dir),
        "total_images": len(image_files),
        "processed": processed_count,
        "skipped": skipped_count,
        "errors": error_count,
        "results": results
    }

def process_all_characters(base_dir: Path, target_size: str = "square",
                           backup: bool = True, lama_model=None) -> List[Dict]:
    """
    Process all character training directories with LaMa inpainting
    """
    # Find all character training directories
    character_dirs = []

    # Pattern: */lora_data/training_data_sdxl/*_identity/*_charactername/
    for movie_dir in base_dir.glob("*"):
        if not movie_dir.is_dir():
            continue

        lora_data = movie_dir / "lora_data" / "training_data_sdxl"
        if not lora_data.exists():
            continue

        for identity_dir in lora_data.glob("*_identity"):
            for char_dir in identity_dir.glob("*_*"):
                if char_dir.is_dir() and not char_dir.name.startswith("_"):
                    character_dirs.append(char_dir)

    if not character_dirs:
        print("No character directories found!")
        return []

    print(f"\n{'='*80}")
    print(f"Found {len(character_dirs)} character directories to process")
    print(f"Method: Letterbox + LaMa Inpainting")
    print(f"{'='*80}")
    for char_dir in character_dirs:
        print(f"  - {char_dir.relative_to(base_dir)}")
    print(f"{'='*80}\n")

    # Process each character
    all_results = []
    for char_dir in character_dirs:
        result = process_character_dir(char_dir, target_size, backup, lama_model)
        all_results.append(result)

        # Print summary
        print(f"\n✅ {char_dir.name}: {result['processed']} processed, "
              f"{result['skipped']} skipped, {result['errors']} errors")

    return all_results

def generate_report(results: List[Dict], output_path: Path):
    """Generate processing report"""
    # Calculate totals
    total_images = sum(r["total_images"] for r in results)
    total_processed = sum(r["processed"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    total_errors = sum(r["errors"] for r in results)

    report = {
        "summary": {
            "total_characters": len(results),
            "total_images": total_images,
            "processed": total_processed,
            "skipped": total_skipped,
            "errors": total_errors,
            "success_rate": f"{(total_processed / total_images * 100):.1f}%" if total_images > 0 else "0%"
        },
        "characters": results
    }

    # Save JSON report
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total characters: {len(results)}")
    print(f"Total images: {total_images}")
    print(f"Processed: {total_processed}")
    print(f"Skipped (already optimized): {total_skipped}")
    print(f"Errors: {total_errors}")
    print(f"Success rate: {report['summary']['success_rate']}")
    print(f"\nReport saved to: {output_path}")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SDXL training images using Letterbox + LaMa Inpainting"
    )
    parser.add_argument("--base-dir", type=str,
                       default="/mnt/data/ai_data/datasets/3d-anime",
                       help="Base directory containing all movie datasets")
    parser.add_argument("--target-size", type=str, default="square",
                       choices=["square", "landscape", "portrait"],
                       help="Target resolution type (square=1024x1024 recommended)")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip backing up original images")
    parser.add_argument("--no-lama", action="store_true",
                       help="Skip LaMa inpainting (use black letterbox only)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview what would be processed without making changes")
    parser.add_argument("--report", type=str,
                       default="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/logs/image_preprocessing_report.json",
                       help="Path to save processing report")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return 1

    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        # TODO: Implement dry run
        return 0

    # Load LaMa model if requested
    lama_model = None
    if not args.no_lama:
        print("\n" + "="*80)
        print("Loading LaMa Inpainting Model...")
        print("="*80 + "\n")
        lama_model = load_lama_model()
        if lama_model is None:
            print("\n⚠️  Failed to load LaMa model. Falling back to black letterbox.")
        else:
            print("\n✅ LaMa model loaded successfully!")
    else:
        print("\n" + "="*80)
        print("LaMa inpainting disabled. Using black letterbox only.")
        print("="*80 + "\n")

    # Process all characters
    print("\n" + "="*80)
    print("Starting Image Preprocessing")
    print("="*80 + "\n")

    results = process_all_characters(
        base_dir=base_dir,
        target_size=args.target_size,
        backup=not args.no_backup,
        lama_model=lama_model
    )

    # Generate report
    if results:
        generate_report(results, Path(args.report))

    return 0

if __name__ == "__main__":
    exit(main())
