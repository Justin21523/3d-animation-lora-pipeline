#!/usr/bin/env python3
"""
Generic Character LoRA Training Pipeline

Prepares character instances for SDXL LoRA training with complete preprocessing:
1. Smart upscale to ~1024px (RealESRGAN, preserve aspect ratio)
2. Quality enhancement (CLAHE + denoise + sharpen)
3. LaMa inpainting (natural background for transparent regions)
4. Letterbox padding to 1024x1024 (black borders, LAST STEP)
5. LLMProvider API caption generation (character-specific)
6. Organize into Kohya training format (repeat_concept/)

GENERIC - Works for any 3D animation project!

Usage:
    python scripts/generic/training/character_lora_pipeline.py \\
        --input-base /path/to/characters \\
        --output-base /path/to/training_data \\
        --characters char1 char2 char3

Author: LLMProvider Tooling
Date: 2025-12-13
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List
from PIL import Image

# ============================================================================
# Pipeline Stage Functions
# ============================================================================

def smart_upscale(character: str, input_base: Path, output_base: Path,
                  scripts_dir: Path, min_size: int = 1024, model: str = "RealESRGAN_x2plus") -> bool:
    """
    Stage 1: Smart upscale to ~1024px (preserve aspect ratio)
    Uses RealESRGAN for high-quality upscaling
    """
    input_dir = input_base / character
    output_dir = output_base / "upscaled" / character

    if not input_dir.exists():
        print(f"⚠️  Character directory not found: {input_dir}")
        return False

    super_resolution_script = scripts_dir / "generic/inpainting/super_resolution.py"
    if not super_resolution_script.exists():
        print(f"❌ Super-resolution script not found: {super_resolution_script}")
        return False

    cmd = [
        "conda", "run", "-n", "ai_env",
        "python", str(super_resolution_script),
        str(input_dir),
        "--output-dir", str(output_dir),
        "--model", model,
        "--min-size", str(min_size),
        "--tile-size", "256",
        "--device", "cuda"
    ]

    print(f"  Running: {' '.join(str(c) for c in cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✅ Upscaling completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Upscaling failed!")
        print(f"  Error: {e.stderr}")
        return False


def quality_enhancement(character: str, output_base: Path, scripts_dir: Path,
                       sharpen: float = 1.2, denoise: int = 5,
                       clahe_clip: float = 2.0, clahe_grid: int = 8) -> bool:
    """
    Stage 2: Quality enhancement (CLAHE + denoise + sharpen)
    """
    input_dir = output_base / "upscaled" / character
    output_dir = output_base / "enhanced" / character

    if not input_dir.exists():
        print(f"⚠️  Input directory not found: {input_dir}")
        return False

    enhancement_script = scripts_dir / "generic/inpainting/instance_enhancement.py"
    if not enhancement_script.exists():
        print(f"❌ Enhancement script not found: {enhancement_script}")
        return False

    cmd = [
        "conda", "run", "-n", "ai_env",
        "python", str(enhancement_script),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--sharpen", str(sharpen),
        "--denoise", str(denoise),
        "--clahe-clip", str(clahe_clip),
        "--clahe-grid", str(clahe_grid),
        "--skip-existing"
    ]

    print(f"  Running: {' '.join(str(c) for c in cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✅ Quality enhancement completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Quality enhancement failed!")
        print(f"  Error: {e.stderr}")
        return False


def lama_inpainting(character: str, output_base: Path, scripts_dir: Path,
                   batch_size: int = 16) -> bool:
    """
    Stage 3: LaMa background inpainting
    """
    input_dir = output_base / "enhanced" / character
    output_dir = output_base / "characters_inpainted" / character

    if not input_dir.exists():
        print(f"⚠️  Input directory not found: {input_dir}")
        return False

    lama_script = scripts_dir / "generic/inpainting/lama_batch_optimized.py"
    if not lama_script.exists():
        print(f"❌ LaMa script not found: {lama_script}")
        return False

    cmd = [
        "conda", "run", "-n", "ai_env",
        "python", str(lama_script),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--flat-input",
        "--batch-size", str(batch_size),
        "--device", "cuda",
        "--skip-existing"
    ]

    print(f"  Running: {' '.join(str(c) for c in cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✅ LaMa inpainting completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ LaMa inpainting failed!")
        print(f"  Error: {e.stderr}")
        return False


def letterbox_padding(character: str, output_base: Path, target_size: int = 1024) -> bool:
    """
    Stage 4: Letterbox padding to target_size x target_size (black borders)
    THIS IS THE LAST PREPROCESSING STEP!
    """
    input_dir = output_base / "characters_inpainted" / character
    output_dir = output_base / "padded" / character

    if not input_dir.exists():
        print(f"⚠️  Input directory not found: {input_dir}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

    if not image_files:
        print(f"⚠️  No images found in {input_dir}")
        return False

    print(f"  Processing {len(image_files)} images...")
    processed = 0

    for img_path in image_files:
        try:
            img = Image.open(img_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size
            max_dim = max(width, height)

            # Scale down if larger than target
            if max_dim > target_size:
                scale = target_size / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                width, height = new_width, new_height

            # Create black canvas
            canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))

            # Center image
            x_offset = (target_size - width) // 2
            y_offset = (target_size - height) // 2
            canvas.paste(img, (x_offset, y_offset))

            # Save
            output_path = output_dir / img_path.name
            canvas.save(output_path, quality=95)
            processed += 1

        except Exception as e:
            print(f"    ⚠️  Failed to process {img_path.name}: {e}")
            continue

    print(f"  ✅ Successfully padded {processed}/{len(image_files)} images")
    return processed > 0


def generate_captions(character: str, output_base: Path, final_output_base: Path,
                     scripts_dir: Path, lora_type: str = "character",
                     batch_size: int = 10, workers: int = 4, skip_padding: bool = False) -> bool:
    """
    Stage 5: LLMProvider API caption generation
    """
    # Use enhanced directory if padding is skipped
    if skip_padding:
        input_dir = output_base / "enhanced" / character / "upscaled"
    else:
        input_dir = output_base / "padded" / character
    output_dir = final_output_base / character / f"10_{character}"  # Kohya format

    if "LLM_VENDOR_API_KEY" not in os.environ:
        print("❌ LLM_VENDOR_API_KEY not set! Please export it first.")
        return False

    recaption_script = scripts_dir / "generic/training/caption_engines/recaption_with_llm_provider.py"
    if not recaption_script.exists():
        print(f"❌ Recaption script not found: {recaption_script}")
        return False

    cmd = [
        "conda", "run", "-n", "ai_env",
        "python", str(recaption_script),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--lora-type", lora_type,
        "--batch-size", str(batch_size),
        "--workers", str(workers)
    ]

    print(f"  Running: {' '.join(str(c) for c in cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✅ Captions generated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Caption generation failed!")
        print(f"  Error: {e.stderr}")
        return False


# ============================================================================
# Main Pipeline
# ============================================================================

def process_character(character: str, args, scripts_dir: Path) -> bool:
    """Process a single character through the complete pipeline."""
    print(f"\n{'=' * 70}")
    print(f"🎯 Processing Character: {character.upper()}")
    print(f"{'=' * 70}")

    # Count original images
    char_dir = args.input_base / character
    if not char_dir.exists():
        print(f"⚠️  Character directory not found: {char_dir}")
        return False

    orig_count = len(list(char_dir.glob("*.png"))) + len(list(char_dir.glob("*.jpg")))
    print(f"Original images: {orig_count}")

    # Stage 1: Smart Upscale
    print(f"\n[1/5] Smart Upscale to ~{args.min_size}px (RealESRGAN)...")
    if not smart_upscale(character, args.input_base, args.intermediate_base,
                         scripts_dir, args.min_size, args.upscale_model):
        print(f"❌ Pipeline failed at upscaling for {character}")
        return False

    # Stage 2: Quality Enhancement
    print(f"\n[2/5] Quality Enhancement (CLAHE + Denoise + Sharpen)...")
    if not quality_enhancement(character, args.intermediate_base, scripts_dir,
                               args.sharpen, args.denoise, args.clahe_clip, args.clahe_grid):
        print(f"❌ Pipeline failed at quality enhancement for {character}")
        return False

    # Stage 3: LaMa Inpainting (optional - skip for SAM2 instances)
    if args.skip_lama:
        print(f"\n[3/5] LaMa Background Inpainting... ⏭️  SKIPPED (SAM2 instances already have clean backgrounds)")
        # Copy enhanced to inpainted directory (from upscaled subdirectory)
        import shutil
        src = args.intermediate_base / "enhanced" / character / "upscaled"
        dst = args.intermediate_base / "characters_inpainted" / character
        if src.exists():
            # Create dst parent directory
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Copy the upscaled subdirectory contents to dst
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  ✅ Copied enhanced images from: {src}")
            print(f"     to: {dst}")
        else:
            print(f"  ⚠️  Source directory not found: {src}")
            return False
    else:
        print(f"\n[3/5] LaMa Background Inpainting...")
        if not lama_inpainting(character, args.intermediate_base, scripts_dir, args.batch_size):
            print(f"❌ Pipeline failed at LaMa inpainting for {character}")
            return False

    # Stage 4: Letterbox Padding (optional - can be skipped for faster processing)
    if not args.skip_padding:
        print(f"\n[4/5] Letterbox Padding to {args.target_size}x{args.target_size}...")
        if not letterbox_padding(character, args.intermediate_base, args.target_size):
            print(f"❌ Pipeline failed at letterbox padding for {character}")
            return False
    else:
        print(f"\n[4/5] Letterbox Padding... ⏭️  SKIPPED (using enhanced images directly)")

    # Stage 5: Caption Generation
    stage_num = "5/5" if not args.skip_padding else "3/3"
    print(f"\n[{stage_num}] LLMProvider API Caption Generation...")
    if not generate_captions(character, args.intermediate_base, args.output_base,
                             scripts_dir, args.lora_type, args.caption_batch, args.caption_workers, args.skip_padding):
        print(f"❌ Pipeline failed at caption generation for {character}")
        return False

    print(f"\n✅ {character} completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generic Character LoRA Training Pipeline"
    )

    # Required arguments
    parser.add_argument(
        "--input-base",
        type=Path,
        required=True,
        help="Input base directory containing character folders (e.g., /path/to/lora_data/characters)"
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        required=True,
        help="Final output directory for training data (e.g., /path/to/lora_data/training_data)"
    )
    parser.add_argument(
        "--characters",
        nargs="+",
        required=True,
        help="List of character names to process"
    )

    # Optional arguments
    parser.add_argument(
        "--intermediate-base",
        type=Path,
        default=None,
        help="Intermediate processing directory (default: same as input-base parent)"
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/scripts"),
        help="Scripts directory (default: auto-detect)"
    )

    # Pipeline parameters
    parser.add_argument("--min-size", type=int, default=1024, help="Minimum size for upscaling (default: 1024)")
    parser.add_argument("--target-size", type=int, default=1024, help="Target size for letterbox (default: 1024)")
    parser.add_argument("--upscale-model", default="RealESRGAN_x2plus", help="Upscale model (default: RealESRGAN_x2plus)")
    parser.add_argument("--sharpen", type=float, default=1.2, help="Sharpening strength (default: 1.2)")
    parser.add_argument("--denoise", type=int, default=5, help="Denoising strength (default: 5)")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit (default: 2.0)")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE grid size (default: 8)")
    parser.add_argument("--batch-size", type=int, default=16, help="LaMa batch size (default: 16)")
    parser.add_argument("--lora-type", default="character", help="LoRA type for captions (default: character)")
    parser.add_argument("--caption-batch", type=int, default=10, help="Caption batch size (default: 10)")
    parser.add_argument("--caption-workers", type=int, default=4, help="Caption workers (default: 4)")
    parser.add_argument("--skip-lama", action="store_true", help="Skip LaMa inpainting (for SAM2 instances with clean backgrounds)")
    parser.add_argument("--skip-padding", action="store_true", help="Skip letterbox padding (use enhanced images directly for faster processing)")

    args = parser.parse_args()

    # Set intermediate base if not provided
    if args.intermediate_base is None:
        args.intermediate_base = args.input_base.parent

    # Auto-detect scripts directory if it doesn't exist
    if not args.scripts_dir.exists():
        # Try to find it relative to this script
        current_file = Path(__file__).resolve()
        args.scripts_dir = current_file.parents[2]  # Go up to project root, then scripts

    print("=" * 70)
    print("Generic Character LoRA Training Pipeline")
    print("=" * 70)
    print()
    print("Pipeline Stages:")
    stages = []
    stages.append("  1. Smart Upscale to ~1024px (RealESRGAN, preserve aspect ratio)")
    stages.append("  2. Quality Enhancement (CLAHE + denoise + sharpen)")

    if not args.skip_lama:
        stages.append("  3. LaMa Background Inpainting")
    else:
        stages.append("  3. LaMa Background Inpainting (SKIPPED)")

    if not args.skip_padding:
        stages.append("  4. Letterbox Padding to 1024x1024 (black borders)")
        stages.append("  5. LLMProvider API Caption Generation (character-specific)")
    else:
        stages.append("  4. Letterbox Padding (SKIPPED)")
        stages.append("  3. LLMProvider API Caption Generation (character-specific)")

    for stage in stages:
        print(stage)
    print()
    print(f"Input:  {args.input_base}")
    print(f"Output: {args.output_base}")
    print(f"Intermediate: {args.intermediate_base}")
    print(f"Characters: {len(args.characters)}")
    print("  " + ", ".join(args.characters))
    print()

    # Check API key
    if "LLM_VENDOR_API_KEY" not in os.environ:
        print("❌ ERROR: LLM_VENDOR_API_KEY not set!")
        print()
        print("Please set your LLMProvider API key:")
        print("  export LLM_VENDOR_API_KEY='sk-ant-...'")
        print()
        return 1

    # Create base directories
    (args.intermediate_base / "upscaled").mkdir(parents=True, exist_ok=True)
    (args.intermediate_base / "enhanced").mkdir(parents=True, exist_ok=True)
    (args.intermediate_base / "characters_inpainted").mkdir(parents=True, exist_ok=True)
    (args.intermediate_base / "padded").mkdir(parents=True, exist_ok=True)
    args.output_base.mkdir(parents=True, exist_ok=True)

    # Process all characters
    successful = []
    failed = []

    for i, character in enumerate(args.characters, 1):
        print(f"\n[{i}/{len(args.characters)}] Starting {character}...")

        if process_character(character, args, args.scripts_dir):
            successful.append(character)
        else:
            failed.append(character)
            print(f"⚠️  Continuing to next character...")

    # Final summary
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print(f"✅ Successful: {len(successful)}/{len(args.characters)}")
    if successful:
        print("   Characters ready for training:")
        for char in successful:
            print(f"   - {char}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for char in failed:
            print(f"   - {char}")

    print("\nNext steps:")
    print("  1. Review training data in:", args.output_base)
    print("  2. Generate SDXL training configs")
    print("  3. Start LoRA training with Kohya_ss")

    return 0 if not failed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
