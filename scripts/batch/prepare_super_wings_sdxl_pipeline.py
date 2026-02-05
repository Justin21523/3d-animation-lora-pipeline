#!/usr/bin/env python3
"""
Super Wings SDXL LoRA Training - Complete Pipeline Orchestrator

Prepares all 9 Super Wings characters for SDXL LoRA training through:
1. Intelligent upscaling (RealESRGAN 2x/4x)
2. Quality enhancement (CodeFormer + CLAHE + denoising)
3. Letterbox padding to 1024x1024 (black borders)
4. LLMProvider API caption generation (Haiku 3.5)
5. Data augmentation (ensure ≤300 images per character)
6. SDXL training config generation

Based on successful Pixar character LoRA results:
- Best checkpoints at Epoch 1-2
- network_dim=64, alpha=32
- AdamW8bit + BF16 + gradient checkpointing (16GB VRAM)

Author: LLMProvider Tooling
Date: 2025-12-13
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Set
import shutil

# ============================================================================
# Configuration
# ============================================================================

# Paths
CHARACTERS_BASE = Path("/mnt/data/datasets/general/super-wings/lora_data/characters")
PROCESSED_BASE = Path("/mnt/data/datasets/general/super-wings/processed")
TRAINING_DATA_BASE = Path("/mnt/data/datasets/general/super-wings/training_data")
SCRIPTS_BASE = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/scripts")
CONFIGS_BASE = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs")

# Character list (9 characters from actual clustering results)
CHARACTERS = [
    "beard", "bello", "chase", "donnie", "flip",
    "jerome", "jett", "paul", "todd"
]

# Pipeline settings
MAX_IMAGES_PER_CHARACTER = 300  # User requirement
MIN_IMAGE_SIZE = 1024           # Target size before padding
TARGET_RESOLUTION = (1024, 1024)  # Final SDXL resolution

# Tool paths
SUPER_RESOLUTION_SCRIPT = SCRIPTS_BASE / "generic/inpainting/super_resolution.py"
FRAME_ENHANCEMENT_SCRIPT = SCRIPTS_BASE / "generic/inpainting/frame_enhancement.py"
PREPROCESS_SDXL_SCRIPT = SCRIPTS_BASE / "batch/preprocess_images_for_sdxl.py"
RECAPTION_LLM_PROVIDER_SCRIPT = SCRIPTS_BASE / "generic/training/caption_engines/recaption_with_llm_provider.py"

# Enhancement config (3D character optimized)
ENHANCEMENT_CONFIG = CONFIGS_BASE / "stages/enhancement/3d_character_enhancement.yaml"

# ============================================================================
# Helper Functions
# ============================================================================

def run_command(cmd: List[str], description: str, env: Dict = None) -> bool:
    """Run a command and return success status"""
    print(f"\n{'=' * 70}")
    print(f"🔧 {description}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})}
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def get_character_image_count(char_dir: Path) -> int:
    """Count images in character directory"""
    if not char_dir.exists():
        return 0
    return len(list(char_dir.glob("*.png"))) + len(list(char_dir.glob("*.jpg")))

def augment_character_dataset(input_dir: Path, output_dir: Path, target_count: int) -> bool:
    """
    Augment character dataset to reach target count
    Uses horizontal flip + slight rotation only (safe for 3D characters)
    """
    from PIL import Image
    import random

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy original images
    original_images = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    current_count = len(original_images)

    print(f"  Original images: {current_count}")
    print(f"  Target count: {target_count}")

    if current_count >= target_count:
        print(f"  ⏭️  No augmentation needed (already have {current_count} images)")
        # Just copy originals
        for img_path in original_images[:target_count]:
            shutil.copy(img_path, output_dir / img_path.name)
        return True

    # Copy all originals first
    for img_path in original_images:
        shutil.copy(img_path, output_dir / img_path.name)

    # Calculate how many augmented images needed
    needed = target_count - current_count
    print(f"  Need {needed} augmented images")

    # Create augmented images
    augmented = 0
    for i in range(needed):
        # Pick random source image
        source_img = random.choice(original_images)
        img = Image.open(source_img)

        # Random augmentation (safe for 3D)
        aug_type = random.choice(['flip', 'rotate_small'])

        if aug_type == 'flip' and random.random() > 0.7:  # Only flip 30% of time
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            suffix = "flip"
        elif aug_type == 'rotate_small':
            angle = random.uniform(-5, 5)  # Very small rotation
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
            suffix = f"rot{int(angle)}"
        else:
            # Brightness adjustment
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.9, 1.1)
            img = enhancer.enhance(factor)
            suffix = "bright"

        # Save augmented image
        aug_name = f"{source_img.stem}_aug{i:03d}_{suffix}{source_img.suffix}"
        img.save(output_dir / aug_name)
        augmented += 1

    print(f"  ✅ Created {augmented} augmented images")
    final_count = len(list(output_dir.glob("*.png"))) + len(list(output_dir.glob("*.jpg")))
    print(f"  Final count: {final_count}")

    return True

# ============================================================================
# Pipeline Stages
# ============================================================================

def stage_upscaling(character: str) -> bool:
    """
    Stage: Intelligent Upscaling
    Uses RealESRGAN with auto 2x/4x decision
    """
    input_dir = CHARACTERS_BASE / character
    output_dir = PROCESSED_BASE / "upscaled" / character

    if not input_dir.exists():
        print(f"⚠️  Character directory not found: {input_dir}")
        return False

    cmd = [
        "conda", "run", "-n", "ai_env",
        "python", str(SUPER_RESOLUTION_SCRIPT),
        str(input_dir),  # Positional argument (not --input-dir)
        "--output-dir", str(output_dir),
        "--model", "RealESRGAN_x4plus",  # Best quality model
        "--min-size", str(MIN_IMAGE_SIZE),
        "--tile-size", "512",  # Not --tile
        "--device", "cuda"
    ]

    return run_command(cmd, f"Upscaling {character} images")

def stage_enhancement(character: str) -> bool:
    """
    Stage: Quality Enhancement
    Uses CodeFormer + CLAHE + CNN denoising + sharpening
    """
    input_dir = PROCESSED_BASE / "upscaled" / character
    output_dir = PROCESSED_BASE / "enhanced" / character

    if not ENHANCEMENT_CONFIG.exists():
        print(f"⚠️  Enhancement config not found: {ENHANCEMENT_CONFIG}")
        print(f"  Skipping enhancement for {character}")
        # Copy upscaled to enhanced
        if input_dir.exists():
            shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
        return True

    cmd = [
        "conda", "run", "-n", "ai_env",
        "python", str(FRAME_ENHANCEMENT_SCRIPT),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--config", str(ENHANCEMENT_CONFIG),
        "--mode", "quality",
        "--device", "cuda"
    ]

    return run_command(cmd, f"Enhancing {character} images")

def stage_letterbox_padding(character: str) -> bool:
    """
    Stage: Letterbox Padding to 1024x1024
    Preserves aspect ratio with black borders
    Uses PIL for simple, reliable processing
    """
    from PIL import Image

    # Skip upscaling/enhancement - use original character images directly
    input_dir = CHARACTERS_BASE / character
    output_dir = PROCESSED_BASE / "padded" / character

    if not input_dir.exists():
        print(f"⚠️  Input directory not found: {input_dir}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all images
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

    if not image_files:
        print(f"⚠️  No images found in {input_dir}")
        return False

    print(f"  Processing {len(image_files)} images...")
    processed = 0

    for img_path in image_files:
        try:
            # Load image
            img = Image.open(img_path)

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Calculate scaling to fit within 1024x1024
            width, height = img.size
            max_dim = max(width, height)

            if max_dim > 1024:
                # Scale down to fit
                scale = 1024 / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                width, height = new_width, new_height

            # Create black 1024x1024 canvas
            canvas = Image.new('RGB', (1024, 1024), (0, 0, 0))

            # Calculate paste position (center)
            x_offset = (1024 - width) // 2
            y_offset = (1024 - height) // 2

            # Paste image onto canvas
            canvas.paste(img, (x_offset, y_offset))

            # Save
            output_path = output_dir / img_path.name
            canvas.save(output_path, quality=95)
            processed += 1

        except Exception as e:
            print(f"    ⚠️  Failed to process {img_path.name}: {e}")
            continue

    print(f"  ✅ Successfully processed {processed}/{len(image_files)} images")
    return processed > 0

def stage_augmentation(character: str) -> bool:
    """
    Stage: Data Augmentation
    Ensure each character has exactly ≤300 images
    """
    input_dir = PROCESSED_BASE / "padded" / character
    output_dir = PROCESSED_BASE / "augmented" / character

    current_count = get_character_image_count(input_dir)

    if current_count > MAX_IMAGES_PER_CHARACTER:
        print(f"  {character}: {current_count} images (will sample {MAX_IMAGES_PER_CHARACTER})")
        # Random sample
        output_dir.mkdir(parents=True, exist_ok=True)
        all_images = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        import random
        selected = random.sample(all_images, MAX_IMAGES_PER_CHARACTER)
        for img in selected:
            shutil.copy(img, output_dir / img.name)
        return True
    elif current_count < MAX_IMAGES_PER_CHARACTER:
        print(f"  {character}: {current_count} images (will augment to {MAX_IMAGES_PER_CHARACTER})")
        return augment_character_dataset(input_dir, output_dir, MAX_IMAGES_PER_CHARACTER)
    else:
        print(f"  {character}: {current_count} images (perfect, no changes needed)")
        shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
        return True

def stage_caption_generation(character: str) -> bool:
    """
    Stage: LLMProvider API Caption Generation
    Uses Haiku 3.5 for cost-effective, high-quality captions
    """
    input_dir = PROCESSED_BASE / "augmented" / character
    output_dir = TRAINING_DATA_BASE / character / f"10_{character}"  # Kohya format: repeat_concept

    # Check for API key
    if "LLM_VENDOR_API_KEY" not in os.environ:
        print("❌ LLM_VENDOR_API_KEY not set! Please export it first.")
        print("   export LLM_VENDOR_API_KEY='sk-ant-...'")
        return False

    cmd = [
        "conda", "run", "-n", "ai_env",
        "python", str(RECAPTION_LLM_PROVIDER_SCRIPT),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--model", "llm_provider-3-5-haiku-20241022",  # Cost-effective
        "--character-name", character.capitalize(),
        "--lora-type", "character",  # Character identity LoRA
        "--batch-size", "10",
        "--cache-system-prompt",
        "--output-format", "kohya"
    ]

    # Add character-specific prompt if config exists
    caption_config = CONFIGS_BASE / "training/super_wings_caption_config.yaml"
    if caption_config.exists():
        cmd.extend(["--character-config", str(caption_config)])

    return run_command(cmd, f"Generating captions for {character} with LLMProvider API")

# ============================================================================
# Main Pipeline
# ============================================================================

def process_character(character: str, skip_stages: Set[str] = None) -> bool:
    """Process a single character through the entire pipeline"""
    skip_stages = skip_stages or set()

    print("\n" + "=" * 70)
    print(f"🎯 Processing Character: {character.upper()}")
    print("=" * 70)

    # Count original images
    orig_count = get_character_image_count(CHARACTERS_BASE / character)
    print(f"Original images: {orig_count}")

    # Run pipeline stages (skip upscaling/enhancement due to basicsr issues)
    stages = [
        # ("upscaling", stage_upscaling),  # SKIPPED: basicsr dependency issues
        # ("enhancement", stage_enhancement),  # SKIPPED: not needed for SAM2 instances
        ("letterbox", stage_letterbox_padding),
        ("augmentation", stage_augmentation),
        ("captions", stage_caption_generation)
    ]

    for stage_name, stage_func in stages:
        if stage_name in skip_stages:
            print(f"⏭️  Skipping {stage_name} (user requested)")
            continue

        success = stage_func(character)
        if not success:
            print(f"❌ Pipeline failed at {stage_name} for {character}")
            return False

    print(f"\n✅ {character} completed successfully!")
    return True

def main():
    print("=" * 70)
    print("Super Wings SDXL LoRA Training - Pipeline Orchestrator")
    print("=" * 70)
    print()
    print("Pipeline Stages:")
    print("  1. Letterbox Padding to 1024x1024 (black borders)")
    print("  2. Data Augmentation (≤300 images per character)")
    print("  3. LLMProvider API Caption Generation (Haiku 3.5)")
    print()
    print(f"Characters to process: {len(CHARACTERS)}")
    print("  " + ", ".join(CHARACTERS))
    print()
    print("⏭️  Skipping initial image counting to avoid hangs")
    print("    Image counts will be shown per character during processing")
    print()

    # Optional: Skip certain stages
    skip_stages = set()

    # Skip enhancement automatically (already commented out in pipeline stages)
    # No interactive prompt needed since enhancement is not in the pipeline
    skip_stages.add("enhancement")

    # Process all characters
    successful = []
    failed = []

    for i, character in enumerate(CHARACTERS, 1):
        print(f"\n[{i}/{len(CHARACTERS)}] Starting {character}...")

        if process_character(character, skip_stages):
            successful.append(character)
        else:
            failed.append(character)
            print(f"⚠️  Continuing to next character...")

    # Final summary
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print(f"✅ Successful: {len(successful)}/{len(CHARACTERS)}")
    if successful:
        print("   Characters ready for training:")
        for char in successful:
            print(f"   - {char}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for char in failed:
            print(f"   - {char}")

    print("\nNext steps:")
    print("  1. Review training data in:", TRAINING_DATA_BASE)
    print("  2. Generate SDXL training configs:")
    print("     python scripts/batch/generate_super_wings_sdxl_configs.py")
    print("  3. Start training:")
    print("     python scripts/batch/train_super_wings_sdxl_loras.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
