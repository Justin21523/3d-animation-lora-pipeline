#!/usr/bin/env python3
"""
Tyler SDXL Caption Generation with Augmented Caption Copying
Only generates captions for 46 original images, then copies to 230 augmented versions
"""

import os
import sys
from pathlib import Path
import shutil
import llm_vendor
import time
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.utils.logger import setup_logger

logger = setup_logger(__name__)

SD15_DIR = Path("/mnt/data/ai_data/datasets/3d-anime/turning-red/lora_data/training_data/tyler_identity/15_tyler")
SDXL_DIR = Path("/mnt/data/ai_data/datasets/3d-anime/turning-red/lora_data/training_data_sdxl/tyler_identity/15_tyler")

LLM_VENDOR_API_KEY = os.getenv("LLM_VENDOR_API_KEY")
if not LLM_VENDOR_API_KEY:
    raise ValueError("LLM_VENDOR_API_KEY environment variable not set")

client = llm_vendor.LLMVendor(api_key=LLM_VENDOR_API_KEY)

def expand_caption_to_sdxl(original_caption: str, character_name: str = "Tyler") -> str:
    """Expand SD1.5 caption (77 tokens) to SDXL caption (225 tokens)"""

    prompt = f"""You are expanding a training caption for Stable Diffusion XL (SDXL).

Original SD1.5 caption (max 77 tokens):
{original_caption}

Expand this to a richer SDXL caption (up to 225 tokens) for character '{character_name}' from Pixar's "Turning Red".

Requirements:
1. Keep the core trigger word and character identity
2. Add more descriptive details about:
   - Facial features (eyes, hair, expression details)
   - Clothing/outfit specifics
   - Pose and body language
   - Lighting and atmosphere
   - Art style details (Pixar 3D animation, smooth shading, etc.)
3. Maintain the same overall scene/context
4. Use natural, flowing language
5. Stay under 225 tokens

Return ONLY the expanded caption, no explanations."""

    try:
        message = client.messages.create(
            model="llm_provider-3-5-haiku-20241022",
            max_tokens=300,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        expanded = message.content[0].text.strip()
        return expanded

    except Exception as e:
        logger.error(f"API error: {e}")
        return original_caption  # Fallback to original


def get_original_images():
    """Get list of original (non-augmented) images"""
    all_images = list(SD15_DIR.glob("*.png"))
    original_images = [img for img in all_images if "_aug" not in img.stem]
    return sorted(original_images)


def get_augmented_versions(original_image: Path):
    """Get all augmented versions of an original image"""
    base_stem = original_image.stem
    aug_pattern = f"{base_stem}_aug*.png"
    return sorted(SD15_DIR.glob(aug_pattern))


def main():
    logger.info("=" * 80)
    logger.info("TYLER SDXL CAPTION GENERATION (WITH AUGMENTED COPY)")
    logger.info("=" * 80)

    # Create output directory
    SDXL_DIR.mkdir(parents=True, exist_ok=True)

    # Get original images
    original_images = get_original_images()
    logger.info(f"Found {len(original_images)} original images")

    # Count augmented images
    total_augmented = 0
    for orig_img in original_images:
        aug_versions = get_augmented_versions(orig_img)
        total_augmented += len(aug_versions)

    logger.info(f"Found {total_augmented} augmented images")
    logger.info(f"Total images: {len(original_images) + total_augmented}")
    logger.info(f"API calls needed: {len(original_images)} (original only)")
    logger.info(f"Estimated cost: ~${len(original_images) * 0.02:.2f}")
    logger.info("")

    # Process each original image
    processed_count = 0
    copied_count = 0

    for orig_img in tqdm(original_images, desc="Processing originals"):
        orig_caption_path = SD15_DIR / f"{orig_img.stem}.txt"
        sdxl_caption_path = SDXL_DIR / f"{orig_img.stem}.txt"

        # Skip if SDXL caption already exists
        if sdxl_caption_path.exists():
            logger.debug(f"SDXL caption exists: {orig_img.stem}")
        else:
            # Read SD1.5 caption
            if not orig_caption_path.exists():
                logger.warning(f"SD1.5 caption not found: {orig_caption_path}")
                continue

            sd15_caption = orig_caption_path.read_text().strip()

            # Expand to SDXL
            logger.info(f"Expanding: {orig_img.stem}")
            sdxl_caption = expand_caption_to_sdxl(sd15_caption, "Tyler")

            # Save SDXL caption
            sdxl_caption_path.write_text(sdxl_caption)
            processed_count += 1

            # Rate limiting
            time.sleep(0.5)

        # Copy SDXL caption to all augmented versions
        aug_versions = get_augmented_versions(orig_img)
        for aug_img in aug_versions:
            aug_caption_src = sdxl_caption_path
            aug_caption_dst = SDXL_DIR / f"{aug_img.stem}.txt"

            if not aug_caption_dst.exists():
                shutil.copy2(aug_caption_src, aug_caption_dst)
                copied_count += 1

    # Copy all images (original + augmented) to SDXL directory
    logger.info("Copying images to SDXL directory...")
    for img in SD15_DIR.glob("*.png"):
        dst = SDXL_DIR / img.name
        if not dst.exists():
            shutil.copy2(img, dst)

    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Original captions generated: {processed_count}")
    logger.info(f"Augmented captions copied: {copied_count}")
    logger.info(f"Total SDXL captions: {processed_count + copied_count}")
    logger.info(f"Output directory: {SDXL_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
