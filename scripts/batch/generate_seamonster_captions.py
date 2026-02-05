#!/usr/bin/env python3
"""
Sea Monster Caption Generation

Generate captions for Alberto and Luca sea monster forms using LLMProvider Haiku.
Specialized for underwater/sea monster characteristics.

Author: LLMProvider Tooling
Date: 2025-11-28
"""

import argparse
import base64
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import llm_vendor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SeaMonsterCaptionGenerator:
    """Generate captions for sea monster characters using LLMProvider Haiku."""

    # Sea monster character descriptions
    SEAMONSTER_DESCRIPTIONS = {
        'alberto_seamonster': {
            'character_name': 'alberto',
            'species': 'sea monster',
            'film': 'luca (pixar)',
            'key_features': [
                'blue-green scaly skin with iridescent sheen',
                'aquatic fins on arms and head',
                'webbed hands',
                'large expressive eyes',
                'playful energetic personality',
                'underwater creature design',
                '3d animated pixar style',
            ],
            'trigger_word': 'alberto_seamonster',
        },
        'luca_seamonster': {
            'character_name': 'luca',
            'species': 'sea monster',
            'film': 'luca (pixar)',
            'key_features': [
                'blue-green scaly skin with smooth texture',
                'aquatic fins on head and body',
                'webbed hands and feet',
                'curious gentle expression',
                'younger more timid appearance',
                'underwater creature design',
                '3d animated pixar style',
            ],
            'trigger_word': 'luca_seamonster',
        },
    }

    def __init__(self, api_key: str = None):
        """
        Initialize sea monster caption generator.

        Args:
            api_key: LLMVendor API key (or from LLM_VENDOR_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('LLM_VENDOR_API_KEY')
        if not self.api_key:
            raise ValueError("LLM_VENDOR_API_KEY not found in environment")

        self.client = llm_vendor.LLMVendor(api_key=self.api_key)

    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, 'rb') as f:
            return base64.standard_b64encode(f.read()).decode('utf-8')

    def generate_caption(
        self,
        image_path: Path,
        character_config: Dict
    ) -> str:
        """
        Generate caption for a single sea monster image.

        Args:
            image_path: Path to image file
            character_config: Character configuration dict

        Returns:
            Generated caption string
        """
        # Encode image
        image_data = self.encode_image(image_path)

        # Determine media type
        suffix = image_path.suffix.lower()
        media_type = "image/png" if suffix == ".png" else "image/jpeg"

        # Build prompt
        char_name = character_config['character_name']
        trigger = character_config['trigger_word']
        features = ', '.join(character_config['key_features'][:5])

        prompt = f"""You are a SDXL LoRA training caption expert. Your task is to generate a training caption for this image.

This image shows {char_name} in SEA MONSTER form from Pixar's "Luca". This IS the sea monster form - do not question this. Sea monsters in this film have humanoid bodies with aquatic features (scales, fins, large eyes, blue-purple skin).

Character: {char_name} (sea monster form - CONFIRMED)
Key features: {features}
Trigger word: {trigger}

Generate a concise training caption (40-60 words) following this EXACT structure:

1. START with trigger word "{trigger}"
2. State species: "sea monster"
3. Describe visible aquatic features: skin color/texture, fins, eyes, teeth
4. Note pose/expression if clear
5. Mention background/setting if visible
6. END with: "pixar style 3d animated character"

Example:
"{trigger}, a sea monster with blue-purple scaly skin, large yellow-green eyes, fin-like crest on head, smiling expression, against blue underwater background, pixar style 3d animated character"

CRITICAL RULES:
- NO questions or refusals - just generate the caption
- Keep under 60 words
- Focus on VISIBLE details only
- Be specific and descriptive
- This IS sea monster form - treat it as such

Generate the caption now (no preamble, just the caption):"""

        try:
            # Call LLMProvider Haiku
            response = self.client.messages.create(
                model="llm_provider-3-5-haiku-20241022",
                max_tokens=150,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )

            # Extract caption
            caption = response.content[0].text.strip()

            # Clean up caption (remove quotes if present)
            caption = caption.strip('"').strip("'").strip()

            return caption

        except Exception as e:
            logger.error(f"Error generating caption for {image_path.name}: {e}")
            # Fallback caption
            return f"{trigger}, a sea monster character, pixar style 3d animated"

    def process_directory(
        self,
        image_dir: Path,
        character_name: str,
        overwrite: bool = False
    ) -> Dict:
        """
        Process all images in directory and generate captions.

        Args:
            image_dir: Directory containing images
            character_name: Character identifier (alberto_seamonster or luca_seamonster)
            overwrite: Whether to overwrite existing captions

        Returns:
            Statistics dict
        """
        if character_name not in self.SEAMONSTER_DESCRIPTIONS:
            raise ValueError(f"Unknown character: {character_name}")

        config = self.SEAMONSTER_DESCRIPTIONS[character_name]

        # Find all images
        image_extensions = {'.png', '.jpg', '.jpeg'}
        images = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {character_name}")
        logger.info(f"{'='*70}")
        logger.info(f"Images found: {len(images)}")
        logger.info(f"Character: {config['character_name']} (sea monster)")
        logger.info(f"Trigger word: {config['trigger_word']}")
        logger.info(f"{'='*70}\n")

        # Process each image
        stats = {
            'total': len(images),
            'processed': 0,
            'skipped': 0,
            'errors': 0
        }

        for img_path in tqdm(images, desc=f"Generating captions for {character_name}"):
            # Caption file path (same name, .txt extension)
            caption_path = img_path.with_suffix('.txt')

            # Skip if caption exists and not overwriting
            if caption_path.exists() and not overwrite:
                stats['skipped'] += 1
                continue

            try:
                # Generate caption
                caption = self.generate_caption(img_path, config)

                # Save caption
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)

                stats['processed'] += 1

            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
                stats['errors'] += 1

        logger.info(f"\n✅ {character_name} Complete:")
        logger.info(f"   Processed: {stats['processed']}")
        logger.info(f"   Skipped: {stats['skipped']}")
        logger.info(f"   Errors: {stats['errors']}\n")

        return stats


def main():
    parser = argparse.ArgumentParser(description="Generate sea monster captions with LLMProvider Haiku")
    parser.add_argument("--image-dirs", nargs="+", required=True,
                        help="Image directories to process")
    parser.add_argument("--character-names", nargs="+", required=True,
                        help="Character names (alberto_seamonster, luca_seamonster)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing captions")
    parser.add_argument("--api-key", type=str, required=False,
                        help="LLMVendor API key (or use LLM_VENDOR_API_KEY env var)")

    args = parser.parse_args()

    if len(args.image_dirs) != len(args.character_names):
        logger.error("ERROR: Number of image dirs must match character names")
        sys.exit(1)

    # Initialize generator
    try:
        generator = SeaMonsterCaptionGenerator(api_key=args.api_key)
    except ValueError as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)

    # Process each character
    all_stats = []

    for img_dir_str, char_name in zip(args.image_dirs, args.character_names):
        img_dir = Path(img_dir_str)

        if not img_dir.exists():
            logger.warning(f"WARNING: {img_dir} does not exist, skipping...")
            continue

        stats = generator.process_directory(
            image_dir=img_dir,
            character_name=char_name,
            overwrite=args.overwrite
        )

        all_stats.append({
            'character': char_name,
            'directory': str(img_dir),
            **stats
        })

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("CAPTION GENERATION COMPLETE")
    logger.info(f"{'='*70}")
    for stats in all_stats:
        logger.info(f"{stats['character']}: {stats['processed']} captions generated")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()
