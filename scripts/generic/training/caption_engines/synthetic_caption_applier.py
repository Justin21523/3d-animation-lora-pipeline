#!/usr/bin/env python3
"""
Synthetic Caption Applier

Applies generated caption templates to all synthetic images deterministically.
This is the second step in the hybrid caption generation strategy.

Process:
1. Load templates and vocabularies from template generator
2. Match images to original prompts.json (by round/img number)
3. Extract pose/action/expression from original prompt metadata
4. Select appropriate template and fill with vocabulary
5. Remove ALL character identity information (critical)
6. Generate .txt caption file for each image
7. Validate token length (100-225 tokens, SDXL optimized)

Output:
- One .txt file per .png image (matching filenames)
- Caption statistics and validation report

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticCaptionApplier:
    """
    Apply caption templates to all synthetic images.

    Strategy:
    1. Load templates and vocabulary from JSON
    2. Match image filename to original prompt
    3. Fill template with vocabulary based on type
    4. Remove identity information
    5. Validate and save caption
    """

    # Critical: Regex patterns to remove character identity
    IDENTITY_REMOVAL_PATTERNS = [
        # Character names (case-insensitive)
        r'\b(luca|alberto|giulia|miguel|bryce|caleb|elio|orion|russell|tyler|ian|barley)\b',

        # Film references
        r'\([^)]*from[^)]*Pixar[^)]*\)',
        r'\([^)]*from[^)]*Disney[^)]*\)',
        r'\([^)]*from[^)]*DreamWorks[^)]*\)',

        # Demographic descriptors
        r'\b(Italian|Mexican|American|European|Asian|African)\s+(boy|girl|kid|child|teen|teenager|youth)\b',
        r'\b(boy|girl|kid|child|teen|teenager|youth)\b',
        r'\b(male|female|man|woman|person|guy|lady)\b',

        # Age descriptors
        r'\b(young|old|elderly|mature|adult|adolescent|juvenile)\s+(boy|girl|person|character)\b',

        # Physical appearance
        r'\b(wavy|curly|spiky|straight|messy|neat|short|long|brown|black|blonde|red|auburn|dark|light)\s+(hair)\b',
        r'\b(blue|brown|green|hazel|dark|bright|deep)\s+(eyes)\b',
        r'\b(tanned|pale|dark|light|fair|olive)\s+(skin)\b',
        r'\b(freckles|dimples|moles|birthmarks|scars|wrinkles)\b',

        # Species-specific (for fantasy characters)
        r'\b(elf|elven|fairy|dwarf|giant|monster|creature|beast)\b',
        r'\b(pointed|long|short|furry|scaly|feathered)\s+(ears|tail|wings)\b',
        r'\bsea\s+monster\s+(form|transformation|appearance)\b',

        # Names in possessive form
        r"\b(luca|alberto|giulia|miguel|bryce|caleb|elio|orion|russell|tyler|ian|barley)'s\b",
    ]

    # Replacement patterns for cleaning
    CLEANING_PATTERNS = [
        (r'\s+,', ','),  # Remove space before comma
        (r',\s*,', ','),  # Remove double commas
        (r'\s{2,}', ' '),  # Collapse multiple spaces
        (r'^\s*,\s*', ''),  # Remove leading comma
        (r',\s*$', ''),  # Remove trailing comma
    ]

    def __init__(
        self,
        vocabulary_dir: Path,
        min_tokens: int = 100,
        max_tokens: int = 225,
        target_tokens: int = 170
    ):
        """
        Initialize caption applier.

        Args:
            vocabulary_dir: Directory containing templates and vocabularies
            min_tokens: Minimum acceptable caption length
            max_tokens: Maximum caption length (SDXL limit)
            target_tokens: Target caption length (sweet spot)
        """
        self.vocabulary_dir = vocabulary_dir
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens

        # Storage
        self.templates = {}
        self.vocabularies = {}
        self.exemplars = {}

        # Statistics
        self.stats = defaultdict(int)

        # Load templates and vocabularies
        self._load_resources()

        logger.info(f"✓ Caption Applier initialized")
        logger.info(f"  Token range: {min_tokens}-{max_tokens} (target: {target_tokens})")

    def _load_resources(self):
        """Load templates, vocabularies, and exemplars."""
        logger.info("Loading caption resources...")

        lora_types = ['pose', 'action', 'expression']

        for lora_type in lora_types:
            # Load vocabularies
            vocab_file = self.vocabulary_dir / f"caption_vocabulary_{lora_type}.json"
            if vocab_file.exists():
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    self.vocabularies[lora_type] = json.load(f)
                logger.info(f"  ✓ Loaded {lora_type} vocabulary")

            # Load exemplars (use as templates if templates not available)
            exemplar_file = self.vocabulary_dir / f"exemplar_captions_{lora_type}.json"
            if exemplar_file.exists():
                with open(exemplar_file, 'r', encoding='utf-8') as f:
                    self.exemplars[lora_type] = json.load(f)
                logger.info(f"  ✓ Loaded {len(self.exemplars[lora_type])} exemplar captions for {lora_type}")

        # Use exemplars as templates (pick diverse examples)
        for lora_type in lora_types:
            if lora_type in self.exemplars:
                self.templates[lora_type] = [ex['caption'] for ex in self.exemplars[lora_type]]

        logger.info("✓ Resources loaded successfully")

    def _parse_filename(self, filename: str) -> Optional[Tuple[int, int]]:
        """
        Parse filename to extract round and image number.

        Args:
            filename: Image filename (e.g., 'pose_round001_img005.png')

        Returns:
            (round_num, img_num) or None if failed
        """
        # Pattern: {type}_round{RRR}_img{III}.png
        match = re.match(r'(?:pose|action|expression)_round(\d+)_img(\d+)\.png', filename)

        if match:
            round_num = int(match.group(1))
            img_num = int(match.group(2))
            return (round_num, img_num)

        return None

    def _load_original_prompts(
        self,
        character: str,
        lora_type: str,
        data_root: Path
    ) -> Optional[Dict]:
        """
        Load original prompts.json for a character/type.

        Args:
            character: Character name
            lora_type: LoRA type
            data_root: Root of synthetic data

        Returns:
            Prompts dictionary or None
        """
        prompts_file = data_root / character / lora_type / "prompts_converted.json"

        # Try converted first, fallback to original
        if not prompts_file.exists():
            prompts_file = data_root / character / lora_type / "prompts.json"

        if not prompts_file.exists():
            logger.warning(f"Prompts file not found for {character} {lora_type}")
            return None

        with open(prompts_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _remove_identity_info(self, caption: str) -> str:
        """
        Remove ALL character identity information from caption.

        Args:
            caption: Original caption

        Returns:
            Cleaned caption with identity removed
        """
        cleaned = caption

        # Apply all identity removal patterns
        for pattern in self.IDENTITY_REMOVAL_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Apply cleaning patterns
        for pattern, replacement in self.CLEANING_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned)

        return cleaned.strip()

    def _validate_caption(self, caption: str) -> Tuple[bool, Optional[str]]:
        """
        Validate caption meets requirements.

        Args:
            caption: Caption to validate

        Returns:
            (is_valid, error_message)
        """
        # Check for identity leakage
        identity_check = caption.lower()
        for pattern in self.IDENTITY_REMOVAL_PATTERNS[:11]:  # Check first few critical patterns
            if re.search(pattern, identity_check):
                return False, f"Identity leakage detected: {pattern}"

        # Check token count
        token_count = len(caption.split())

        if token_count < self.min_tokens:
            return False, f"Too short: {token_count} tokens (min {self.min_tokens})"

        if token_count > self.max_tokens:
            return False, f"Too long: {token_count} tokens (max {self.max_tokens})"

        # Check for error patterns
        error_patterns = [
            'error', 'failed', 'none', 'null', 'undefined',
            '\\n', '```', '[', ']', '{', '}'
        ]

        for error_pattern in error_patterns:
            if error_pattern in caption.lower():
                return False, f"Contains error pattern: {error_pattern}"

        # Check minimum descriptive content
        if caption.count(',') < 3:
            return False, "Too simple (fewer than 3 descriptive clauses)"

        return True, None

    def _generate_caption_from_template(
        self,
        lora_type: str,
        original_prompt: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate caption by selecting and filling a template.

        Args:
            lora_type: LoRA type (pose/action/expression)
            original_prompt: Original prompt from generation
            metadata: Metadata from original prompt

        Returns:
            Generated caption
        """
        # Select a template randomly
        if lora_type in self.templates and self.templates[lora_type]:
            base_caption = random.choice(self.templates[lora_type])
        else:
            # Fallback to generic template if no exemplars
            base_caption = f"a 3d animated character, high quality, detailed, professional rendering"

        # Remove any identity info that might be in exemplar
        caption = self._remove_identity_info(base_caption)

        # If too short, enhance with vocabulary
        if len(caption.split()) < self.target_tokens:
            caption = self._enhance_caption(caption, lora_type)

        # Truncate if too long
        words = caption.split()
        if len(words) > self.max_tokens:
            caption = ' '.join(words[:self.max_tokens])

        return caption

    def _enhance_caption(self, caption: str, lora_type: str) -> str:
        """
        Enhance caption with additional vocabulary if too short.

        Args:
            caption: Base caption
            lora_type: LoRA type

        Returns:
            Enhanced caption
        """
        if lora_type not in self.vocabularies:
            return caption

        vocab = self.vocabularies[lora_type]
        additions = []

        # Add type-specific terms
        if lora_type == 'pose' and vocab.get('camera_terms'):
            additions.append(random.choice(vocab['camera_terms']))

        if lora_type == 'action' and vocab.get('action_terms'):
            additions.append(random.choice(vocab['action_terms']))

        if lora_type == 'expression' and vocab.get('expression_terms'):
            additions.append(random.choice(vocab['expression_terms']))

        # Add lighting terms (universal)
        if vocab.get('lighting_terms'):
            additions.append(random.choice(vocab['lighting_terms']))

        # Add quality terms (universal)
        if vocab.get('quality_terms'):
            additions.append(random.choice(vocab['quality_terms']))

        # Combine
        if additions:
            enhanced = f"{caption}, {', '.join(additions)}"
        else:
            enhanced = caption

        return enhanced

    def process_character_type(
        self,
        character: str,
        lora_type: str,
        data_root: Path,
        output_dir: Path,
        force: bool = False
    ) -> Dict:
        """
        Process all images for a character/type combination.

        Args:
            character: Character name
            lora_type: LoRA type
            data_root: Root of synthetic data
            output_dir: Output directory for captions
            force: Overwrite existing captions

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_images': 0,
            'captions_generated': 0,
            'captions_skipped': 0,
            'validation_failed': 0,
            'errors': []
        }

        # Get image directory
        image_dir = data_root / character / lora_type / "generated"

        if not image_dir.exists():
            logger.warning(f"Image directory not found: {image_dir}")
            return stats

        # Load original prompts
        original_prompts = self._load_original_prompts(character, lora_type, data_root)

        # Get all images
        images = sorted(image_dir.glob("*.png"))
        stats['total_images'] = len(images)

        logger.info(f"Processing {len(images)} images for {character} {lora_type}")

        # Create output directory
        output_subdir = output_dir / character / lora_type
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Process each image
        for img_path in tqdm(images, desc=f"{character} {lora_type}"):
            # Check if caption already exists
            caption_path = output_subdir / f"{img_path.stem}.txt"

            if caption_path.exists() and not force:
                stats['captions_skipped'] += 1
                continue

            try:
                # Generate caption
                caption = self._generate_caption_from_template(
                    lora_type=lora_type,
                    original_prompt=None,  # Not using original for now
                    metadata=None
                )

                # Validate caption
                is_valid, error_msg = self._validate_caption(caption)

                if not is_valid:
                    logger.warning(f"Validation failed for {img_path.name}: {error_msg}")
                    stats['validation_failed'] += 1
                    stats['errors'].append({
                        'image': str(img_path.name),
                        'error': error_msg
                    })
                    continue

                # Save caption
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)

                stats['captions_generated'] += 1

            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
                stats['errors'].append({
                    'image': str(img_path.name),
                    'error': str(e)
                })

        return stats

    def process_all_characters(
        self,
        characters: List[str],
        lora_types: List[str],
        data_root: Path,
        output_dir: Path,
        force: bool = False
    ):
        """
        Process all characters and LoRA types.

        Args:
            characters: List of character names
            lora_types: List of LoRA types
            data_root: Root of synthetic data
            output_dir: Output directory
            force: Overwrite existing captions
        """
        logger.info(f"\n{'='*60}")
        logger.info("CAPTION APPLICATION - ALL CHARACTERS")
        logger.info(f"{'='*60}\n")

        all_stats = defaultdict(lambda: defaultdict(dict))

        for lora_type in lora_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {lora_type.upper()}")
            logger.info(f"{'='*60}\n")

            for character in characters:
                stats = self.process_character_type(
                    character=character,
                    lora_type=lora_type,
                    data_root=data_root,
                    output_dir=output_dir,
                    force=force
                )

                all_stats[lora_type][character] = stats

                logger.info(
                    f"  ✓ {character}: "
                    f"{stats['captions_generated']}/{stats['total_images']} captions "
                    f"(skipped: {stats['captions_skipped']}, failed: {stats['validation_failed']})"
                )

        # Save overall statistics
        self._save_statistics(output_dir, all_stats)

        # Print summary
        self._print_summary(all_stats)

    def _save_statistics(self, output_dir: Path, stats: Dict):
        """Save caption generation statistics."""
        stats_file = output_dir / "caption_application_statistics.json"

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✓ Statistics saved to {stats_file}")

    def _print_summary(self, stats: Dict):
        """Print caption generation summary."""
        logger.info(f"\n{'='*60}")
        logger.info("CAPTION APPLICATION SUMMARY")
        logger.info(f"{'='*60}\n")

        for lora_type, char_stats in stats.items():
            total_images = sum(s['total_images'] for s in char_stats.values())
            total_generated = sum(s['captions_generated'] for s in char_stats.values())
            total_skipped = sum(s['captions_skipped'] for s in char_stats.values())
            total_failed = sum(s['validation_failed'] for s in char_stats.values())

            logger.info(f"{lora_type.upper()}:")
            logger.info(f"  Total images: {total_images}")
            logger.info(f"  Captions generated: {total_generated}")
            logger.info(f"  Skipped (existing): {total_skipped}")
            logger.info(f"  Validation failed: {total_failed}")
            logger.info(f"  Success rate: {total_generated/total_images*100:.1f}%\n")

        logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Apply caption templates to all synthetic images"
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default='/mnt/data/ai_data/synthetic_lora_data/generated_data',
        help='Root directory of synthetic generated data'
    )
    parser.add_argument(
        '--vocabulary-dir',
        type=Path,
        default='/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training',
        help='Directory containing templates and vocabularies'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='/mnt/data/ai_data/synthetic_lora_data/captioned_data',
        help='Output directory for generated captions'
    )
    parser.add_argument(
        '--characters',
        nargs='+',
        default=['alberto', 'bryce', 'caleb', 'elio', 'giulia', 'ian_lightfoot',
                 'luca', 'miguel', 'orion', 'russell', 'tyler',
                 'alberto_seamonster', 'luca_seamonster', 'barley_lightfoot'],
        help='Character names to process'
    )
    parser.add_argument(
        '--lora-types',
        nargs='+',
        default=['pose', 'action', 'expression'],
        help='LoRA types to process'
    )
    parser.add_argument(
        '--min-tokens',
        type=int,
        default=100,
        help='Minimum caption token count'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=225,
        help='Maximum caption token count (SDXL limit)'
    )
    parser.add_argument(
        '--target-tokens',
        type=int,
        default=170,
        help='Target caption token count (sweet spot)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing captions'
    )

    args = parser.parse_args()

    # Initialize applier
    applier = SyntheticCaptionApplier(
        vocabulary_dir=args.vocabulary_dir,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        target_tokens=args.target_tokens
    )

    # Process all characters
    applier.process_all_characters(
        characters=args.characters,
        lora_types=args.lora_types,
        data_root=args.data_root,
        output_dir=args.output_dir,
        force=args.force
    )

    logger.info("✓ Caption application complete!")


if __name__ == '__main__':
    main()
