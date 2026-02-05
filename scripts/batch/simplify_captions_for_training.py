#!/usr/bin/env python3
"""
Simplify overly technical captions for LoRA training.

This script removes technical rendering terms and simplifies captions to focus on:
- Character appearance (hair, eyes, skin, clothing)
- Pose and view angle (frontal, three-quarter, profile)
- Expression and emotion
- Basic lighting description
- Pixar/3D animation style markers

Problem: Overly technical captions with terms like "8K", "PBR", "subsurface scattering",
"photorealistic", "hyper-realistic" cause models to learn incorrect texture representations,
resulting in watercolor-like blurring and color bleeding.

Solution: Simplify captions to 30-50 tokens focusing on visual content, not rendering tech.
"""

import re
import argparse
from pathlib import Path
import shutil
from typing import List, Tuple
import sys

# Technical terms to remove (these cause watercolor artifacts)
TECHNICAL_TERMS = [
    # Resolution/quality terms
    r'\b8k\b', r'\b4k\b', r'\bhigh-resolution\b', r'\bhigh resolution\b',
    r'\bphotorealistic\b', r'\bhyper-realistic\b', r'\bhyperrealistic\b',
    r'\baward-winning\b', r'\bprofessional-grade\b',

    # Rendering technique terms
    r'\bphysically-based rendering\b', r'\bPBR\b', r'\bpbr\b',
    r'\bsubsurface scattering\b', r'\bSSS\b',
    r'\badvanced rendering techniques\b', r'\brendering techniques\b',
    r'\bambient occlusion\b', r'\bAO\b',
    r'\bglobal illumination\b', r'\bray tracing\b', r'\bpath tracing\b',

    # Material/shader terms
    r'\bskin shader\b', r'\bdetailed skin shader\b',
    r'\badvanced materials\b', r'\bPBR materials\b',
    r'\bintricate\b', r'\bmeticulous\b', r'\bmeticulously\b',
    r'\bhyper-detailed\b', r'\bhyperdetailed\b',

    # Excessive quality descriptors
    r'\bexceptional detail\b', r'\bexceptional visual fidelity\b',
    r'\bunprecedented realism\b', r'\bstunning\b',
    r'\bnuanced\b', r'\bminute details\b',

    # Overly technical lighting terms (keep basic lighting terms)
    r'\bthree-point lighting setup\b', r'\bthree-point studio lighting\b',
    r'\bsophisticated.*?lighting\b', r'\bprofessional studio lighting setup\b',

    # Composition terms that are too technical
    r'\bcinematographic composition\b', r'\bprofessional.*?composition\b',

    # Render/quality descriptors
    r'\brender\b', r'\brendered\b', r'\brendering\b',
    r'\bat \d+px\b', r'\bat \d+k resolution\b',

    # Filler words that add no value
    r'\bresulting in\b', r'\bshowcasing\b', r'\bcapturing\b',
    r'\bcreating\b', r'\benhancing\b', r'\bemphasizing\b',
    r'\breveal\b', r'\brevealing\b', r'\bdrawing focus to\b',
]

# Verbose phrases to simplify
VERBOSE_PHRASES = {
    # Lighting descriptions
    r'soft, directional three-point illumination': 'studio lighting',
    r'soft, directional key light': 'soft key light',
    r'gentle fill light reducing shadows': 'fill light',
    r'subtle rim light creating depth': 'rim light',
    r'warm, natural lighting': 'warm lighting',

    # Background descriptions
    r'a softly blurred.*?background': 'blurred background',
    r'provides context while maintaining.*?focus': '',
    r'evoking the seaside ambiance': '',

    # Character descriptions
    r'a (young|teenage) (Italian )?boy': 'a teenage boy',
    r'wild, (tousled )?curly (golden-)?brown hair': 'curly brown hair',
    r'vibrant emerald green eyes': 'green eyes',
    r'sun-kissed tanned skin': 'tan skin',
    r'lean, (athletic|muscular) physique': 'lean build',

    # View/pose descriptions
    r'captured in a.*?(three-quarter|close-up|medium)': 'in a $1',
    r'positioned in a': 'in a',
    r'(alberto )?stands in a': 'standing,',
    r'(alberto )?leans forward with': 'leaning forward,',

    # Style markers
    r'pixar-inspired animation style': 'pixar style',
    r'pixar-style 3d animation': 'pixar style, 3d animated character',
    r'reminiscent of pixar': 'pixar style',
}

# Core elements to preserve/prioritize
CORE_ELEMENTS = {
    'character_name': r'\b(alberto|luca|giulia)\b',
    'appearance': r'\b(hair|eyes|skin|face|body|physique|build)\b',
    'clothing': r'\b(tank top|shirt|shorts|trunks|outfit|wearing)\b',
    'view_angle': r'\b(frontal|three-quarter|profile|side|close-up|medium|full body)\b',
    'expression': r'\b(smiling|grin|neutral|thoughtful|excited|mischievous)\b',
    'pose': r'\b(standing|sitting|leaning|walking|raised arms)\b',
    'lighting': r'\b(studio lighting|soft lighting|natural lighting|dramatic lighting|warm light)\b',
    'style': r'\b(3d animated|pixar style|animated character)\b',
}


def extract_core_info(caption: str) -> dict:
    """Extract core visual information from caption."""
    core_info = {}
    caption_lower = caption.lower()

    for category, pattern in CORE_ELEMENTS.items():
        matches = re.findall(pattern, caption_lower, re.IGNORECASE)
        if matches:
            core_info[category] = matches

    return core_info


def simplify_caption(caption: str, character_name: str = "alberto") -> str:
    """
    Simplify a caption by removing technical terms and focusing on core visual elements.

    Args:
        caption: Original caption text
        character_name: Name of the character (e.g., "alberto", "luca")

    Returns:
        Simplified caption
    """
    # Start with original
    simplified = caption

    # Remove technical terms
    for term in TECHNICAL_TERMS:
        simplified = re.sub(term, '', simplified, flags=re.IGNORECASE)

    # Replace verbose phrases
    for verbose, simple in VERBOSE_PHRASES.items():
        simplified = re.sub(verbose, simple, simplified, flags=re.IGNORECASE)

    # Clean up extra spaces, commas, and punctuation
    simplified = re.sub(r'\s+', ' ', simplified)
    simplified = re.sub(r'\s*,\s*,\s*', ', ', simplified)
    simplified = re.sub(r'\s*\.\s*\.\s*', '. ', simplified)
    simplified = re.sub(r',\s*\.\s*', '. ', simplified)
    simplified = re.sub(r'\s+([.,;:])', r'\1', simplified)

    # Remove phrases that start with verbs describing the render process
    simplified = re.sub(r'\.\s*The (image|scene|composition).*?\.', '.', simplified)

    # If caption is still too long (> 150 words), extract core elements
    word_count = len(simplified.split())
    if word_count > 100:
        # Extract key visual descriptors
        core_info = extract_core_info(caption)

        # Build simplified caption from core elements
        parts = [character_name]

        # Add appearance details
        if core_info.get('appearance'):
            # Find hair and eye color in original
            hair_match = re.search(r'(curly|wild|tousled)?\s*(golden-)?brown\s+hair', caption, re.IGNORECASE)
            if hair_match:
                parts.append(hair_match.group(0))

            eye_match = re.search(r'(bright|vibrant|emerald)?\s*green\s+eyes', caption, re.IGNORECASE)
            if eye_match:
                parts.append(eye_match.group(0))

            skin_match = re.search(r'(tan|tanned|sun-kissed)\s+skin', caption, re.IGNORECASE)
            if skin_match:
                parts.append('tan skin')

        # Add clothing
        if core_info.get('clothing'):
            clothing_match = re.search(r'(yellow|white|brown|blue).*?(tank top|shirt|shorts|trunks)', caption, re.IGNORECASE)
            if clothing_match:
                parts.append(clothing_match.group(0))

        # Add style markers
        parts.append('3d animated character')
        parts.append('pixar style')

        # Add view angle
        if core_info.get('view_angle'):
            view = core_info['view_angle'][0]
            parts.append(f'{view} view')

        # Add lighting
        if 'studio lighting' in caption.lower():
            parts.append('studio lighting')
        elif 'natural lighting' in caption.lower():
            parts.append('natural lighting')
        elif 'soft lighting' in caption.lower():
            parts.append('soft lighting')

        # Add expression if present
        if core_info.get('expression'):
            expr = core_info['expression'][0]
            parts.append(f'{expr} expression')

        simplified = ', '.join(parts)

    # Final cleanup
    simplified = simplified.strip()
    simplified = re.sub(r'\s*,\s*$', '', simplified)  # Remove trailing comma
    simplified = re.sub(r'^[,\s]+', '', simplified)  # Remove leading comma/spaces

    # Ensure it starts with character name (lowercase for consistency)
    if not simplified.lower().startswith(character_name.lower()):
        simplified = f"{character_name}, {simplified}"

    return simplified


def process_directory(input_dir: Path, backup: bool = True, character_name: str = "alberto", dry_run: bool = False):
    """
    Process all .txt caption files in a directory.

    Args:
        input_dir: Directory containing caption files
        backup: Whether to backup original captions
        character_name: Name of the character
        dry_run: If True, only show what would be changed without modifying files
    """
    txt_files = list(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"❌ No .txt files found in {input_dir}")
        return

    print(f"📁 Found {len(txt_files)} caption files")

    # Create backup if requested
    if backup and not dry_run:
        backup_dir = input_dir / "_original_captions"
        backup_dir.mkdir(exist_ok=True)
        print(f"💾 Backing up originals to {backup_dir}")

        for txt_file in txt_files:
            shutil.copy2(txt_file, backup_dir / txt_file.name)

    # Process each caption
    processed = 0
    shortened = 0

    for txt_file in txt_files:
        original = txt_file.read_text(encoding='utf-8').strip()
        simplified = simplify_caption(original, character_name)

        original_words = len(original.split())
        simplified_words = len(simplified.split())

        if original != simplified:
            if dry_run:
                print(f"\n📄 {txt_file.name}")
                print(f"  Original ({original_words} words): {original[:100]}...")
                print(f"  Simplified ({simplified_words} words): {simplified}")
            else:
                txt_file.write_text(simplified, encoding='utf-8')
                processed += 1
                if simplified_words < original_words * 0.5:
                    shortened += 1

    if not dry_run:
        print(f"\n✅ Processed {processed}/{len(txt_files)} captions")
        print(f"📉 Significantly shortened: {shortened} captions")
    else:
        print(f"\n🔍 Dry run complete. Run without --dry-run to apply changes.")


def main():
    parser = argparse.ArgumentParser(
        description="Simplify overly technical captions for LoRA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without modifying files
  python simplify_captions_for_training.py /path/to/captions --dry-run

  # Process Alberto's captions with backup
  python simplify_captions_for_training.py \\
    /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/training_data_sdxl/alberto_identity/5_alberto \\
    --character alberto --backup

  # Process without backup
  python simplify_captions_for_training.py /path/to/captions --no-backup
        """
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .txt caption files"
    )

    parser.add_argument(
        "--character",
        type=str,
        default="alberto",
        help="Character name (default: alberto)"
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Backup original captions (default: True)"
    )

    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Don't backup original captions"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"❌ Error: Directory not found: {args.input_dir}")
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"❌ Error: Not a directory: {args.input_dir}")
        sys.exit(1)

    print("🔧 Caption Simplification Tool")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Character name: {args.character}")
    print(f"Backup originals: {args.backup}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)
    print()

    process_directory(
        args.input_dir,
        backup=args.backup,
        character_name=args.character,
        dry_run=args.dry_run
    )

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
