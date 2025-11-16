#!/usr/bin/env python3
"""
Apply Captions from AI Analysis JSON
Simple script to generate .txt caption files from AI analysis results
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

def apply_captions(analysis_json_path: Path, image_dir: Path, trigger_word: str = ""):
    """
    Create .txt caption files from AI analysis JSON

    Args:
        analysis_json_path: Path to ai_deep_analysis.json
        image_dir: Directory containing images
        trigger_word: Optional trigger word to prepend to captions
    """
    print(f"Loading AI analysis from {analysis_json_path}")
    with open(analysis_json_path, 'r') as f:
        analysis_data = json.load(f)

    print(f"Processing {len(analysis_data)} entries...")

    # Build mapping from frame_name to caption
    caption_map = {}
    for item in analysis_data:
        frame_name = item['frame_name']

        # Combine character and background captions
        char_caption = item.get('character_caption', '')
        bg_caption = item.get('background_caption', '')

        # Format caption
        if char_caption and bg_caption:
            caption = f"{char_caption}, background: {bg_caption}"
        elif char_caption:
            caption = char_caption
        elif bg_caption:
            caption = bg_caption
        else:
            caption = "anime character"

        # Add trigger word if provided
        if trigger_word:
            caption = f"{trigger_word}, {caption}"

        caption_map[frame_name] = caption

    # Find all images in directory
    image_files = list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpg"))

    print(f"Found {len(image_files)} images in {image_dir}")

    # Create caption files
    created = 0
    skipped = 0

    for img_path in tqdm(image_files, desc="Creating captions"):
        frame_name = img_path.name

        if frame_name in caption_map:
            caption = caption_map[frame_name]

            # Write caption file
            caption_file = img_path.with_suffix('.txt')
            caption_file.write_text(caption, encoding='utf-8')

            created += 1
        else:
            skipped += 1
            # Create generic caption for images not in analysis
            caption = f"{trigger_word}, anime character" if trigger_word else "anime character"
            caption_file = img_path.with_suffix('.txt')
            caption_file.write_text(caption, encoding='utf-8')

    print(f"\n{'='*60}")
    print(f"✅ Caption generation complete!")
    print(f"{'='*60}")
    print(f"Created: {created} captions from AI analysis")
    print(f"Generic: {skipped} captions for images not in analysis")
    print(f"Total: {created + skipped} caption files")
    print(f"Output: {image_dir}")

    # Show samples
    print(f"\n{'='*60}")
    print("Sample Captions:")
    print(f"{'='*60}")
    for img_path in list(image_files)[:3]:
        caption_file = img_path.with_suffix('.txt')
        if caption_file.exists():
            caption = caption_file.read_text(encoding='utf-8')
            print(f"{img_path.name}:")
            print(f"  {caption}\n")


def main():
    parser = argparse.ArgumentParser(description="Apply Captions from AI Analysis")
    parser.add_argument('analysis_json', type=Path, help='Path to ai_deep_analysis.json')
    parser.add_argument('image_dir', type=Path, help='Directory containing images')
    parser.add_argument('--trigger-word', type=str, default='', help='Trigger word to prepend')

    args = parser.parse_args()

    if not args.analysis_json.exists():
        print(f"Error: Analysis JSON not found: {args.analysis_json}")
        return 1

    if not args.image_dir.exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        return 1

    apply_captions(args.analysis_json, args.image_dir, args.trigger_word)
    return 0


if __name__ == '__main__':
    exit(main())
