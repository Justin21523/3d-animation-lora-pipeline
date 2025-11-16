"""
Prepare training data for LoRA training with character information integration
Generates captions and organizes in kohya_ss format with 3D animation support
"""

import sys
import shutil
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from core.utils.character_loader import load_character, get_character_tags
    CHARACTER_LOADER_AVAILABLE = True
except ImportError:
    CHARACTER_LOADER_AVAILABLE = False
    print("Warning: character_loader not available, using fallback mode")


def generate_character_caption_prefix(character_name: str, form: str = "human") -> str:
    """
    Generate caption prefix from character information

    Args:
        character_name: Character identifier (e.g., 'luca', 'alberto')
        form: Character form ('human' or 'sea_monster')

    Returns:
        Caption prefix string with character-specific tags
    """
    if not CHARACTER_LOADER_AVAILABLE:
        return f"{character_name}, "

    try:
        char_info = load_character(character_name)

        # Build prefix from character info
        parts = [char_info.name.lower()]

        # Add form-specific appearance tags
        if form == "human" and char_info.human_form:
            app = char_info.human_form.appearance
            if app.hair:
                parts.append(app.hair)
            if app.eyes:
                parts.append(app.eyes)
            # Add clothing
            parts.extend(app.clothing[:2])  # Limit to 2 clothing items
        elif form == "sea_monster" and char_info.sea_monster_form:
            app = char_info.sea_monster_form.appearance
            if app.skin:
                parts.append(app.skin)
            if app.eyes:
                parts.append(app.eyes)
            # Add distinctive features
            parts.extend(app.distinctive_features[:2])

        # Join with commas
        return ", ".join(parts) + ", "

    except Exception as e:
        print(f"Warning: Could not load character info for {character_name}: {e}")
        return f"{character_name}, "


def generate_caption_with_character_context(
    image_path: Path,
    character_name: str,
    form: str = "human",
    style: str = "3d_animation",
    additional_tags: Optional[List[str]] = None
) -> str:
    """
    Generate caption with character context

    Args:
        image_path: Path to image
        character_name: Character identifier
        form: Character form
        style: Caption style
        additional_tags: Additional tags to include

    Returns:
        Generated caption string
    """
    caption_parts = []

    # Start with character prefix
    prefix = generate_character_caption_prefix(character_name, form)
    caption_parts.append(prefix)

    # Add style-specific tags
    if style == "3d_animation":
        caption_parts.append("pixar style 3d animation")
        caption_parts.append("smooth shading")
        caption_parts.append("cinematic lighting")

    # Add additional tags if provided
    if additional_tags:
        caption_parts.extend(additional_tags)

    return ", ".join(caption_parts)


def prepare_training_data(
    character_dirs: List[Path],
    output_dir: Path,
    character_name: str,
    generate_captions: bool = True,
    caption_prefix: Optional[str] = None,
    caption_model: str = "blip2",
    augment: bool = False,
    target_size: Optional[int] = None,
    repeat_count: int = 10,
    use_character_info: bool = True,
    character_form: str = "human"
):
    """
    Prepare training data from clustered character images

    Args:
        character_dirs: List of directories containing character images
        output_dir: Output directory for training data
        character_name: Character name for captions
        generate_captions: Whether to generate captions automatically
        caption_prefix: Optional caption prefix (overrides auto-generated)
        caption_model: Caption model to use ('blip2', 'qwen2_vl', etc.)
        augment: Whether to apply data augmentation
        target_size: Target dataset size (will sample if provided)
        repeat_count: Repeat count for kohya_ss training format
        use_character_info: Use character_loader for enhanced captions
        character_form: Character form for caption generation
    """
    print("=" * 80)
    print(f"Preparing Training Data: {character_name}")
    print("=" * 80)

    # Create output structure
    images_dir = output_dir / "images"
    captions_dir = output_dir / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images
    all_images = []
    for char_dir in character_dirs:
        if not char_dir.exists():
            print(f"Warning: Directory not found: {char_dir}")
            continue

        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            all_images.extend(char_dir.glob(f'*{ext}'))
            all_images.extend(char_dir.glob(f'*{ext.upper()}'))

    print(f"✓ Found {len(all_images)} images from {len(character_dirs)} directories")

    # Sample if target size specified
    if target_size and len(all_images) > target_size:
        import random
        random.seed(42)
        all_images = random.sample(all_images, target_size)
        print(f"✓ Sampled {target_size} images")

    # Generate caption prefix
    if caption_prefix is None and use_character_info and CHARACTER_LOADER_AVAILABLE:
        caption_prefix = generate_character_caption_prefix(character_name, character_form)
        print(f"✓ Auto-generated caption prefix: {caption_prefix}")
    elif caption_prefix is None:
        caption_prefix = f"{character_name}, "

    # Copy images and generate/copy captions
    print("\nProcessing images...")
    copied_count = 0

    for img_path in all_images:
        # Copy image
        dst_img = images_dir / f"{character_name}_{copied_count:04d}{img_path.suffix}"
        shutil.copy2(img_path, dst_img)

        # Handle caption
        caption_file = img_path.with_suffix('.txt')
        dst_caption = captions_dir / f"{character_name}_{copied_count:04d}.txt"

        if caption_file.exists() and not generate_captions:
            # Use existing caption
            shutil.copy2(caption_file, dst_caption)
        else:
            # Generate caption
            if generate_captions:
                if caption_model == "blip2":
                    # Placeholder for actual BLIP2 generation
                    caption = f"{caption_prefix}high quality image, detailed"
                else:
                    # For other models, use character context
                    caption = generate_caption_with_character_context(
                        img_path, character_name, character_form
                    )

                dst_caption.write_text(caption, encoding='utf-8')
            else:
                # Create minimal caption
                dst_caption.write_text(caption_prefix.strip(), encoding='utf-8')

        copied_count += 1

        if copied_count % 50 == 0:
            print(f"  Processed {copied_count}/{len(all_images)} images...")

    print(f"✓ Processed {copied_count} images")

    # Organize for kohya_ss format
    print("\nOrganizing for kohya_ss training...")

    # Format: <repeat>_<class_name>
    training_dir = output_dir.parent / f"{repeat_count}_{character_name}"
    if training_dir.exists():
        shutil.rmtree(training_dir)
    training_dir.mkdir(parents=True)

    # Copy images and captions to training directory
    for img_file in images_dir.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
            shutil.copy2(img_file, training_dir / img_file.name)

            # Copy corresponding caption
            caption_file = captions_dir / img_file.with_suffix('.txt').name
            if caption_file.exists():
                shutil.copy2(caption_file, training_dir / caption_file.name)

    # Create metadata
    metadata = {
        "character_name": character_name,
        "character_form": character_form,
        "total_images": copied_count,
        "repeat_count": repeat_count,
        "effective_size": copied_count * repeat_count,
        "caption_prefix": caption_prefix,
        "caption_model": caption_model if generate_captions else "manual",
        "source_directories": [str(d) for d in character_dirs],
        "output_directory": str(training_dir),
        "used_character_loader": use_character_info and CHARACTER_LOADER_AVAILABLE
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 80)
    print("Training Data Ready!")
    print("=" * 80)
    print(f"Location: {training_dir}")
    print(f"Images: {copied_count}")
    print(f"Repeats per epoch: {repeat_count}")
    print(f"Effective dataset size: {copied_count * repeat_count} per epoch")
    print(f"Caption prefix: {caption_prefix}")
    print(f"Metadata: {metadata_file}")
    print("=" * 80)

    # Character info summary
    if use_character_info and CHARACTER_LOADER_AVAILABLE:
        try:
            char_info = load_character(character_name)
            print(f"\n✓ Character Info Loaded:")
            print(f"  Name: {char_info.full_name}")
            print(f"  Age: {char_info.age}")
            print(f"  Personality: {', '.join(char_info.personality_traits[:5])}")
            if char_info.signature_phrases:
                print(f"  Signature phrases: {', '.join(char_info.signature_phrases[:3])}")
        except:
            pass

    return metadata


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Prepare training data for LoRA training with character info support"
    )
    parser.add_argument(
        "--character-dirs",
        type=str,
        nargs='+',
        required=True,
        help="Directories containing character images (clustered output)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for training data"
    )
    parser.add_argument(
        "--character-name",
        type=str,
        required=True,
        help="Character name (e.g., 'luca', 'alberto')"
    )
    parser.add_argument(
        "--generate-captions",
        action="store_true",
        help="Generate captions automatically"
    )
    parser.add_argument(
        "--caption-model",
        type=str,
        default="blip2",
        choices=["blip2", "qwen2_vl", "internvl2", "manual"],
        help="Caption model to use (default: blip2)"
    )
    parser.add_argument(
        "--caption-prefix",
        type=str,
        help="Custom caption prefix (overrides auto-generated)"
    )
    parser.add_argument(
        "--character-form",
        type=str,
        default="human",
        choices=["human", "sea_monster"],
        help="Character form for Luca characters (default: human)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        help="Target dataset size (will sample if exceeded)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Repeat count for kohya_ss format (default: 10)"
    )
    parser.add_argument(
        "--no-character-info",
        action="store_true",
        help="Disable character_loader integration"
    )

    args = parser.parse_args()

    # Convert paths
    character_dirs = [Path(d) for d in args.character_dirs]
    output_dir = Path(args.output_dir)

    # Prepare training data
    prepare_training_data(
        character_dirs=character_dirs,
        output_dir=output_dir,
        character_name=args.character_name,
        generate_captions=args.generate_captions,
        caption_prefix=args.caption_prefix,
        caption_model=args.caption_model,
        augment=args.augment,
        target_size=args.target_size,
        repeat_count=args.repeat,
        use_character_info=not args.no_character_info,
        character_form=args.character_form
    )


if __name__ == "__main__":
    main()
