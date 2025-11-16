#!/usr/bin/env python3
"""
Augment dataset to balance shot types and scenarios
Creates synthetic variations to address imbalances
"""

import argparse
from pathlib import Path
import shutil
import json
from PIL import Image
from typing import List, Dict
import random


def identify_underrepresented_samples(analysis_json: Path) -> Dict[str, List[str]]:
    """Load analysis and identify which samples to augment"""

    with open(analysis_json, 'r') as f:
        data = json.load(f)

    samples = data["caption_analysis"]["samples_by_category"]

    return {
        "full_body": samples.get("full-body", []),
        "medium": samples.get("medium", []),
        "multi_char": samples.get("multi-character", []),
        "occlusion": samples.get("occlusion", []),
    }


def create_caption_variants(original_caption: str, variant_type: str) -> List[str]:
    """Generate caption variations to improve diversity"""

    variants = [original_caption]  # Keep original

    if variant_type == "shot_variation":
        # Add shot type descriptions
        variants.extend([
            f"{original_caption}, medium shot",
            f"{original_caption}, three-quarter view",
            f"{original_caption}, waist-up view",
        ])

    elif variant_type == "context_addition":
        # Add contextual elements
        contexts = [
            ", italian coastal town background",
            ", portorosso street scene",
            ", with colorful buildings in background",
            ", detailed environmental setting",
        ]
        variants.extend([original_caption + ctx for ctx in contexts])

    elif variant_type == "multi_character":
        # Suggest multi-character scenarios (requires manual filtering later)
        multi_variants = [
            f"{original_caption}, with alberto scorfano nearby",
            f"{original_caption}, talking with giulia marcovaldo",
            f"{original_caption}, in conversation with another character",
        ]
        variants.extend(multi_variants)

    elif variant_type == "occlusion":
        # Add occlusion descriptions
        occlusion_variants = [
            f"{original_caption}, partially hidden behind object",
            f"{original_caption}, with foreground elements",
            f"{original_caption}, partially obscured by environment",
        ]
        variants.extend(occlusion_variants)

    return variants


def apply_image_transforms(img: Image.Image, transform_type: str) -> List[Image.Image]:
    """Apply transforms to create variations"""

    variations = [img]  # Keep original

    if transform_type == "crop_reframe":
        # Crop to create different shot types from full-body
        w, h = img.size

        # Medium shot (crop to upper 60%)
        medium_crop = img.crop((0, 0, w, int(h * 0.6)))
        variations.append(medium_crop.resize((w, h), Image.Resampling.LANCZOS))

        # Close-up (crop to upper 40%)
        closeup_crop = img.crop((w//4, 0, w*3//4, int(h * 0.4)))
        variations.append(closeup_crop.resize((w, h), Image.Resampling.LANCZOS))

    elif transform_type == "zoom_out":
        # Add border to simulate wider shot
        w, h = img.size
        new_w, new_h = int(w * 1.3), int(h * 1.3)

        # Create new image with border
        new_img = Image.new("RGB", (new_w, new_h), (120, 140, 160))  # Sky-ish color
        new_img.paste(img, ((new_w - w) // 2, (new_h - h) // 2))
        variations.append(new_img.resize((w, h), Image.Resampling.LANCZOS))

    return variations


def augment_dataset(
    dataset_dir: Path,
    analysis_json: Path,
    output_dir: Path,
    augmentation_strategy: str = "balanced"
):
    """
    Create augmented dataset with better balance

    Args:
        dataset_dir: Original dataset directory
        analysis_json: Analysis results from analyze_training_dataset.py
        output_dir: Output directory for augmented dataset
        augmentation_strategy: 'conservative' or 'balanced' or 'aggressive'
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy all original files first
    print(f"📋 Copying original dataset...")
    for img_file in dataset_dir.glob("*.png"):
        shutil.copy2(img_file, output_dir / img_file.name)

    for txt_file in dataset_dir.glob("*.txt"):
        shutil.copy2(txt_file, output_dir / txt_file.name)

    original_count = len(list(output_dir.glob("*.png")))
    print(f"✓ Copied {original_count} original images")

    # Identify samples to augment
    underrepresented = identify_underrepresented_samples(analysis_json)

    augmentation_targets = {
        "conservative": {"full_body": 2, "medium": 3, "multi_char": 0, "occlusion": 0},
        "balanced": {"full_body": 3, "medium": 4, "multi_char": 2, "occlusion": 2},
        "aggressive": {"full_body": 5, "medium": 6, "multi_char": 3, "occlusion": 3},
    }

    targets = augmentation_targets[augmentation_strategy]

    print(f"\n🔄 Generating augmented samples ({augmentation_strategy} strategy)...")

    aug_count = 0

    # Augment full-body shots
    for sample_name in underrepresented["full_body"][:targets["full_body"]]:
        img_path = dataset_dir / f"{sample_name}.png"
        caption_path = dataset_dir / f"{sample_name}.txt"

        if not img_path.exists():
            continue

        with open(caption_path, 'r') as f:
            original_caption = f.read().strip()

        img = Image.open(img_path)

        # Create cropped variations
        variations = apply_image_transforms(img, "crop_reframe")
        caption_variants = create_caption_variants(original_caption, "shot_variation")

        for i, (var_img, var_caption) in enumerate(zip(variations[1:], caption_variants[1:]), 1):
            new_name = f"{sample_name}_aug_shot{i}"
            var_img.save(output_dir / f"{new_name}.png")
            with open(output_dir / f"{new_name}.txt", 'w') as f:
                f.write(var_caption)
            aug_count += 1

    # Add context variations for simple backgrounds
    simple_samples = [dataset_dir / f"{p.stem}.png"
                      for p in dataset_dir.glob("*.txt")][:targets["medium"]]

    for img_path in simple_samples:
        caption_path = img_path.with_suffix('.txt')

        if not caption_path.exists():
            continue

        with open(caption_path, 'r') as f:
            original_caption = f.read().strip()

        caption_variants = create_caption_variants(original_caption, "context_addition")

        # Copy image with new captions
        for i, var_caption in enumerate(caption_variants[1:], 1):
            new_name = f"{img_path.stem}_aug_ctx{i}"
            shutil.copy2(img_path, output_dir / f"{new_name}.png")
            with open(output_dir / f"{new_name}.txt", 'w') as f:
                f.write(var_caption)
            aug_count += 1

    final_count = len(list(output_dir.glob("*.png")))

    print(f"\n✅ AUGMENTATION COMPLETE")
    print(f"  Original:   {original_count}")
    print(f"  Augmented:  +{aug_count}")
    print(f"  Total:      {final_count}")
    print(f"\n📁 Output: {output_dir}")
    print(f"\n💡 Next steps:")
    print(f"  1. Review augmented samples in {output_dir}")
    print(f"  2. Run analyze_training_dataset.py to verify balance")
    print(f"  3. Update training config to use augmented dataset")
    print(f"  4. Retrain with better balance")


def main():
    parser = argparse.ArgumentParser(description="Augment dataset for better balance")
    parser.add_argument("dataset_dir", type=Path, help="Original dataset directory")
    parser.add_argument("analysis_json", type=Path, help="Analysis JSON from analyze_training_dataset.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--strategy",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Augmentation strategy"
    )

    args = parser.parse_args()

    if not args.dataset_dir.exists():
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        return

    if not args.analysis_json.exists():
        print(f"❌ Analysis JSON not found: {args.analysis_json}")
        return

    augment_dataset(
        args.dataset_dir,
        args.analysis_json,
        args.output_dir,
        args.strategy
    )


if __name__ == "__main__":
    main()
