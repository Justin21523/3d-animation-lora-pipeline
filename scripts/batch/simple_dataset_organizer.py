#!/usr/bin/env python3
"""
Simple Dataset Organizer for Synthetic LoRA Training

Organizes filtered images into Kohya_ss format without complex dependencies.
Creates 45 datasets: 42 character-specific + 3 universal

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def organize_dataset(
    source_dir: Path,
    output_dir: Path,
    concept_name: str
) -> Tuple[int, int]:
    """
    Organize a single dataset into Kohya format

    Returns:
        (images_copied, captions_copied)
    """
    # Create Kohya format directory: {repeat}_{concept_name}
    kohya_dir = output_dir / f"1_{concept_name}"
    kohya_dir.mkdir(parents=True, exist_ok=True)

    images_copied = 0
    captions_copied = 0

    # Copy all PNG files and their captions
    for img_path in source_dir.glob("*.png"):
        caption_path = img_path.with_suffix('.txt')

        # Skip if caption missing
        if not caption_path.exists():
            continue

        # Copy image
        img_dest = kohya_dir / img_path.name
        shutil.copy2(img_path, img_dest)
        images_copied += 1

        # Copy caption
        cap_dest = kohya_dir / caption_path.name
        shutil.copy2(caption_path, cap_dest)
        captions_copied += 1

    return images_copied, captions_copied


def main():
    print("=" * 60)
    print("Simple Dataset Organization - Phase 3")
    print("=" * 60)
    print()

    FILTERED_DATA = Path("/mnt/data/ai_data/synthetic_lora_data/filtered_data")
    OUTPUT_ROOT = Path("/mnt/data/ai_data/synthetic_lora_data/datasets")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    CHARACTERS = [
        "alberto", "bryce", "caleb", "elio", "giulia",
        "ian_lightfoot", "luca", "miguel", "orion", "russell",
        "tyler", "alberto_seamonster", "luca_seamonster", "barley_lightfoot"
    ]

    LORA_TYPES = ["pose", "action", "expression"]

    results = {
        "character_specific": {},
        "universal": {},
        "summary": {"success": 0, "failed": 0, "total_images": 0, "total_captions": 0}
    }

    # Part 1: Character-Specific Datasets (42)
    print("=== Part 1: Character-Specific Datasets (42) ===")
    print()

    dataset_num = 0
    for character in CHARACTERS:
        for lora_type in LORA_TYPES:
            dataset_num += 1
            dataset_name = f"{character}_{lora_type}"

            print(f"[{dataset_num}/45] {dataset_name}")

            source_dir = FILTERED_DATA / character / lora_type / "tier_a"
            output_dir = OUTPUT_ROOT / dataset_name

            if not source_dir.exists():
                print(f"  ⚠️  Source not found, skipping")
                results["summary"]["failed"] += 1
                continue

            try:
                images, captions = organize_dataset(source_dir, output_dir, dataset_name)

                if images == 0:
                    print(f"  ⚠️  No images found")
                    results["summary"]["failed"] += 1
                else:
                    print(f"  ✅ {images} images, {captions} captions")
                    results["character_specific"][dataset_name] = {
                        "images": images,
                        "captions": captions
                    }
                    results["summary"]["success"] += 1
                    results["summary"]["total_images"] += images
                    results["summary"]["total_captions"] += captions

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results["summary"]["failed"] += 1

    print()
    print(f"Character-specific complete: {results['summary']['success']} success, {results['summary']['failed']} failed")
    print()

    # Part 2: Universal Datasets (3)
    print("=== Part 2: Universal Cross-Character Datasets (3) ===")
    print()

    for lora_type in LORA_TYPES:
        dataset_num += 1
        dataset_name = f"universal_{lora_type}"

        print(f"[{dataset_num}/45] {dataset_name}")

        # Create temp collection directory
        temp_dir = Path(f"/tmp/synthetic_universal_{lora_type}")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Collect from all characters
        collected = 0
        for character in CHARACTERS:
            source_dir = FILTERED_DATA / character / lora_type / "tier_a"

            if source_dir.exists():
                for img_path in source_dir.glob("*.png"):
                    caption_path = img_path.with_suffix('.txt')

                    if caption_path.exists():
                        # Copy with unique name
                        unique_name = f"{character}_{img_path.name}"
                        shutil.copy2(img_path, temp_dir / unique_name)
                        shutil.copy2(caption_path, temp_dir / unique_name.replace('.png', '.txt'))
                        collected += 1

        print(f"  Collected {collected} images from all characters")

        if collected == 0:
            print(f"  ⚠️  No images collected")
            shutil.rmtree(temp_dir)
            results["summary"]["failed"] += 1
            continue

        try:
            output_dir = OUTPUT_ROOT / dataset_name
            images, captions = organize_dataset(temp_dir, output_dir, dataset_name)

            print(f"  ✅ {images} images, {captions} captions")
            results["universal"][dataset_name] = {
                "images": images,
                "captions": captions
            }
            results["summary"]["success"] += 1
            results["summary"]["total_images"] += images
            results["summary"]["total_captions"] += captions

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["summary"]["failed"] += 1

        finally:
            shutil.rmtree(temp_dir)

    print()
    print("=" * 60)
    print("Dataset Organization Complete!")
    print("=" * 60)
    print()
    print(f"Total datasets: {dataset_num}")
    print(f"  ✅ Success: {results['summary']['success']}")
    print(f"  ❌ Failed: {results['summary']['failed']}")
    print(f"  📊 Total images: {results['summary']['total_images']}")
    print(f"  📝 Total captions: {results['summary']['total_captions']}")
    print()

    # Save report
    report_path = OUTPUT_ROOT / "organization_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Report saved: {report_path}")
    print()
    print("✅ Phase 3 Complete! Ready for Phase 4")

    return 0 if results["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
