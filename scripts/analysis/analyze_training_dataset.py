#!/usr/bin/env python3
"""
Analyze training dataset composition and identify imbalances
"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict
import json
from typing import Dict, List
from PIL import Image


def analyze_captions(dataset_dir: Path) -> Dict:
    """Analyze caption content to identify scene types"""

    caption_files = list(dataset_dir.glob("*.txt"))

    results = {
        "total_images": len(caption_files),
        "scene_types": Counter(),
        "shot_types": Counter(),
        "character_count": Counter(),
        "background_complexity": Counter(),
        "occlusion_mentions": 0,
        "full_body_mentions": 0,
        "samples_by_category": defaultdict(list)
    }

    # Keywords for classification
    close_up_keywords = ["close-up", "close up", "portrait", "face", "head shot", "shoulder"]
    medium_keywords = ["medium shot", "waist up", "half body", "three-quarter"]
    full_body_keywords = ["full body", "full-body", "entire body", "whole body", "standing"]
    far_keywords = ["far", "distance", "distant", "wide shot", "long shot"]

    multi_char_keywords = ["alberto", "giulia", "two people", "multiple", "with others"]
    complex_bg_keywords = ["town", "street", "plaza", "crowd", "detailed background", "busy"]
    occlusion_keywords = ["behind", "partial", "obscured", "covered", "hidden"]

    for caption_file in caption_files:
        with open(caption_file, 'r', encoding='utf-8') as f:
            caption = f.read().lower()

        img_name = caption_file.stem

        # Classify shot type
        if any(kw in caption for kw in close_up_keywords):
            results["shot_types"]["close-up"] += 1
            results["samples_by_category"]["close-up"].append(img_name)
        elif any(kw in caption for kw in medium_keywords):
            results["shot_types"]["medium"] += 1
            results["samples_by_category"]["medium"].append(img_name)
        elif any(kw in caption for kw in full_body_keywords):
            results["shot_types"]["full-body"] += 1
            results["samples_by_category"]["full-body"].append(img_name)
        elif any(kw in caption for kw in far_keywords):
            results["shot_types"]["far"] += 1
            results["samples_by_category"]["far"].append(img_name)
        else:
            results["shot_types"]["unclassified"] += 1

        # Check for multiple characters
        if any(kw in caption for kw in multi_char_keywords):
            results["character_count"]["multiple"] += 1
            results["samples_by_category"]["multi-character"].append(img_name)
        else:
            results["character_count"]["single"] += 1

        # Check background complexity
        if any(kw in caption for kw in complex_bg_keywords):
            results["background_complexity"]["complex"] += 1
            results["samples_by_category"]["complex-bg"].append(img_name)
        else:
            results["background_complexity"]["simple"] += 1

        # Check occlusion
        if any(kw in caption for kw in occlusion_keywords):
            results["occlusion_mentions"] += 1
            results["samples_by_category"]["occlusion"].append(img_name)

        # Full body count
        if any(kw in caption for kw in full_body_keywords):
            results["full_body_mentions"] += 1

    return results


def analyze_image_sizes(dataset_dir: Path) -> Dict:
    """Analyze image dimensions and aspect ratios"""

    image_files = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg"))

    sizes = Counter()
    aspect_ratios = Counter()

    for img_file in image_files:
        try:
            with Image.open(img_file) as img:
                w, h = img.size
                sizes[f"{w}x{h}"] += 1

                # Classify aspect ratio
                ratio = w / h
                if ratio > 1.3:
                    aspect_ratios["landscape"] += 1
                elif ratio < 0.77:
                    aspect_ratios["portrait"] += 1
                else:
                    aspect_ratios["square"] += 1
        except Exception as e:
            print(f"Error reading {img_file}: {e}")

    return {
        "sizes": dict(sizes.most_common(10)),
        "aspect_ratios": dict(aspect_ratios)
    }


def print_report(caption_analysis: Dict, image_analysis: Dict):
    """Print formatted analysis report"""

    total = caption_analysis["total_images"]

    print("\n" + "="*80)
    print("TRAINING DATASET ANALYSIS REPORT")
    print("="*80)

    print(f"\n📊 DATASET SIZE: {total} images")

    # Shot type distribution
    print(f"\n📷 SHOT TYPE DISTRIBUTION:")
    for shot_type, count in caption_analysis["shot_types"].most_common():
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {shot_type:15s}: {count:4d} ({percentage:5.1f}%) {bar}")

    # Character count
    print(f"\n👥 CHARACTER COUNT:")
    for char_type, count in caption_analysis["character_count"].items():
        percentage = (count / total) * 100
        print(f"  {char_type:15s}: {count:4d} ({percentage:5.1f}%)")

    # Background complexity
    print(f"\n🌆 BACKGROUND COMPLEXITY:")
    for bg_type, count in caption_analysis["background_complexity"].items():
        percentage = (count / total) * 100
        print(f"  {bg_type:15s}: {count:4d} ({percentage:5.1f}%)")

    # Special features
    print(f"\n🔍 SPECIAL FEATURES:")
    print(f"  Full-body shots:  {caption_analysis['full_body_mentions']:4d} ({caption_analysis['full_body_mentions']/total*100:.1f}%)")
    print(f"  Occlusion cases:  {caption_analysis['occlusion_mentions']:4d} ({caption_analysis['occlusion_mentions']/total*100:.1f}%)")

    # Image sizes
    print(f"\n📐 IMAGE DIMENSIONS:")
    for size, count in list(image_analysis["sizes"].items())[:5]:
        percentage = (count / total) * 100
        print(f"  {size:12s}: {count:4d} ({percentage:5.1f}%)")

    print(f"\n📏 ASPECT RATIOS:")
    for ratio, count in image_analysis["aspect_ratios"].items():
        percentage = (count / total) * 100
        print(f"  {ratio:12s}: {count:4d} ({percentage:5.1f}%)")

    # Recommendations
    print(f"\n" + "="*80)
    print("⚠️  IMBALANCE WARNINGS & RECOMMENDATIONS")
    print("="*80)

    close_up_pct = caption_analysis["shot_types"].get("close-up", 0) / total * 100
    full_body_pct = caption_analysis["shot_types"].get("full-body", 0) / total * 100
    multi_char_pct = caption_analysis["character_count"].get("multiple", 0) / total * 100

    if close_up_pct > 60:
        print(f"\n❌ Close-up shots are {close_up_pct:.1f}% (SEVERELY OVER-REPRESENTED)")
        print(f"   → Add more medium and full-body shots")

    if full_body_pct < 20:
        print(f"\n❌ Full-body shots are only {full_body_pct:.1f}% (UNDER-REPRESENTED)")
        print(f"   → Need at least 25-30% full-body shots for good body consistency")

    if multi_char_pct < 10:
        print(f"\n❌ Multi-character scenes are only {multi_char_pct:.1f}% (CRITICALLY LOW)")
        print(f"   → Add scenes with other characters to improve context handling")

    if caption_analysis["occlusion_mentions"] < total * 0.05:
        print(f"\n❌ Occlusion examples are {caption_analysis['occlusion_mentions']/total*100:.1f}% (TOO LOW)")
        print(f"   → Include partially occluded samples to prevent distortion")

    print(f"\n✅ RECOMMENDED DATASET COMPOSITION:")
    print(f"   Close-up:   30-40%  (current: {close_up_pct:.1f}%)")
    print(f"   Medium:     25-30%  (current: {caption_analysis['shot_types'].get('medium', 0)/total*100:.1f}%)")
    print(f"   Full-body:  25-30%  (current: {full_body_pct:.1f}%)")
    print(f"   Far/Wide:   5-10%   (current: {caption_analysis['shot_types'].get('far', 0)/total*100:.1f}%)")
    print(f"   Multi-char: 15-20%  (current: {multi_char_pct:.1f}%)")
    print(f"   Occlusion:  8-12%   (current: {caption_analysis['occlusion_mentions']/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze LoRA training dataset")
    parser.add_argument("dataset_dir", type=Path, help="Training dataset directory")
    parser.add_argument("--output-json", type=Path, help="Save results to JSON")

    args = parser.parse_args()

    if not args.dataset_dir.exists():
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        return

    print(f"🔍 Analyzing dataset: {args.dataset_dir}")

    # Run analyses
    caption_analysis = analyze_captions(args.dataset_dir)
    image_analysis = analyze_image_sizes(args.dataset_dir)

    # Print report
    print_report(caption_analysis, image_analysis)

    # Save JSON if requested
    if args.output_json:
        results = {
            "dataset_dir": str(args.dataset_dir),
            "caption_analysis": {
                k: dict(v) if isinstance(v, Counter) else
                   {k2: list(v2) if isinstance(v2, list) else v2 for k2, v2 in v.items()} if isinstance(v, defaultdict) else v
                for k, v in caption_analysis.items()
            },
            "image_analysis": image_analysis
        }

        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
