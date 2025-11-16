#!/usr/bin/env python3
"""
Intelligent Data Augmentation for Small Clusters

Augments character images while preserving character identity and features.
Designed for important characters/yokai with limited samples (5-20 images).

Augmentation strategies:
- Horizontal flip (for symmetric yokai)
- Slight rotation (-5° to +5°)
- Brightness adjustment (0.8x to 1.2x)
- Contrast adjustment (0.9x to 1.1x)
- Hue shift (±10°, preserving character colors)
- Crop variation (95%-100%, keeping character complete)

Target multipliers:
- 5-10 images → 6-8x augmentation (40-60 final images)
- 11-15 images → 4-5x augmentation (50-70 final images)
- 16-20 images → 3-4x augmentation (60-80 final images)
- 21+ images → 1.5-2x augmentation (minimal, if requested)
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
from datetime import datetime
import random
import cv2
import warnings

warnings.filterwarnings("ignore")


class CharacterPreservingAugmentor:
    """Augments images while preserving character identity"""

    def __init__(self, seed: int = 42):
        """
        Initialize augmentor

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def horizontal_flip(self, image: Image.Image) -> Image.Image:
        """Horizontal flip"""
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def rotate(self, image: Image.Image, angle: float) -> Image.Image:
        """
        Slight rotation while keeping character complete

        Args:
            angle: Rotation angle in degrees (-5 to +5)
        """
        # Rotate with expansion to avoid cropping
        return image.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

    def adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        """
        Adjust brightness

        Args:
            factor: 0.8 to 1.2 (darker to brighter)
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def adjust_contrast(self, image: Image.Image, factor: float) -> Image.Image:
        """
        Adjust contrast

        Args:
            factor: 0.9 to 1.1
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def adjust_hue(self, image: Image.Image, shift: int) -> Image.Image:
        """
        Slight hue shift to preserve character colors

        Args:
            shift: -10 to +10 degrees
        """
        # Convert to HSV
        img_array = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Shift hue
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180

        # Convert back
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result = Image.fromarray(rgb)

        # Restore alpha channel if present
        if image.mode == "RGBA":
            result.putalpha(image.split()[3])

        return result

    def random_crop_resize(self, image: Image.Image, crop_ratio: float) -> Image.Image:
        """
        Random crop then resize back to original size

        Args:
            crop_ratio: 0.95 to 1.0 (how much to keep)
        """
        width, height = image.size

        # Calculate crop size
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        # Random offset
        max_offset_x = width - crop_width
        max_offset_y = height - crop_height

        if max_offset_x > 0:
            offset_x = random.randint(0, max_offset_x)
        else:
            offset_x = 0

        if max_offset_y > 0:
            offset_y = random.randint(0, max_offset_y)
        else:
            offset_y = 0

        # Crop
        cropped = image.crop(
            (offset_x, offset_y, offset_x + crop_width, offset_y + crop_height)
        )

        # Resize back
        return cropped.resize((width, height), Image.Resampling.LANCZOS)

    def apply_slight_blur(self, image: Image.Image) -> Image.Image:
        """Apply very slight gaussian blur"""
        return image.filter(ImageFilter.GaussianBlur(radius=0.5))

    def generate_augmentation(
        self, image: Image.Image, aug_intensity: str = "medium"
    ) -> Image.Image:
        """
        Generate one augmented version

        Args:
            image: Source image
            aug_intensity: "light", "medium", or "heavy"

        Returns:
            Augmented image
        """
        result = image.copy()

        # Define augmentation probabilities based on intensity
        if aug_intensity == "light":
            flip_prob = 0.3
            rotate_range = (-3, 3)
            brightness_range = (0.9, 1.1)
            contrast_range = (0.95, 1.05)
            hue_range = (-5, 5)
            crop_range = (0.98, 1.0)
            blur_prob = 0.1
        elif aug_intensity == "heavy":
            flip_prob = 0.5
            rotate_range = (-5, 5)
            brightness_range = (0.8, 1.2)
            contrast_range = (0.9, 1.1)
            hue_range = (-10, 10)
            crop_range = (0.95, 1.0)
            blur_prob = 0.3
        else:  # medium
            flip_prob = 0.4
            rotate_range = (-4, 4)
            brightness_range = (0.85, 1.15)
            contrast_range = (0.92, 1.08)
            hue_range = (-8, 8)
            crop_range = (0.96, 1.0)
            blur_prob = 0.2

        # Apply augmentations randomly
        # Horizontal flip
        if random.random() < flip_prob:
            result = self.horizontal_flip(result)

        # Rotation
        if random.random() < 0.7:
            angle = random.uniform(*rotate_range)
            result = self.rotate(result, angle)

        # Brightness
        if random.random() < 0.8:
            factor = random.uniform(*brightness_range)
            result = self.adjust_brightness(result, factor)

        # Contrast
        if random.random() < 0.7:
            factor = random.uniform(*contrast_range)
            result = self.adjust_contrast(result, factor)

        # Hue shift
        if random.random() < 0.6:
            shift = random.randint(*hue_range)
            result = self.adjust_hue(result, shift)

        # Crop variation
        if random.random() < 0.5:
            ratio = random.uniform(*crop_range)
            result = self.random_crop_resize(result, ratio)

        # Slight blur
        if random.random() < blur_prob:
            result = self.apply_slight_blur(result)

        return result

    def augment_cluster(
        self,
        cluster_dir: Path,
        output_dir: Path,
        target_count: int = None,
        aug_intensity: str = "medium",
    ) -> Dict:
        """
        Augment all images in a cluster

        Args:
            cluster_dir: Source cluster directory
            output_dir: Output directory for augmented images
            target_count: Target total images (original + augmented)
            aug_intensity: "light", "medium", or "heavy"

        Returns:
            Augmentation statistics
        """
        # Find original images
        image_files = list(cluster_dir.glob("*.png"))

        if not image_files:
            return {"success": False, "error": "No images found"}

        num_original = len(image_files)

        # Determine target count if not specified
        if target_count is None:
            if num_original <= 10:
                target_count = num_original * 6  # 6x for smallest
            elif num_original <= 15:
                target_count = num_original * 4  # 4x
            elif num_original <= 20:
                target_count = num_original * 3  # 3x
            else:
                target_count = int(num_original * 1.5)  # 1.5x for larger

        num_augmented_needed = max(0, target_count - num_original)
        augmentations_per_image = num_augmented_needed // num_original + 1

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy original images
        copied_count = 0
        for img_path in image_files:
            output_path = output_dir / img_path.name
            Image.open(img_path).save(output_path)
            copied_count += 1

        # Generate augmented images
        augmented_count = 0
        for img_path in tqdm(image_files, desc=f"  Augmenting {cluster_dir.name}"):
            image = Image.open(img_path)

            # Generate multiple augmentations per image
            for i in range(augmentations_per_image):
                if augmented_count >= num_augmented_needed:
                    break

                # Generate augmented version
                aug_image = self.generate_augmentation(image, aug_intensity)

                # Save with unique name
                stem = img_path.stem
                output_name = f"{stem}_aug{i:03d}.png"
                output_path = output_dir / output_name
                aug_image.save(output_path)

                augmented_count += 1

            if augmented_count >= num_augmented_needed:
                break

        return {
            "success": True,
            "cluster_name": cluster_dir.name,
            "num_original": num_original,
            "num_augmented": augmented_count,
            "total_images": copied_count + augmented_count,
            "target_count": target_count,
            "aug_intensity": aug_intensity,
        }


def augment_small_clusters(
    input_dir: Path,
    output_dir: Path,
    max_original: int = 30,
    min_original: int = 5,
    aug_intensity: str = "medium",
    target_multiplier: float = None,
):
    """
    Augment clusters with limited samples

    Args:
        input_dir: Directory containing character clusters
        output_dir: Output directory for augmented clusters
        max_original: Only augment clusters with <= this many images
        min_original: Minimum images required to augment
        aug_intensity: Augmentation intensity
        target_multiplier: Custom target multiplier (overrides auto)
    """
    print(f"\n{'='*80}")
    print("INTELLIGENT DATA AUGMENTATION FOR SMALL CLUSTERS")
    print(f"{'='*80}\n")

    cluster_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("cluster_")]
    )

    if not cluster_dirs:
        print(f"❌ No clusters found in {input_dir}")
        return

    print(f"📊 Configuration:")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Max original images: {max_original}")
    print(f"  Min original images: {min_original}")
    print(f"  Augmentation intensity: {aug_intensity}")
    if target_multiplier:
        print(f"  Target multiplier: {target_multiplier}x")
    print()

    # Filter clusters needing augmentation
    clusters_to_augment = []
    for cluster_dir in cluster_dirs:
        num_images = len(list(cluster_dir.glob("*.png")))
        if min_original <= num_images <= max_original:
            clusters_to_augment.append((cluster_dir, num_images))

    print(f"Found {len(clusters_to_augment)} clusters to augment\n")

    if not clusters_to_augment:
        print("✓ No clusters need augmentation")
        return

    # Sort by image count (smallest first - they need most help)
    clusters_to_augment.sort(key=lambda x: x[1])

    # Create augmentor
    augmentor = CharacterPreservingAugmentor()

    # Augment clusters
    results = []
    for cluster_dir, num_images in tqdm(
        clusters_to_augment, desc="Augmenting clusters"
    ):
        cluster_output = output_dir / cluster_dir.name

        # Calculate target count
        if target_multiplier:
            target_count = int(num_images * target_multiplier)
        else:
            target_count = None  # Auto-determine

        result = augmentor.augment_cluster(
            cluster_dir,
            cluster_output,
            target_count=target_count,
            aug_intensity=aug_intensity,
        )

        if result["success"]:
            results.append(result)

    # Generate summary
    total_original = sum(r["num_original"] for r in results)
    total_augmented = sum(r["num_augmented"] for r in results)
    total_final = sum(r["total_images"] for r in results)

    print(f"\n{'='*80}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Clusters augmented: {len(results)}")
    print(f"  Original images: {total_original}")
    print(f"  Augmented images: {total_augmented}")
    print(f"  Total final images: {total_final}")
    print(f"  Average multiplier: {total_final/total_original:.1f}x")
    print(f"\n  Output: {output_dir}")
    print(f"{'='*80}\n")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_clusters": len(results),
        "total_original": total_original,
        "total_augmented": total_augmented,
        "total_final": total_final,
        "aug_intensity": aug_intensity,
        "results": results,
    }

    metadata_path = output_dir / "augmentation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved: {metadata_path}\n")

    # Show top augmented clusters
    results.sort(key=lambda x: x["num_augmented"], reverse=True)
    print(f"Top 10 most augmented clusters:")
    for i, result in enumerate(results[:10], 1):
        print(
            f"  {i}. {result['cluster_name']}: "
            f"{result['num_original']} → {result['total_images']} "
            f"({result['num_augmented']} augmented, "
            f"{result['total_images']/result['num_original']:.1f}x)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent data augmentation for small character clusters"
    )

    parser.add_argument(
        "input_dir", type=Path, help="Directory containing character clusters"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for augmented clusters",
    )
    parser.add_argument(
        "--max-original",
        type=int,
        default=30,
        help="Only augment clusters with <= this many images (default: 30)",
    )
    parser.add_argument(
        "--min-original",
        type=int,
        default=5,
        help="Minimum images required to augment (default: 5)",
    )
    parser.add_argument(
        "--aug-intensity",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Augmentation intensity (default: medium)",
    )
    parser.add_argument(
        "--target-multiplier",
        type=float,
        default=None,
        help="Custom target multiplier (e.g., 4.0 for 4x)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Run augmentation
    augment_small_clusters(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_original=args.max_original,
        min_original=args.min_original,
        aug_intensity=args.aug_intensity,
        target_multiplier=args.target_multiplier,
    )


if __name__ == "__main__":
    main()
