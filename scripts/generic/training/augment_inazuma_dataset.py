#!/usr/bin/env python3
"""
Data augmentation for Inazuma Eleven character LoRA datasets.
Expands each character to >= 200 images using traditional augmentation techniques.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import shutil


class DataAugmentor:
    """Applies augmentation to images while preserving captions."""

    def __init__(self, target_count: int = 200):
        self.target_count = target_count
        self.augmentation_count = 0

    def random_rotation(self, img: Image.Image, max_angle: float = 15) -> Image.Image:
        """Random rotation augmentation."""
        angle = random.uniform(-max_angle, max_angle)
        return img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    def random_crop_and_resize(self, img: Image.Image, crop_ratio_range: Tuple[float, float] = (0.85, 0.95)) -> Image.Image:
        """Crop and resize back to original size."""
        ratio = random.uniform(*crop_ratio_range)
        w, h = img.size

        crop_w = int(w * ratio)
        crop_h = int(h * ratio)

        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)

        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((w, h), Image.Resampling.LANCZOS)

    def random_flip(self, img: Image.Image) -> Image.Image:
        """Random horizontal flip."""
        if random.random() < 0.5:
            return ImageOps.mirror(img)
        return img

    def random_color_jitter(
        self,
        img: Image.Image,
        brightness_factor: Tuple[float, float] = (0.85, 1.15),
        contrast_factor: Tuple[float, float] = (0.85, 1.15),
        saturation_factor: Tuple[float, float] = (0.85, 1.15)
    ) -> Image.Image:
        """Random brightness, contrast, and saturation adjustments."""
        # Brightness
        brightness_enhancer = ImageEnhance.Brightness(img)
        img = brightness_enhancer.enhance(random.uniform(*brightness_factor))

        # Contrast
        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(random.uniform(*contrast_factor))

        # Saturation
        saturation_enhancer = ImageEnhance.Color(img)
        img = saturation_enhancer.enhance(random.uniform(*saturation_factor))

        return img

    def random_blur(self, img: Image.Image, blur_type: str = "gaussian") -> Image.Image:
        """Apply minimal blur to simulate anti-aliasing artifacts."""
        if random.random() < 0.3:
            return img.filter(ImageOps.gaussian_blur(img, radius=0.5))
        return img

    def apply_augmentation(self, img: Image.Image, augmentation_type: int) -> Image.Image:
        """Apply a single augmentation based on type."""
        if augmentation_type == 0:
            return self.random_rotation(img, max_angle=8)
        elif augmentation_type == 1:
            return self.random_crop_and_resize(img, crop_ratio_range=(0.90, 0.98))
        elif augmentation_type == 2:
            return self.random_flip(img)
        elif augmentation_type == 3:
            return self.random_color_jitter(
                img,
                brightness_factor=(0.90, 1.10),
                contrast_factor=(0.90, 1.10),
                saturation_factor=(0.90, 1.15)
            )
        elif augmentation_type == 4:
            # Combo: rotation + color
            img = self.random_rotation(img, max_angle=5)
            img = self.random_color_jitter(
                img,
                brightness_factor=(0.92, 1.08),
                contrast_factor=(0.92, 1.08),
                saturation_factor=(0.92, 1.12)
            )
            return img
        else:
            # Combo: crop + flip + color
            img = self.random_crop_and_resize(img, crop_ratio_range=(0.92, 0.97))
            img = self.random_flip(img)
            img = self.random_color_jitter(
                img,
                brightness_factor=(0.93, 1.07),
                contrast_factor=(0.93, 1.07),
                saturation_factor=(0.93, 1.10)
            )
            return img

    def augment_character_dataset(
        self,
        character_dir: Path,
        output_augmented_dir: Path,
        caption: str
    ) -> int:
        """Augment a character's dataset to target count."""
        output_augmented_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        original_images = [
            img for img in character_dir.iterdir()
            if img.suffix.lower() in image_extensions
        ]

        if not original_images:
            return 0

        current_count = len(original_images)
        augmented_count = 0

        # Copy originals
        for img_path in original_images:
            shutil.copy(img_path, output_augmented_dir / img_path.name)

            # Copy or create caption
            caption_path = character_dir / (img_path.stem + ".txt")
            output_caption_path = output_augmented_dir / (img_path.stem + ".txt")

            if caption_path.exists():
                shutil.copy(caption_path, output_caption_path)
            else:
                output_caption_path.write_text(caption, encoding="utf-8")

        # Generate augmented versions until target
        aug_index = 0

        while current_count < self.target_count:
            source_img = random.choice(original_images)

            try:
                pil_img = Image.open(source_img).convert("RGB")
            except Exception as e:
                print(f"  ⚠️  Failed to open {source_img.name}: {e}")
                continue

            # Apply augmentation
            aug_type = aug_index % 6
            augmented_img = self.apply_augmentation(pil_img, aug_type)

            # Save augmented image
            output_name = f"{source_img.stem}_aug{aug_index:04d}{source_img.suffix}"
            output_path = output_augmented_dir / output_name

            augmented_img.save(output_path, quality=95)

            # Create caption for augmented image
            caption_path = output_augmented_dir / (output_name.replace(source_img.suffix, ".txt"))
            caption_path.write_text(caption, encoding="utf-8")

            current_count += 1
            augmented_count += 1
            aug_index += 1

        return augmented_count


def load_captions_from_jsonl(jsonl_path: Path) -> Dict[str, str]:
    """Load character ID to caption mapping from JSONL."""
    captions = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            char_id = entry.get("character_id")
            caption = entry.get("caption")

            if char_id and caption and char_id not in captions:
                captions[char_id] = caption

    return captions


def main():
    """Main execution."""

    base_dir = Path("/mnt/data/datasets/general/inazuma-eleven/lora_data/characters")
    output_base_dir = Path("/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented")
    captions_jsonl = Path("/mnt/data/datasets/general/inazuma-eleven/lora_data/captions_output/captions_all_characters.jsonl")

    target_count = 200

    print("=" * 70)
    print("Inazuma Eleven Dataset Augmentation")
    print("=" * 70)

    # Load captions
    print("\n📖 Loading captions from JSONL...")
    captions_map = load_captions_from_jsonl(captions_jsonl)
    print(f"✓ Loaded {len(captions_map)} unique captions")

    # Character ID to name mapping
    character_mapping = {
        "endou_mamoru": "Endou Mamoru",
        "fudou_akio": "Fudou Akio",
        "gouenji_shuuya": "Gouenji Shuuya",
        "inamori_asuto": "Inamori Asuto",
        "matsukaze_tenma": "Matsukaze Tenma",
        "nosaka_yuuma": "Nosaka Yuuma",
        "utsunomiya_toramaru": "Utsunomiya Toramaru",
    }

    augmentor = DataAugmentor(target_count=target_count)

    print(f"\n🔄 Augmenting datasets to target: {target_count} images per character\n")

    total_original = 0
    total_augmented = 0
    stats = []

    for char_id, char_name in character_mapping.items():
        char_dir = base_dir / char_name

        if not char_dir.exists():
            print(f"⚠️  {char_name}: directory not found")
            continue

        # Get caption
        caption = captions_map.get(char_id, "")
        if not caption:
            print(f"⚠️  {char_name}: no caption found")
            continue

        # Augment
        output_dir = output_base_dir / char_name
        original_count = len([
            f for f in char_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ])

        print(f"Augmenting {char_name}...", end=" ", flush=True)
        aug_count = augmentor.augment_character_dataset(char_dir, output_dir, caption)
        final_count = original_count + aug_count

        print(f"✓ ({original_count} original + {aug_count} augmented = {final_count} total)")

        total_original += original_count
        total_augmented += aug_count

        stats.append({
            "character": char_name,
            "original_count": original_count,
            "augmented_count": aug_count,
            "final_count": final_count
        })

    # Summary report
    print("\n" + "=" * 70)
    print("📊 Augmentation Summary")
    print("=" * 70)
    print(f"Total original images: {total_original}")
    print(f"Total augmented images: {total_augmented}")
    print(f"Total final images: {total_original + total_augmented}")
    print(f"Output directory: {output_base_dir}")

    # Save stats
    stats_path = output_base_dir / "augmentation_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats saved: {stats_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
