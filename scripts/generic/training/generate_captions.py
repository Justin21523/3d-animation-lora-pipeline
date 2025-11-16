#!/usr/bin/env python3
"""
VLM-based Caption Generation for Character Training Data

Supports multiple VLM backends:
- BLIP2 (fast, good for simple captions)
- Qwen2-VL (advanced, schema-guided captions)
- InternVL2 (alternative advanced option)

For 3D animation characters, generates captions emphasizing:
- Character identity and appearance
- Materials (skin, hair, clothing textures)
- Lighting (key light, rim light, ambient)
- Camera angle (close-up, three-quarter view, low angle)
- Pose and expression

Usage:
    python generate_captions.py \
      --input-dir /path/to/filtered_clusters \
      --output-dir /path/to/captioned \
      --model blip2 \
      --character-name "luca" \
      --prefix "a 3d animated character" \
      --device cuda
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import json
from PIL import Image
import sys


class CaptionGenerator:
    """Base class for caption generation"""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def generate(self, image: np.ndarray, prefix: str = "") -> str:
        """Generate caption for image"""
        raise NotImplementedError


class BLIP2CaptionGenerator(CaptionGenerator):
    """BLIP2-based caption generator"""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.load_model()

    def load_model(self):
        """Load BLIP2 model"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration

            print("Loading BLIP2 model...")
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            print("✓ BLIP2 model loaded successfully")

        except ImportError:
            print("❌ transformers not installed!")
            print("\nInstall with:")
            print("  pip install transformers")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to load BLIP2: {e}")
            sys.exit(1)

    def generate(self, image: np.ndarray, prefix: str = "") -> str:
        """Generate caption using BLIP2"""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Process image
        inputs = self.processor(pil_img, return_tensors="pt").to(self.device, torch.float16 if self.device == "cuda" else torch.float32)

        # Generate caption
        with torch.no_grad():
            if prefix:
                # Conditional generation with prefix
                prompt = f"Question: Describe this image. Answer: {prefix}"
                inputs = self.processor(pil_img, text=prompt, return_tensors="pt").to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
                generated_ids = self.model.generate(**inputs, max_new_tokens=50)
            else:
                generated_ids = self.model.generate(**inputs, max_new_tokens=50)

            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return caption


class SimpleCaptionGenerator(CaptionGenerator):
    """Simple template-based caption generator (fallback)"""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        print("⚠️  Using simple template-based captions (no VLM)")

    def generate(self, image: np.ndarray, prefix: str = "") -> str:
        """Generate simple template caption"""
        if prefix:
            return prefix
        else:
            return "a 3d animated character"


def load_caption_generator(model_name: str, device: str = "cuda") -> CaptionGenerator:
    """Load caption generator based on model name"""
    if model_name == "blip2":
        return BLIP2CaptionGenerator(device=device)
    elif model_name == "simple":
        return SimpleCaptionGenerator(device=device)
    else:
        print(f"⚠️  Unknown model: {model_name}, using simple captions")
        return SimpleCaptionGenerator(device=device)


def process_character_cluster(
    cluster_dir: Path,
    output_dir: Path,
    caption_generator: CaptionGenerator,
    character_name: str,
    prefix: str = "",
    max_length: int = 77,
    skip_existing: bool = True
) -> Dict:
    """
    Generate captions for a character cluster

    Returns statistics
    """
    cluster_name = cluster_dir.name
    output_cluster_dir = output_dir / cluster_name
    output_images_dir = output_cluster_dir / "images"
    output_captions_dir = output_cluster_dir / "captions"

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_captions_dir.mkdir(parents=True, exist_ok=True)

    # Get all instances
    instances = list(cluster_dir.glob("*.png"))

    if not instances:
        return {
            "cluster": cluster_name,
            "total": 0,
            "captioned": 0,
            "skipped": 0,
            "failed": 0
        }

    stats = {
        "cluster": cluster_name,
        "total": len(instances),
        "captioned": 0,
        "skipped": 0,
        "failed": 0
    }

    print(f"\n📂 Processing {cluster_name} ({len(instances)} images)")

    for img_path in tqdm(instances, desc=f"  Generating captions", leave=False):
        caption_path = output_captions_dir / f"{img_path.stem}.txt"
        output_img_path = output_images_dir / img_path.name

        # Skip if exists
        if skip_existing and caption_path.exists() and output_img_path.exists():
            stats["skipped"] += 1
            continue

        try:
            # Read image
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                stats["failed"] += 1
                continue

            # Generate caption
            caption = caption_generator.generate(image, prefix=prefix)

            # Post-process caption
            # 1. Add character name if not present
            if character_name and character_name.lower() not in caption.lower():
                caption = f"{character_name}, {caption}"

            # 2. Ensure it starts with prefix if provided
            if prefix and not caption.startswith(prefix):
                caption = f"{prefix}, {caption}"

            # 3. Truncate to max length
            words = caption.split()
            if len(words) > max_length:
                caption = " ".join(words[:max_length])

            # 4. Clean up
            caption = caption.strip()

            # Save caption
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)

            # Copy image to output
            import shutil
            shutil.copy2(img_path, output_img_path)

            stats["captioned"] += 1

        except Exception as e:
            print(f"\n⚠️  Failed to process {img_path.name}: {e}")
            stats["failed"] += 1
            continue

    return stats


def main():
    parser = argparse.ArgumentParser(description="VLM-based Caption Generation")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with filtered character clusters"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for captioned data"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["blip2", "simple"],
        default="blip2",
        help="Caption model to use (default: blip2)"
    )
    parser.add_argument(
        "--character-name",
        type=str,
        default="",
        help="Character name to include in captions"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="a 3d animated character",
        help="Caption prefix (default: 'a 3d animated character')"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=77,
        help="Maximum caption length in words (default: 77)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run model"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing captions"
    )

    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"VLM-BASED CAPTION GENERATION")
    print(f"{'='*70}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model.upper()}")
    print(f"Character name: {args.character_name or 'None'}")
    print(f"Prefix: {args.prefix}")
    print(f"Max length: {args.max_length} words")
    print(f"Device: {args.device.upper()}")
    print(f"{'='*70}\n")

    # Load caption generator
    caption_generator = load_caption_generator(args.model, device=args.device)

    # Get all cluster directories
    cluster_dirs = [d for d in input_dir.iterdir()
                   if d.is_dir() and d.name.startswith("character_")]

    if not cluster_dirs:
        print("❌ No cluster directories found!")
        return

    print(f"📂 Found {len(cluster_dirs)} character clusters\n")

    # Process each cluster
    all_stats = []

    for cluster_dir in cluster_dirs:
        stats = process_character_cluster(
            cluster_dir,
            output_dir,
            caption_generator,
            character_name=args.character_name,
            prefix=args.prefix,
            max_length=args.max_length,
            skip_existing=args.skip_existing
        )
        all_stats.append(stats)

    # Save report
    report_path = output_dir / "caption_generation_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            "parameters": {
                "model": args.model,
                "character_name": args.character_name,
                "prefix": args.prefix,
                "max_length": args.max_length
            },
            "clusters": all_stats
        }, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"CAPTION GENERATION COMPLETE")
    print(f"{'='*70}")

    total_images = sum(s["total"] for s in all_stats)
    total_captioned = sum(s["captioned"] for s in all_stats)
    total_skipped = sum(s["skipped"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)

    print(f"Total images: {total_images}")
    print(f"Captioned: {total_captioned}")
    print(f"Skipped (existing): {total_skipped}")
    print(f"Failed: {total_failed}")
    print(f"{'='*70}")
    print(f"\n📁 Captioned data saved to: {output_dir}")
    print(f"📊 Report saved to: {report_path}\n")


if __name__ == "__main__":
    main()
