#!/usr/bin/env python3
"""
Instance Categorizer

Purpose: Auto-suggest categories for SAM2 instances using CLIP
Categories: character, object, furniture, background, vehicle, prop
Use Cases: Pre-filter instances before manual review

Usage:
    python instance_categorizer.py \
        --instances-dir /path/to/instances \
        --output-dir /path/to/categorized \
        --confidence-threshold 0.7 \
        --project luca
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import shutil


@dataclass
class CategorizationConfig:
    """Configuration for instance categorization"""
    model_name: str = "openai/clip-vit-base-patch32"
    confidence_threshold: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 128  # Optimized for RTX 5080 16GB (was 32)
    categories: List[str] = None
    create_category_folders: bool = True
    create_hardlinks: bool = False
    save_metadata: bool = True
    num_workers: int = 4  # CPU workers for data loading

    def __post_init__(self):
        if self.categories is None:
            self.categories = [
                "character",
                "human person",
                "object",
                "furniture",
                "background",
                "vehicle",
                "prop",
                "accessory",
            ]


class InstanceCategorizer:
    """Categorize instances using CLIP zero-shot classification"""

    def __init__(self, config: CategorizationConfig):
        """
        Initialize categorizer

        Args:
            config: Categorization configuration
        """
        self.config = config

        print(f"\n🔧 Loading CLIP model: {config.model_name}")
        self.model = CLIPModel.from_pretrained(config.model_name).to(config.device)
        self.processor = CLIPProcessor.from_pretrained(config.model_name)
        self.model.eval()

        # Prepare category prompts
        self.category_prompts = [
            f"a photo of a {cat}" for cat in config.categories
        ]
        print(f"✅ Model loaded on {config.device}")
        print(f"📋 Categories: {', '.join(config.categories)}\n")

    @torch.no_grad()
    def categorize_batch(
        self,
        image_paths: List[Path]
    ) -> List[Tuple[str, float]]:
        """
        Categorize a batch of images

        Args:
            image_paths: List of image paths

        Returns:
            List of (category, confidence) tuples
        """
        # Load images
        images = []
        valid_indices = []

        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"⚠️ Error loading {img_path.name}: {e}")
                continue

        if not images:
            return [(None, 0.0)] * len(image_paths)

        # Process with CLIP
        inputs = self.processor(
            text=self.category_prompts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.config.device)

        # Get predictions
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        # Extract results
        results = [(None, 0.0)] * len(image_paths)

        for i, valid_idx in enumerate(valid_indices):
            category_probs = probs[i].cpu().numpy()
            max_idx = category_probs.argmax()
            max_prob = float(category_probs[max_idx])

            category = self.config.categories[max_idx]
            results[valid_idx] = (category, max_prob)

        return results

    def categorize_all(
        self,
        image_files: List[Path]
    ) -> Dict[str, Dict]:
        """
        Categorize all instances

        Args:
            image_files: List of image paths

        Returns:
            Dictionary mapping filename to category info
        """
        print(f"\n🔍 Categorizing {len(image_files)} instances...")

        results = {}

        # Process in batches
        for i in tqdm(range(0, len(image_files), self.config.batch_size), desc="Categorizing"):
            batch = image_files[i:i + self.config.batch_size]
            batch_results = self.categorize_batch(batch)

            for img_path, (category, confidence) in zip(batch, batch_results):
                # Apply confidence threshold
                if confidence >= self.config.confidence_threshold:
                    final_category = category
                    status = "confident"
                else:
                    final_category = "uncertain"
                    status = "low_confidence"

                results[img_path.name] = {
                    "filename": img_path.name,
                    "category": final_category,
                    "raw_category": category,
                    "confidence": confidence,
                    "status": status,
                }

        return results

    def categorize_instances(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main categorization pipeline

        Args:
            input_dir: Directory with instances
            output_dir: Directory to save categorized instances

        Returns:
            Statistics dictionary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all instances
        image_files = sorted(
            list(input_dir.glob("*.png")) +
            list(input_dir.glob("*.jpg"))
        )

        print(f"\n📊 Found {len(image_files)} instances in {input_dir}")

        # Categorize
        results = self.categorize_all(image_files)

        # Organize by category
        if self.config.create_category_folders:
            print(f"\n📁 Organizing into category folders...")

            category_counts = {}

            for img_path in tqdm(image_files, desc="Organizing files"):
                filename = img_path.name
                result = results.get(filename)

                if not result:
                    continue

                category = result["category"]

                # Create category folder
                category_dir = output_dir / category
                category_dir.mkdir(exist_ok=True)

                # Copy/link file
                dst_path = category_dir / filename

                if self.config.create_hardlinks:
                    try:
                        dst_path.hardlink_to(img_path)
                    except:
                        shutil.copy2(img_path, dst_path)
                else:
                    shutil.copy2(img_path, dst_path)

                # Update counts
                category_counts[category] = category_counts.get(category, 0) + 1

        else:
            # Just count categories
            category_counts = {}
            for result in results.values():
                category = result["category"]
                category_counts[category] = category_counts.get(category, 0) + 1

        # Compute statistics
        confident_count = sum(1 for r in results.values() if r["status"] == "confident")
        uncertain_count = sum(1 for r in results.values() if r["status"] == "low_confidence")

        stats = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "model": self.config.model_name,
            "confidence_threshold": self.config.confidence_threshold,
            "total_instances": len(image_files),
            "confident_predictions": confident_count,
            "uncertain_predictions": uncertain_count,
            "confidence_rate": confident_count / len(image_files) if len(image_files) > 0 else 0,
            "category_distribution": category_counts,
            "timestamp": datetime.now().isoformat(),
        }

        # Save results
        if self.config.save_metadata:
            results_path = output_dir / "categorization_results.json"
            with open(results_path, 'w') as f:
                json.dump({
                    "statistics": stats,
                    "instances": results,
                }, f, indent=2)

            print(f"\n📄 Results saved to: {results_path}")

        # Print summary
        print(f"\n✅ Categorization complete!")
        print(f"   Total instances: {len(image_files)}")
        print(f"   Confident predictions: {confident_count} ({stats['confidence_rate']:.1%})")
        print(f"   Uncertain predictions: {uncertain_count}")

        print(f"\n📊 Category distribution:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {category}: {count}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Auto-categorize instances using CLIP (Film-Agnostic)"
    )
    parser.add_argument(
        "--instances-dir",
        type=str,
        required=True,
        help="Directory with SAM2 instances"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save categorized instances"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP model name (default: openai/clip-vit-base-patch32)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold (default: 0.7)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="Custom categories (default: character, human person, object, furniture, etc.)"
    )
    parser.add_argument(
        "--no-folders",
        action="store_true",
        help="Don't create category folders (only generate metadata)"
    )
    parser.add_argument(
        "--hardlinks",
        action="store_true",
        help="Use hardlinks instead of copying"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (for logging)"
    )

    args = parser.parse_args()

    # Create config
    config = CategorizationConfig(
        model_name=args.model,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        batch_size=args.batch_size,
        categories=args.categories,
        create_category_folders=not args.no_folders,
        create_hardlinks=args.hardlinks,
        save_metadata=True
    )

    # Run categorization
    categorizer = InstanceCategorizer(config)
    stats = categorizer.categorize_instances(
        input_dir=Path(args.instances_dir),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
