#!/usr/bin/env python3
"""
LoRA Quality Metrics Evaluation
Comprehensive quality assessment using multiple quantitative metrics
"""

import torch
import clip
from PIL import Image
from pathlib import Path
import json
import argparse
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torchvision.transforms as transforms


class LoRAQualityEvaluator:
    """Evaluate LoRA-generated images using multiple metrics"""

    def __init__(self, device: str = "cuda"):
        """
        Initialize evaluator with necessary models

        Args:
            device: Device to use (cuda/cpu)
        """
        self.device = device
        print(f"🔧 Initializing evaluator on {device}...")

        # Load CLIP model for prompt matching and consistency
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-L/14", device=device
        )

        # Image preprocessing for CLIP
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            ),
        ])

        print("✓ Evaluator initialized")

    def calculate_clip_score(
        self,
        image_path: Path,
        prompt: str
    ) -> float:
        """
        Calculate CLIP score between image and prompt

        Args:
            image_path: Path to image
            prompt: Text prompt

        Returns:
            CLIP score (0-100)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Encode text
        text_input = clip.tokenize([prompt]).to(self.device)

        # Calculate features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            similarity = (image_features @ text_features.T).squeeze()

        # Convert to 0-100 scale
        score = float(similarity.cpu()) * 100
        return score

    def calculate_character_consistency(
        self,
        image_paths: List[Path]
    ) -> Dict[str, float]:
        """
        Calculate consistency across multiple images of same character

        Args:
            image_paths: List of image paths (variations)

        Returns:
            Dictionary with consistency metrics
        """
        if len(image_paths) < 2:
            return {
                "mean_similarity": 100.0,
                "std_similarity": 0.0,
                "min_similarity": 100.0,
                "max_similarity": 100.0
            }

        # Encode all images
        features = []
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feat = self.clip_model.encode_image(image_input)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                features.append(feat)

        # Stack features
        features = torch.cat(features, dim=0)

        # Calculate pairwise similarities
        similarities = []
        n = len(features)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float((features[i] @ features[j].T).cpu())
                similarities.append(sim * 100)  # Convert to 0-100 scale

        similarities = np.array(similarities)

        return {
            "mean_similarity": float(similarities.mean()),
            "std_similarity": float(similarities.std()),
            "min_similarity": float(similarities.min()),
            "max_similarity": float(similarities.max()),
        }

    def calculate_image_quality_metrics(
        self,
        image_path: Path
    ) -> Dict[str, float]:
        """
        Calculate basic image quality metrics

        Args:
            image_path: Path to image

        Returns:
            Dictionary with quality metrics
        """
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)

        # Calculate metrics
        metrics = {
            "resolution": image.width * image.height,
            "aspect_ratio": image.width / image.height,
            "mean_brightness": float(img_array.mean()),
            "std_brightness": float(img_array.std()),
            "saturation": float(img_array.std(axis=-1).mean()),
        }

        return metrics

    def evaluate_lora_output(
        self,
        output_dir: Path,
        metadata_path: Path = None,
    ) -> Dict:
        """
        Evaluate all images in a LoRA output directory

        Args:
            output_dir: Directory with generated images
            metadata_path: Path to generation metadata JSON

        Returns:
            Comprehensive evaluation results
        """
        print(f"\n{'='*80}")
        print(f"Evaluating LoRA Output: {output_dir.name}")
        print(f"{'='*80}\n")

        # Load metadata if available
        metadata = None
        if metadata_path and metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        elif (output_dir / "generation_metadata.json").exists():
            with open(output_dir / "generation_metadata.json", 'r') as f:
                metadata = json.load(f)

        # Find all images
        image_files = sorted(output_dir.glob("*.png")) + sorted(output_dir.glob("*.jpg"))

        if not image_files:
            print(f"⚠️  No images found in {output_dir}")
            return {}

        print(f"Found {len(image_files)} images")

        # Group images by prompt
        prompt_groups = {}
        for img_path in image_files:
            # Parse filename: img_00000_p000_v00.png
            # Extract prompt index
            parts = img_path.stem.split('_')
            if len(parts) >= 3:
                prompt_idx = int(parts[2][1:])  # p000 -> 0
                if prompt_idx not in prompt_groups:
                    prompt_groups[prompt_idx] = []
                prompt_groups[prompt_idx].append(img_path)

        # Evaluate each prompt group
        results = {
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(output_dir),
            "total_images": len(image_files),
            "num_prompts": len(prompt_groups),
            "per_prompt_results": [],
            "overall_metrics": {}
        }

        all_clip_scores = []
        all_consistency_scores = []

        print("\n📊 Evaluating per-prompt metrics...")
        for prompt_idx, images in tqdm(sorted(prompt_groups.items())):
            # Get prompt text from metadata
            prompt_text = "unknown"
            if metadata and "images" in metadata:
                for img_meta in metadata["images"]:
                    if img_meta.get("prompt_idx") == prompt_idx:
                        prompt_text = img_meta.get("prompt", "unknown")
                        break

            # Calculate CLIP scores for all images in this group
            clip_scores = []
            for img_path in images:
                score = self.calculate_clip_score(img_path, prompt_text)
                clip_scores.append(score)
                all_clip_scores.append(score)

            # Calculate character consistency within this prompt
            consistency = self.calculate_character_consistency(images)
            all_consistency_scores.append(consistency["mean_similarity"])

            # Calculate image quality for first image in group
            quality = self.calculate_image_quality_metrics(images[0])

            prompt_result = {
                "prompt_idx": prompt_idx,
                "prompt_text": prompt_text,
                "num_variations": len(images),
                "clip_score": {
                    "mean": float(np.mean(clip_scores)),
                    "std": float(np.std(clip_scores)),
                    "min": float(np.min(clip_scores)),
                    "max": float(np.max(clip_scores)),
                },
                "character_consistency": consistency,
                "image_quality": quality,
            }

            results["per_prompt_results"].append(prompt_result)

        # Calculate overall metrics
        results["overall_metrics"] = {
            "clip_score": {
                "mean": float(np.mean(all_clip_scores)),
                "std": float(np.std(all_clip_scores)),
                "min": float(np.min(all_clip_scores)),
                "max": float(np.max(all_clip_scores)),
            },
            "character_consistency": {
                "mean": float(np.mean(all_consistency_scores)),
                "std": float(np.std(all_consistency_scores)),
            }
        }

        return results

    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}\n")

        overall = results.get("overall_metrics", {})

        print(f"📊 Overall CLIP Score (Prompt Matching):")
        clip = overall.get("clip_score", {})
        print(f"  Mean:  {clip.get('mean', 0):.2f} / 100")
        print(f"  Std:   {clip.get('std', 0):.2f}")
        print(f"  Range: {clip.get('min', 0):.2f} - {clip.get('max', 0):.2f}")

        print(f"\n🎭 Character Consistency (across variations):")
        consistency = overall.get("character_consistency", {})
        print(f"  Mean:  {consistency.get('mean', 0):.2f} / 100")
        print(f"  Std:   {consistency.get('std', 0):.2f}")

        print(f"\n✅ Interpretation:")
        clip_mean = clip.get('mean', 0)
        if clip_mean >= 30:
            print(f"  CLIP Score: Excellent (≥30) - Strong prompt adherence")
        elif clip_mean >= 25:
            print(f"  CLIP Score: Good (25-30) - Adequate prompt adherence")
        else:
            print(f"  CLIP Score: Needs improvement (<25)")

        cons_mean = consistency.get('mean', 0)
        if cons_mean >= 85:
            print(f"  Consistency: Excellent (≥85%) - Very consistent character")
        elif cons_mean >= 75:
            print(f"  Consistency: Good (75-85%) - Fairly consistent")
        else:
            print(f"  Consistency: Variable (<75%) - Consider retraining")

        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA quality using quantitative metrics"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory with generated images"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Path to generation metadata JSON (optional)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = LoRAQualityEvaluator(device=args.device)

    # Run evaluation
    results = evaluator.evaluate_lora_output(
        args.output_dir,
        args.metadata
    )

    # Print summary
    evaluator.print_summary(results)

    # Save to JSON
    output_json = args.output_json or args.output_dir / "quality_evaluation.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✓ Full results saved to: {output_json}")

    # Print top and bottom performing prompts
    if results.get("per_prompt_results"):
        print(f"\n{'='*80}")
        print("TOP 5 BEST PERFORMING PROMPTS (by CLIP score)")
        print(f"{'='*80}\n")

        sorted_prompts = sorted(
            results["per_prompt_results"],
            key=lambda x: x["clip_score"]["mean"],
            reverse=True
        )

        for i, prompt in enumerate(sorted_prompts[:5], 1):
            print(f"{i}. Score: {prompt['clip_score']['mean']:.2f}")
            print(f"   Prompt: {prompt['prompt_text'][:80]}...")
            print()


if __name__ == "__main__":
    main()
