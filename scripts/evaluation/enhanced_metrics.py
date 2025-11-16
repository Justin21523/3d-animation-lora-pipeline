#!/usr/bin/env python3
"""
Enhanced Evaluation Metrics for LoRA Quality Assessment
Implements LPIPS, CLIP consistency, and other advanced metrics
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Optional imports for advanced metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")


class EnhancedMetrics:
    """
    Advanced metrics for LoRA evaluation:
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - CLIP consistency (text-image alignment)
    - CLIP diversity (avoiding mode collapse)
    - Image quality metrics (brightness, contrast, saturation)
    - Character consistency (face similarity across samples)
    """

    def __init__(self, device="cuda"):
        self.device = device

        # Load LPIPS model (perceptual similarity)
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.lpips_model.eval()
        else:
            self.lpips_model = None

        # Load CLIP model (text-image consistency)
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
        else:
            self.clip_model = None
            self.clip_preprocess = None

        # Image preprocessing for LPIPS
        self.lpips_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def calculate_lpips_diversity(self, image_paths: List[str]) -> Dict[str, float]:
        """
        Calculate LPIPS diversity between generated images
        Higher LPIPS = more diverse (avoiding mode collapse)

        Returns:
            - mean_lpips: Average pairwise LPIPS distance
            - std_lpips: Standard deviation of LPIPS distances
            - min_lpips: Minimum distance (closest pair)
            - max_lpips: Maximum distance (most different pair)
        """
        if not LPIPS_AVAILABLE or self.lpips_model is None:
            return {
                "mean_lpips": 0.0,
                "std_lpips": 0.0,
                "min_lpips": 0.0,
                "max_lpips": 0.0,
            }

        # Load and preprocess images
        images = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.lpips_transform(img).unsqueeze(0).to(self.device)
            images.append(img_tensor)

        # Calculate pairwise LPIPS distances
        distances = []
        n = len(images)

        with torch.no_grad():
            for i in range(n):
                for j in range(i + 1, n):
                    dist = self.lpips_model(images[i], images[j]).item()
                    distances.append(dist)

        if not distances:
            return {
                "mean_lpips": 0.0,
                "std_lpips": 0.0,
                "min_lpips": 0.0,
                "max_lpips": 0.0,
            }

        return {
            "mean_lpips": float(np.mean(distances)),
            "std_lpips": float(np.std(distances)),
            "min_lpips": float(np.min(distances)),
            "max_lpips": float(np.max(distances)),
        }

    def calculate_clip_consistency(
        self,
        image_paths: List[str],
        prompts: List[str]
    ) -> Dict[str, float]:
        """
        Calculate CLIP text-image consistency
        Higher score = better alignment between prompt and generated image

        Returns:
            - mean_clip_score: Average CLIP similarity
            - std_clip_score: Consistency of scores
            - min_clip_score: Worst alignment
            - max_clip_score: Best alignment
        """
        if not CLIP_AVAILABLE or self.clip_model is None:
            return {
                "mean_clip_score": 0.0,
                "std_clip_score": 0.0,
                "min_clip_score": 0.0,
                "max_clip_score": 0.0,
            }

        scores = []

        with torch.no_grad():
            for img_path, prompt in zip(image_paths, prompts):
                # Load and preprocess image
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)

                # Tokenize text
                text_tokens = clip.tokenize([prompt]).to(self.device)

                # Get features
                image_features = self.clip_model.encode_image(img_tensor)
                text_features = self.clip_model.encode_text(text_tokens)

                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                # Calculate cosine similarity
                similarity = (image_features @ text_features.T).item()
                scores.append(similarity)

        if not scores:
            return {
                "mean_clip_score": 0.0,
                "std_clip_score": 0.0,
                "min_clip_score": 0.0,
                "max_clip_score": 0.0,
            }

        return {
            "mean_clip_score": float(np.mean(scores)),
            "std_clip_score": float(np.std(scores)),
            "min_clip_score": float(np.min(scores)),
            "max_clip_score": float(np.max(scores)),
        }

    def calculate_image_quality_metrics(self, image_paths: List[str]) -> Dict[str, float]:
        """
        Calculate basic image quality metrics:
        - Brightness (mean pixel value)
        - Contrast (std of pixel values)
        - Saturation (mean saturation in HSV)

        For Pixar style, we expect:
        - Brightness: 0.4-0.6 (moderate)
        - Contrast: 0.15-0.25 (low, smooth shading)
        - Saturation: 0.3-0.5 (moderate, not oversaturated)
        """
        brightness_values = []
        contrast_values = []
        saturation_values = []

        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)

            # Brightness (mean of RGB)
            brightness = np.mean(img_array) / 255.0
            brightness_values.append(brightness)

            # Contrast (std of RGB)
            contrast = np.std(img_array) / 255.0
            contrast_values.append(contrast)

            # Saturation (mean S channel in HSV)
            img_hsv = img.convert("HSV")
            hsv_array = np.array(img_hsv)
            saturation = np.mean(hsv_array[:, :, 1]) / 255.0
            saturation_values.append(saturation)

        return {
            # Brightness metrics
            "mean_brightness": float(np.mean(brightness_values)),
            "std_brightness": float(np.std(brightness_values)),
            "brightness_in_range": sum(0.4 <= b <= 0.6 for b in brightness_values) / len(brightness_values),

            # Contrast metrics
            "mean_contrast": float(np.mean(contrast_values)),
            "std_contrast": float(np.std(contrast_values)),
            "contrast_in_range": sum(0.15 <= c <= 0.25 for c in contrast_values) / len(contrast_values),

            # Saturation metrics
            "mean_saturation": float(np.mean(saturation_values)),
            "std_saturation": float(np.std(saturation_values)),
            "saturation_in_range": sum(0.3 <= s <= 0.5 for s in saturation_values) / len(saturation_values),
        }

    def calculate_pixar_style_score(self, image_quality_metrics: Dict[str, float]) -> float:
        """
        Calculate overall Pixar-style adherence score (0-1)

        Weighted combination of:
        - Brightness in optimal range (weight: 0.3)
        - Contrast in optimal range (weight: 0.4) - most important for Pixar
        - Saturation in optimal range (weight: 0.2)
        - Consistency (low std, weight: 0.1)
        """
        brightness_score = image_quality_metrics.get("brightness_in_range", 0)
        contrast_score = image_quality_metrics.get("contrast_in_range", 0)
        saturation_score = image_quality_metrics.get("saturation_in_range", 0)

        # Consistency bonus (lower std = more consistent = better)
        brightness_std = image_quality_metrics.get("std_brightness", 0)
        contrast_std = image_quality_metrics.get("std_contrast", 0)

        # Normalize std to 0-1 score (lower std = higher score)
        brightness_consistency = max(0, 1 - brightness_std / 0.1)  # Expect std < 0.1
        contrast_consistency = max(0, 1 - contrast_std / 0.05)     # Expect std < 0.05
        consistency_score = (brightness_consistency + contrast_consistency) / 2

        # Weighted combination
        pixar_score = (
            0.3 * brightness_score +
            0.4 * contrast_score +
            0.2 * saturation_score +
            0.1 * consistency_score
        )

        return float(pixar_score)

    def evaluate_checkpoint(
        self,
        image_paths: List[str],
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """
        Comprehensive evaluation of a checkpoint

        Args:
            image_paths: List of paths to generated images
            prompts: Optional list of prompts used (for CLIP consistency)

        Returns:
            Dictionary with all metrics
        """
        results = {}

        # Image quality metrics (always available)
        quality_metrics = self.calculate_image_quality_metrics(image_paths)
        results["quality"] = quality_metrics

        # Pixar style score
        results["pixar_style_score"] = self.calculate_pixar_style_score(quality_metrics)

        # LPIPS diversity (if available)
        if LPIPS_AVAILABLE:
            lpips_metrics = self.calculate_lpips_diversity(image_paths)
            results["lpips"] = lpips_metrics

        # CLIP consistency (if available and prompts provided)
        if CLIP_AVAILABLE and prompts is not None:
            clip_metrics = self.calculate_clip_consistency(image_paths, prompts)
            results["clip"] = clip_metrics

        return results

    def save_metrics(self, metrics: Dict, output_path: str):
        """Save metrics to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"✅ Metrics saved to: {output_path}")


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate enhanced metrics for LoRA evaluation")
    parser.add_argument("--image-dir", required=True, help="Directory containing generated images")
    parser.add_argument("--prompts-file", help="Optional JSON file with prompts")
    parser.add_argument("--output", required=True, help="Output JSON file for metrics")
    parser.add_argument("--device", default="cuda", help="Device to use")

    args = parser.parse_args()

    # Load image paths
    image_dir = Path(args.image_dir)
    image_paths = sorted(str(p) for p in image_dir.glob("*.png"))

    if not image_paths:
        print(f"❌ No images found in {image_dir}")
        return

    print(f"📊 Evaluating {len(image_paths)} images...")

    # Load prompts if provided
    prompts = None
    if args.prompts_file:
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
            prompts = [p["prompt"] for p in prompts_data]

    # Calculate metrics
    evaluator = EnhancedMetrics(device=args.device)
    metrics = evaluator.evaluate_checkpoint(image_paths, prompts)

    # Save results
    evaluator.save_metrics(metrics, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("ENHANCED METRICS SUMMARY")
    print("=" * 60)
    print(f"Pixar Style Score: {metrics['pixar_style_score']:.3f}")
    print(f"Brightness: {metrics['quality']['mean_brightness']:.3f} ± {metrics['quality']['std_brightness']:.3f}")
    print(f"Contrast: {metrics['quality']['mean_contrast']:.3f} ± {metrics['quality']['std_contrast']:.3f}")

    if "lpips" in metrics:
        print(f"LPIPS Diversity: {metrics['lpips']['mean_lpips']:.3f}")

    if "clip" in metrics:
        print(f"CLIP Consistency: {metrics['clip']['mean_clip_score']:.3f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
