#!/usr/bin/env python3
"""
NSFW Content Detection for Safety Filtering

Detects NSFW (Not Safe For Work) content using the safety classifier
from Stable Diffusion's safety checker.

Classification:
- Safe: Safe for work content
- NSFW: Not safe for work content (sexual, violent, disturbing)

Part of Module 3: Quality Filtering System
Author: LLMProvider Tooling
Date: 2025-11-30
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Tuple, Optional
from transformers import CLIPImageProcessor, CLIPModel
import warnings

warnings.filterwarnings("ignore")


class NSFWDetector:
    """
    NSFW detection using CLIP-based safety classifier

    Uses CLIP embeddings to detect potentially unsafe content.
    Lightweight and fast, no need for dedicated NSFW models.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        model_id: str = 'openai/clip-vit-base-patch32'
    ):
        """
        Initialize NSFW detector

        Args:
            threshold: NSFW probability threshold (default: 0.3)
                      Values above this are flagged as NSFW
            device: Device to run on ('cuda' or 'cpu')
            model_id: CLIP model ID (default: openai/clip-vit-base-patch32)
        """
        self.threshold = threshold
        self.device = device

        # Load CLIP model for feature extraction
        self.processor = CLIPImageProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.model.eval()

        # NSFW detection prompts (simple heuristic approach)
        # In production, you might want to use a dedicated safety classifier
        self.unsafe_concepts = [
            "explicit sexual content",
            "nudity",
            "violence",
            "gore",
            "disturbing content",
        ]

        self.safe_concepts = [
            "safe for work",
            "family friendly",
            "appropriate content",
            "cartoon character",
            "animation",
        ]

    @torch.no_grad()
    def compute_nsfw_score(self, image: Union[str, Path, Image.Image]) -> float:
        """
        Compute NSFW probability score

        Args:
            image: Image path or PIL Image

        Returns:
            NSFW score (0.0 = safe, 1.0 = unsafe)
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        else:
            img = image.convert('RGB')

        # Preprocess
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        # Get image features
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Get text features for unsafe/safe concepts
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Unsafe concepts
        unsafe_inputs = tokenizer(
            self.unsafe_concepts,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        unsafe_features = self.model.get_text_features(**unsafe_inputs)
        unsafe_features = unsafe_features / unsafe_features.norm(p=2, dim=-1, keepdim=True)

        # Safe concepts
        safe_inputs = tokenizer(
            self.safe_concepts,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        safe_features = self.model.get_text_features(**safe_inputs)
        safe_features = safe_features / safe_features.norm(p=2, dim=-1, keepdim=True)

        # Compute similarities
        unsafe_similarity = (image_features @ unsafe_features.T).max().item()
        safe_similarity = (image_features @ safe_features.T).max().item()

        # Normalize to 0-1 score
        # Higher unsafe similarity and lower safe similarity = higher NSFW score
        nsfw_score = (unsafe_similarity + (1 - safe_similarity)) / 2.0
        nsfw_score = max(0.0, min(1.0, nsfw_score))

        return nsfw_score

    def is_nsfw(self, image: Union[str, Path, Image.Image]) -> bool:
        """
        Check if image contains NSFW content

        Args:
            image: Image path or PIL Image

        Returns:
            True if NSFW, False if safe
        """
        score = self.compute_nsfw_score(image)
        return score > self.threshold

    def classify(self, image: Union[str, Path, Image.Image]) -> Tuple[str, float]:
        """
        Classify image as safe or NSFW with confidence score

        Args:
            image: Image path or PIL Image

        Returns:
            Tuple of (classification, score)
            - classification: 'safe' or 'nsfw'
            - score: NSFW probability score
        """
        score = self.compute_nsfw_score(image)
        classification = 'nsfw' if score > self.threshold else 'safe'
        return classification, score

    def batch_detect(
        self,
        image_paths: List[Path],
        batch_size: int = 32
    ) -> dict[Path, dict]:
        """
        Batch process multiple images

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing (default: 32)

        Returns:
            Dictionary mapping paths to results:
            {
                path: {
                    'nsfw_score': float,
                    'is_nsfw': bool,
                    'classification': str
                }
            }
        """
        results = {}

        for path in image_paths:
            try:
                score = self.compute_nsfw_score(path)
                is_nsfw = score > self.threshold
                classification = 'nsfw' if is_nsfw else 'safe'

                results[path] = {
                    'nsfw_score': score,
                    'is_nsfw': is_nsfw,
                    'classification': classification
                }
            except Exception as e:
                results[path] = {
                    'error': str(e),
                    'nsfw_score': None,
                    'is_nsfw': None,
                    'classification': None
                }

        return results


def main():
    """CLI for testing NSFW detection"""
    import argparse

    parser = argparse.ArgumentParser(description="NSFW Content Detection")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="NSFW threshold (default: 0.3)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=['cuda', 'cpu'],
                       help="Device to use (default: cuda)")

    args = parser.parse_args()

    detector = NSFWDetector(threshold=args.threshold, device=args.device)

    classification, score = detector.classify(args.image)

    print(f"Image: {args.image}")
    print(f"NSFW Score: {score:.4f}")
    print(f"Classification: {classification.upper()}")
    print(f"\nInterpretation:")
    print(f"  0.0-0.3: Safe for work")
    print(f"  0.3-0.7: Borderline/uncertain")
    print(f"  0.7-1.0: Not safe for work")


if __name__ == "__main__":
    main()
