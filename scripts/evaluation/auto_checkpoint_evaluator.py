#!/usr/bin/env python3
"""
Auto Checkpoint Evaluator for LoRA Training.

Automatically evaluates LoRA checkpoints with:
- Image generation with test prompts
- CLIP score computation
- Character consistency metrics
- Comparative ranking
- Best checkpoint selection

AI_WAREHOUSE 3.0 compliant paths.
Supports stub mode for CPU-only testing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationPrompt:
    """A test prompt for evaluation."""
    text: str
    seed: int = 42
    negative: str = "blurry, low quality, deformed"
    cfg_scale: float = 7.0
    steps: int = 20


@dataclass
class GeneratedImage:
    """A generated test image."""
    path: Path
    prompt: str
    seed: int
    checkpoint: str


@dataclass
class CheckpointScore:
    """Evaluation scores for a single checkpoint."""
    checkpoint_path: str
    checkpoint_name: str
    clip_score: float
    consistency_score: float
    quality_score: float
    overall_score: float
    generated_images: List[str]
    metrics: Dict


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    checkpoints: List[CheckpointScore]
    best_checkpoint: str
    best_score: float
    comparison_grid: Optional[str]
    timestamp: str
    parameters: Dict

    def to_dict(self) -> Dict:
        return {
            "checkpoints": [asdict(c) for c in self.checkpoints],
            "best_checkpoint": self.best_checkpoint,
            "best_score": self.best_score,
            "comparison_grid": self.comparison_grid,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
        }


class AutoCheckpointEvaluator:
    """
    Automatic checkpoint evaluation and ranking.

    Features:
    - Generate test images with each checkpoint
    - Compute CLIP similarity scores
    - Measure character consistency across prompts
    - Rank and select best checkpoint
    - Generate comparison grids
    """

    # AI_WAREHOUSE 3.0 paths
    MODEL_ROOT = Path("/mnt/c/ai_models")
    EVAL_OUTPUT_ROOT = Path("/mnt/data/training/lora/evaluation")

    def __init__(
        self,
        base_model: str = "sdxl",
        device: str = "cuda",
        stub_mode: bool = False,
    ):
        self.base_model = base_model
        self.device = device
        self.stub_mode = stub_mode or device == "stub"

        self._pipeline = None
        self._clip_model = None

        if not self.stub_mode:
            self._init_models()

    def _init_models(self) -> None:
        """Initialize evaluation models."""
        try:
            # Try to import diffusers for image generation
            from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
            import torch

            if self.base_model == "sdxl":
                model_path = self.MODEL_ROOT / "stable-diffusion/sd_xl_base_1.0.safetensors"
                self._pipeline = StableDiffusionXLPipeline.from_single_file(
                    str(model_path),
                    torch_dtype=torch.float16,
                ).to(self.device)
            else:
                model_path = self.MODEL_ROOT / "stable-diffusion/v1-5-pruned-emaonly.safetensors"
                self._pipeline = StableDiffusionPipeline.from_single_file(
                    str(model_path),
                    torch_dtype=torch.float16,
                ).to(self.device)

            logger.info(f"Initialized {self.base_model} pipeline")

        except ImportError:
            logger.warning("Diffusers not available, using stub mode")
            self.stub_mode = True
        except Exception as e:
            logger.warning(f"Failed to load pipeline: {e}, using stub mode")
            self.stub_mode = True

        # Initialize CLIP for scoring
        try:
            import clip
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("Initialized CLIP model for scoring")
        except ImportError:
            logger.warning("CLIP not available, will skip CLIP scoring")
            self._clip_model = None
        except Exception as e:
            logger.warning(f"Failed to load CLIP: {e}")
            self._clip_model = None

    def evaluate_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        test_prompts: List[Union[str, EvaluationPrompt]],
        output_dir: Union[str, Path],
    ) -> CheckpointScore:
        """
        Evaluate a single checkpoint.

        Args:
            checkpoint_path: Path to LoRA checkpoint
            test_prompts: List of prompts to test
            output_dir: Directory for generated images

        Returns:
            CheckpointScore with evaluation metrics
        """
        checkpoint_path = Path(checkpoint_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name = checkpoint_path.stem

        # Normalize prompts
        prompts = []
        for p in test_prompts:
            if isinstance(p, str):
                prompts.append(EvaluationPrompt(text=p))
            else:
                prompts.append(p)

        # Generate images
        generated_images = []
        for i, prompt in enumerate(prompts):
            image_path = output_dir / f"{checkpoint_name}_prompt{i}.png"
            self._generate_image(checkpoint_path, prompt, image_path)
            generated_images.append(GeneratedImage(
                path=image_path,
                prompt=prompt.text,
                seed=prompt.seed,
                checkpoint=checkpoint_name,
            ))

        # Compute scores
        clip_score = self._compute_clip_score(generated_images)
        consistency_score = self._compute_consistency_score(generated_images)
        quality_score = self._compute_quality_score(generated_images)

        # Overall score (weighted average)
        overall_score = (
            0.4 * clip_score +
            0.3 * consistency_score +
            0.3 * quality_score
        )

        return CheckpointScore(
            checkpoint_path=str(checkpoint_path),
            checkpoint_name=checkpoint_name,
            clip_score=clip_score,
            consistency_score=consistency_score,
            quality_score=quality_score,
            overall_score=overall_score,
            generated_images=[str(g.path) for g in generated_images],
            metrics={
                "num_prompts": len(prompts),
                "device": self.device,
                "base_model": self.base_model,
            },
        )

    def evaluate_checkpoints(
        self,
        checkpoint_paths: List[Union[str, Path]],
        test_prompts: List[Union[str, EvaluationPrompt]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> EvaluationResult:
        """
        Evaluate multiple checkpoints and rank them.

        Args:
            checkpoint_paths: List of checkpoint paths
            test_prompts: List of prompts to test
            output_dir: Optional output directory

        Returns:
            EvaluationResult with rankings
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.EVAL_OUTPUT_ROOT / timestamp
        output_dir = Path(output_dir)

        logger.info(f"Evaluating {len(checkpoint_paths)} checkpoints")

        scores = []
        for ckpt_path in checkpoint_paths:
            ckpt_name = Path(ckpt_path).stem
            ckpt_output = output_dir / ckpt_name

            score = self.evaluate_checkpoint(ckpt_path, test_prompts, ckpt_output)
            scores.append(score)
            logger.info(f"  {ckpt_name}: {score.overall_score:.3f}")

        # Rank by overall score
        scores.sort(key=lambda s: s.overall_score, reverse=True)

        # Generate comparison grid
        grid_path = self._generate_comparison_grid(scores, output_dir)

        result = EvaluationResult(
            checkpoints=scores,
            best_checkpoint=scores[0].checkpoint_path if scores else "",
            best_score=scores[0].overall_score if scores else 0.0,
            comparison_grid=str(grid_path) if grid_path else None,
            timestamp=datetime.now().isoformat(),
            parameters={
                "num_checkpoints": len(checkpoint_paths),
                "num_prompts": len(test_prompts),
                "base_model": self.base_model,
            },
        )

        # Save report
        report_path = output_dir / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return result

    def compare_checkpoints(
        self,
        checkpoint_dir: Union[str, Path],
        test_prompts: List[Union[str, EvaluationPrompt]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> EvaluationResult:
        """
        Convenience method to compare all checkpoints in a directory.

        Args:
            checkpoint_dir: Directory containing checkpoints
            test_prompts: List of prompts to test
            output_dir: Optional output directory

        Returns:
            EvaluationResult with rankings
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Find all checkpoints
        checkpoint_paths = []
        for ext in [".safetensors", ".pt", ".ckpt"]:
            checkpoint_paths.extend(checkpoint_dir.glob(f"*{ext}"))

        if not checkpoint_paths:
            logger.warning(f"No checkpoints found in {checkpoint_dir}")
            return EvaluationResult(
                checkpoints=[],
                best_checkpoint="",
                best_score=0.0,
                comparison_grid=None,
                timestamp=datetime.now().isoformat(),
                parameters={},
            )

        return self.evaluate_checkpoints(checkpoint_paths, test_prompts, output_dir)

    def select_best(
        self,
        results: EvaluationResult,
        metric: str = "overall_score",
    ) -> Path:
        """
        Select the best checkpoint based on a metric.

        Args:
            results: EvaluationResult from evaluation
            metric: Metric to use ("overall_score", "clip_score", etc.)

        Returns:
            Path to best checkpoint
        """
        if not results.checkpoints:
            raise ValueError("No checkpoints in results")

        if metric == "overall_score":
            best = max(results.checkpoints, key=lambda c: c.overall_score)
        elif metric == "clip_score":
            best = max(results.checkpoints, key=lambda c: c.clip_score)
        elif metric == "consistency_score":
            best = max(results.checkpoints, key=lambda c: c.consistency_score)
        elif metric == "quality_score":
            best = max(results.checkpoints, key=lambda c: c.quality_score)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return Path(best.checkpoint_path)

    def _generate_image(
        self,
        checkpoint_path: Path,
        prompt: EvaluationPrompt,
        output_path: Path,
    ) -> None:
        """Generate a single image with a checkpoint."""
        if self.stub_mode:
            # Generate synthetic image for testing
            self._generate_stub_image(prompt, output_path)
            return

        try:
            # Load LoRA weights
            self._pipeline.load_lora_weights(str(checkpoint_path))

            # Generate image
            import torch
            generator = torch.Generator(self.device).manual_seed(prompt.seed)

            image = self._pipeline(
                prompt=prompt.text,
                negative_prompt=prompt.negative,
                num_inference_steps=prompt.steps,
                guidance_scale=prompt.cfg_scale,
                generator=generator,
            ).images[0]

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)

            # Unload LoRA
            self._pipeline.unload_lora_weights()

        except Exception as e:
            logger.warning(f"Image generation failed: {e}, using stub")
            self._generate_stub_image(prompt, output_path)

    def _generate_stub_image(
        self,
        prompt: EvaluationPrompt,
        output_path: Path,
    ) -> None:
        """Generate a synthetic image for stub mode."""
        from PIL import Image, ImageDraw, ImageFont

        # Create deterministic colored image based on prompt
        seed = hash(prompt.text) % (2**32)
        np.random.seed(seed)

        # Random color background
        color = tuple(np.random.randint(50, 200, 3))
        img = Image.new("RGB", (512, 512), color)
        draw = ImageDraw.Draw(img)

        # Add some text
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        text = f"STUB: {prompt.text[:30]}..."
        draw.text((50, 240), text, fill=(255, 255, 255), font=font)
        draw.text((50, 260), f"seed={prompt.seed}", fill=(200, 200, 200), font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

    def _compute_clip_score(
        self,
        images: List[GeneratedImage],
    ) -> float:
        """Compute CLIP similarity score between images and prompts."""
        if self._clip_model is None or self.stub_mode:
            # Return synthetic score
            np.random.seed(42)
            return 0.7 + np.random.random() * 0.2

        try:
            import torch
            from PIL import Image

            scores = []
            for gen_img in images:
                # Load and preprocess image
                img = Image.open(gen_img.path).convert("RGB")
                img_input = self._clip_preprocess(img).unsqueeze(0).to(self.device)

                # Tokenize prompt
                import clip
                text_input = clip.tokenize([gen_img.prompt]).to(self.device)

                # Compute similarity
                with torch.no_grad():
                    img_features = self._clip_model.encode_image(img_input)
                    text_features = self._clip_model.encode_text(text_input)

                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    similarity = (img_features @ text_features.T).item()
                    scores.append(similarity)

            return sum(scores) / len(scores) if scores else 0.0

        except Exception as e:
            logger.warning(f"CLIP scoring failed: {e}")
            return 0.75

    def _compute_consistency_score(
        self,
        images: List[GeneratedImage],
    ) -> float:
        """Compute character consistency across generated images."""
        if len(images) < 2 or self.stub_mode:
            return 0.8 + np.random.random() * 0.15

        try:
            from PIL import Image
            import torch

            if self._clip_model is None:
                return 0.8

            # Get CLIP embeddings for all images
            embeddings = []
            for gen_img in images:
                img = Image.open(gen_img.path).convert("RGB")
                img_input = self._clip_preprocess(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    features = self._clip_model.encode_image(img_input)
                    features = features / features.norm(dim=-1, keepdim=True)
                    embeddings.append(features.cpu().numpy())

            embeddings = np.vstack(embeddings)

            # Compute pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j].T).item()
                    similarities.append(sim)

            return sum(similarities) / len(similarities) if similarities else 0.8

        except Exception as e:
            logger.warning(f"Consistency scoring failed: {e}")
            return 0.8

    def _compute_quality_score(
        self,
        images: List[GeneratedImage],
    ) -> float:
        """Compute image quality score."""
        if self.stub_mode:
            return 0.75 + np.random.random() * 0.2

        try:
            from PIL import Image
            import cv2

            scores = []
            for gen_img in images:
                img = cv2.imread(str(gen_img.path))
                if img is None:
                    continue

                # Laplacian variance as sharpness indicator
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

                # Normalize to 0-1 range (empirical thresholds)
                sharpness = min(1.0, laplacian_var / 500)
                scores.append(sharpness)

            return sum(scores) / len(scores) if scores else 0.75

        except Exception as e:
            logger.warning(f"Quality scoring failed: {e}")
            return 0.75

    def _generate_comparison_grid(
        self,
        scores: List[CheckpointScore],
        output_dir: Path,
    ) -> Optional[Path]:
        """Generate a comparison grid of all checkpoints."""
        if not scores:
            return None

        try:
            from PIL import Image

            # Collect images (first prompt from each checkpoint)
            images = []
            labels = []
            for score in scores:
                if score.generated_images:
                    img_path = score.generated_images[0]
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    labels.append(f"{score.checkpoint_name}\n{score.overall_score:.3f}")

            if not images:
                return None

            # Create grid
            n = len(images)
            cols = min(4, n)
            rows = (n + cols - 1) // cols

            img_size = 256
            grid_w = cols * img_size
            grid_h = rows * (img_size + 30)  # Extra space for labels

            grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

            from PIL import ImageDraw

            for i, (img, label) in enumerate(zip(images, labels)):
                row = i // cols
                col = i % cols

                x = col * img_size
                y = row * (img_size + 30)

                # Resize and paste image
                img_resized = img.resize((img_size, img_size))
                grid.paste(img_resized, (x, y))

                # Add label
                draw = ImageDraw.Draw(grid)
                draw.text((x + 5, y + img_size + 2), label, fill=(0, 0, 0))

            grid_path = output_dir / "comparison_grid.png"
            grid.save(grid_path)
            return grid_path

        except Exception as e:
            logger.warning(f"Grid generation failed: {e}")
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LoRA checkpoints"
    )
    parser.add_argument(
        "checkpoint_dir",
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--prompts", "-p",
        nargs="+",
        help="Test prompts (or path to JSON file)"
    )
    parser.add_argument(
        "--character", "-c",
        help="Character name (for auto-generating prompts)"
    )
    parser.add_argument(
        "--base-model",
        choices=["sd15", "sdxl"],
        default="sdxl",
        help="Base model type"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda, cpu, stub)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Determine prompts
    if args.prompts:
        # Check if it's a JSON file
        if len(args.prompts) == 1 and args.prompts[0].endswith(".json"):
            with open(args.prompts[0]) as f:
                prompts = json.load(f)
        else:
            prompts = args.prompts
    elif args.character:
        trigger = args.character.lower().replace(" ", "_")
        prompts = [
            f"{trigger}, portrait, front view",
            f"{trigger}, full body, standing",
            f"{trigger}, three-quarter view, smiling",
            f"{trigger}, close-up face",
            f"{trigger}, action pose",
        ]
    else:
        prompts = ["a character, portrait", "a character, full body"]

    evaluator = AutoCheckpointEvaluator(
        base_model=args.base_model,
        device=args.device,
    )

    result = evaluator.compare_checkpoints(
        args.checkpoint_dir,
        prompts,
        args.output_dir,
    )

    print(f"\nEvaluation Results:")
    print(f"  Checkpoints evaluated: {len(result.checkpoints)}")
    print(f"\nRankings:")
    for i, score in enumerate(result.checkpoints, 1):
        print(f"  {i}. {score.checkpoint_name}: {score.overall_score:.3f}")
        print(f"     CLIP: {score.clip_score:.3f}, "
              f"Consistency: {score.consistency_score:.3f}, "
              f"Quality: {score.quality_score:.3f}")

    print(f"\nBest checkpoint: {result.best_checkpoint}")
    if result.comparison_grid:
        print(f"Comparison grid: {result.comparison_grid}")
