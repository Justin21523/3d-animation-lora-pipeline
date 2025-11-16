#!/usr/bin/env python3
"""
Test LoRA Composition (Multi-LoRA Loading)
Generate images using multiple LoRA models simultaneously.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.utils.logger import setup_logger


class LoRACompositionTester:
    """Test multiple LoRA models loaded simultaneously."""

    def __init__(self, base_model: str, device: str = "cuda"):
        """
        Initialize tester.

        Args:
            base_model: Path to base model
            device: Device for inference
        """
        self.device = device
        logging.info(f"Loading base model: {base_model}")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)

        # Use DPM++ 2M Karras scheduler (faster, high quality)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True
        )

        self.loaded_loras: Dict[str, Path] = {}

    def load_loras(self, lora_configs: List[Dict[str, any]]):
        """
        Load multiple LoRA models.

        Args:
            lora_configs: List of dicts with 'path', 'name', 'weight'
        """
        adapter_names = []
        adapter_weights = []

        for config in lora_configs:
            lora_path = Path(config["path"])
            adapter_name = config["name"]
            weight = config.get("weight", 1.0)

            if not lora_path.exists():
                logging.warning(f"LoRA not found: {lora_path}, skipping")
                continue

            logging.info(f"Loading LoRA: {adapter_name} from {lora_path} (weight={weight})")

            self.pipe.load_lora_weights(
                str(lora_path.parent),
                weight_name=lora_path.name,
                adapter_name=adapter_name
            )

            adapter_names.append(adapter_name)
            adapter_weights.append(weight)
            self.loaded_loras[adapter_name] = lora_path

        # Set adapters with weights
        if adapter_names:
            self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
            logging.info(f"✅ Loaded {len(adapter_names)} LoRAs: {adapter_names}")
            logging.info(f"   Weights: {adapter_weights}")
        else:
            logging.warning("No LoRAs loaded!")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512
    ) -> List[Image.Image]:
        """
        Generate images with loaded LoRAs.

        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            num_images: Number of images to generate
            steps: Number of inference steps
            guidance_scale: CFG scale
            seed: Random seed (optional)
            width: Image width
            height: Image height

        Returns:
            List of generated images
        """
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        logging.info(f"Generating {num_images} images...")
        logging.info(f"  Prompt: {prompt[:100]}...")
        logging.info(f"  Steps: {steps}, CFG: {guidance_scale}, Seed: {seed}")

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height
        ).images

        return images

    def test_composition(
        self,
        lora_configs: List[Dict],
        prompts: List[str],
        output_dir: Path,
        negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy",
        num_samples: int = 4,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed_start: int = 42,
        width: int = 512,
        height: int = 512
    ) -> dict:
        """
        Test LoRA composition with multiple prompts.

        Args:
            lora_configs: LoRA configuration list
            prompts: List of test prompts
            output_dir: Output directory
            negative_prompt: Negative prompt
            num_samples: Samples per prompt
            steps: Inference steps
            guidance_scale: CFG scale
            seed_start: Starting seed
            width: Image width
            height: Image height

        Returns:
            Test results metadata
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load LoRAs
        self.load_loras(lora_configs)

        results = {
            "lora_configs": lora_configs,
            "base_model": str(self.pipe.name_or_path),
            "settings": {
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height
            },
            "prompts": []
        }

        # Test each prompt
        for prompt_idx, prompt in enumerate(prompts):
            logging.info(f"\n{'='*80}")
            logging.info(f"Testing prompt {prompt_idx+1}/{len(prompts)}")
            logging.info(f"{'='*80}")

            prompt_dir = output_dir / f"prompt_{prompt_idx:02d}"
            prompt_dir.mkdir(exist_ok=True)

            # Save prompt
            with open(prompt_dir / "prompt.txt", 'w') as f:
                f.write(prompt)

            prompt_results = {
                "prompt": prompt,
                "samples": []
            }

            # Generate samples
            for sample_idx in range(num_samples):
                seed = seed_start + prompt_idx * num_samples + sample_idx

                images = self.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images=1,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    width=width,
                    height=height
                )

                # Save image
                image = images[0]
                image_path = prompt_dir / f"sample_{sample_idx:02d}_seed{seed}.png"
                image.save(image_path)

                prompt_results["samples"].append({
                    "path": str(image_path),
                    "seed": seed
                })

                logging.info(f"  ✅ Sample {sample_idx+1}/{num_samples} saved: {image_path.name}")

            # Create grid
            self._create_grid(
                [Image.open(s["path"]) for s in prompt_results["samples"]],
                prompt_dir / "grid.png"
            )

            results["prompts"].append(prompt_results)

        # Save metadata
        metadata_path = output_dir / "composition_test_results.json"
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"\n✅ Test completed! Results saved to: {output_dir}")
        logging.info(f"   Metadata: {metadata_path}")

        return results

    def _create_grid(self, images: List[Image.Image], output_path: Path):
        """Create image grid."""
        n = len(images)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        width, height = images[0].size
        grid = Image.new('RGB', (width * cols, height * rows), color=(255, 255, 255))

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            grid.paste(img, (col * width, row * height))

        grid.save(output_path)
        logging.info(f"  📊 Grid saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Test LoRA composition (multi-LoRA loading)"
    )

    # LoRA paths
    parser.add_argument("--character-lora", type=Path, default=None,
                       help="Character LoRA path")
    parser.add_argument("--background-lora", type=Path, default=None,
                       help="Background LoRA path")
    parser.add_argument("--pose-lora", type=Path, default=None,
                       help="Pose LoRA path")
    parser.add_argument("--expression-lora", type=Path, default=None,
                       help="Expression LoRA path")
    parser.add_argument("--style-lora", type=Path, default=None,
                       help="Style LoRA path")

    # LoRA weights
    parser.add_argument("--character-weight", type=float, default=1.0,
                       help="Character LoRA weight")
    parser.add_argument("--background-weight", type=float, default=0.8,
                       help="Background LoRA weight")
    parser.add_argument("--pose-weight", type=float, default=0.7,
                       help="Pose LoRA weight")
    parser.add_argument("--expression-weight", type=float, default=0.6,
                       help="Expression LoRA weight")
    parser.add_argument("--style-weight", type=float, default=0.9,
                       help="Style LoRA weight")

    # Generation settings
    parser.add_argument("--base-model", type=str, required=True,
                       help="Base model path")
    parser.add_argument("--prompts", type=str, nargs="+", required=True,
                       help="Test prompts")
    parser.add_argument("--negative-prompt", type=str,
                       default="blurry, low quality, distorted, ugly, bad anatomy",
                       help="Negative prompt")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory")
    parser.add_argument("--num-samples", type=int, default=4,
                       help="Samples per prompt")
    parser.add_argument("--steps", type=int, default=30,
                       help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                       help="CFG scale")
    parser.add_argument("--seed-start", type=int, default=42,
                       help="Starting seed")
    parser.add_argument("--width", type=int, default=512,
                       help="Image width")
    parser.add_argument("--height", type=int, default=512,
                       help="Image height")
    parser.add_argument("--device", default="cuda",
                       help="Device for inference")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(
        "lora_composition",
        log_file=args.output_dir / "composition_test.log",
        console_level=logging.INFO
    )

    logger.info("=" * 80)
    logger.info("LoRA Composition Testing")
    logger.info("=" * 80)

    # Build LoRA configs
    lora_configs = []

    if args.character_lora:
        lora_configs.append({
            "path": str(args.character_lora),
            "name": "character",
            "weight": args.character_weight
        })

    if args.background_lora:
        lora_configs.append({
            "path": str(args.background_lora),
            "name": "background",
            "weight": args.background_weight
        })

    if args.pose_lora:
        lora_configs.append({
            "path": str(args.pose_lora),
            "name": "pose",
            "weight": args.pose_weight
        })

    if args.expression_lora:
        lora_configs.append({
            "path": str(args.expression_lora),
            "name": "expression",
            "weight": args.expression_weight
        })

    if args.style_lora:
        lora_configs.append({
            "path": str(args.style_lora),
            "name": "style",
            "weight": args.style_weight
        })

    if not lora_configs:
        logger.error("No LoRAs specified! Provide at least one LoRA.")
        return

    logger.info(f"Testing {len(lora_configs)} LoRAs:")
    for cfg in lora_configs:
        logger.info(f"  - {cfg['name']}: {cfg['path']} (weight={cfg['weight']})")

    # Initialize tester
    tester = LoRACompositionTester(base_model=args.base_model, device=args.device)

    # Run test
    results = tester.test_composition(
        lora_configs=lora_configs,
        prompts=args.prompts,
        output_dir=args.output_dir,
        negative_prompt=args.negative_prompt,
        num_samples=args.num_samples,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed_start=args.seed_start,
        width=args.width,
        height=args.height
    )

    logger.info("\n" + "=" * 80)
    logger.info("✅ LoRA Composition Test Completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
