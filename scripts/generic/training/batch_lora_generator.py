#!/usr/bin/env python3
"""
Batch LoRA Testing and Image Generation
Generate large batches of images using trained LoRA models to evaluate quality
"""

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from pathlib import Path
import argparse
from typing import List, Dict, Optional
import json
from datetime import datetime
from tqdm import tqdm
import gc


class BatchLoRAGenerator:
    """Generate batches of images using LoRA models for testing and evaluation"""

    def __init__(
        self,
        base_model_path: str,
        lora_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the generator

        Args:
            base_model_path: Path to base Stable Diffusion model
            lora_path: Path to trained LoRA weights
            device: Device to use (cuda/cpu)
            dtype: Data type for inference
        """
        self.device = device
        self.dtype = dtype

        print(f"Loading base model: {base_model_path}")

        # Check if it's a single file (.safetensors/.ckpt) or a directory
        if Path(base_model_path).is_file():
            # Use from_single_file for .safetensors/.ckpt files
            self.pipe = StableDiffusionPipeline.from_single_file(
                base_model_path,
                torch_dtype=dtype,
                safety_checker=None,
            )
        else:
            # Use from_pretrained for directories or HF model IDs
            self.pipe = StableDiffusionPipeline.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )

        # Use DPM++ solver for faster/better sampling
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe = self.pipe.to(device)

        # Load LoRA weights
        print(f"Loading LoRA weights: {lora_path}")
        self.pipe.load_lora_weights(lora_path)

        # Enable memory optimizations
        self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()

        print(f"✓ Generator initialized on {device}")

    def generate_batch(
        self,
        prompts: List[str],
        negative_prompt: str = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate batch of images

        Args:
            prompts: List of prompts to generate
            negative_prompt: Negative prompt for all images
            num_images_per_prompt: How many variations per prompt
            num_inference_steps: Sampling steps
            guidance_scale: CFG scale
            width: Image width
            height: Image height
            seed: Random seed (None for random)

        Returns:
            List of generation results with metadata
        """
        results = []

        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
            for variation in range(num_images_per_prompt):
                # Set seed for reproducibility if provided
                if seed is not None:
                    generator = torch.Generator(device=self.device).manual_seed(
                        seed + prompt_idx * 1000 + variation
                    )
                else:
                    generator = None

                # Generate image
                with torch.autocast(self.device):
                    output = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        generator=generator,
                    )

                image = output.images[0]

                # Store result with metadata
                result = {
                    "image": image,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "prompt_idx": prompt_idx,
                    "variation": variation,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "seed": (
                        seed + prompt_idx * 1000 + variation
                        if seed is not None
                        else None
                    ),
                }

                results.append(result)

                # Clear CUDA cache periodically
                if len(results) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        return results

    def save_results(
        self, results: List[Dict], output_dir: Path, save_metadata: bool = True
    ):
        """
        Save generated images and metadata

        Args:
            results: List of generation results
            output_dir: Output directory
            save_metadata: Whether to save JSON metadata
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_list = []

        for idx, result in enumerate(tqdm(results, desc="Saving images")):
            image = result["image"]

            # Create filename
            prompt_id = result["prompt_idx"]
            variation = result["variation"]
            filename = f"img_{idx:05d}_p{prompt_id:03d}_v{variation:02d}.png"

            # Save image
            image_path = output_dir / filename
            image.save(image_path, quality=95)

            # Prepare metadata (without PIL image object)
            if save_metadata:
                meta = {k: v for k, v in result.items() if k != "image"}
                meta["filename"] = filename
                meta["file_path"] = str(image_path)
                metadata_list.append(meta)

        # Save metadata JSON
        if save_metadata:
            metadata_path = output_dir / "generation_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "total_images": len(results),
                        "images": metadata_list,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"✓ Metadata saved to: {metadata_path}")

    def cleanup(self):
        """Free GPU memory"""
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()


def load_prompt_library(library_path: Path) -> List[str]:
    """Load prompts from file (one per line) or JSON"""
    if not library_path.exists():
        raise FileNotFoundError(f"Prompt library not found: {library_path}")

    if library_path.suffix == ".json":
        with open(library_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("prompts", [])
    else:
        # Text file, one prompt per line
        with open(library_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
            return prompts


def main():
    parser = argparse.ArgumentParser(description="Batch LoRA image generation")

    parser.add_argument(
        "lora_path", type=Path, help="Path to LoRA weights file (.safetensors)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/anything-v4.5-vae-swapped.safetensors",
        help="Path to base model",
    )
    parser.add_argument(
        "--prompts", type=Path, help="Path to prompt library file (.txt or .json)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        action="append",
        help="Single prompt (can use multiple times)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--negative",
        type=str,
        default="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        help="Negative prompt",
    )
    parser.add_argument(
        "--variations", type=int, default=1, help="Number of variations per prompt"
    )
    parser.add_argument("--steps", type=int, default=28, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (for reproducibility)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Collect prompts
    prompts = []

    if args.prompts:
        prompts.extend(load_prompt_library(args.prompts))

    if args.prompt:
        prompts.extend(args.prompt)

    if not prompts:
        raise ValueError("No prompts provided! Use --prompts or --prompt")

    print(f"\n{'='*80}")
    print(f"Batch LoRA Image Generation")
    print(f"{'='*80}")
    print(f"LoRA: {args.lora_path}")
    print(f"Base Model: {args.base_model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Variations per prompt: {args.variations}")
    print(f"Total images: {len(prompts) * args.variations}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Initialize generator
    generator = BatchLoRAGenerator(
        base_model_path=args.base_model,
        lora_path=str(args.lora_path),
        device=args.device,
    )

    # Generate images
    print(f"\n🎨 Generating {len(prompts) * args.variations} images...")
    results = generator.generate_batch(
        prompts=prompts,
        negative_prompt=args.negative,
        num_images_per_prompt=args.variations,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    # Save results
    print(f"\n💾 Saving images...")
    generator.save_results(results, args.output_dir)

    # Cleanup
    generator.cleanup()

    print(f"\n✅ Complete!")
    print(f"Generated {len(results)} images in {args.output_dir}")
    print(f"Check generation_metadata.json for details")


if __name__ == "__main__":
    main()
