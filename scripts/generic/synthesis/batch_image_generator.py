#!/usr/bin/env python3
"""
Batch SDXL Image Generator with Checkpointing

Production-grade synthetic dataset generation system:
- SDXL pipeline with LoRA loading
- Deterministic seed tracking for reproducible resume
- Checkpoint every 50 images (max 3min data loss)
- GPU memory management and OOM recovery
- Real-time progress monitoring with ETA

Part of Module 2: SDXL Generation Engine
Author: LLMProvider Tooling
Date: 2025-11-30
"""

import json
import torch
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Diffusion
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# Import checkpoint manager from core utils
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.core.utils.checkpoint_manager import IndexCheckpointManager


@dataclass
class PromptSpec:
    """Specification for a single prompt to generate"""
    prompt: str
    seed: int
    categories: Dict[str, str]  # {expression: 'happy', pose: 'standing', ...}
    negative_prompt: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for image generation"""
    num_inference_steps: int = 40
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    checkpoint_interval: int = 50  # Checkpoint every N images
    lora_scale: float = 1.0


@dataclass
class GenerationReport:
    """Report for completed generation job"""
    character: str
    lora_path: str
    total_prompts: int
    images_generated: int
    images_rejected: int
    start_time: str
    end_time: str
    duration_seconds: float
    checkpoint_saves: int
    resumed_from_index: int


# CheckpointManager moved to scripts.core.utils.checkpoint_manager.IndexCheckpointManager


class SDXLSyntheticGenerator:
    """
    Main generation engine with checkpointing

    Features:
    - Deterministic generation (same seed → same image)
    - Checkpoint every N images for fault tolerance
    - GPU memory management
    - Progress monitoring with ETA
    """

    def __init__(
        self,
        base_model_path: str,
        checkpoint_dir: Path,
        vae_path: Optional[str] = None,
        device: str = 'cuda',
        use_fp16: bool = True,
        enable_xformers: bool = True,
    ):
        self.device = device
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.checkpoint_mgr = IndexCheckpointManager(checkpoint_dir, filename="generation_checkpoint.json")

        print("\n" + "="*80)
        print("🚀 SDXL SYNTHETIC GENERATOR - Production Edition")
        print("="*80)

        # Load SDXL pipeline
        self._load_sdxl_pipeline(base_model_path, vae_path, enable_xformers)

        print("="*80)
        print("✅ Generator ready")
        print("="*80 + "\n")

    def _load_sdxl_pipeline(
        self,
        base_model_path: str,
        vae_path: Optional[str],
        enable_xformers: bool
    ):
        """Load SDXL generation pipeline"""

        print(f"📊 Loading SDXL pipeline from {base_model_path}...")

        # Load SDXL pipeline from single file
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            base_model_path,
            torch_dtype=self.dtype,
            use_safetensors=True,
        ).to(self.device)

        # Load custom VAE if specified
        if vae_path:
            print(f"  → Loading custom VAE: {vae_path}")
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=self.dtype).to(self.device)
            self.pipe.vae = vae

        # Enable optimizations
        if enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("  ✓ xformers enabled")
            except Exception as e:
                print(f"  ✗ xformers failed: {e}")

        # Use model CPU offload for memory efficiency
        try:
            self.pipe.enable_model_cpu_offload()
            print("  ✓ Model CPU offload enabled")
        except Exception as e:
            print(f"  ✗ Model CPU offload failed: {e}")

        print("  ✓ SDXL pipeline loaded")

    def load_lora(self, lora_path: str, lora_scale: float = 1.0) -> bool:
        """Load LoRA weights into pipeline"""

        print(f"📦 Loading LoRA: {Path(lora_path).name}")

        # Unload previous LoRA if any
        try:
            self.pipe.unload_lora_weights()
        except:
            pass

        # Load new LoRA
        try:
            self.pipe.load_lora_weights(str(lora_path))
            self.pipe.fuse_lora(lora_scale=lora_scale)
            print(f"  ✓ LoRA loaded (scale={lora_scale})")
            return True
        except Exception as e:
            print(f"  ✗ LoRA loading failed: {e}")
            return False

    def validate_lora(
        self,
        test_prompt: str = "a pixar style 3d animated character, high quality",
        seed: int = 42
    ) -> bool:
        """
        Generate 1 test image to validate LoRA before batch

        Returns: True if generation succeeds, False otherwise
        """

        print(f"🔍 Validating LoRA with test generation...")

        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)

            _ = self.pipe(
                prompt=test_prompt,
                num_inference_steps=10,  # Fast test
                guidance_scale=7.5,
                generator=generator,
            ).images[0]

            print(f"  ✓ Validation passed")
            return True

        except Exception as e:
            print(f"  ✗ Validation failed: {e}")
            return False

    @torch.no_grad()
    def generate_single(
        self,
        prompt: str,
        seed: int,
        negative_prompt: str,
        config: GenerationConfig
    ) -> Image.Image:
        """Generate single image with deterministic seed"""

        generator = torch.Generator(device=self.device).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            width=config.width,
            height=config.height,
            generator=generator,
        ).images[0]

        return image

    def generate_batch(
        self,
        prompts: List[PromptSpec],
        lora_path: str,
        character: str,
        output_dir: Path,
        config: GenerationConfig,
        default_negative_prompt: str = "multiple people, duplicate, clone, two characters, extra limbs, extra arms, extra legs, extra hands, deformed, distorted, disfigured, bad anatomy, wrong anatomy, mutation, mutated, ugly, blurry, low quality, jpeg artifacts, watermark, text, bad proportions, gross proportions"
    ) -> GenerationReport:
        """
        Generate batch of images with automatic checkpointing

        Features:
        - Resume from checkpoint if exists
        - Save checkpoint every N images
        - Deterministic (same seed → same image)
        - Real-time ETA tracking
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = datetime.now()

        # Load LoRA
        if not self.load_lora(lora_path, config.lora_scale):
            raise RuntimeError(f"Failed to load LoRA: {lora_path}")

        # Validate LoRA with test generation
        if not self.validate_lora():
            raise RuntimeError("LoRA validation failed")

        # Check for existing checkpoint
        checkpoint = self.checkpoint_mgr.load()
        if checkpoint and checkpoint['character'] == character:
            start_idx = checkpoint['last_completed_index'] + 1
            seeds_generated = set(checkpoint['seeds_generated'])
            print(f"📂 Resuming from checkpoint: index {start_idx}/{len(prompts)}")
            print(f"   Previously generated: {len(seeds_generated)} images")
        else:
            start_idx = 0
            seeds_generated = set()
            print(f"🆕 Starting fresh generation: {len(prompts)} prompts")

        # Track metrics
        images_generated = 0
        images_rejected = 0
        checkpoint_saves = 0
        seeds_generated_list = list(seeds_generated)

        # Progress bar
        pbar = tqdm(
            range(start_idx, len(prompts)),
            desc=f"Generating {character}",
            initial=start_idx,
            total=len(prompts)
        )

        for idx in pbar:
            prompt_spec = prompts[idx]

            # Skip if seed already generated (idempotency check)
            if prompt_spec.seed in seeds_generated:
                continue

            # Generate image
            try:
                negative = prompt_spec.negative_prompt or default_negative_prompt

                image = self.generate_single(
                    prompt=prompt_spec.prompt,
                    seed=prompt_spec.seed,
                    negative_prompt=negative,
                    config=config
                )

                # Save image
                image_filename = f"{character}_{idx:06d}_seed{prompt_spec.seed}.png"
                image.save(output_dir / image_filename)

                # Save metadata
                metadata = {
                    "prompt": prompt_spec.prompt,
                    "seed": prompt_spec.seed,
                    "categories": prompt_spec.categories,
                    "negative_prompt": negative,
                    "index": idx,
                }
                metadata_filename = f"{character}_{idx:06d}_seed{prompt_spec.seed}.json"
                with open(output_dir / metadata_filename, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Track
                seeds_generated.add(prompt_spec.seed)
                seeds_generated_list.append(prompt_spec.seed)
                images_generated += 1

            except Exception as e:
                print(f"\n⚠️  Failed to generate image {idx}: {e}")
                images_rejected += 1
                continue

            # Checkpoint every N images
            if (idx + 1) % config.checkpoint_interval == 0:
                self.checkpoint_mgr.save(
                    last_completed_index=idx,
                    total_items=len(prompts),
                    character=character,
                    lora_path=str(lora_path),
                    seeds_generated=seeds_generated_list,
                    config=asdict(config)
                )
                checkpoint_saves += 1

                # Update progress bar with ETA
                elapsed = (datetime.now() - start_time).total_seconds()
                images_per_sec = images_generated / elapsed if elapsed > 0 else 0
                remaining = len(prompts) - (idx + 1)
                eta_seconds = remaining / images_per_sec if images_per_sec > 0 else 0
                eta = timedelta(seconds=int(eta_seconds))

                pbar.set_postfix({
                    'generated': images_generated,
                    'rejected': images_rejected,
                    'img/s': f'{images_per_sec:.2f}',
                    'ETA': str(eta)
                })

        # Final checkpoint
        self.checkpoint_mgr.save(
            last_completed_index=len(prompts) - 1,
            total_items=len(prompts),
            character=character,
            lora_path=str(lora_path),
            seeds_generated=seeds_generated_list,
            config=asdict(config)
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Clear checkpoint (job complete)
        self.checkpoint_mgr.clear()

        # Generate report
        report = GenerationReport(
            character=character,
            lora_path=str(lora_path),
            total_prompts=len(prompts),
            images_generated=images_generated,
            images_rejected=images_rejected,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            checkpoint_saves=checkpoint_saves,
            resumed_from_index=start_idx
        )

        # Save report
        with open(output_dir / "generation_report.json", 'w') as f:
            json.dump(asdict(report), f, indent=2)

        print(f"\n{'='*80}")
        print(f"✅ GENERATION COMPLETE: {character}")
        print(f"{'='*80}")
        print(f"  Images generated: {images_generated}/{len(prompts)}")
        print(f"  Images rejected:  {images_rejected}")
        print(f"  Duration:         {duration/60:.1f} minutes")
        print(f"  Speed:            {images_generated/duration:.2f} img/sec")
        print(f"  Checkpoints:      {checkpoint_saves}")
        print(f"  Output:           {output_dir}")
        print(f"{'='*80}\n")

        return report


def load_prompt_specs(prompts_file: Path) -> List[PromptSpec]:
    """Load prompt specifications from JSON file"""

    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file, 'r') as f:
        data = json.load(f)

    # Support both formats:
    # 1. List of dicts with 'prompt', 'seed', 'categories'
    # 2. Dict with 'prompts' key
    if isinstance(data, list):
        prompts_data = data
    elif isinstance(data, dict) and 'prompts' in data:
        prompts_data = data['prompts']
    else:
        raise ValueError("Invalid prompts file format")

    # Convert to PromptSpec objects
    prompt_specs = []
    for item in prompts_data:
        spec = PromptSpec(
            prompt=item['prompt'],
            seed=item['seed'],
            categories=item.get('categories', {}),
            negative_prompt=item.get('negative_prompt')
        )
        prompt_specs.append(spec)

    return prompt_specs


def main():
    parser = argparse.ArgumentParser(
        description="Batch SDXL Image Generator with Checkpointing"
    )

    # Required arguments
    parser.add_argument("prompts_file", type=str, help="JSON file with prompt specifications")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to LoRA weights")
    parser.add_argument("--character", type=str, required=True, help="Character name")
    parser.add_argument("--base-model", type=str, required=True, help="Path to SDXL base model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Checkpoint directory")

    # Optional arguments
    parser.add_argument("--vae", type=str, default=None, help="Custom VAE path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--steps", type=int, default=40, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Checkpoint every N images")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--no-xformers", action="store_true", help="Disable xformers")
    parser.add_argument("--negative-prompt", type=str,
                       default="multiple people, duplicate, clone, two characters, extra limbs, extra arms, extra legs, extra hands, deformed, distorted, disfigured, bad anatomy, wrong anatomy, mutation, mutated, ugly, blurry, low quality, jpeg artifacts, watermark, text, bad proportions, gross proportions",
                       help="Default negative prompt")

    args = parser.parse_args()

    # Setup paths
    prompts_file = Path(args.prompts_file)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    # Load prompts
    print(f"📝 Loading prompts from {prompts_file}...")
    prompt_specs = load_prompt_specs(prompts_file)
    print(f"  ✓ Loaded {len(prompt_specs)} prompts")

    # Initialize generator
    generator = SDXLSyntheticGenerator(
        base_model_path=args.base_model,
        checkpoint_dir=checkpoint_dir,
        vae_path=args.vae,
        device=args.device,
        use_fp16=not args.no_fp16,
        enable_xformers=not args.no_xformers,
    )

    # Generation config
    config = GenerationConfig(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        checkpoint_interval=args.checkpoint_interval,
        lora_scale=args.lora_scale,
    )

    # Generate batch
    report = generator.generate_batch(
        prompts=prompt_specs,
        lora_path=args.lora_path,
        character=args.character,
        output_dir=output_dir,
        config=config,
        default_negative_prompt=args.negative_prompt,
    )

    print(f"✅ Report saved to {output_dir}/generation_report.json")


if __name__ == "__main__":
    main()
