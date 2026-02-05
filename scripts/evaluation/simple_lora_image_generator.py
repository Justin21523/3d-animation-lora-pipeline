#!/usr/bin/env python3
"""
Simple SDXL LoRA Image Generator
Generates test images for multiple checkpoints without complex metrics
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
import json

def main():
    parser = argparse.ArgumentParser(description='Generate test images for SDXL LoRA checkpoints')
    parser.add_argument('lora_dir', type=str, help='Directory containing LoRA checkpoints')
    parser.add_argument('--base-model', type=str, required=True, help='Path to base SDXL model')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--prompts-file', type=str, required=True, help='File with test prompts')
    parser.add_argument('--num-images-per-prompt', type=int, default=4, help='Images per prompt')
    parser.add_argument('--steps', type=int, default=30, help='Inference steps')
    parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    with open(args.prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"📋 Loaded {len(prompts)} prompts")

    # Find all checkpoints
    checkpoints = sorted(lora_dir.glob("*.safetensors"))
    checkpoints = [c for c in checkpoints if not c.name.startswith("BEST_")]

    print(f"📦 Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"   - {cp.name}")

    # Load base pipeline
    print(f"\n🔧 Loading SDXL pipeline from {args.base_model}...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        args.base_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(args.device)

    # Enable optimizations
    try:
        pipe.enable_model_cpu_offload()
        print("  ✓ Model CPU offload enabled")
    except:
        print("  → CPU offload not available")

    print(f"\n{'='*80}")
    print(f"🎨 GENERATING IMAGES")
    print(f"{'='*80}\n")

    # Process each checkpoint
    for checkpoint in checkpoints:
        checkpoint_name = checkpoint.stem
        checkpoint_output = output_dir / checkpoint_name
        checkpoint_output.mkdir(parents=True, exist_ok=True)

        print(f"\n📦 Checkpoint: {checkpoint_name}")
        print(f"   Output: {checkpoint_output}")

        # Load LoRA
        print(f"   Loading LoRA...")
        pipe.load_lora_weights(str(checkpoint))

        # Generate images
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

        image_idx = 0
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="   Generating")):
            for var_idx in range(args.num_images_per_prompt):
                # Generate
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator,
                ).images[0]

                # Save
                image.save(checkpoint_output / f"image_{image_idx:04d}.png")
                image_idx += 1

        # Save metadata
        metadata = {
            "checkpoint": checkpoint_name,
            "checkpoint_path": str(checkpoint),
            "num_prompts": len(prompts),
            "num_images": image_idx,
            "num_images_per_prompt": args.num_images_per_prompt,
            "steps": args.steps,
            "guidance_scale": args.guidance,
            "seed": args.seed,
            "prompts": prompts,
        }

        with open(checkpoint_output / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"   ✓ Generated {image_idx} images")

        # Unload LoRA for next checkpoint
        pipe.unload_lora_weights()

    print(f"\n{'='*80}")
    print(f"✅ ALL CHECKPOINTS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nView images:")
    for checkpoint in checkpoints:
        checkpoint_name = checkpoint.stem
        print(f"  {checkpoint_name}/")
        print(f"    ls {output_dir / checkpoint_name}")

if __name__ == '__main__':
    main()
