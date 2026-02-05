#!/usr/bin/env python3
"""
Quick LoRA Testing with Full Negative Prompts
Tests specific checkpoints with proper negative prompts to avoid artifacts
"""

import torch
import argparse
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora-path', type=str, required=True, help='Path to LoRA checkpoint')
    parser.add_argument('--base-model', type=str, required=True, help='Path to base SDXL model')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--prompt', type=str, required=True, help='Positive prompt')
    parser.add_argument('--negative-prompt', type=str, default='', help='Negative prompt')
    parser.add_argument('--num-samples', type=int, default=4, help='Number of images')
    parser.add_argument('--steps', type=int, default=30, help='Inference steps')
    parser.add_argument('--guidance-scale', type=float, default=7.5, help='CFG scale')
    parser.add_argument('--lora-scale', type=float, default=1.0, help='LoRA scale')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    args = parser.parse_args()

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print(f"Loading SDXL pipeline from {args.base_model}...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        args.base_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to('cuda')

    pipe.enable_model_cpu_offload()

    # Load LoRA
    print(f"Loading LoRA from {args.lora_path}...")
    pipe.load_lora_weights(args.lora_path)
    pipe.fuse_lora(lora_scale=args.lora_scale)

    # Generate images
    print(f"Generating {args.num_samples} images...")
    print(f"Prompt: {args.prompt}")
    print(f"Negative: {args.negative_prompt}")

    generator = torch.Generator(device='cuda').manual_seed(args.seed)

    results = []
    for i in range(args.num_samples):
        print(f"  Generating image {i+1}/{args.num_samples}...")
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            width=args.width,
            height=args.height,
        ).images[0]

        # Save image
        lora_name = Path(args.lora_path).stem
        output_path = output_dir / f"{lora_name}_sample_{i+1}.png"
        image.save(output_path)
        print(f"    Saved: {output_path}")
        results.append(str(output_path))

    # Save metadata
    metadata = {
        'lora_path': args.lora_path,
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'num_samples': args.num_samples,
        'steps': args.steps,
        'guidance_scale': args.guidance_scale,
        'lora_scale': args.lora_scale,
        'seed': args.seed,
        'images': results,
    }

    metadata_path = output_dir / f"{lora_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Done! Generated {len(results)} images")
    print(f"Output: {output_dir}")

if __name__ == '__main__':
    main()
