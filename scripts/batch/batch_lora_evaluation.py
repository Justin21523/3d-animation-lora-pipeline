#!/usr/bin/env python3
"""
Batch LoRA Checkpoint Evaluation
批次測試所有 LoRA checkpoints 並選出最佳模型

Author: LLMProvider Tooling
Date: 2025-11-22
"""

import argparse
import json
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from datetime import datetime
import sys

# Character-specific prompts
CHARACTER_PROMPTS = {
    'caleb': [
        "caleb, a 3d animated teenage boy character from Pixar's Elio, portrait, neutral expression, soft studio lighting, detailed face, pixar style, smooth shading",
        "caleb, a 3d animated character from Pixar's Elio, close-up face, smiling, warm lighting, cinematic quality, detailed eyes, pixar animation",
        "caleb, a 3d animated character from Pixar's Elio, full body, standing confidently, natural lighting, high quality 3d render, pixar style",
        "caleb, a 3d animated character from Pixar's Elio, three-quarter view, casual pose, soft lighting, high resolution, pixar quality",
        "caleb, a 3d animated character from Pixar's Elio, walking forward, dynamic pose, natural lighting, smooth animation style",
    ],
    'bryce': [
        "bryce markwell, a 3d animated teenage boy character from Pixar's Elio, portrait, confident expression, soft studio lighting, detailed face, pixar style, smooth shading",
        "bryce markwell, a 3d animated character from Pixar's Elio, close-up face, friendly smile, warm lighting, cinematic quality, detailed eyes, pixar animation",
        "bryce markwell, a 3d animated character from Pixar's Elio, full body, standing tall, natural lighting, high quality 3d render, pixar style",
        "bryce markwell, a 3d animated character from Pixar's Elio, three-quarter view, relaxed pose, soft lighting, high resolution, pixar quality",
        "bryce markwell, a 3d animated character from Pixar's Elio, athletic pose, dynamic lighting, smooth animation style",
    ]
}

# Enhanced negative prompts
NEGATIVE_PROMPT = """low quality, worst quality, bad anatomy, bad hands, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra arms, extra fingers, extra legs, extra limbs, fused fingers, gross proportions, jpeg artifacts, long neck, low resolution, lowres, malformed limbs, missing arms, missing legs, morbid, mutated hands, mutation, mutilated, out of frame, poorly drawn face, poorly drawn hands, signature, text, ugly, username, watermark, 2d, anime, cartoon sketch, drawing, painting, black and white, monochrome, grayscale, dark, underexposed, overexposed, realistic, photorealistic, photograph, cropped head, cut off head, incomplete head, head out of frame, partial head, multiple people, two people, crowd, group, extra person""".strip()


def test_single_checkpoint(
    lora_path: Path,
    character_name: str,
    base_model: str,
    output_dir: Path,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 28,
    seed: int = 42,
    device: str = "cuda"
):
    """Test a single checkpoint with character-specific prompts."""

    checkpoint_name = lora_path.stem
    print(f"\n{'='*70}")
    print(f"Testing: {checkpoint_name}")
    print(f"{'='*70}\n")

    # Create checkpoint-specific output directory
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get character prompts
    prompts = CHARACTER_PROMPTS.get(character_name, CHARACTER_PROMPTS['caleb'])

    # Load pipeline
    print(f"Loading base model...")
    pipe = StableDiffusionPipeline.from_single_file(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        load_safety_checker=False
    ).to(device)

    # Load LoRA
    print(f"Loading LoRA: {lora_path.name}")
    pipe.load_lora_weights(str(lora_path))

    # Test metadata
    metadata = {
        "checkpoint": str(lora_path),
        "character": character_name,
        "timestamp": datetime.now().isoformat(),
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "images": []
    }

    # Generate test images
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Generating...")
        print(f"Prompt: {prompt[:60]}...")

        generator = torch.Generator(device=device).manual_seed(seed + idx)
        image = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        # Save image
        filename = f"test_{idx:02d}.png"
        image_path = checkpoint_dir / filename
        image.save(image_path)
        print(f"Saved: {filename}")

        metadata["images"].append({
            "filename": filename,
            "prompt": prompt,
            "seed": seed + idx
        })

    # Save metadata
    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Cleanup
    del pipe
    torch.cuda.empty_cache()

    print(f"\n✅ Checkpoint test complete: {checkpoint_name}\n")
    return metadata


def batch_evaluate_character(
    lora_dir: Path,
    character_name: str,
    base_model: str,
    output_root: Path,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 28,
    seed: int = 42,
    device: str = "cuda"
):
    """Evaluate all checkpoints for a character."""

    # Find all checkpoint files
    checkpoints = sorted(lora_dir.glob("*.safetensors"))

    if not checkpoints:
        print(f"❌ No checkpoints found in {lora_dir}")
        return

    print(f"\n{'='*70}")
    print(f"BATCH EVALUATION: {character_name.upper()}")
    print(f"{'='*70}")
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Output directory: {output_root}")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir = output_root / character_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test each checkpoint
    batch_results = {
        "character": character_name,
        "lora_directory": str(lora_dir),
        "timestamp": datetime.now().isoformat(),
        "total_checkpoints": len(checkpoints),
        "checkpoints": []
    }

    for i, checkpoint_path in enumerate(checkpoints, 1):
        print(f"\n{'='*70}")
        print(f"Progress: {i}/{len(checkpoints)}")
        print(f"{'='*70}")

        try:
            result = test_single_checkpoint(
                lora_path=checkpoint_path,
                character_name=character_name,
                base_model=base_model,
                output_dir=output_dir,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                device=device
            )

            batch_results["checkpoints"].append({
                "name": checkpoint_path.name,
                "path": str(checkpoint_path),
                "status": "success"
            })

        except Exception as e:
            print(f"❌ Error testing {checkpoint_path.name}: {e}")
            batch_results["checkpoints"].append({
                "name": checkpoint_path.name,
                "path": str(checkpoint_path),
                "status": "failed",
                "error": str(e)
            })

    # Save batch summary
    summary_path = output_dir / "batch_evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(batch_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ BATCH EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Character: {character_name}")
    print(f"Tested: {len(checkpoints)} checkpoints")
    print(f"Results: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluate LoRA checkpoints for a character',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Evaluate all Caleb checkpoints
  python batch_lora_evaluation.py \\
    /mnt/data/ai_data/models/lora/elio/caleb_identity \\
    --character caleb \\
    --output-dir /mnt/data/ai_data/lora_evaluation/caleb

  # Evaluate all Bryce checkpoints
  python batch_lora_evaluation.py \\
    /mnt/data/ai_data/models/lora/elio/bryce_identity \\
    --character bryce \\
    --output-dir /mnt/data/ai_data/lora_evaluation/bryce
        """
    )

    parser.add_argument(
        'lora_dir',
        type=Path,
        help='Directory containing LoRA checkpoints'
    )
    parser.add_argument(
        '--character',
        type=str,
        required=True,
        choices=['caleb', 'bryce'],
        help='Character name'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors',
        help='Path to base SD model'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/mnt/data/ai_data/lora_evaluation'),
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=7.5,
        help='Guidance scale (default: 7.5)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=28,
        help='Number of inference steps (default: 28)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (default: cuda)'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.lora_dir.exists():
        print(f"❌ LoRA directory not found: {args.lora_dir}")
        return 1

    # Run batch evaluation
    try:
        batch_evaluate_character(
            lora_dir=args.lora_dir,
            character_name=args.character,
            base_model=args.base_model,
            output_root=args.output_dir,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            seed=args.seed,
            device=args.device
        )
        return 0

    except Exception as e:
        print(f"❌ Batch evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
