#!/usr/bin/env python3
"""
Quick LoRA Checkpoint Tester for SDXL
Tests a single LoRA checkpoint with various prompts
"""

import sys
import torch
from pathlib import Path
from diffusers import DiffusionPipeline, AutoencoderKL
from datetime import datetime
import argparse

def test_lora_checkpoint(
    lora_path: str,
    character_name: str,
    output_dir: str,
    base_model: str = "/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors",
    num_images: int = 4,
    device: str = "cuda"
):
    """Test a LoRA checkpoint with various prompts"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"🧪 Testing LoRA Checkpoint: {character_name}")
    print("=" * 70)
    print(f"\nLoRA: {lora_path}")
    print(f"Base Model: {base_model}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print()

    # Load SDXL pipeline
    print("📦 Loading SDXL pipeline...")
    try:
        pipe = DiffusionPipeline.from_single_file(
            base_model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        pipe = pipe.to(device)
        print("✓ SDXL pipeline loaded")
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        print("\nTrying alternative loading method...")
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        pipe = pipe.to(device)
        print("✓ SDXL pipeline loaded from pretrained")

    # Load LoRA weights
    print(f"\n🎨 Loading LoRA weights: {Path(lora_path).name}...")
    pipe.load_lora_weights(lora_path)
    print("✓ LoRA weights loaded")

    # Test prompts
    test_prompts = [
        {
            "prompt": f"{character_name}, a 3d animated character, standing pose, neutral expression, front view, studio lighting, white background",
            "name": "front_view_neutral"
        },
        {
            "prompt": f"{character_name}, a 3d animated character, three-quarter view, happy expression, soft lighting, simple background",
            "name": "three_quarter_happy"
        },
        {
            "prompt": f"{character_name}, a 3d animated character, side profile view, looking to the side, outdoor lighting, sky background",
            "name": "side_profile_outdoor"
        },
        {
            "prompt": f"{character_name}, a 3d animated character, close-up portrait, smiling, warm lighting, blurred background",
            "name": "closeup_portrait"
        }
    ]

    # Generation settings
    generator = torch.Generator(device=device).manual_seed(42)

    print(f"\n🖼️  Generating {len(test_prompts)} test images...")
    print()

    results = []
    for idx, test in enumerate(test_prompts, 1):
        print(f"[{idx}/{len(test_prompts)}] {test['name']}...")
        print(f"  Prompt: {test['prompt'][:80]}...")

        try:
            image = pipe(
                prompt=test['prompt'],
                negative_prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text",
                num_inference_steps=30,
                guidance_scale=7.5,
                width=1024,
                height=1024,
                generator=generator
            ).images[0]

            # Save image
            output_file = output_path / f"{character_name}_epoch1_{test['name']}.png"
            image.save(output_file)
            print(f"  ✓ Saved: {output_file.name}")
            results.append(str(output_file))

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print()
    print("=" * 70)
    print("✅ Testing Complete!")
    print("=" * 70)
    print(f"\nGenerated {len(results)} images:")
    for img in results:
        print(f"  • {img}")
    print()
    print(f"Output directory: {output_dir}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Test LoRA checkpoint")
    parser.add_argument("--lora-path", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--character", required=True, help="Character name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--base-model", default="/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors")
    parser.add_argument("--device", default="cuda", help="Device to use")

    args = parser.parse_args()

    test_lora_checkpoint(
        lora_path=args.lora_path,
        character_name=args.character,
        output_dir=args.output_dir,
        base_model=args.base_model,
        device=args.device
    )


if __name__ == "__main__":
    main()
