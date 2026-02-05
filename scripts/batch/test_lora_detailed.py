#!/usr/bin/env python3
"""
Detailed LoRA Checkpoint Tester with High-Quality Prompts
Tests with professional, detailed prompts and comprehensive negative prompts
"""

import sys
import torch
from pathlib import Path
from diffusers import DiffusionPipeline
import argparse

def test_lora_detailed(
    lora_path: str,
    character_name: str,
    output_dir: str,
    device: str = "cuda"
):
    """Test LoRA with highly detailed prompts"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🧪 Detailed LoRA Testing: {character_name}")
    print("=" * 80)
    print(f"\nLoRA: {Path(lora_path).name}")
    print(f"Output: {output_dir}")
    print()

    # Load SDXL pipeline
    print("📦 Loading SDXL pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe = pipe.to(device)
    print("✓ SDXL pipeline loaded")

    # Load LoRA weights
    print(f"\n🎨 Loading LoRA weights...")
    pipe.load_lora_weights(lora_path)
    print("✓ LoRA weights loaded")

    # Comprehensive negative prompt
    negative_prompt = """
low quality, worst quality, bad quality, lowres, low resolution,
blurry, blurred, blur, out of focus, bokeh, depth of field,
distorted, deformed, disfigured, mutated, mutation, malformed,
ugly, disgusting, amateur, draft, unfinished,
jpeg artifacts, compression artifacts, noise, noisy, grainy, chromatic aberration,
watermark, text, signature, username, artist name, copyright, logo,
bad anatomy, bad proportions, extra limbs, missing limbs, bad hands, bad fingers, extra fingers, fused fingers,
bad face, bad eyes, cross-eyed, bad mouth, bad teeth,
cut off, cropped, frame, framed, border,
multiple views, duplicate, repetitive,
bad lighting, overexposed, underexposed, harsh lighting,
flat colors, oversaturated, undersaturated, color banding,
3d render artifact, mesh visible, polygon visible, uncanny valley
    """.strip()

    # Detailed test prompts with professional quality
    test_prompts = [
        {
            "prompt": f"""
{character_name}, a 3d animated character from Super Wings,
small red and white jet plane character with big expressive blue eyes,
compact sporty jet aircraft proportions with rounded friendly nose cone,
glossy red body paint with crisp white racing stripes and accents,
professional studio lighting with soft key light and subtle rim light,
clean white background with gentle gradient, centered composition,
three-quarter front view angle, neutral standing pose,
high quality 3d render, Pixar style, smooth shading, clean edges,
professional product photography lighting, octane render quality
            """.strip(),
            "name": "studio_professional",
            "steps": 40,
            "cfg": 8.0
        },
        {
            "prompt": f"""
{character_name}, a 3d animated character,
red and white jet plane with friendly personality,
cheerful happy expression with bright blue eyes looking at camera,
dynamic three-quarter angle view, slight tilt for energy,
outdoor scene with clear blue sky and white clouds background,
natural sunlight, warm golden hour lighting, soft shadows,
vibrant colors, cinematic composition, rule of thirds,
high quality 3d animation still, DreamWorks quality,
professional color grading, depth and dimension
            """.strip(),
            "name": "outdoor_dynamic",
            "steps": 40,
            "cfg": 7.5
        },
        {
            "prompt": f"""
{character_name}, 3d animated jet plane character,
close-up portrait shot, filling frame,
detailed view of face with large expressive blue eyes,
red glossy paint with realistic reflections and specularity,
white accent stripes with crisp clean edges,
soft diffused lighting from above and sides, minimal shadows,
simple gradient background, professional headshot style,
extremely high quality render, perfect focus, sharp details,
Pixar level detail, smooth anti-aliased edges,
8k resolution quality, masterpiece
            """.strip(),
            "name": "closeup_portrait",
            "steps": 45,
            "cfg": 8.5
        },
        {
            "prompt": f"""
{character_name}, 3d animated character design,
full body view, complete character visible,
standing on runway or landing pad surface,
side profile angle showing sleek jet aircraft design,
red and white color scheme with blue eye detail,
environment lighting, realistic outdoor illumination,
airport or hangar background with depth, slightly blurred,
cinematic wide shot, establishing shot composition,
professional animation frame, Disney/Pixar quality,
clean render, perfect geometry, smooth gradients,
high production value, feature film quality
            """.strip(),
            "name": "full_body_side",
            "steps": 40,
            "cfg": 7.5
        },
        {
            "prompt": f"""
{character_name}, Super Wings character,
red jet plane with white markings and blue eyes,
excited happy expression, dynamic pose,
action shot, mid-flight or taking off,
dramatic lighting with strong key light creating depth,
colorful background with sky and landscape elements,
high energy composition, diagonal lines for movement,
professional 3d animation still frame,
vibrant saturated colors, high contrast,
sharp focus on character, cinematic quality,
Pixar rendering engine quality, perfect textures
            """.strip(),
            "name": "action_dynamic",
            "steps": 40,
            "cfg": 8.0
        },
        {
            "prompt": f"""
{character_name}, 3d character model,
technical three-quarter view for character reference,
neutral grey background, even lighting all around,
red and white jet aircraft design, compact proportions,
bright blue expressive eyes, friendly appearance,
model sheet quality, turnaround reference,
professional character design presentation,
clean render, no shadows, flat even illumination,
high detail, perfect for reference,
character concept art quality, studio quality
            """.strip(),
            "name": "model_sheet",
            "steps": 40,
            "cfg": 7.0
        }
    ]

    print(f"\n🖼️  Generating {len(test_prompts)} high-quality test images...")
    print()

    results = []
    for idx, test in enumerate(test_prompts, 1):
        print(f"[{idx}/{len(test_prompts)}] {test['name']}...")
        print(f"  Steps: {test['steps']}, CFG: {test['cfg']}")
        print(f"  Prompt: {test['prompt'][:100]}...")

        try:
            generator = torch.Generator(device=device).manual_seed(42)

            image = pipe(
                prompt=test['prompt'],
                negative_prompt=negative_prompt,
                num_inference_steps=test['steps'],
                guidance_scale=test['cfg'],
                width=1024,
                height=1024,
                generator=generator
            ).images[0]

            # Save image
            output_file = output_path / f"{character_name}_detailed_{test['name']}.png"
            image.save(output_file)
            print(f"  ✓ Saved: {output_file.name}")
            results.append(str(output_file))

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print()
    print("=" * 80)
    print("✅ Detailed Testing Complete!")
    print("=" * 80)
    print(f"\nGenerated {len(results)} high-quality images:")
    for img in results:
        print(f"  • {img}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    test_lora_detailed(
        lora_path=args.lora_path,
        character_name=args.character,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
