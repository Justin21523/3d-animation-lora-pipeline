#!/usr/bin/env python3
"""
Super Wings Character LoRA Tester
Designed to avoid common issues: color confusion, extra eyes, multiple characters, human appearance
"""

import torch
from pathlib import Path
from diffusers import DiffusionPipeline
import argparse

def test_super_wings_lora(
    lora_path: str,
    character_name: str,
    character_colors: str,
    output_dir: str,
    device: str = "cuda"
):
    """Test Super Wings LoRA with strict quality control"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🧪 Super Wings LoRA Testing: {character_name}")
    print("=" * 80)
    print(f"\nLoRA: {Path(lora_path).name}")
    print(f"Character Colors: {character_colors}")
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

    # CRITICAL: Super Wings specific negative prompt
    negative_prompt = f"""
human, person, people, boy, girl, man, woman, child, humanoid,
human face, human body, human features, realistic human,
multiple characters, two characters, group, crowd, duo,
extra eyes, three eyes, four eyes, multiple eyes, extra limbs,
wrong colors, incorrect colors, color swap, mismatched colors,
not {character_colors}, different color scheme,
blurry, low quality, worst quality, bad quality, lowres,
distorted, deformed, disfigured, mutated, malformed,
ugly, amateur, draft, unfinished, bad anatomy, bad proportions,
jpeg artifacts, watermark, text, signature,
2d, anime style, cartoon, illustration, painting, drawing,
photographic, photo, photograph, real life,
cropped, cut off, frame, border,
noise, grainy, chromatic aberration,
multiple views, character sheet, reference sheet
    """.strip()

    # Test prompts with STRICT character specifications
    test_prompts = [
        {
            "prompt": f"""
{character_name}, solo character, single {character_colors} jet plane,
Super Wings 3d animated mechanical character,
small cute jet aircraft with exactly two blue eyes,
{character_colors} glossy paint body with smooth metallic surface,
compact jet plane proportions, rounded nose cone, small wings,
friendly expressive face with two eyes only,
mechanical robot transforming jet, not human,
professional studio lighting, clean white background,
centered composition, front three-quarter view,
high quality 3d render, Pixar animation style,
perfect focus, sharp details, smooth anti-aliased edges
            """.strip(),
            "name": "studio_front_view",
            "steps": 45,
            "cfg": 9.0
        },
        {
            "prompt": f"""
{character_name}, one single character only,
{character_colors} mechanical jet plane from Super Wings,
3d animated robot aircraft with two blue eyes,
cute friendly jet design, {character_colors} color scheme,
glossy vehicle paint, metallic jet body,
small compact jet proportions, rounded features,
solo character portrait, no other characters,
soft studio lighting, neutral grey background,
close-up view, centered framing,
extremely high quality render, masterpiece,
Pixar level 3d animation, smooth shading
            """.strip(),
            "name": "portrait_closeup",
            "steps": 45,
            "cfg": 8.5
        },
        {
            "prompt": f"""
{character_name} character, single {character_colors} jet aircraft,
Super Wings mechanical transforming robot jet plane,
exactly two expressive blue eyes, friendly face,
{character_colors} glossy body paint with clean edges,
small cute jet design, vehicle mode,
one character only, no humans, no people,
outdoor scene, blue sky background with white clouds,
natural lighting, three-quarter angle view,
professional 3d animation quality,
DreamWorks style rendering, vibrant colors,
perfect character consistency
            """.strip(),
            "name": "outdoor_scene",
            "steps": 40,
            "cfg": 8.0
        },
        {
            "prompt": f"""
{character_name}, solo Super Wings jet character,
{character_colors} mechanical aircraft with two eyes,
3d animated transforming robot jet plane,
cute compact jet proportions, {character_colors} paint,
glossy metallic surface, rounded friendly design,
single character focus, vehicle form,
side profile view showing full body,
simple gradient background, professional lighting,
high quality 3d render, feature film quality,
Pixar rendering style, clean perfect geometry
            """.strip(),
            "name": "side_profile",
            "steps": 40,
            "cfg": 7.5
        },
        {
            "prompt": f"""
{character_name} from Super Wings, one character,
{character_colors} jet plane robot with two blue eyes,
3d animated mechanical character, not human,
small friendly jet aircraft design,
{character_colors} vehicle paint with white accents,
glossy finish, smooth 3d model,
dynamic action pose, mid-flight or landing,
colorful background, energetic composition,
professional animation frame, Disney Pixar quality,
sharp focus on character, vibrant saturated colors
            """.strip(),
            "name": "action_pose",
            "steps": 40,
            "cfg": 8.0
        },
        {
            "prompt": f"""
{character_name}, technical character reference,
single {character_colors} Super Wings jet aircraft,
3d model turnaround, mechanical robot jet,
exactly two eyes, {character_colors} color scheme only,
professional character design presentation,
clean render, neutral background, even lighting,
model sheet quality, character concept art,
Pixar 3d character model, perfect for reference,
high detail, accurate proportions, solo character
            """.strip(),
            "name": "model_reference",
            "steps": 40,
            "cfg": 7.0
        }
    ]

    print(f"\n🖼️  Generating {len(test_prompts)} test images with strict quality control...")
    print()

    results = []
    for idx, test in enumerate(test_prompts, 1):
        print(f"[{idx}/{len(test_prompts)}] {test['name']}...")
        print(f"  Steps: {test['steps']}, CFG: {test['cfg']}")

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
            output_file = output_path / f"{character_name}_epoch15_{test['name']}.png"
            image.save(output_file)
            print(f"  ✓ Saved: {output_file.name}")
            results.append(str(output_file))

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print()
    print("=" * 80)
    print("✅ Testing Complete!")
    print("=" * 80)
    print(f"\nGenerated {len(results)} images:")
    for img in results:
        print(f"  • {img}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--colors", required=True, help="Character color description (e.g., 'red and white')")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    test_super_wings_lora(
        lora_path=args.lora_path,
        character_name=args.character,
        character_colors=args.colors,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
