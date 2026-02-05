#!/usr/bin/env python3
"""
Super Wings Comprehensive LoRA Testing
Tests each character with 30 diverse prompts (10 actions + 10 expressions + 10 poses)
Multiple LoRA strengths and seeds for thorough evaluation
"""

import torch
from pathlib import Path
from diffusers import DiffusionPipeline
import argparse
import yaml
from typing import Dict, List
import time


def load_config(config_path: str) -> Dict:
    """Load YAML configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def fill_prompt_template(template: str, character_data: Dict) -> str:
    """Fill prompt template with character-specific data"""
    return template.format(
        character=character_data['name'],
        colors=character_data['colors'],
        personality=character_data['personality'],
        features=character_data['features']
    )


def test_character_comprehensive(
    character_name: str,
    lora_path: str,
    config: Dict,
    output_base_dir: str,
    device: str = "cuda"
):
    """Comprehensive testing for one character"""

    character_data = config['characters'][character_name]
    test_config = config['test_config']
    negative_prompt = config['negative_prompt'].strip()

    output_dir = Path(output_base_dir) / character_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🎬 Comprehensive LoRA Test: {character_name.upper()}")
    print("=" * 80)
    print(f"\nCharacter: {character_data['name']}")
    print(f"Colors: {character_data['colors']}")
    print(f"LoRA: {Path(lora_path).name}")
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

    # Load LoRA
    print(f"\n🎨 Loading LoRA weights...")
    pipe.load_lora_weights(lora_path)
    print("✓ LoRA weights loaded")

    # Collect all prompts
    all_prompts = []

    # Actions (10)
    for action in config['actions']:
        prompt = fill_prompt_template(action['prompt_template'], character_data)
        all_prompts.append({
            'category': 'action',
            'name': action['name'],
            'prompt': prompt
        })

    # Expressions (10)
    for expression in config['expressions']:
        prompt = fill_prompt_template(expression['prompt_template'], character_data)
        all_prompts.append({
            'category': 'expression',
            'name': expression['name'],
            'prompt': prompt
        })

    # Poses (10)
    for pose in config['poses']:
        prompt = fill_prompt_template(pose['prompt_template'], character_data)
        all_prompts.append({
            'category': 'pose',
            'name': pose['name'],
            'prompt': prompt
        })

    print(f"\n📝 Total prompts: {len(all_prompts)}")
    print(f"   - Actions: 10")
    print(f"   - Expressions: 10")
    print(f"   - Poses: 10")
    print()

    print(f"🔧 Test configuration:")
    print(f"   - LoRA strengths: {test_config['lora_strengths']}")
    print(f"   - Seeds per prompt: {test_config['seeds']}")
    print(f"   - Steps: {test_config['steps']}")
    print(f"   - CFG scale: {test_config['cfg_scale']}")
    print()

    total_images = len(all_prompts) * len(test_config['lora_strengths']) * len(test_config['seeds'])
    print(f"📊 Total images to generate: {total_images}")
    print(f"   ({len(all_prompts)} prompts × {len(test_config['lora_strengths'])} strengths × {len(test_config['seeds'])} seeds)")
    print()

    # Generate images
    print("🖼️  Starting image generation...")
    print()

    results = []
    count = 0
    start_time = time.time()

    for prompt_data in all_prompts:
        category = prompt_data['category']
        name = prompt_data['name']
        prompt = prompt_data['prompt']

        print(f"[{category.upper()}] {name}")

        for lora_strength in test_config['lora_strengths']:
            for seed in test_config['seeds']:
                count += 1

                # Generate with LoRA strength
                generator = torch.Generator(device=device).manual_seed(seed)

                try:
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=test_config['steps'],
                        guidance_scale=test_config['cfg_scale'],
                        width=test_config['width'],
                        height=test_config['height'],
                        generator=generator,
                        cross_attention_kwargs={"scale": lora_strength}
                    ).images[0]

                    # Save with descriptive filename
                    filename = f"{category}_{name}_lora{lora_strength}_seed{seed}.png"
                    output_path = output_dir / filename
                    image.save(output_path)

                    results.append(str(output_path))

                    # Progress indicator
                    elapsed = time.time() - start_time
                    avg_time = elapsed / count
                    remaining = (total_images - count) * avg_time

                    print(f"  [{count}/{total_images}] LoRA {lora_strength} | Seed {seed} | "
                          f"ETA: {remaining/60:.1f}m", end='\r')

                except Exception as e:
                    print(f"\n  ❌ Error: {e}")

        print()  # New line after each prompt

    # Summary
    elapsed_total = time.time() - start_time
    print()
    print("=" * 80)
    print("✅ Comprehensive Testing Complete!")
    print("=" * 80)
    print(f"\nCharacter: {character_name}")
    print(f"Generated: {len(results)}/{total_images} images")
    print(f"Time: {elapsed_total/60:.1f} minutes")
    print(f"Output: {output_dir}")
    print()

    # Stats by category
    print("📊 Breakdown:")
    print(f"   Actions: {len([p for p in all_prompts if p['category'] == 'action'])} prompts "
          f"× {len(test_config['lora_strengths'])} strengths × {len(test_config['seeds'])} seeds "
          f"= {len([p for p in all_prompts if p['category'] == 'action']) * len(test_config['lora_strengths']) * len(test_config['seeds'])} images")
    print(f"   Expressions: {len([p for p in all_prompts if p['category'] == 'expression'])} prompts "
          f"× {len(test_config['lora_strengths'])} strengths × {len(test_config['seeds'])} seeds "
          f"= {len([p for p in all_prompts if p['category'] == 'expression']) * len(test_config['lora_strengths']) * len(test_config['seeds'])} images")
    print(f"   Poses: {len([p for p in all_prompts if p['category'] == 'pose'])} prompts "
          f"× {len(test_config['lora_strengths'])} strengths × {len(test_config['seeds'])} seeds "
          f"= {len([p for p in all_prompts if p['category'] == 'pose']) * len(test_config['lora_strengths']) * len(test_config['seeds'])} images")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Super Wings LoRA Testing")
    parser.add_argument("--character", required=True,
                       choices=["jett", "jerome", "donnie"],
                       help="Character to test")
    parser.add_argument("--lora-path", required=True,
                       help="Path to LoRA checkpoint file")
    parser.add_argument("--config",
                       default="/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/evaluation/super_wings_comprehensive_test.yaml",
                       help="Path to test configuration YAML")
    parser.add_argument("--output-dir", required=True,
                       help="Base output directory")
    parser.add_argument("--device", default="cuda",
                       help="Device to use")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Run comprehensive test
    test_character_comprehensive(
        character_name=args.character,
        lora_path=args.lora_path,
        config=config,
        output_base_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
