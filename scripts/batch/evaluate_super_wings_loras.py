#!/usr/bin/env python3
"""
Evaluate All Super Wings SDXL LoRAs

Generates test images for all trained Super Wings character LoRAs using
reference prompts from character documentation. Tests multiple checkpoints
per character to identify the best performing ones.

Characters are anthropomorphic planes/helicopters with distinct designs,
colors, and personalities (referencing docs/films/super-wings/characters/).

Author: LLMProvider Tooling
Date: 2025-12-13
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import yaml

# Diffusers imports
try:
    import torch
    from diffusers import StableDiffusionXLPipeline
    from PIL import Image
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("   Please install: pip install diffusers transformers accelerate")
    sys.exit(1)

# Paths
LORA_BASE_DIR = Path("/mnt/c/ai_models/lora_sdxl/super-wings")
CAPTION_CONFIG = Path("/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/super_wings_caption_config.yaml")
OUTPUT_BASE = Path("/mnt/data/outputs/lora_evaluation/super_wings")
SDXL_MODEL = Path("/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors")
SDXL_VAE = Path("/mnt/c/ai_models/stable-diffusion/vae/sdxl_vae.safetensors")

# Characters
CHARACTERS = [
    "beard", "bello", "flip", "jerone", "jet",
    "paul", "shark", "tank", "tony"
]

# Test parameters
INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
LORA_SCALES = [0.7, 0.8, 0.9, 1.0]  # Test multiple LoRA strengths
SEEDS = [42, 123, 456]  # Multiple seeds for consistency testing

def load_character_info() -> Dict:
    """Load character descriptions from caption config"""
    if not CAPTION_CONFIG.exists():
        print(f"⚠️  Character config not found: {CAPTION_CONFIG}")
        return {}

    with open(CAPTION_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    return config.get('characters', {})

def generate_test_prompts(character_name: str, char_info: Dict) -> List[str]:
    """
    Generate test prompts for a character based on their description.

    Super Wings characters are anthropomorphic planes/helicopters with:
    - Distinct colors and designs
    - Vehicle type (jet/helicopter/propeller plane)
    - Personality traits reflected in appearance
    """

    # Default fallback
    base_desc = f"{character_name}, anthropomorphic plane character"
    colors = "colorful"
    vehicle_type = "plane"

    # Get character-specific info
    if char_info:
        base_desc = char_info.get('description', base_desc)
        colors = char_info.get('colors', colors)
        style = char_info.get('style', 'friendly')

    # Test prompts covering different scenarios
    prompts = [
        # Basic identity test
        f"a 3d animated character, {base_desc}, {colors}, Pixar-style rendering, smooth shading, glossy materials, front view, neutral pose, clean white background, studio lighting",

        # Three-quarter view test
        f"a 3d animated character, {base_desc}, {colors}, three-quarter view, slight tilt, cinematic lighting, soft shadows, high quality 3d render",

        # Action/flying pose
        f"a 3d animated character, {base_desc}, flying pose, dynamic angle, motion blur on propeller, blue sky background, warm sunlight, heroic perspective",

        # Close-up detail test
        f"a 3d animated character, {base_desc}, close-up view, detailed facial features, glossy metallic surface, dramatic side lighting, high contrast",

        # Scene context test
        f"a 3d animated character, {base_desc}, standing on runway, airport background, soft morning light, depth of field, bokeh background"
    ]

    return prompts

def get_character_checkpoints(character: str) -> List[Path]:
    """Get all safetensors checkpoints for a character"""
    char_dir = LORA_BASE_DIR / character

    if not char_dir.exists():
        return []

    checkpoints = sorted(char_dir.glob("*.safetensors"))
    return checkpoints

def evaluate_checkpoint(
    pipeline: StableDiffusionXLPipeline,
    lora_path: Path,
    prompts: List[str],
    character: str,
    checkpoint_name: str,
    output_dir: Path
) -> Dict:
    """
    Evaluate a single LoRA checkpoint with multiple prompts and settings.

    Returns evaluation metrics and generated image paths.
    """

    print(f"  Testing checkpoint: {checkpoint_name}")

    checkpoint_dir = output_dir / checkpoint_name.replace('.safetensors', '')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load LoRA weights
    pipeline.load_lora_weights(str(lora_path))

    results = {
        "checkpoint": checkpoint_name,
        "character": character,
        "lora_path": str(lora_path),
        "generated_images": [],
        "prompts_tested": len(prompts),
        "total_images": 0
    }

    # Test each prompt with multiple LoRA scales and seeds
    for prompt_idx, prompt in enumerate(prompts, 1):
        prompt_dir = checkpoint_dir / f"prompt_{prompt_idx:02d}"
        prompt_dir.mkdir(exist_ok=True)

        for scale in LORA_SCALES:
            for seed_idx, seed in enumerate(SEEDS, 1):
                # Generate image
                generator = torch.Generator(device=pipeline.device).manual_seed(seed)

                image = pipeline(
                    prompt=prompt,
                    num_inference_steps=INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                    cross_attention_kwargs={"scale": scale}
                ).images[0]

                # Save image
                img_name = f"scale{scale:.1f}_seed{seed}.png"
                img_path = prompt_dir / img_name
                image.save(img_path)

                results["generated_images"].append({
                    "path": str(img_path.relative_to(output_dir)),
                    "prompt_idx": prompt_idx,
                    "lora_scale": scale,
                    "seed": seed
                })
                results["total_images"] += 1

        # Save prompt text
        with open(prompt_dir / "prompt.txt", 'w') as f:
            f.write(prompt)

    # Unload LoRA for next checkpoint
    pipeline.unload_lora_weights()

    print(f"    ✅ Generated {results['total_images']} test images")

    return results

def evaluate_character(character: str, char_info: Dict, pipeline: StableDiffusionXLPipeline) -> Dict:
    """Evaluate all checkpoints for a single character"""

    print(f"\n{'=' * 70}")
    print(f"📊 Evaluating: {character.upper()}")
    print(f"{'=' * 70}")

    # Get checkpoints
    checkpoints = get_character_checkpoints(character)

    if not checkpoints:
        print(f"⚠️  No checkpoints found for {character}")
        return None

    print(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.name}")

    # Generate test prompts
    prompts = generate_test_prompts(character, char_info)
    print(f"\nGenerated {len(prompts)} test prompts")

    # Create output directory
    output_dir = OUTPUT_BASE / character
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save prompts for reference
    with open(output_dir / "test_prompts.json", 'w') as f:
        json.dump(prompts, f, indent=2)

    # Evaluate each checkpoint
    checkpoint_results = []

    for i, ckpt_path in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] Testing {ckpt_path.name}...")

        try:
            result = evaluate_checkpoint(
                pipeline=pipeline,
                lora_path=ckpt_path,
                prompts=prompts,
                character=character,
                checkpoint_name=ckpt_path.name,
                output_dir=output_dir
            )
            checkpoint_results.append(result)

        except Exception as e:
            print(f"  ❌ Error testing {ckpt_path.name}: {e}")
            continue

    # Save evaluation results
    eval_report = {
        "character": character,
        "character_info": char_info,
        "total_checkpoints": len(checkpoints),
        "successful_tests": len(checkpoint_results),
        "test_prompts": prompts,
        "checkpoint_results": checkpoint_results,
        "test_settings": {
            "inference_steps": INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "lora_scales": LORA_SCALES,
            "seeds": SEEDS
        }
    }

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(eval_report, f, indent=2)

    print(f"\n✅ Evaluation complete for {character}")
    print(f"   Output: {output_dir}")
    print(f"   Report: {report_path}")

    return eval_report

def initialize_pipeline(device: str = "cuda") -> StableDiffusionXLPipeline:
    """Initialize SDXL pipeline for inference"""

    print("Initializing SDXL pipeline...")

    if not SDXL_MODEL.exists():
        print(f"❌ SDXL model not found: {SDXL_MODEL}")
        sys.exit(1)

    # Load pipeline
    pipeline = StableDiffusionXLPipeline.from_single_file(
        str(SDXL_MODEL),
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )

    # Load VAE if available
    if SDXL_VAE.exists():
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_single_file(str(SDXL_VAE), torch_dtype=torch.float16)
        pipeline.vae = vae

    pipeline = pipeline.to(device)

    # Enable optimizations
    pipeline.enable_attention_slicing()

    # RTX 5080 optimizations
    if torch.cuda.get_device_capability()[0] >= 9:  # Blackwell architecture
        print("  Detected RTX 5080 Blackwell - using native SDPA")
        pipeline.enable_model_cpu_offload()
    else:
        # Try xformers for older GPUs
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("  Enabled xFormers optimization")
        except:
            pass

    print("✅ Pipeline initialized")

    return pipeline

def main():
    print("=" * 70)
    print("Super Wings SDXL LoRA Evaluation")
    print("=" * 70)
    print()
    print("Anthropomorphic plane/helicopter characters from Super Wings")
    print("Testing character identity and visual consistency")
    print()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Load character information
    char_info_dict = load_character_info()

    # Initialize pipeline
    pipeline = initialize_pipeline()

    # Evaluate each character
    all_results = []

    for i, character in enumerate(CHARACTERS, 1):
        print(f"\n[{i}/{len(CHARACTERS)}] Processing {character}...")

        char_info = char_info_dict.get(character, {})

        result = evaluate_character(character, char_info, pipeline)

        if result:
            all_results.append(result)

    # Generate final summary report
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_characters": len(CHARACTERS),
        "characters_evaluated": len(all_results),
        "output_directory": str(OUTPUT_BASE),
        "character_results": all_results
    }

    summary_path = OUTPUT_BASE / "evaluation_summary.json"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Evaluated {len(all_results)} characters")
    print(f"   Output directory: {OUTPUT_BASE}")
    print(f"   Summary report: {summary_path}")

    # Print per-character stats
    print("\nPer-Character Results:")
    for result in all_results:
        char = result['character']
        successful = result['successful_tests']
        total = result['total_checkpoints']
        print(f"  - {char}: {successful}/{total} checkpoints tested")

    print("\nNext steps:")
    print("1. Review generated images in:", OUTPUT_BASE)
    print("2. Compare checkpoints visually")
    print("3. Select best performing checkpoint per character")
    print("4. Copy best checkpoints to production:")
    print(f"   cp /mnt/c/ai_models/lora_sdxl/super-wings/CHARACTER/BEST.safetensors \\")
    print(f"      /mnt/c/ai_models/lora_sdxl/production/")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
