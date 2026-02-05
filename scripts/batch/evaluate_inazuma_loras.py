#!/usr/bin/env python3
"""
Evaluate Inazuma Eleven SDXL LoRA checkpoints with timeline-specific prompts.
Tests identity consistency, timeline conditioning, and prompt responsiveness.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from PIL import Image
import os

# Character metadata and test prompts
CHARACTERS = {
    "endou_mamoru": {
        "name": "Endou Mamoru (円堂守)",
        "trigger": "inazuma_endou_mamoru",
        "test_prompts": [
            "inazuma_endou_mamoru, timeline_original, goalkeeper_uniform, orange_headband, stadium, action_pose",
            "inazuma_endou_mamoru, timeline_go, adult, coach_outfit, confident_smile, training_ground",
            "inazuma_endou_mamoru, timeline_ares, soccer_match, diving_save, determined_expression",
            "inazuma_endou_mamoru, anime_style, close_up_portrait, orange_headband, spiky_brown_hair",
            "inazuma_endou_mamoru, timeline_orion, captain_armband, team_huddle, leadership_pose",
            "anime_boy, soccer_player, generic_style"  # Negative test (should NOT trigger character)
        ]
    },
    "gouenji_shuuya": {
        "name": "Gouenji Shuuya (豪炎寺修也)",
        "trigger": "inazuma_gouenji_shuuya",
        "test_prompts": [
            "inazuma_gouenji_shuuya, timeline_original, forward_position, white_headband, cool_expression",
            "inazuma_gouenji_shuuya, timeline_go, masked_identity, ishido_shuuji, authority_figure",
            "inazuma_gouenji_shuuya, timeline_ares, kidokawa_seishuu_uniform, ace_striker",
            "inazuma_gouenji_shuuya, fire_element, shooting_pose, intense_focus",
            "inazuma_gouenji_shuuya, anime_style, silver_hair, sharp_eyes, portrait",
            "anime_boy, soccer_player, generic_style"
        ]
    },
    "fudou_akio": {
        "name": "Fudou Akio (不動明王)",
        "trigger": "inazuma_fudou_akio",
        "test_prompts": [
            "inazuma_fudou_akio, timeline_original, midfielder, mohawk_hair, intimidating_expression",
            "inazuma_fudou_akio, timeline_go, defensive_stance, sharp_eyes, aggressive_pose",
            "inazuma_fudou_akio, mountain_element, powerful_presence, action_shot",
            "inazuma_fudou_akio, anime_style, portrait, mohawk_visible, confident_smirk",
            "anime_boy, soccer_player, generic_style"
        ]
    },
    "matsukaze_tenma": {
        "name": "Matsukaze Tenma (松風天馬)",
        "trigger": "inazuma_matsukaze_tenma",
        "test_prompts": [
            "inazuma_matsukaze_tenma, timeline_go, midfielder, brown_spiky_hair, goggles",
            "inazuma_matsukaze_tenma, wind_element, energetic_expression, running_pose",
            "inazuma_matsukaze_tenma, anime_style, cheerful_smile, close_up, goggles_visible",
            "anime_boy, soccer_player, generic_style"
        ]
    },
    "inamori_asuto": {
        "name": "Inamori Asuto (稲森明日人)",
        "trigger": "inazuma_inamori_asuto",
        "test_prompts": [
            "inazuma_inamori_asuto, timeline_ares, forward_position, orange_hair, cheerful_expression",
            "inazuma_inamori_asuto, wind_element, bright_eyes, energetic_pose",
            "inazuma_inamori_asuto, anime_style, portrait, orange_hair_visible, happy_face",
            "anime_boy, soccer_player, generic_style"
        ]
    },
    "nosaka_yuuma": {
        "name": "Nosaka Yuuma (野坂悠馬)",
        "trigger": "inazuma_nosaka_yuuma",
        "test_prompts": [
            "inazuma_nosaka_yuuma, timeline_orion, midfielder, purple_hair, glasses, strategic_expression",
            "inazuma_nosaka_yuuma, fire_element, intelligent_look, tactical_pose",
            "inazuma_nosaka_yuuma, anime_style, portrait, purple_hair, glasses_visible",
            "anime_boy, soccer_player, generic_style"
        ]
    },
    "utsunomiya_toramaru": {
        "name": "Utsunomiya Toramaru (宇都宮虎丸)",
        "trigger": "inazuma_utsunomiya_toramaru",
        "test_prompts": [
            "inazuma_utsunomiya_toramaru, timeline_original, forward_position, spiky_orange_hair, feral_expression",
            "inazuma_utsunomiya_toramaru, fire_element, energetic_pose, wild_look",
            "inazuma_utsunomiya_toramaru, anime_style, portrait, orange_spiky_hair_visible",
            "anime_boy, soccer_player, generic_style"
        ]
    }
}

# Negative prompt (standard SDXL)
NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, ugly, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, missing limb, floating limbs, disconnected limbs, malformed hands, long neck, long body, disgusting, poorly drawn, mutilated, mangled, old, surreal, 3d render, realistic, photo"


def load_pipeline(base_model_path: str, vae_path: str, device: str = "cuda"):
    """Load SDXL pipeline with custom VAE."""
    print(f"Loading SDXL pipeline from {base_model_path}...")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        torch_dtype=torch.float16
    )

    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained(
        base_model_path,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.to(device)

    # Enable optimizations
    pipe.enable_xformers_memory_efficient_attention()

    print("✓ Pipeline loaded successfully")
    return pipe


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str,
    lora_path: str = None,
    lora_scale: float = 0.8,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42
):
    """Generate single image with optional LoRA."""

    # Load LoRA if provided
    if lora_path:
        pipe.load_lora_weights(lora_path)
        print(f"  LoRA loaded: {Path(lora_path).name}")

    # Set seed for reproducibility
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Generate
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        cross_attention_kwargs={"scale": lora_scale} if lora_path else None
    ).images[0]

    # Unload LoRA
    if lora_path:
        pipe.unload_lora_weights()

    return image


def evaluate_character(
    pipe,
    char_id: str,
    lora_dir: Path,
    output_dir: Path,
    checkpoint_epoch: int = None
):
    """Evaluate all checkpoints for a character."""

    char_info = CHARACTERS.get(char_id)
    if not char_info:
        print(f"Warning: Unknown character {char_id}, skipping...")
        return

    print(f"\n{'='*60}")
    print(f"Evaluating: {char_info['name']}")
    print(f"{'='*60}")

    # Find checkpoints
    if checkpoint_epoch:
        checkpoint_pattern = f"*-{checkpoint_epoch:06d}.safetensors"
    else:
        checkpoint_pattern = "*.safetensors"

    checkpoints = sorted(lora_dir.glob(checkpoint_pattern))

    if not checkpoints:
        print(f"No checkpoints found in {lora_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoint(s)")

    # Create output directory for this character
    char_output_dir = output_dir / char_id
    char_output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each checkpoint
    for ckpt_path in checkpoints:
        ckpt_name = ckpt_path.stem
        print(f"\nTesting checkpoint: {ckpt_name}")

        ckpt_output_dir = char_output_dir / ckpt_name
        ckpt_output_dir.mkdir(exist_ok=True)

        # Test each prompt
        for i, prompt in enumerate(char_info["test_prompts"]):
            print(f"  Prompt {i+1}/{len(char_info['test_prompts'])}: {prompt[:60]}...")

            try:
                image = generate_image(
                    pipe,
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    lora_path=str(ckpt_path),
                    seed=42 + i  # Different seed per prompt
                )

                # Save image
                output_path = ckpt_output_dir / f"prompt_{i+1:02d}.png"
                image.save(output_path)

                # Save prompt
                prompt_path = ckpt_output_dir / f"prompt_{i+1:02d}.txt"
                prompt_path.write_text(prompt)

            except Exception as e:
                print(f"    Error: {e}")
                continue

    print(f"\n✓ Evaluation completed for {char_info['name']}")
    print(f"  Output: {char_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Inazuma Eleven SDXL LoRA checkpoints"
    )
    parser.add_argument(
        "--lora-base-dir",
        type=str,
        default="/mnt/data/ai_data/models/lora_sdxl/inazuma-eleven",
        help="Base directory containing character LoRA folders"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/lora_evaluation/inazuma_sdxl",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="/mnt/data/ai_data/models/sdxl/stable-diffusion-xl-base-1.0",
        help="Path to SDXL base model"
    )
    parser.add_argument(
        "--vae",
        type=str,
        default="/mnt/data/ai_data/models/sdxl/sdxl_vae.safetensors",
        help="Path to SDXL VAE"
    )
    parser.add_argument(
        "--character",
        type=str,
        default=None,
        help="Specific character to evaluate (default: all)"
    )
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=None,
        help="Specific checkpoint epoch to evaluate (e.g., 12 for final)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Setup
    lora_base_dir = Path(args.lora_base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    pipe = load_pipeline(args.base_model, args.vae, args.device)

    # Determine characters to evaluate
    if args.character:
        characters_to_eval = [args.character]
    else:
        characters_to_eval = list(CHARACTERS.keys())

    print(f"\nEvaluating {len(characters_to_eval)} character(s)")
    if args.checkpoint_epoch:
        print(f"Target checkpoint: epoch {args.checkpoint_epoch}")

    # Evaluate each character
    for char_id in characters_to_eval:
        lora_dir = lora_base_dir / f"{char_id}_identity"

        if not lora_dir.exists():
            print(f"Warning: LoRA directory not found: {lora_dir}")
            continue

        evaluate_character(
            pipe,
            char_id,
            lora_dir,
            output_dir,
            args.checkpoint_epoch
        )

    print(f"\n{'='*60}")
    print("Evaluation completed!")
    print(f"Results saved to: {output_dir}")
    print("='*60}")


if __name__ == "__main__":
    main()
