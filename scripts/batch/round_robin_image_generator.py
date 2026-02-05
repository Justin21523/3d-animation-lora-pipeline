#!/usr/bin/env python3
"""
Round-Robin Image Generator for Synthetic Data Pipeline
========================================================

Generates images in round-robin fashion across all character/lora combinations.
Each round generates 1 image per combination.
This allows seeing results from all characters quickly rather than waiting
for one character to complete entirely.

Round structure:
- num_characters × num_lora_types = combinations per round
- num_prompts × num_images_per_prompt = rounds total
- combinations × rounds = total images

Usage:
    python round_robin_image_generator.py --config configs/batch/synthetic_data_generation.yaml

Author: LLMProvider Tooling
Date: 2025-12-06
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Disable xformers to use SDPA instead
os.environ["XFORMERS_DISABLED"] = "1"

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RoundRobinGenerator:
    """Round-robin image generator for synthetic data pipeline."""

    def __init__(self, config_path: str, resume: bool = True):
        self.config = self._load_config(config_path)
        self.resume = resume

        # Extract settings
        self.workspace_root = Path(self.config['workspace']['root'])
        self.generated_data_dir = self.workspace_root / self.config['workspace']['subdirs']['generated_data']
        self.checkpoint_dir = self.workspace_root / self.config['workspace']['subdirs']['checkpoints']
        rr_cfg = self.config.get("round_robin", {}) or {}
        checkpoint_name = rr_cfg.get("checkpoint_filename", "round_robin_progress.json")
        self.filename_suffix = str(rr_cfg.get("filename_suffix", "")).strip()
        self.checkpoint_file = self.checkpoint_dir / checkpoint_name

        # Model settings
        self.base_model_path = self.config['models']['base_model']
        self.lora_dir = Path(self.config['models']['identity_loras_dir'])

        # Generation settings
        self.gen_config = self.config['image_generation']
        self.num_images_per_prompt = self.gen_config['num_images_per_prompt']  # 10
        self.num_inference_steps = self.gen_config['num_inference_steps']
        self.guidance_scale = self.gen_config['guidance_scale']
        self.height = self.gen_config['height']
        self.width = self.gen_config['width']
        self.negative_prompt = self.gen_config['negative_prompt']
        self.lora_scale = self.gen_config['lora_scale']

        # Characters and lora types
        self.characters = self.config['characters']
        self.lora_types = self.config['lora_types']
        self.num_prompts = self.config['vocabulary_generation']['num_prompts_per_type']  # 100

        # Calculate totals
        self.num_combinations = len(self.characters) * len(self.lora_types)
        self.total_rounds = self.num_prompts * self.num_images_per_prompt
        self.total_images = self.num_combinations * self.total_rounds

        # Pipeline (lazy load)
        self.pipe = None
        self.current_lora = None

        # Progress tracking
        self.progress = self._load_progress()

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_progress(self) -> dict:
        """Load or initialize progress tracking."""
        if self.resume and self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                progress = json.load(f)
                print(f"📂 Resumed from checkpoint: round {progress.get('current_round', 0)}")
                return progress

        return {
            'current_round': 0,
            'completed_images': 0,
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'combination_progress': {}  # {char_type: image_count}
        }

    def _save_progress(self):
        """Save progress to checkpoint."""
        self.progress['last_update'] = datetime.now().isoformat()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def _load_pipeline(self):
        """Load SDXL pipeline."""
        if self.pipe is not None:
            return

        print("🔄 Loading SDXL pipeline...")
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

        self.pipe = StableDiffusionXLPipeline.from_single_file(
            self.base_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        # Use DPM++ 2M Karras scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )

        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

        print("✅ Pipeline loaded")

    def _load_lora(self, character: str):
        """Load LoRA for character."""
        if self.current_lora == character:
            return True

        # Unload previous LoRA
        if self.current_lora is not None:
            self.pipe.unload_lora_weights()

        # Find LoRA file with EXACT character name matching
        # Use precise pattern: BEST_{character}_lora_sdxl.safetensors
        lora_pattern = f"BEST_{character}_lora_sdxl.safetensors"
        lora_path = self.lora_dir / lora_pattern

        if not lora_path.exists():
            # Fallback: try other patterns but ensure exact character match
            # Avoid matching "alberto" with "alberto_seamonster"
            all_loras = list(self.lora_dir.glob("BEST_*.safetensors"))
            matching_loras = []
            for lora in all_loras:
                # Extract character name from filename: BEST_{name}_lora_sdxl.safetensors or BEST_{name}_*.safetensors
                name = lora.name.replace("BEST_", "").replace("_lora_sdxl.safetensors", "").replace(".safetensors", "")
                # Handle cases like BEST_character_epoch5_lora_sdxl.safetensors
                name_parts = name.split("_")
                # Check if the character name matches exactly
                if name == character or (len(name_parts) >= 1 and "_".join(name_parts[:len(character.split("_"))]) == character):
                    # Additional check: make sure we're not matching a prefix
                    # e.g., "alberto" should not match "alberto_seamonster"
                    remaining = name[len(character):]
                    if remaining == "" or remaining.startswith("_lora") or remaining.startswith("_epoch"):
                        matching_loras.append(lora)

            if not matching_loras:
                print(f"⚠️ No LoRA found for {character}, skipping...")
                self.current_lora = None
                return False

            lora_path = matching_loras[0]

        print(f"🎨 Loading LoRA: {lora_path.name}")

        self.pipe.load_lora_weights(str(lora_path))
        self.current_lora = character
        return True

    def _get_prompts(self, character: str, lora_type: str) -> List[dict]:
        """Load prompts for character/lora_type combination."""
        prompts_file = self.generated_data_dir / character / lora_type / "prompts.json"

        if not prompts_file.exists():
            print(f"⚠️ Prompts file not found: {prompts_file}")
            return []

        with open(prompts_file, 'r') as f:
            data = json.load(f)

        return data.get('prompts', [])

    def _get_output_path(self, character: str, lora_type: str, prompt_idx: int, image_idx: int) -> Path:
        """Get output path for generated image."""
        output_dir = self.generated_data_dir / character / lora_type / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{self.filename_suffix}" if self.filename_suffix else ""
        return output_dir / f"prompt_{prompt_idx:04d}_img_{image_idx:02d}{suffix}.png"

    def _get_caption_path(self, character: str, lora_type: str, prompt_idx: int, image_idx: int) -> Path:
        """Get caption path for generated image."""
        output_dir = self.generated_data_dir / character / lora_type / "txt_captions"
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{self.filename_suffix}" if self.filename_suffix else ""
        return output_dir / f"prompt_{prompt_idx:04d}_img_{image_idx:02d}{suffix}.txt"

    def _get_caption_for_prompt(self, character: str, lora_type: str, prompt_idx: int) -> Optional[str]:
        """Get caption for a specific prompt index."""
        captions_file = self.generated_data_dir / character / lora_type / "captions.json"

        if not captions_file.exists():
            return None

        with open(captions_file, 'r') as f:
            data = json.load(f)

        captions = data.get('captions', [])
        if prompt_idx < len(captions):
            return captions[prompt_idx].get('caption', None)

        return None

    def generate_round(self, round_num: int) -> int:
        """Generate one image per combination for this round.

        Returns number of images generated.
        """
        # Calculate which prompt and image index this round corresponds to
        prompt_idx = round_num // self.num_images_per_prompt  # 0-99
        image_idx = round_num % self.num_images_per_prompt    # 0-9

        print(f"\n{'='*70}")
        print(
            f"🔄 ROUND {round_num + 1}/{self.total_rounds} | "
            f"Prompt {prompt_idx + 1}/{self.num_prompts}, "
            f"Image {image_idx + 1}/{self.num_images_per_prompt}"
        )
        print(f"{'='*70}")

        images_generated = 0

        for char_idx, character in enumerate(self.characters):
            # Load LoRA for this character
            if not self._load_lora(character):
                continue

            # Get prompts for this character
            prompts_cache = {}

            for type_idx, lora_type in enumerate(self.lora_types):
                combo_key = f"{character}_{lora_type}"
                combo_num = char_idx * len(self.lora_types) + type_idx + 1

                # Check if already generated
                output_path = self._get_output_path(character, lora_type, prompt_idx, image_idx)
                if output_path.exists():
                    print(f"  [{combo_num}/{self.num_combinations}] {combo_key}: Already exists, skipping")
                    continue

                # Get prompts if not cached
                if character not in prompts_cache:
                    prompts_cache[character] = {}
                if lora_type not in prompts_cache[character]:
                    prompts_cache[character][lora_type] = self._get_prompts(character, lora_type)

                prompts = prompts_cache[character][lora_type]
                if prompt_idx >= len(prompts):
                    print(f"  [{combo_num}/{self.num_combinations}] {combo_key}: No prompt at index {prompt_idx}")
                    continue

                prompt = prompts[prompt_idx]['prompt']

                # Generate image
                print(f"  [{combo_num}/{self.num_combinations}] {combo_key}: Generating...", end=" ", flush=True)

                try:
                    # Random seed for diversity
                    seed = random.randint(0, 2**32 - 1)
                    generator = torch.Generator(device="cuda").manual_seed(seed)

                    image = self.pipe(
                        prompt=prompt,
                        negative_prompt=self.negative_prompt,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        height=self.height,
                        width=self.width,
                        generator=generator,
                        cross_attention_kwargs={"scale": self.lora_scale},
                    ).images[0]

                    # Save image
                    image.save(output_path)

                    # Save caption as txt file
                    caption = self._get_caption_for_prompt(character, lora_type, prompt_idx)
                    if caption:
                        caption_path = self._get_caption_path(character, lora_type, prompt_idx, image_idx)
                        with open(caption_path, 'w') as f:
                            f.write(caption)

                    print(f"✅ Saved (seed={seed})")
                    images_generated += 1

                    # Update progress
                    self.progress['completed_images'] += 1
                    if combo_key not in self.progress['combination_progress']:
                        self.progress['combination_progress'][combo_key] = 0
                    self.progress['combination_progress'][combo_key] += 1

                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue

                # Clear CUDA cache periodically
                if images_generated % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        return images_generated

    def run(self, start_round_override: Optional[int] = None):
        """Run the round-robin generation."""
        print("="*70)
        print("🚀 ROUND-ROBIN IMAGE GENERATION")
        print("="*70)
        print(f"Characters: {len(self.characters)}")
        print(f"LoRA Types: {len(self.lora_types)}")
        print(f"Combinations per round: {self.num_combinations}")
        print(f"Prompts per combination: {self.num_prompts}")
        print(f"Images per prompt: {self.num_images_per_prompt}")
        print(f"Total rounds: {self.total_rounds}")
        print(f"Total images: {self.total_images:,}")
        print(f"Starting from round: {self.progress['current_round'] + 1}")
        print("="*70)

        # Load pipeline
        self._load_pipeline()

        if start_round_override is not None:
            self.progress['current_round'] = int(start_round_override)
            self._save_progress()

        start_round = int(self.progress.get('current_round', 0))
        if start_round >= self.total_rounds and int(self.progress.get('completed_images', 0)) < int(self.total_images):
            # Finished iterating rounds, but some images failed and remain missing.
            # Re-scan from round 0 and only generate missing files (existing files are skipped).
            print("♻️  Rounds completed but dataset incomplete; re-scanning from round 0 to fill missing images.")
            start_round = 0
            self.progress['current_round'] = 0
            self._save_progress()

        start_time = time.time()

        for round_num in range(start_round, self.total_rounds):
            round_start = time.time()

            images_generated = self.generate_round(round_num)

            round_time = time.time() - round_start
            total_time = time.time() - start_time

            # Update progress
            self.progress['current_round'] = round_num + 1
            self._save_progress()

            # Print stats
            completed = self.progress['completed_images']
            remaining = self.total_images - completed
            avg_time_per_image = total_time / max(1, completed - (start_round * self.num_combinations))
            eta_seconds = remaining * avg_time_per_image
            eta_hours = eta_seconds / 3600

            print(f"\n📊 Round {round_num + 1} complete: {images_generated} images in {round_time:.1f}s")
            print(f"   Total: {completed:,}/{self.total_images:,} ({100*completed/self.total_images:.1f}%)")
            print(f"   ETA: {eta_hours:.1f} hours")

            # Clear memory between rounds
            torch.cuda.empty_cache()
            gc.collect()

        print("\n" + "="*70)
        print("🎉 GENERATION COMPLETE!")
        print(f"   Total images: {self.progress['completed_images']:,}")
        print(f"   Total time: {(time.time() - start_time)/3600:.1f} hours")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Round-Robin Image Generator")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="Resume from checkpoint (default)")
    group.add_argument("--no-resume", action="store_true", help="Start fresh, ignore checkpoints")
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Re-scan from round 0 and generate only missing files (does not overwrite existing images).",
    )
    args = parser.parse_args()

    generator = RoundRobinGenerator(
        config_path=args.config,
        resume=not args.no_resume
    )
    generator.run(start_round_override=0 if args.fill_missing else None)


if __name__ == "__main__":
    main()
