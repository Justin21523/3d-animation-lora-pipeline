#!/usr/bin/env python3
"""
Launch SDXL LoRA training for Inazuma Eleven characters.
Uses proper command-line arguments with Kohya_ss sd-scripts.
"""

import subprocess
import sys
from pathlib import Path
from typing import List

# Character configurations
CHARACTERS = {
    "Endou Mamoru": "endou_mamoru",
    "Fudou Akio": "fudou_akio",
    "Gouenji Shuuya": "gouenji_shuuya",
    "Inamori Asuto": "inamori_asuto",
    "Matsukaze Tenma": "matsukaze_tenma",
    "Nosaka Yuuma": "nosaka_yuuma",
    "Utsunomiya Toramaru": "utsunomiya_toramaru",
}

TRAINING_BASE_DIR = Path("/mnt/data/training/lora/inazuma_eleven")
DATASET_BASE_DIR = Path("/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented")
BASE_MODEL = Path("/mnt/c/ai_models/stable-diffusion/checkpoints/novaAnimeXL_ilV140.safetensors")
KOHYA_DIR = Path("/mnt/c/ai_projects/kohya_ss")

# Training parameters
TRAIN_PARAMS = {
    "batch_size": 2,
    "max_epochs": 10,
    "learning_rate": 1e-4,
    "network_dim": 64,
    "network_alpha": 32,
    "resolution": "1024,1024",
    "save_every_n_steps": 200,
    "save_every_n_epochs": 1,
}


def build_training_command(
    character_name: str,
    character_id: str,
) -> List[str]:
    """Build Kohya_ss training command."""

    # Use parent directory as train_data_dir (Kohya expects this structure)
    # The actual images are in: DATASET_BASE_DIR/1_character_id/
    train_data_dir = DATASET_BASE_DIR
    output_dir = TRAINING_BASE_DIR / character_id
    logging_dir = output_dir / "logs"

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "conda",
        "run",
        "-n",
        "kohya_ss",
        "python",
        str(KOHYA_DIR / "sd-scripts" / "train_network.py"),

        # Model
        "--pretrained_model_name_or_path", str(BASE_MODEL),
        "--output_name", f"{character_id}_sdxl_lora",
        "--train_data_dir", str(train_data_dir),
        "--save_model_as", "safetensors",
        "--save_precision", "bf16",

        # Output
        "--output_dir", str(output_dir),
        "--logging_dir", str(logging_dir),

        # Dataset
        "--caption_extension", ".txt",
        "--enable_bucket",
        "--resolution", TRAIN_PARAMS["resolution"],
        "--cache_latents",
        "--bucket_reso_steps", "64",
        "--bucket_no_upscale",
        "--min_bucket_reso", "256",
        "--max_bucket_reso", "2048",

        # Training
        "--mixed_precision", "bf16",
        "--full_bf16",
        "--gradient_checkpointing",
        "--max_train_epochs", str(TRAIN_PARAMS["max_epochs"]),
        "--max_train_steps", "1000",
        "--train_batch_size", str(TRAIN_PARAMS["batch_size"]),
        "--learning_rate", str(TRAIN_PARAMS["learning_rate"]),
        "--lr_scheduler", "cosine",
        "--optimizer_type", "AdamW8bit",
        "--save_every_n_steps", str(TRAIN_PARAMS["save_every_n_steps"]),
        "--save_every_n_epochs", str(TRAIN_PARAMS["save_every_n_epochs"]),
        "--clip_skip", "2",
        "--seed", "42",

        # Advanced
        "--noise_offset", "0.1",
        "--loss_type", "l2",
        "--max_token_length", "225",
        "--keep_tokens", "0",

        # Network (LoRA)
        "--network_module", "lycoris.kohya",
        "--network_dim", str(TRAIN_PARAMS["network_dim"]),
        "--network_alpha", str(TRAIN_PARAMS["network_alpha"]),
        "--network_dropout", "0",

        # Logging
        "--log_with", "tensorboard",
        "--log_prefix", f"{character_id}_training",

        # Sampling
        "--sample_every_n_epochs", "1",
        "--sample_prompts", f"1girl, inazuma_{character_id}, anime_style, looking_at_viewer, masterpiece",
        "--sample_sampler", "euler_a",
    ]

    return cmd


def run_training(character_name: str, character_id: str) -> int:
    """Run training for a single character."""
    print(f"\n{'='*70}")
    print(f"Starting SDXL LoRA training for {character_name}")
    print(f"{'='*70}\n")

    cmd = build_training_command(character_name, character_id)

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running training: {e}", file=sys.stderr)
        return 1


def main():
    """Run sequential training for all characters."""
    print("=" * 70)
    print("Inazuma Eleven SDXL LoRA Training Pipeline")
    print("=" * 70)
    print(f"Training Base Dir: {TRAINING_BASE_DIR}")
    print(f"Dataset Base Dir: {DATASET_BASE_DIR}")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Characters to train: {len(CHARACTERS)}")
    print()

    # Verify prerequisites
    if not BASE_MODEL.exists():
        print(f"ERROR: Base model not found: {BASE_MODEL}", file=sys.stderr)
        return 1

    if not DATASET_BASE_DIR.exists():
        print(f"ERROR: Dataset directory not found: {DATASET_BASE_DIR}", file=sys.stderr)
        return 1

    # Run training for each character
    results = {}
    for character_name, character_id in CHARACTERS.items():
        # Check for reorganized directory: 1_character_id
        train_data_subdir = DATASET_BASE_DIR / f"1_{character_id}"
        if not train_data_subdir.exists():
            print(f"WARNING: Training data not found for {character_name} ({train_data_subdir}), skipping", file=sys.stderr)
            results[character_name] = "SKIPPED"
            continue

        returncode = run_training(character_name, character_id)
        results[character_name] = "SUCCESS" if returncode == 0 else "FAILED"

    # Print summary
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    for character_name, status in results.items():
        print(f"{character_name}: {status}")
    print("=" * 70)

    return 0 if all(s in ["SUCCESS", "SKIPPED"] for s in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
