#!/usr/bin/env python3
"""
Automatic LoRA Checkpoint Evaluation
Monitors LoRA output directory and automatically generates test images for new checkpoints

Usage:
    python auto_evaluate_checkpoints.py \
        --lora-dir /path/to/lora/output \
        --base-model /path/to/base_model.safetensors \
        --prompts-file prompts.txt \
        --output-dir evaluations/

Author: LLMProvider Tooling
Date: 2025-11-21
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_checkpoints(lora_dir: Path) -> List[Path]:
    """Find all .safetensors checkpoint files."""
    return sorted(lora_dir.glob("*.safetensors"))


def load_prompts(prompts_file: Path) -> List[str]:
    """Load test prompts from file (one per line)."""
    if not prompts_file.exists():
        logger.warning(f"Prompts file not found: {prompts_file}")
        return []

    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    return prompts


def load_evaluated_checkpoints(state_file: Path) -> Dict:
    """Load list of already evaluated checkpoints."""
    if not state_file.exists():
        return {}

    try:
        with open(state_file) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading state file: {e}")
        return {}


def save_evaluated_checkpoints(state_file: Path, state: Dict):
    """Save evaluated checkpoints state."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def evaluate_checkpoint(
    checkpoint_path: Path,
    base_model: str,
    prompts: List[str],
    output_dir: Path,
    device: str = "cuda"
) -> bool:
    """
    Generate test images for a checkpoint.

    Returns:
        True if evaluation succeeded
    """
    try:
        import torch
        from diffusers import StableDiffusionPipeline

        logger.info(f"Evaluating checkpoint: {checkpoint_path.name}")

        # Create output directory for this checkpoint
        checkpoint_name = checkpoint_path.stem
        checkpoint_output = output_dir / checkpoint_name
        checkpoint_output.mkdir(parents=True, exist_ok=True)

        # Load pipeline
        logger.info(f"Loading base model: {base_model}")
        pipe = StableDiffusionPipeline.from_single_file(
            base_model,
            torch_dtype=torch.float16,
            safety_checker=None,
            load_safety_checker=False
        ).to(device)

        # Load LoRA weights
        logger.info(f"Loading LoRA: {checkpoint_path}")
        pipe.load_lora_weights(str(checkpoint_path))

        # Generate images for each prompt
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating image {i+1}/{len(prompts)}: {prompt[:60]}...")

            image = pipe(
                prompt,
                num_inference_steps=28,
                guidance_scale=7.5,
                generator=torch.Generator(device=device).manual_seed(42 + i)
            ).images[0]

            # Save image
            output_path = checkpoint_output / f"sample_{i:03d}.png"
            image.save(output_path)
            logger.info(f"Saved: {output_path}")

        # Save prompts used
        prompts_file = checkpoint_output / "prompts.txt"
        with open(prompts_file, 'w') as f:
            for prompt in prompts:
                f.write(f"{prompt}\n")

        # Cleanup
        del pipe
        torch.cuda.empty_cache()

        logger.info(f"✅ Evaluation complete: {checkpoint_name}")
        return True

    except Exception as e:
        logger.error(f"❌ Evaluation failed for {checkpoint_path.name}: {e}")
        return False


def monitor_and_evaluate(
    lora_dir: Path,
    base_model: str,
    prompts: List[str],
    output_dir: Path,
    state_file: Path,
    check_interval: int = 300,
    device: str = "cuda",
    once: bool = False
):
    """
    Monitor LoRA directory and evaluate new checkpoints.

    Args:
        lora_dir: Directory containing LoRA checkpoints
        base_model: Path to base SD model
        prompts: List of test prompts
        output_dir: Output directory for evaluations
        state_file: JSON file tracking evaluated checkpoints
        check_interval: Seconds between checks (default 5 minutes)
        device: Device to use for inference
        once: If True, evaluate once and exit
    """
    logger.info("=" * 60)
    logger.info("Automatic LoRA Checkpoint Evaluation")
    logger.info("=" * 60)
    logger.info(f"Monitoring: {lora_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Prompts: {len(prompts)}")
    logger.info(f"Check interval: {check_interval}s")
    logger.info("=" * 60)

    # Load state
    evaluated = load_evaluated_checkpoints(state_file)
    logger.info(f"Already evaluated: {len(evaluated)} checkpoints")

    while True:
        # Find all checkpoints
        checkpoints = find_checkpoints(lora_dir)
        logger.info(f"Found {len(checkpoints)} checkpoint(s)")

        # Evaluate new checkpoints
        new_evaluations = 0
        for checkpoint in checkpoints:
            checkpoint_key = checkpoint.name

            if checkpoint_key in evaluated:
                logger.debug(f"Skipping (already evaluated): {checkpoint_key}")
                continue

            # Evaluate this checkpoint
            logger.info(f"New checkpoint detected: {checkpoint_key}")
            success = evaluate_checkpoint(
                checkpoint,
                base_model,
                prompts,
                output_dir,
                device
            )

            # Record evaluation
            evaluated[checkpoint_key] = {
                "timestamp": time.time(),
                "success": success,
                "output_dir": str(output_dir / checkpoint.stem)
            }
            save_evaluated_checkpoints(state_file, evaluated)

            new_evaluations += 1

        if new_evaluations > 0:
            logger.info(f"✅ Evaluated {new_evaluations} new checkpoint(s)")

        # Exit if once mode
        if once:
            logger.info("One-time evaluation complete, exiting")
            break

        # Wait before next check
        logger.info(f"Waiting {check_interval}s before next check...")
        time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(
        description='Automatically evaluate LoRA checkpoints'
    )
    parser.add_argument(
        '--lora-dir',
        type=Path,
        required=True,
        help='Directory containing LoRA checkpoints'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors',
        help='Path to base Stable Diffusion model'
    )
    parser.add_argument(
        '--prompts-file',
        type=Path,
        help='File with test prompts (one per line). If not provided, uses default prompts from config.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for evaluations (default: lora_dir/evaluations)'
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=300,
        help='Seconds between checks (default: 300 = 5 minutes)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for inference (default: cuda)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Evaluate once and exit (don\'t monitor)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.lora_dir.exists():
        logger.error(f"LoRA directory not found: {args.lora_dir}")
        return

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.lora_dir / "evaluations"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # State file
    state_file = args.output_dir / "evaluation_state.json"

    # Load prompts
    if args.prompts_file and args.prompts_file.exists():
        prompts = load_prompts(args.prompts_file)
    else:
        # Default prompts
        logger.info("Using default prompts")
        prompts = [
            "a 3d animated character, pixar style, portrait, neutral expression, studio lighting",
            "a 3d animated character, pixar style, full body, standing, natural lighting",
            "a 3d animated character, pixar style, close-up face, smiling, soft lighting",
            "a 3d animated character, pixar style, three-quarter view, cinematic lighting"
        ]

    if not prompts:
        logger.error("No prompts provided!")
        return

    # Start monitoring
    monitor_and_evaluate(
        lora_dir=args.lora_dir,
        base_model=args.base_model,
        prompts=prompts,
        output_dir=args.output_dir,
        state_file=state_file,
        check_interval=args.check_interval,
        device=args.device,
        once=args.once
    )


if __name__ == '__main__':
    main()
