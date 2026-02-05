"""
Round-robin synthetic data generator
Generates 1 image per character per type in rotation for easier review
"""

import json
import logging
from pathlib import Path
import sys
import argparse
from datetime import datetime
import torch
from diffusers import StableDiffusionXLPipeline

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CHARACTERS = [
    # "alberto",  # Skipped - already completed
    # "alberto_seamonster",  # Skipped - handle later
    "barley_lightfoot", "bryce", "caleb",
    "elio", "giulia", "ian_lightfoot", "luca", "luca_seamonster",
    "miguel", "orion", "russell", "tyler"
]

TYPES = ["pose", "expression", "action"]

def load_config(config_path):
    """Load YAML config"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_lora_path(char_name, best_dir):
    """Get LoRA path for character"""
    lora_file = best_dir / f"BEST_{char_name}_lora_sdxl.safetensors"
    if lora_file.exists():
        return str(lora_file)
    return None

def load_checkpoint(checkpoint_path):
    """Load generation checkpoint to resume from where we left off"""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {
        "current_round": 0,
        "current_char_index": 0,
        "current_type_index": 0,
        "current_prompt_index": {},  # char/type -> prompt_index
        "total_generated": 0
    }

def save_checkpoint(checkpoint_path, checkpoint_data):
    """Save generation checkpoint"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def round_robin_generate(config_path, total_rounds=10):
    """
    Generate images in round-robin fashion:
    - Round 1: barley/pose(1), barley/expr(1), barley/action(1), bryce/pose(1), ...
    - Round 2: barley/pose(2), barley/expr(2), barley/action(2), ...
    - ...
    """
    config = load_config(config_path)

    # Paths - Updated to match YAML structure
    workspace_root = Path(config['workspace']['root'])
    data_dir = workspace_root / config['workspace']['subdirs']['generated_data']
    best_lora_dir = Path(config['models']['identity_loras_dir'])
    base_model = config['models']['base_model']
    checkpoint_dir = workspace_root / config['workspace']['subdirs']['checkpoints']
    checkpoint_file = checkpoint_dir / "round_robin_checkpoint.json"

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    logger.info(f"📍 Resuming from Round {checkpoint['current_round'] + 1}")

    # Initialize pipeline
    img_config = config['image_generation']
    logger.info(f"Loading SDXL base model from {base_model}")

    pipeline = StableDiffusionXLPipeline.from_single_file(
        base_model,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(img_config.get('device', 'cuda'))

    # Enable memory efficient attention
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        logger.info("✓ Enabled xformers memory efficient attention")
    except Exception as e:
        logger.warning(f"Could not enable xformers: {e}")

    # Generation parameters
    gen_params = {
        'height': img_config.get('height', 1024),
        'width': img_config.get('width', 1024),
        'guidance_scale': img_config.get('guidance_scale', 7.5),
        'num_inference_steps': img_config.get('num_inference_steps', 40),
    }

    negative_prompt = img_config.get('negative_prompt', '')
    current_lora = None

    # Round-robin generation
    for round_num in range(checkpoint['current_round'], total_rounds):
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 ROUND {round_num + 1}/{total_rounds}")
        logger.info(f"{'='*60}")

        for char_idx in range(checkpoint.get('current_char_index', 0), len(CHARACTERS)):
            char = CHARACTERS[char_idx]

            # Get LoRA
            lora_path = get_lora_path(char, best_lora_dir)
            if not lora_path:
                logger.warning(f"⚠️  Skipping {char}: LoRA not found")
                continue

            # Load LoRA if different from current
            if current_lora != lora_path:
                # Unload previous LoRA
                if current_lora is not None:
                    try:
                        pipeline.unfuse_lora()
                        pipeline.unload_lora_weights()
                    except:
                        pass

                # Load new LoRA
                logger.info(f"\n👤 Character: {char} - Loading LoRA...")
                pipeline.load_lora_weights(lora_path)
                pipeline.fuse_lora(lora_scale=img_config.get('lora_scale', 1.0))
                current_lora = lora_path
                logger.info(f"✓ LoRA loaded and fused")
            else:
                logger.info(f"\n👤 Character: {char} (LoRA already loaded)")

            for type_idx in range(checkpoint.get('current_type_index', 0), len(TYPES)):
                ptype = TYPES[type_idx]

                # Load prompts
                prompt_file = data_dir / char / ptype / "prompts_converted.json"
                if not prompt_file.exists():
                    logger.warning(f"  ⚠️  {ptype}: prompts not found at {prompt_file}")
                    continue

                with open(prompt_file, 'r') as f:
                    prompt_data = json.load(f)

                prompts = prompt_data['prompts']

                # Get current prompt index for this char/type (with cycling)
                key = f"{char}/{ptype}"
                raw_idx = checkpoint['current_prompt_index'].get(key, 0)
                prompt_idx = raw_idx % len(prompts)  # Cycle through prompts

                cycle_num = raw_idx // len(prompts) + 1
                if raw_idx >= len(prompts):
                    logger.info(f"  🔄 {ptype}: cycling prompts (cycle {cycle_num}, prompt {prompt_idx + 1}/{len(prompts)})")

                # Generate 1 image
                prompt = prompts[prompt_idx]
                output_dir = data_dir / char / ptype / "generated"
                output_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"  🎨 {ptype}: generating image {prompt_idx + 1}/{len(prompts)}")

                try:
                    # Generate with pipeline
                    output = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        **gen_params
                    )

                    image = output.images[0]

                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = output_dir / f"{ptype}_round{round_num + 1:03d}_img{prompt_idx + 1:03d}_{timestamp}.png"
                    image.save(image_path, quality=95, optimize=True)

                    # Update checkpoint
                    checkpoint['current_prompt_index'][key] = prompt_idx + 1
                    checkpoint['total_generated'] += 1
                    save_checkpoint(checkpoint_file, checkpoint)

                    logger.info(f"    ✓ Saved: {image_path.name}")

                    # Clean up GPU memory
                    del output
                    del image
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"    ✗ Error generating {char}/{ptype}: {e}")
                    torch.cuda.empty_cache()

            # Reset type index for next character
            checkpoint['current_type_index'] = 0

        # Move to next round
        checkpoint['current_round'] = round_num + 1
        checkpoint['current_char_index'] = 0
        save_checkpoint(checkpoint_file, checkpoint)

        logger.info(f"\n✓ Round {round_num + 1} completed. Total images: {checkpoint['total_generated']}")

    logger.info(f"\n🎉 All {total_rounds} rounds completed!")
    logger.info(f"📊 Total images generated: {checkpoint['total_generated']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round-robin synthetic data generation")
    parser.add_argument('--config', required=True, help="Path to config YAML")
    parser.add_argument('--rounds', type=int, default=10, help="Number of rounds (default: 10)")

    args = parser.parse_args()

    round_robin_generate(args.config, total_rounds=args.rounds)
