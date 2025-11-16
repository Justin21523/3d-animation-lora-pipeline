#!/usr/bin/env python3
"""
Character-Aware Inpainting for 3D Animation

Purpose: High-quality inpainting with character awareness
Methods:
  - PowerPaint (ECCV 2024): Text-guided inpainting with character descriptions
  - LaMa (fallback): Traditional large mask inpainting

Usage:
    python character_inpainting.py \
      --input-dir /path/to/clustered \
      --output-dir /path/to/inpainted \
      --model powerpaint \
      --character-info docs/films/luca/characters/ \
      --batch-size 4
"""

import argparse
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict
import sys


# ============================================================================
# Model Loading
# ============================================================================

def load_powerpaint_model(config_path: str, device: str = "cuda"):
    """
    Load PowerPaint model for character-aware inpainting

    PowerPaint (ECCV 2024): Text-guided multi-task inpainting
    - Supports object removal, insertion, and outpainting
    - Can use character descriptions as negative prompts
    """
    try:
        from diffusers import StableDiffusionInpaintPipeline
        from diffusers import DPMSolverMultistepScheduler

        print(f"Loading PowerPaint model on {device}...")

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        inpaint_config = config['inpainting']
        model_checkpoint = inpaint_config['checkpoint']

        # Load pipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_checkpoint,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        )

        # Set scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        pipe = pipe.to(device)

        # Enable optimizations
        if device == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_vae_tiling()

        print("✓ PowerPaint model loaded successfully")
        return pipe, config

    except ImportError as e:
        print(f"❌ PowerPaint dependencies not installed: {e}")
        print("\nInstall with:")
        print("  pip install diffusers transformers accelerate")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load PowerPaint model: {e}")
        sys.exit(1)


def load_lama_model(device: str = "cuda"):
    """
    Load LaMa inpainting model (fallback)
    """
    try:
        from simple_lama_inpainting import SimpleLama

        print(f"Loading LaMa model on {device}...")
        model = SimpleLama(device=device)
        print("✓ LaMa model loaded successfully")
        return model

    except ImportError:
        print("❌ simple-lama-inpainting not installed!")
        print("\nInstall with:")
        print("  pip install simple-lama-inpainting")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load LaMa model: {e}")
        sys.exit(1)


# ============================================================================
# Character Information Loading
# ============================================================================

def load_character_info(character_dir: Path, cluster_name: str) -> Optional[Dict]:
    """
    Load character description for inpainting prompts

    Args:
        character_dir: Directory with character markdown files
        cluster_name: Name like "character_0", "luca_human", etc.

    Returns:
        dict with character description or None
    """
    if not character_dir or not character_dir.exists():
        return None

    # Try to find matching character file
    possible_names = [
        f"{cluster_name}.md",
        f"character_{cluster_name}.md",
        "character_luca.md" if "luca" in cluster_name.lower() else None,
        "character_alberto.md" if "alberto" in cluster_name.lower() else None,
    ]

    for name in possible_names:
        if name:
            char_file = character_dir / name
            if char_file.exists():
                # Extract description
                with open(char_file) as f:
                    content = f.read()
                    lines = content.split('\n')
                    description = []
                    for line in lines:
                        if line.strip() and not line.startswith('#') and len(line) > 20:
                            description.append(line.strip())
                            if len(description) >= 2:
                                break

                    if description:
                        return {
                            'name': cluster_name,
                            'description': ' '.join(description)
                        }

    return None


# ============================================================================
# Mask Creation
# ============================================================================

def create_inpainting_mask(image: np.ndarray, dilate_size: int = 8) -> np.ndarray:
    """
    Create inpainting mask from alpha channel
    """
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
    else:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    binary_mask = (alpha < 240).astype(np.uint8) * 255

    if dilate_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    return binary_mask


# ============================================================================
# PowerPaint Inpainting
# ============================================================================

def inpaint_with_powerpaint(
    image: np.ndarray,
    mask: np.ndarray,
    pipe,
    config: Dict,
    character_info: Optional[Dict] = None
) -> np.ndarray:
    """
    Inpaint using PowerPaint with character-aware prompts
    """
    from PIL import Image

    # Get config
    inpaint_cfg = config['inpainting']
    prompt_cfg = inpaint_cfg['prompts']
    inference_cfg = inpaint_cfg['inference']

    # Prepare image
    original_alpha = None
    if image.shape[2] == 4:
        original_alpha = image[:, :, 3].copy()
        image_rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(image_rgb)
    pil_mask = Image.fromarray(mask)

    # Build prompts
    positive_prompt = prompt_cfg['background_template'].format(
        scene_context="Italian coastal town, summer afternoon",
        animation_style="Pixar 3D animation",
        location="cobblestone street",
        time_of_day="warm sunlight"
    )

    negative_prompt = "character, person, figure, "
    if character_info:
        negative_prompt += f"Do not include: {character_info['description']}"
    else:
        negative_prompt += "people, characters, humans, artifacts, blur"

    # Run inference
    output = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=pil_image,
        mask_image=pil_mask,
        num_inference_steps=inference_cfg['num_steps'],
        guidance_scale=inference_cfg['guidance_scale'],
        strength=inference_cfg['strength'],
    )

    result_rgb = np.array(output.images[0])
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    # Restore alpha
    if original_alpha is not None:
        updated_alpha = original_alpha.copy()
        updated_alpha[mask > 0] = 255
        return np.dstack([result_bgr, updated_alpha])
    else:
        return result_bgr


# ============================================================================
# LaMa Inpainting (Fallback)
# ============================================================================

def inpaint_with_lama(
    image: np.ndarray,
    binary_mask: np.ndarray,
    lama_model,
    preserve_alpha: bool = True
) -> np.ndarray:
    """LaMa inpainting (fallback method)"""
    original_height, original_width = image.shape[:2]
    original_alpha = None

    if image.shape[2] == 4:
        original_alpha = image[:, :, 3].copy()
        alpha = image[:, :, 3:4] / 255.0
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        gray_bg = np.ones_like(rgb) * 128
        rgb_composite = (rgb * alpha + gray_bg * (1 - alpha)).astype(np.uint8)
    else:
        rgb_composite = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result_pil = lama_model(rgb_composite, binary_mask)
    result_rgb = np.array(result_pil)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    if result_bgr.shape[:2] != (original_height, original_width):
        result_bgr = cv2.resize(result_bgr, (original_width, original_height))

    original_bgr = image[:, :, :3]
    output_bgr = original_bgr.copy()
    mask_3ch = np.stack([binary_mask] * 3, axis=2) > 0
    output_bgr[mask_3ch] = result_bgr[mask_3ch]

    if preserve_alpha and original_alpha is not None:
        updated_alpha = original_alpha.copy()
        updated_alpha[binary_mask > 0] = 255
        return np.dstack([output_bgr, updated_alpha])
    else:
        return output_bgr


# ============================================================================
# Processing Pipeline
# ============================================================================

def process_cluster_directory(
    cluster_dir: Path,
    output_dir: Path,
    model,
    model_type: str,
    config: Optional[Dict] = None,
    character_info: Optional[Dict] = None,
    skip_existing: bool = True
):
    """Process all instances in a cluster directory"""
    cluster_name = cluster_dir.name
    output_cluster_dir = output_dir / cluster_name
    output_cluster_dir.mkdir(parents=True, exist_ok=True)

    instances = list(cluster_dir.glob("*.png"))
    stats = {"total": len(instances), "processed": 0, "skipped": 0, "no_mask": 0, "failed": 0}

    for instance_path in tqdm(instances, desc=f"  {cluster_name}", leave=False):
        output_path = output_cluster_dir / instance_path.name

        if skip_existing and output_path.exists():
            stats["skipped"] += 1
            continue

        try:
            image = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                stats["failed"] += 1
                continue

            binary_mask = create_inpainting_mask(image, dilate_size=8)

            if binary_mask.sum() == 0:
                cv2.imwrite(str(output_path), image)
                stats["no_mask"] += 1
                continue

            if model_type == "powerpaint":
                result = inpaint_with_powerpaint(image, binary_mask, model, config, character_info)
            else:
                result = inpaint_with_lama(image, binary_mask, model, preserve_alpha=True)

            cv2.imwrite(str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            stats["processed"] += 1

        except Exception as e:
            print(f"\n⚠️  Failed {instance_path.name}: {e}")
            stats["failed"] += 1

    return stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Character-Aware Inpainting")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["powerpaint", "lama"], default="lama")
    parser.add_argument("--config", type=str, default="configs/stages/inpainting/powerpaint.yaml")
    parser.add_argument("--character-info", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = "cpu"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CHARACTER-AWARE INPAINTING - {args.model.upper()}")
    print(f"{'='*70}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Load model
    config = None
    if args.model == "powerpaint":
        model, config = load_powerpaint_model(args.config, device=args.device)
    else:
        model = load_lama_model(device=args.device)

    cluster_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("character_")]
    if not cluster_dirs:
        print("❌ No clusters found!")
        return

    print(f"📂 Found {len(cluster_dirs)} clusters\n")

    char_info_dir = Path(args.character_info) if args.character_info else None
    total_stats = {"total": 0, "processed": 0, "skipped": 0, "no_mask": 0, "failed": 0}

    for cluster_dir in tqdm(cluster_dirs, desc="Clusters"):
        character_info = load_character_info(char_info_dir, cluster_dir.name) if char_info_dir else None

        stats = process_cluster_directory(
            cluster_dir, output_dir, model, args.model, config, character_info, args.skip_existing
        )

        for key in total_stats:
            total_stats[key] += stats[key]

    print(f"\n{'='*70}")
    print(f"COMPLETE - Processed: {total_stats['processed']}/{total_stats['total']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
