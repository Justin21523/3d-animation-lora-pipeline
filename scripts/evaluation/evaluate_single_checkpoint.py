#!/usr/bin/env python3
"""
Single Checkpoint Evaluator
自動評估單個 LoRA checkpoint 的質量
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# UPDATED Test Prompts for Evaluation
# CRITICAL: Added "single character" and "full body visible" to prevent duplicates and cropping
TEST_PROMPTS = [
    # Frontal views
    "single character, one person only, a 3d animated character, pixar uniform lighting, Luca Paguro, 12-year-old boy, large brown eyes, happy smile, looking at camera, full body visible, centered composition, pixar film quality, smooth shading",

    # Different expressions
    "single character, one person only, a 3d animated character, pixar uniform lighting, Luca Paguro, 12-year-old boy, concerned expression, furrowed brows, full body visible, centered, pixar film quality",

    # Different poses
    "single character, one person only, a 3d animated character, pixar uniform lighting, Luca Paguro, 12-year-old boy, looking up with wonder, excited expression, complete body visible, pixar film quality",

    # Side view
    "single character, one person only, a 3d animated character, pixar uniform lighting, Luca Paguro, 12-year-old boy, profile view, gentle smile, full body in frame, pixar film quality",

    # Different clothing
    "single character, one person only, a 3d animated character, pixar uniform lighting, Luca Paguro, 12-year-old boy, blue plaid shirt, happy expression, full body visible, pixar film quality",

    # Portrait (head and shoulders visible)
    "single character, one person only, a 3d animated character, pixar uniform lighting, Luca Paguro, 12-year-old boy, portrait shot, head and shoulders visible, soft smile, large brown eyes, centered, pixar film quality",

    # Three-quarter view
    "single character, one person only, a 3d animated character, pixar uniform lighting, Luca Paguro, 12-year-old boy, curious expression, three-quarter view, full body visible, pixar film quality",

    # Full character standing
    "single character, one person only, a 3d animated character, pixar uniform lighting, Luca Paguro, 12-year-old boy, standing pose, friendly smile, complete body from head to toe visible, centered composition, pixar film quality"
]

# ENHANCED Negative Prompt for Pixar-style 3D Animation
# CRITICAL: Strong controls for common SD 1.5 + LoRA artifacts
NEGATIVE_PROMPT = (
    # === COMPOSITION ISSUES (HIGHEST PRIORITY) ===
    "multiple people, two people, three people, many people, crowd, group, "
    "duplicate character, clones, twins, multiple boys, multiple persons, "
    "cropped head, cut off head, cropped face, head out of frame, "
    "cropped body, cut off, incomplete, partial body, "
    "bad framing, bad composition, off-center badly, "

    # === ANATOMY & PROPORTION ISSUES ===
    "bad anatomy, wrong anatomy, bad proportions, gross proportions, "
    "unnatural proportions, distorted proportions, asymmetric, "
    "extra limbs, extra arms, extra legs, extra fingers, extra hands, "
    "missing limbs, missing arms, missing legs, missing hands, "
    "floating limbs, disconnected limbs, detached limbs, "
    "malformed limbs, deformed limbs, mutated hands, poorly drawn hands, "
    "long neck, long body, elongated, stretched, "
    "big head, small head, huge eyes, tiny eyes, "

    # === QUALITY & ARTIFACTS ===
    "low quality, worst quality, low res, lowres, blurry, out of focus, "
    "jpeg artifacts, compression artifacts, noise, grainy, pixelated, "
    "watermark, text, signature, username, artist name, "
    "ugly, poorly drawn, bad art, amateur, sketch, "

    # === STYLE MISMATCHES ===
    "realistic, photorealistic, photography, photo, hyperrealistic, "
    "2d anime, manga, comic, cartoon flat, cel shading, line art, "
    "low poly, blocky, minecraft style, voxel, "

    # === LIGHTING & COLOR ISSUES ===
    "overexposed, underexposed, washed out, oversaturated, undersaturated, "
    "harsh lighting, bad lighting, dark shadows, high contrast shadows, "

    # === FACE ISSUES ===
    "deformed face, disfigured face, bad eyes, crossed eyes, lazy eye, "
    "open mouth, teeth, weird expression, "

    # === MISC ===
    "mutation, mutated, horror, creepy, nsfw"
)


def load_checkpoint(checkpoint_path, base_model_path, device="cuda"):
    """Load SD pipeline with LoRA checkpoint"""
    print(f"📦 [DEBUG] Starting load_checkpoint", flush=True)
    print(f"📦 Loading base model: {base_model_path}", flush=True)

    try:
        print(f"   [DEBUG] Attempting from_single_file...", flush=True)
        # Try loading from single file (for safetensors base model)
        pipe = StableDiffusionPipeline.from_single_file(
            base_model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            load_safety_checker=False,
        )
        print(f"   [DEBUG] from_single_file successful", flush=True)
    except Exception as e:
        print(f"   Warning: from_single_file failed ({e}), trying from_pretrained...", flush=True)
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        print(f"   [DEBUG] from_pretrained successful", flush=True)

    print(f"   [DEBUG] Moving pipeline to {device}...", flush=True)
    pipe = pipe.to(device)
    print(f"   [DEBUG] Pipeline moved to {device}", flush=True)

    print(f"📦 Loading LoRA checkpoint: {checkpoint_path}", flush=True)

    try:
        print(f"   [DEBUG] Attempting load_lora_weights...", flush=True)
        # Try to load LoRA weights with adapter_name
        pipe.load_lora_weights(
            os.path.dirname(checkpoint_path),
            weight_name=os.path.basename(checkpoint_path),
            adapter_name="luca"
        )
        print("   ✓ LoRA loaded successfully", flush=True)
    except Exception as e:
        print(f"   Warning: Failed to load LoRA ({e})", flush=True)
        print("   Continuing with base model only", flush=True)

    print(f"   [DEBUG] load_checkpoint completed", flush=True)
    return pipe


def generate_samples(pipe, prompts, output_dir, seed=42, num_inference_steps=40):
    """Generate test samples using the pipeline

    Args:
        num_inference_steps: Increased from 30 to 40 for better quality
    """
    print(f"   [DEBUG] Starting generate_samples", flush=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   [DEBUG] Output directory created: {output_dir}", flush=True)

    print(f"   [DEBUG] Creating generator with seed {seed}", flush=True)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    print(f"   [DEBUG] Generator created", flush=True)

    results = []

    for idx, prompt in enumerate(prompts):
        print(f"🎨 Generating sample {idx+1}/{len(prompts)}", flush=True)
        print(f"   Prompt: {prompt[:80]}...", flush=True)

        print(f"   [DEBUG] Starting pipeline inference...", flush=True)
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=num_inference_steps,  # 40 steps for better quality
            generator=generator,
            guidance_scale=9.0,  # Increased from 7.5 to 9.0 for stronger negative prompt adherence
        ).images[0]
        print(f"   [DEBUG] Pipeline inference completed", flush=True)

        # Save image
        image_path = output_dir / f"sample_{idx:02d}.png"
        print(f"   [DEBUG] Saving image to {image_path}", flush=True)
        image.save(image_path)
        print(f"   [DEBUG] Image saved successfully", flush=True)

        results.append({
            "index": idx,
            "prompt": prompt,
            "image_path": str(image_path),
        })

        print(f"   ✓ Saved: {image_path.name}", flush=True)

    print(f"   [DEBUG] generate_samples completed, {len(results)} samples", flush=True)
    return results


def calculate_basic_metrics(results):
    """Calculate basic image quality metrics"""
    metrics = {
        "total_samples": len(results),
        "avg_brightness": [],
        "avg_contrast": [],
    }

    for result in results:
        img = Image.open(result["image_path"]).convert("RGB")
        img_array = np.array(img)

        # Calculate brightness (mean pixel value)
        brightness = np.mean(img_array) / 255.0
        metrics["avg_brightness"].append(float(brightness))

        # Calculate contrast (std of pixel values)
        contrast = np.std(img_array) / 255.0
        metrics["avg_contrast"].append(float(contrast))

    # Calculate averages
    metrics["mean_brightness"] = float(np.mean(metrics["avg_brightness"]))
    metrics["mean_contrast"] = float(np.mean(metrics["avg_contrast"]))

    return metrics


def save_results(results, metrics, output_dir, checkpoint_name):
    """Save evaluation results and metrics"""
    output_dir = Path(output_dir)

    # Save generation results
    results_file = output_dir / "generation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save metrics
    metrics_file = output_dir / "metrics.json"
    full_metrics = {
        "checkpoint": checkpoint_name,
        "evaluation_time": datetime.now().isoformat(),
        "metrics": metrics,
    }

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(full_metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to: {output_dir}")
    print(f"   - Images: {metrics['total_samples']} samples")
    print(f"   - Metrics: {metrics_file}")

    return results_file, metrics_file


def create_summary_report(output_dir, metrics):
    """Create a human-readable summary report"""
    output_dir = Path(output_dir)
    report_file = output_dir / "EVALUATION_SUMMARY.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("CHECKPOINT EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Checkpoint: {metrics.get('checkpoint', 'Unknown')}\n")
        f.write(f"Evaluation Time: {metrics.get('evaluation_time', 'Unknown')}\n\n")

        f.write("METRICS:\n")
        f.write("-" * 60 + "\n")

        m = metrics.get('metrics', {})
        f.write(f"Total Samples: {m.get('total_samples', 0)}\n")
        f.write(f"Mean Brightness: {m.get('mean_brightness', 0):.4f}\n")
        f.write(f"Mean Contrast: {m.get('mean_contrast', 0):.4f}\n\n")

        f.write("NOTES:\n")
        f.write("-" * 60 + "\n")
        f.write("- Pixar 風格期望：低對比度 (0.15-0.25)\n")
        f.write("- 適度亮度 (0.4-0.6)\n")
        f.write("- 均勻光照特徵\n\n")

        # Analysis
        brightness = m.get('mean_brightness', 0)
        contrast = m.get('mean_contrast', 0)

        f.write("QUICK ANALYSIS:\n")
        f.write("-" * 60 + "\n")

        if 0.4 <= brightness <= 0.6:
            f.write("✓ Brightness: Good (Pixar-like)\n")
        else:
            f.write("⚠ Brightness: Outside optimal range\n")

        if 0.15 <= contrast <= 0.25:
            f.write("✓ Contrast: Good (Pixar-like low contrast)\n")
        else:
            f.write("⚠ Contrast: Outside optimal range\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"📊 Summary report: {report_file}")

    return report_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate single LoRA checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--base-model", default="/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors", help="Base SD model path")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of test samples")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 60)
    print("🔍 SINGLE CHECKPOINT EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)
    print()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load pipeline
    try:
        pipe = load_checkpoint(args.checkpoint, args.base_model, args.device)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        sys.exit(1)

    # Generate samples
    print()
    print("🎨 Generating test samples...")
    print("-" * 60)

    prompts = TEST_PROMPTS[:args.num_samples]
    results = generate_samples(pipe, prompts, args.output_dir, seed=args.seed)

    # Calculate metrics
    print()
    print("📊 Calculating metrics...")
    print("-" * 60)

    metrics = calculate_basic_metrics(results)

    # Save results
    print()
    print("💾 Saving results...")
    print("-" * 60)

    checkpoint_name = os.path.basename(args.checkpoint)
    save_results(results, metrics, args.output_dir, checkpoint_name)

    # Create summary
    summary_metrics = {
        "checkpoint": checkpoint_name,
        "evaluation_time": datetime.now().isoformat(),
        "metrics": metrics,
    }
    create_summary_report(args.output_dir, summary_metrics)

    print()
    print("=" * 60)
    print("✅ EVALUATION COMPLETE")
    print("=" * 60)
    print(f"📁 Results saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
