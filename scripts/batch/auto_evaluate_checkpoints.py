#!/usr/bin/env python3
"""
Automatic Checkpoint Evaluation for Synthetic LoRA Training

Monitors training output directory and automatically evaluates new checkpoints:
- Generates test images with standard prompts
- Saves visual comparison grids
- Tracks quality metrics
- Creates evaluation report

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import gc
import psutil

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image


class CheckpointEvaluator:
    def __init__(
        self,
        lora_dir: Path,
        output_dir: Path,
        base_model: str = "/mnt/c/ai_models/stable-diffusion-xl-base-1.0",
        device: str = "cuda"
    ):
        self.lora_dir = lora_dir
        self.output_dir = output_dir
        self.base_model = base_model
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluation state
        self.state_file = output_dir / "evaluation_state.json"
        self.load_state()

        # Memory management
        self.memory_threshold = 0.90  # Trigger cleanup at 90% usage

    def check_memory(self) -> Dict:
        """Check current memory usage"""
        # System RAM
        ram = psutil.virtual_memory()

        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
            gpu_reserved = torch.cuda.memory_reserved()
            gpu_allocated = torch.cuda.memory_allocated()
        else:
            gpu_mem = 0
            gpu_reserved = 0
            gpu_allocated = 0

        return {
            "ram_percent": ram.percent,
            "ram_available_gb": ram.available / (1024**3),
            "gpu_allocated_gb": gpu_allocated / (1024**3),
            "gpu_reserved_gb": gpu_reserved / (1024**3),
            "gpu_percent": gpu_mem * 100
        }

    def cleanup_memory(self):
        """Aggressive memory cleanup"""
        print("🧹 Cleaning up memory...")

        # Python garbage collection
        gc.collect()

        # PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        mem_info = self.check_memory()
        print(f"  RAM: {mem_info['ram_percent']:.1f}% used, {mem_info['ram_available_gb']:.1f} GB available")
        if torch.cuda.is_available():
            print(f"  GPU: {mem_info['gpu_allocated_gb']:.2f} GB allocated, {mem_info['gpu_reserved_gb']:.2f} GB reserved")
        print()

    def load_state(self):
        """Load evaluation state"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                self.state = json.load(f)
        else:
            self.state = {
                "evaluated_checkpoints": [],
                "last_check": None
            }

    def save_state(self):
        """Save evaluation state"""
        self.state["last_check"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_test_prompts(self, lora_type: str) -> List[str]:
        """Get standard test prompts based on LoRA type"""
        base_prompt = "a 3d animated character, pixar style, smooth shading, studio lighting"

        if lora_type == "pose":
            return [
                f"{base_prompt}, standing with arms crossed",
                f"{base_prompt}, sitting on a chair",
                f"{base_prompt}, running forward dynamically",
                f"{base_prompt}, jumping in mid-air"
            ]
        elif lora_type == "action":
            return [
                f"{base_prompt}, waving hand enthusiastically",
                f"{base_prompt}, reaching out to grab something",
                f"{base_prompt}, pointing at viewer",
                f"{base_prompt}, throwing a ball"
            ]
        else:  # expression
            return [
                f"{base_prompt}, smiling happily",
                f"{base_prompt}, surprised expression with wide eyes",
                f"{base_prompt}, sad and crying",
                f"{base_prompt}, angry and shouting"
            ]

    def detect_new_checkpoints(self) -> List[Path]:
        """Detect new checkpoints that haven't been evaluated"""
        evaluated = set(self.state["evaluated_checkpoints"])

        # Find all .safetensors files
        checkpoints = list(self.lora_dir.glob("*.safetensors"))

        # Filter out already evaluated
        new_checkpoints = [cp for cp in checkpoints if cp.name not in evaluated]

        # Sort by modification time
        new_checkpoints.sort(key=lambda x: x.stat().st_mtime)

        return new_checkpoints

    def evaluate_checkpoint(self, checkpoint_path: Path) -> Dict:
        """
        Evaluate a single checkpoint with OOM protection

        Returns:
            Evaluation results dict
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {checkpoint_path.name}")
        print(f"{'='*70}\n")

        # PRE-CLEANUP: Aggressive memory cleanup before loading
        self.cleanup_memory()

        # Check available memory
        mem_info = self.check_memory()
        if mem_info['ram_percent'] > 85:
            print(f"⚠️  WARNING: RAM usage high ({mem_info['ram_percent']:.1f}%)")
        if mem_info['gpu_allocated_gb'] > 12:  # RTX 5080 has 16GB
            print(f"⚠️  WARNING: GPU memory usage high ({mem_info['gpu_allocated_gb']:.2f} GB)")

        # Infer LoRA type from filename
        name = checkpoint_path.stem
        if "pose" in name.lower():
            lora_type = "pose"
        elif "action" in name.lower():
            lora_type = "action"
        elif "expression" in name.lower():
            lora_type = "expression"
        else:
            lora_type = "unknown"

        # Get test prompts
        prompts = self.get_test_prompts(lora_type)

        # Output directory for this checkpoint
        eval_dir = self.output_dir / checkpoint_path.stem
        eval_dir.mkdir(exist_ok=True)

        print(f"Loading pipeline...")
        print(f"  Base model: {self.base_model}")
        print(f"  LoRA: {checkpoint_path.name}")
        print()

        pipe = None
        try:
            # Load pipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(self.device)

            # Load LoRA weights
            pipe.load_lora_weights(str(checkpoint_path))

            # Generate images
            generated_images = []

            for i, prompt in enumerate(prompts, 1):
                print(f"[{i}/{len(prompts)}] Generating: {prompt[:60]}...")

                image = pipe(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).images[0]

                # Save image
                img_path = eval_dir / f"test_{i:02d}.png"
                image.save(img_path)
                generated_images.append(img_path)

                # Save prompt
                prompt_path = eval_dir / f"test_{i:02d}_prompt.txt"
                with open(prompt_path, 'w') as f:
                    f.write(prompt)

            # Create comparison grid
            print(f"\nCreating comparison grid...")
            grid_path = eval_dir / "comparison_grid.png"
            self.create_comparison_grid(generated_images, grid_path)

            result = {
                "checkpoint": checkpoint_path.name,
                "lora_type": lora_type,
                "evaluated_at": datetime.now().isoformat(),
                "images_generated": len(generated_images),
                "output_dir": str(eval_dir),
                "grid_path": str(grid_path),
                "success": True
            }

            print(f"✅ Evaluation complete: {eval_dir}")
            print()

            return result

        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ CUDA OOM Error: {e}")
            print("   Try reducing inference steps or using CPU offloading")
            print()

            return {
                "checkpoint": checkpoint_path.name,
                "evaluated_at": datetime.now().isoformat(),
                "success": False,
                "error": f"CUDA OOM: {str(e)}"
            }

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            print()

            return {
                "checkpoint": checkpoint_path.name,
                "evaluated_at": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            }

        finally:
            # POST-CLEANUP: Always cleanup, even on error
            if pipe is not None:
                del pipe

            self.cleanup_memory()

    def create_comparison_grid(self, image_paths: List[Path], output_path: Path):
        """Create a comparison grid from images"""
        images = [Image.open(p) for p in image_paths]

        # Calculate grid size
        n = len(images)
        cols = 2
        rows = (n + cols - 1) // cols

        # Get image size
        w, h = images[0].size

        # Create grid
        grid = Image.new('RGB', (w * cols, h * rows), color='white')

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            grid.paste(img, (col * w, row * h))

        grid.save(output_path)

    def monitor_and_evaluate(self, interval: int = 300):
        """
        Monitor directory and evaluate new checkpoints

        Args:
            interval: Check interval in seconds (default: 5 min)
        """
        print("=" * 70)
        print("Checkpoint Monitoring Mode")
        print("=" * 70)
        print(f"Monitoring: {self.lora_dir}")
        print(f"Check interval: {interval} seconds")
        print(f"Output: {self.output_dir}")
        print()
        print("Press Ctrl+C to stop")
        print()

        try:
            while True:
                new_checkpoints = self.detect_new_checkpoints()

                if new_checkpoints:
                    print(f"🔔 Found {len(new_checkpoints)} new checkpoint(s)")

                    for checkpoint in new_checkpoints:
                        result = self.evaluate_checkpoint(checkpoint)

                        # Mark as evaluated
                        self.state["evaluated_checkpoints"].append(checkpoint.name)
                        self.save_state()

                        # Save result
                        result_file = self.output_dir / f"{checkpoint.stem}_result.json"
                        with open(result_file, 'w') as f:
                            json.dump(result, f, indent=2)

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")


def main():
    parser = argparse.ArgumentParser(
        description="Automatic checkpoint evaluation for synthetic LoRAs"
    )
    parser.add_argument(
        "--lora-dir",
        type=Path,
        required=True,
        help="Directory containing LoRA checkpoints"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="/mnt/c/ai_models/stable-diffusion-xl-base-1.0",
        help="Base SDXL model path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor mode: continuously check for new checkpoints"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (monitor mode only)"
    )

    args = parser.parse_args()

    if not args.lora_dir.exists():
        print(f"❌ LoRA directory not found: {args.lora_dir}")
        return 1

    evaluator = CheckpointEvaluator(
        lora_dir=args.lora_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        device=args.device
    )

    if args.monitor:
        evaluator.monitor_and_evaluate(interval=args.interval)
    else:
        # One-time evaluation of all new checkpoints
        new_checkpoints = evaluator.detect_new_checkpoints()

        if not new_checkpoints:
            print("✅ No new checkpoints to evaluate")
            return 0

        print(f"Found {len(new_checkpoints)} new checkpoint(s) to evaluate")
        print()

        for checkpoint in new_checkpoints:
            result = evaluator.evaluate_checkpoint(checkpoint)

            # Mark as evaluated
            evaluator.state["evaluated_checkpoints"].append(checkpoint.name)
            evaluator.save_state()

            # Save result
            result_file = evaluator.output_dir / f"{checkpoint.stem}_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
