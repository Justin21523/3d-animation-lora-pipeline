#!/usr/bin/env python3
"""
SDXL Multi-LoRA Compositor
Generate images using multiple SDXL LoRA models simultaneously with different weight combinations.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from datetime import datetime

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.utils.logger import setup_logger


class SDXLMultiLoRACompositor:
    """Compose multiple SDXL LoRA models with configurable weights."""

    def __init__(self, base_model: str, device: str = "cuda", enable_cpu_offload: bool = True):
        """
        Initialize compositor.

        Args:
            base_model: Path to SDXL base model
            device: Device for inference
            enable_cpu_offload: Enable CPU offloading for memory efficiency
        """
        self.device = device
        logging.info(f"Loading SDXL base model: {base_model}")

        self.pipe = StableDiffusionXLPipeline.from_single_file(
            base_model,
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        # Use DPM++ 2M Karras scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True
        )

        if enable_cpu_offload:
            try:
                # Important: keep pipeline on CPU and let offload manage GPU transfers.
                # Moving the full SDXL pipeline to GPU first can OOM on 16GB.
                self.pipe.enable_model_cpu_offload()
                logging.info("✓ Model CPU offload enabled")
            except Exception as e:
                logging.warning(f"CPU offload failed: {e}")
                self.pipe = self.pipe.to(device)
        else:
            self.pipe = self.pipe.to(device)

        self.loaded_loras: Dict[str, Path] = {}

    def load_loras(self, lora_configs: List[Dict[str, any]]) -> bool:
        """
        Load multiple LoRA models.

        Args:
            lora_configs: List of dicts with 'path', 'name', 'weight'

        Returns:
            True if at least one LoRA loaded successfully
        """
        import sys

        print(f"\n{'='*80}", flush=True)
        print(f"DEBUG: Starting LoRA loading process", flush=True)
        print(f"DEBUG: Config count: {len(lora_configs)}", flush=True)
        logging.info(f"\n{'='*80}")
        logging.info(f"Starting LoRA loading process")
        logging.info(f"Received {len(lora_configs)} LoRA configs")

        # Unload previous LoRAs (only if LoRAs were previously loaded)
        if self.loaded_loras:
            print(f"DEBUG: Unloading {len(self.loaded_loras)} previous LoRAs", flush=True)
            logging.info(f"Unloading {len(self.loaded_loras)} previous LoRAs")
            try:
                self.pipe.unload_lora_weights()
                print("DEBUG: Previous LoRAs unloaded successfully", flush=True)
                logging.info("Previous LoRAs unloaded successfully")
            except ValueError as e:
                print(f"DEBUG: PEFT unload skipped: {e}", flush=True)
                logging.warning(f"PEFT unload skipped: {e}")
        self.loaded_loras.clear()

        adapter_names = []
        adapter_weights = []

        for idx, config in enumerate(lora_configs):
            lora_path = Path(config["path"])
            adapter_name = config["name"]
            weight = config.get("weight", 1.0)

            print(f"\nDEBUG: LoRA {idx+1}/{len(lora_configs)}", flush=True)
            print(f"  Name: {adapter_name}", flush=True)
            print(f"  Path: {lora_path}", flush=True)
            print(f"  Weight: {weight:.2f}", flush=True)
            print(f"  Exists: {lora_path.exists()}", flush=True)

            logging.info(f"\nLoRA {idx+1}/{len(lora_configs)}:")
            logging.info(f"  Name: {adapter_name}")
            logging.info(f"  Path: {lora_path}")
            logging.info(f"  Weight: {weight:.2f}")

            if not lora_path.exists():
                msg = f"❌ LoRA not found: {lora_path}"
                print(f"DEBUG: {msg}", flush=True)
                logging.warning(msg)
                continue

            print(f"DEBUG: Attempting to load {adapter_name}...", flush=True)
            logging.info(f"Loading LoRA: {adapter_name} @ {weight:.2f} from {lora_path.name}")

            try:
                self.pipe.load_lora_weights(
                    str(lora_path),
                    adapter_name=adapter_name
                )

                adapter_names.append(adapter_name)
                adapter_weights.append(weight)
                self.loaded_loras[adapter_name] = lora_path

                print(f"DEBUG: ✓ Successfully loaded {adapter_name}", flush=True)
                logging.info(f"✓ Successfully loaded {adapter_name}")

            except Exception as e:
                msg = f"❌ Failed to load {adapter_name}: {e}"
                print(f"DEBUG: {msg}", flush=True)
                logging.error(msg)
                import traceback
                traceback.print_exc()
                continue

        # Set adapters with weights
        if adapter_names:
            print(f"\nDEBUG: Setting {len(adapter_names)} adapters with weights", flush=True)
            print(f"DEBUG: Adapter names: {adapter_names}", flush=True)
            print(f"DEBUG: Adapter weights: {adapter_weights}", flush=True)
            logging.info(f"\nSetting {len(adapter_names)} adapters")

            self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

            print(f"DEBUG: ✅ Adapters set successfully!", flush=True)
            logging.info(f"✅ Loaded {len(adapter_names)} LoRAs:")
            for name, weight in zip(adapter_names, adapter_weights):
                msg = f"   - {name}: {weight:.2f}"
                print(f"DEBUG: {msg}", flush=True)
                logging.info(msg)

            # Verify adapters are actually set
            if hasattr(self.pipe.unet, 'peft_config'):
                print(f"DEBUG: UNet PEFT config exists: {list(self.pipe.unet.peft_config.keys())}", flush=True)
                logging.info(f"UNet PEFT adapters: {list(self.pipe.unet.peft_config.keys())}")

            print(f"{'='*80}\n", flush=True)
            logging.info(f"{'='*80}")
            return True
        else:
            msg = "❌ No LoRAs loaded!"
            print(f"DEBUG: {msg}", flush=True)
            logging.warning(msg)
            print(f"{'='*80}\n", flush=True)
            return False

    def _detect_needed_loras(self, prompt: str, available_loras: List[str]) -> List[str]:
        """
        Detect which LoRAs are needed based on prompt content.

        Args:
            prompt: The generation prompt
            available_loras: List of available LoRA names

        Returns:
            List of LoRA names that should be active for this prompt
        """
        prompt_lower = prompt.lower()
        needed = []

        for lora_name in available_loras:
            if lora_name.lower() in prompt_lower:
                needed.append(lora_name)

        # If no specific LoRA mentioned, use all (for general scenes)
        if not needed:
            needed = available_loras

        return needed

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy, deformed",
        num_images: int = 1,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        lora_configs: Optional[List[Dict[str, any]]] = None,
        smart_lora_selection: bool = True
    ) -> List[Image.Image]:
        """
        Generate images with loaded LoRAs.

        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            num_images: Number of images to generate
            steps: Number of inference steps
            guidance_scale: CFG scale
            seed: Random seed (optional)
            width: Image width
            height: Image height
            lora_configs: Optional LoRA configs to reload before generation
            smart_lora_selection: If True, only activate LoRAs mentioned in prompt

        Returns:
            List of generated images
        """
        # Reload LoRAs if configs provided
        if lora_configs is not None:
            all_lora_names = [cfg["name"] for cfg in lora_configs]

            if smart_lora_selection:
                # Detect which LoRAs are needed for this prompt
                needed_loras = self._detect_needed_loras(prompt, all_lora_names)

                print(f"\nDEBUG: 🎯 Smart LoRA selection enabled", flush=True)
                print(f"DEBUG: Available LoRAs: {all_lora_names}", flush=True)
                print(f"DEBUG: Detected needed LoRAs: {needed_loras}", flush=True)
                logging.info(f"Smart selection: {needed_loras} from {all_lora_names}")

                # Load only needed LoRAs
                filtered_configs = [cfg for cfg in lora_configs if cfg["name"] in needed_loras]
                self.load_loras(filtered_configs)
            else:
                # Load all LoRAs
                print(f"\nDEBUG: Loading all {len(lora_configs)} LoRAs", flush=True)
                self.load_loras(lora_configs)

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        print(f"\nDEBUG: 🎨 Generating with prompt: {prompt}", flush=True)
        print(f"DEBUG: Seed: {seed}, Steps: {steps}, CFG: {guidance_scale}", flush=True)
        print(f"DEBUG: Active adapters: {list(self.loaded_loras.keys())}", flush=True)
        logging.info(f"Generating with prompt: {prompt[:80]}...")
        logging.info(f"Active LoRAs: {list(self.loaded_loras.keys())}")

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height
        ).images

        print(f"DEBUG: ✅ Generated {len(images)} images successfully", flush=True)
        logging.info(f"Generated {len(images)} images")

        return images

    def test_weight_combinations(
        self,
        lora_paths: List[Path],
        lora_names: List[str],
        weight_combinations: List[List[float]],
        prompts: List[str],
        output_dir: Path,
        negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy",
        num_samples: int = 4,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed_start: int = 42,
        width: int = 1024,
        height: int = 1024
    ) -> dict:
        """
        Test multiple weight combinations.

        Args:
            lora_paths: List of LoRA file paths
            lora_names: List of LoRA names
            weight_combinations: List of weight lists (one per combination)
            prompts: List of test prompts
            output_dir: Output directory
            negative_prompt: Negative prompt
            num_samples: Samples per prompt
            steps: Inference steps
            guidance_scale: CFG scale
            seed_start: Starting seed
            width: Image width
            height: Image height

        Returns:
            Test results metadata
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "timestamp": datetime.now().isoformat(),
            "lora_names": lora_names,
            "lora_paths": [str(p) for p in lora_paths],
            "base_model": "SDXL Base 1.0",
            "settings": {
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height
            },
            "combinations": []
        }

        # Test each weight combination
        for combo_idx, weights in enumerate(weight_combinations):
            logging.info(f"\n{'='*80}")
            logging.info(f"Testing combination {combo_idx+1}/{len(weight_combinations)}")
            logging.info(f"Weights: {' | '.join([f'{name}={w:.2f}' for name, w in zip(lora_names, weights)])}")
            logging.info(f"{'='*80}")

            combo_dir = output_dir / f"combo_{combo_idx:02d}_{'_'.join([f'{w:.1f}'.replace('.', '') for w in weights])}"
            combo_dir.mkdir(exist_ok=True)

            # Build LoRA configs (will be used for smart selection per-prompt)
            lora_configs = [
                {"path": str(path), "name": name, "weight": weight}
                for path, name, weight in zip(lora_paths, lora_names, weights)
            ]

            # NOTE: We don't load LoRAs here anymore - smart selection will handle it per-prompt
            logging.info(f"🎯 Smart LoRA selection enabled - will load LoRAs dynamically per prompt")

            combo_results = {
                "weights": {name: weight for name, weight in zip(lora_names, weights)},
                "prompts": []
            }

            # Test each prompt
            for prompt_idx, prompt in enumerate(prompts):
                prompt_dir = combo_dir / f"prompt_{prompt_idx:02d}"
                prompt_dir.mkdir(exist_ok=True)

                # Save prompt
                with open(prompt_dir / "prompt.txt", 'w') as f:
                    f.write(prompt)

                prompt_results = {
                    "prompt": prompt,
                    "samples": []
                }

                # Generate samples
                for sample_idx in range(num_samples):
                    seed = seed_start + combo_idx * len(prompts) * num_samples + prompt_idx * num_samples + sample_idx

                    images = self.generate(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_images=1,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        seed=seed,
                        width=width,
                        height=height,
                        lora_configs=lora_configs,  # Enable smart LoRA selection
                        smart_lora_selection=True   # Only load LoRAs mentioned in prompt
                    )

                    # Save image
                    image = images[0]
                    image_path = prompt_dir / f"sample_{sample_idx:02d}_seed{seed}.png"
                    image.save(image_path)

                    prompt_results["samples"].append({
                        "path": str(image_path.relative_to(output_dir)),
                        "seed": seed
                    })

                # Create grid
                self._create_grid(
                    [Image.open(output_dir / s["path"]) for s in prompt_results["samples"]],
                    prompt_dir / "grid.png"
                )

                combo_results["prompts"].append(prompt_results)

                logging.info(f"  ✅ Prompt {prompt_idx+1}/{len(prompts)} completed")

            results["combinations"].append(combo_results)

        # Save metadata
        metadata_path = output_dir / "composition_test_results.json"
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"\n✅ All tests completed! Results saved to: {output_dir}")
        logging.info(f"   Metadata: {metadata_path}")

        return results

    def _create_grid(self, images: List[Image.Image], output_path: Path):
        """Create image grid."""
        if not images:
            return

        n = len(images)
        cols = min(4, n)  # Max 4 columns
        rows = int(np.ceil(n / cols))

        width, height = images[0].size
        grid = Image.new('RGB', (width * cols, height * rows), color=(240, 240, 240))

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            grid.paste(img, (col * width, row * height))

        grid.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="SDXL Multi-LoRA Compositor - Test multiple LoRAs with different weight combinations"
    )

    # LoRA configuration
    parser.add_argument("--loras", type=str, nargs="+", required=True,
                       help="Paths to LoRA safetensors files")
    parser.add_argument("--lora-names", type=str, nargs="+", required=True,
                       help="Names for each LoRA (must match --loras count)")
    parser.add_argument("--weight-combos", type=str, nargs="+", required=True,
                       help="Weight combinations as comma-separated lists (e.g., '1.0,0.8' '0.7,1.0')")

    # Prompts
    parser.add_argument("--prompts", type=str, nargs="+",
                       help="Test prompts (alternative to --prompts-file)")
    parser.add_argument("--prompts-file", type=Path,
                       help="File containing test prompts (one per line)")
    parser.add_argument("--negative-prompt", type=str,
                       default="blurry, low quality, distorted, ugly, bad anatomy, deformed",
                       help="Negative prompt")

    # Generation settings
    parser.add_argument("--base-model", type=str, required=True,
                       help="Path to SDXL base model (.safetensors)")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory")
    parser.add_argument("--num-samples", type=int, default=4,
                       help="Samples per prompt")
    parser.add_argument("--steps", type=int, default=30,
                       help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                       help="CFG scale")
    parser.add_argument("--seed-start", type=int, default=42,
                       help="Starting seed")
    parser.add_argument("--width", type=int, default=1024,
                       help="Image width")
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height")
    parser.add_argument("--device", default="cuda",
                       help="Device for inference")
    parser.add_argument("--no-cpu-offload", action="store_true",
                       help="Disable CPU offloading")

    args = parser.parse_args()

    # Validate inputs
    if len(args.loras) != len(args.lora_names):
        parser.error("Number of --loras must match --lora-names")

    # Parse weight combinations
    weight_combinations = []
    for combo_str in args.weight_combos:
        weights = [float(w.strip()) for w in combo_str.split(',')]
        if len(weights) != len(args.loras):
            parser.error(f"Weight combination '{combo_str}' must have {len(args.loras)} values")
        weight_combinations.append(weights)

    # Load prompts
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    elif args.prompts:
        prompts = args.prompts
    else:
        parser.error("Must provide either --prompts or --prompts-file")

    # Setup logging
    logger = setup_logger(
        "sdxl_multi_lora",
        log_file=args.output_dir / "composition_test.log",
        level="INFO"
    )

    logger.info("=" * 80)
    logger.info("SDXL Multi-LoRA Composition Testing")
    logger.info("=" * 80)
    logger.info(f"LoRAs: {', '.join(args.lora_names)}")
    logger.info(f"Weight combinations: {len(weight_combinations)}")
    logger.info(f"Prompts: {len(prompts)}")
    logger.info(f"Samples per prompt: {args.num_samples}")
    logger.info(f"Total images: {len(weight_combinations) * len(prompts) * args.num_samples}")
    logger.info("=" * 80)

    # Initialize compositor
    compositor = SDXLMultiLoRACompositor(
        base_model=args.base_model,
        device=args.device,
        enable_cpu_offload=not args.no_cpu_offload
    )

    # Run test
    results = compositor.test_weight_combinations(
        lora_paths=[Path(p) for p in args.loras],
        lora_names=args.lora_names,
        weight_combinations=weight_combinations,
        prompts=prompts,
        output_dir=args.output_dir,
        negative_prompt=args.negative_prompt,
        num_samples=args.num_samples,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed_start=args.seed_start,
        width=args.width,
        height=args.height
    )

    logger.info("\n" + "=" * 80)
    logger.info("✅ SDXL Multi-LoRA Composition Test Completed!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
