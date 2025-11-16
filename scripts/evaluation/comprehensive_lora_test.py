#!/usr/bin/env python3
"""
Comprehensive LoRA Testing Script
Tests LoRA checkpoint with extensive prompt combinations and negative prompts
Generates comparison grids and quality metrics
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    from safetensors.torch import load_file
except ImportError:
    print("ERROR: Required packages not installed")
    print("Install: pip install diffusers safetensors transformers accelerate")
    sys.exit(1)


class ComprehensiveLoRATester:
    """Comprehensive LoRA testing with extensive prompts and quality checks"""

    def __init__(
        self,
        lora_path: str,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = None,
        device: str = "cuda",
        lora_scale: float = 1.0,
        is_sdxl: bool = False
    ):
        self.lora_path = Path(lora_path)
        self.base_model = base_model
        self.device = device
        self.lora_scale = lora_scale
        self.is_sdxl = is_sdxl

        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"outputs/lora_comprehensive_test/{self.lora_path.stem}_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "grids").mkdir(exist_ok=True)

        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎨 Loading base model: {base_model}")
        print(f"🎯 Model type: {'SDXL' if is_sdxl else 'SD 1.5'}")

        # Load pipeline
        self.pipe = self._load_pipeline()

        # Test categories
        self.test_categories = {
            "portraits": "Close-up portraits with various expressions",
            "full_body": "Full body poses and compositions",
            "angles": "Different camera angles and views",
            "environments": "Various backgrounds and settings",
            "expressions": "Emotional expressions and reactions",
            "actions": "Dynamic poses and activities",
            "clothing": "Different outfit variations",
            "lighting": "Various lighting conditions",
            "compositions": "Different framing and compositions"
        }

    def _load_pipeline(self):
        """Load Stable Diffusion pipeline with LoRA weights"""
        try:
            # Determine pipeline class based on model type
            PipelineClass = StableDiffusionXLPipeline if self.is_sdxl else StableDiffusionPipeline

            # Load base model
            if Path(self.base_model).exists():
                # Local path
                pipe = PipelineClass.from_single_file(
                    self.base_model,
                    torch_dtype=torch.float16,
                    safety_checker=None
                ).to(self.device)
            else:
                # HuggingFace path
                pipe = PipelineClass.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16,
                    safety_checker=None
                ).to(self.device)

            # Use DPM++ 2M Karras scheduler (faster, better quality)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True
            )

            # Load LoRA weights
            print(f"🔧 Loading LoRA: {self.lora_path}")
            if not self.lora_path.exists():
                raise FileNotFoundError(f"LoRA file not found: {self.lora_path}")

            # Load LoRA using safetensors
            lora_state_dict = load_file(str(self.lora_path))

            # Apply LoRA to pipeline
            pipe.load_lora_weights(str(self.lora_path.parent), weight_name=self.lora_path.name)

            print(f"✅ Pipeline ready (LoRA scale: {self.lora_scale})")
            return pipe

        except Exception as e:
            print(f"❌ Error loading pipeline: {e}")
            raise

    def get_comprehensive_prompts(self) -> Dict[str, List[str]]:
        """Get comprehensive test prompts organized by category"""
        # Try to load from JSON file first
        prompts_file = Path(__file__).parent.parent.parent / "prompts" / "luca" / "comprehensive_test_prompts.json"
        if prompts_file.exists():
            try:
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    prompts_dict = {}
                    for category, category_data in data.get("test_categories", {}).items():
                        prompts_dict[category] = category_data.get("prompts", [])
                    if prompts_dict:
                        print(f"✅ Loaded prompts from: {prompts_file.name}")
                        return prompts_dict
            except Exception as e:
                print(f"⚠️  Could not load prompts from JSON: {e}")
                print(f"   Using hardcoded fallback prompts")

        # Fallback to hardcoded prompts
        return {
            "portraits": [
                "a 3d animated character, luca paguro, close-up portrait, smiling warmly, bright eyes, pixar style, smooth shading, studio lighting, high quality, 8k",
                "a 3d animated character, luca paguro, close-up face, concerned expression, furrowed brows, detailed facial features, pixar film quality, soft lighting",
                "a 3d animated character, luca paguro, portrait, looking at viewer, gentle smile, brown eyes, curly dark hair, pixar rendering, professional quality",
                "a 3d animated character, luca paguro, headshot, neutral expression, clear facial features, pixar style, even illumination, sharp focus",
                "a 3d animated character, luca paguro, face close-up, happy expression, wide smile, sparkling eyes, pixar animation quality"
            ],
            "full_body": [
                "a 3d animated character, luca paguro, full body, standing confidently, hands on hips, pixar style, outdoor lighting, high quality",
                "a 3d animated character, luca paguro, full body shot, standing straight, arms at sides, pixar film quality, natural pose",
                "a 3d animated character, luca paguro, full body, sitting casually, relaxed pose, pixar rendering, soft shadows",
                "a 3d animated character, luca paguro, full body, dynamic standing pose, confident stance, pixar animation quality",
                "a 3d animated character, luca paguro, complete figure, balanced composition, pixar style, professional lighting"
            ],
            "angles": [
                "a 3d animated character, luca paguro, three-quarter view, looking to the side, pixar style, even illumination, detailed",
                "a 3d animated character, luca paguro, side profile, standing straight, clear silhouette, pixar film quality, natural lighting",
                "a 3d animated character, luca paguro, front view, facing camera, centered, pixar rendering, balanced lighting",
                "a 3d animated character, luca paguro, slight angle view, natural pose, pixar style, soft lighting",
                "a 3d animated character, luca paguro, over shoulder view, looking back, pixar animation quality"
            ],
            "environments": [
                "a 3d animated character, luca paguro, standing on beach, ocean background, sunny day, pixar style, natural outdoor lighting, vibrant colors",
                "a 3d animated character, luca paguro, in italian village, cobblestone street, warm afternoon light, pixar film quality, atmospheric",
                "a 3d animated character, luca paguro, indoors, cozy room, soft window light, pixar rendering, warm tones",
                "a 3d animated character, luca paguro, outdoor scene, natural environment, bright daylight, pixar style, clear atmosphere",
                "a 3d animated character, luca paguro, simple background, focused lighting, pixar animation quality, clean composition"
            ],
            "expressions": [
                "a 3d animated character, luca paguro, excited expression, eyes wide, big smile, joyful, pixar style, bright lighting",
                "a 3d animated character, luca paguro, thoughtful expression, hand on chin, contemplative look, pixar film quality, soft lighting",
                "a 3d animated character, luca paguro, surprised face, mouth open, eyebrows raised, shocked, pixar rendering, dramatic",
                "a 3d animated character, luca paguro, happy expression, genuine smile, warm eyes, pixar style, friendly",
                "a 3d animated character, luca paguro, curious look, tilted head, interested expression, pixar animation quality",
                "a 3d animated character, luca paguro, angry expression, furrowed brows, frown, upset face, pixar style, dramatic lighting",
                "a 3d animated character, luca paguro, very angry expression, intense glare, clenched teeth, furious, pixar film quality, high contrast lighting",
                "a 3d animated character, luca paguro, frustrated expression, annoyed look, narrowed eyes, irritated, pixar rendering, moody lighting"
            ],
            "actions": [
                "a 3d animated character, luca paguro, waving hello, friendly smile, welcoming gesture, pixar style, outdoor lighting",
                "a 3d animated character, luca paguro, pointing forward, determined look, confident gesture, pixar film quality, dramatic lighting",
                "a 3d animated character, luca paguro, hands clasped together, hopeful expression, pixar rendering, soft lighting",
                "a 3d animated character, luca paguro, reaching out, friendly gesture, open pose, pixar style, natural lighting",
                "a 3d animated character, luca paguro, standing pose, natural stance, relaxed body language, pixar animation quality"
            ],
            "clothing": [
                "a 3d animated character, luca paguro, wearing striped shirt, casual outfit, blue and white stripes, pixar style, natural lighting",
                "a 3d animated character, luca paguro, wearing simple t-shirt, relaxed clothing, plain top, pixar film quality, soft lighting",
                "a 3d animated character, luca paguro, wearing vest, casual formal, layered outfit, pixar rendering, balanced lighting",
                "a 3d animated character, luca paguro, summer outfit, light clothing, casual wear, pixar style, bright lighting",
                "a 3d animated character, luca paguro, typical outfit, character-appropriate clothing, pixar animation quality"
            ],
            "lighting": [
                "a 3d animated character, luca paguro, dramatic side lighting, strong shadows, cinematic look, pixar style, high contrast",
                "a 3d animated character, luca paguro, soft diffused lighting, no harsh shadows, gentle illumination, pixar film quality, even tones",
                "a 3d animated character, luca paguro, bright natural lighting, sunny atmosphere, clear visibility, pixar rendering, vibrant",
                "a 3d animated character, luca paguro, rim lighting, edge highlights, separated from background, pixar style, professional",
                "a 3d animated character, luca paguro, studio lighting setup, three-point lighting, balanced, pixar animation quality"
            ],
            "compositions": [
                "a 3d animated character, luca paguro, medium shot, waist up, centered composition, balanced framing, pixar style",
                "a 3d animated character, luca paguro, close framing, tight composition, focused on character, pixar film quality",
                "a 3d animated character, luca paguro, wide composition, character in environment, contextual, pixar rendering",
                "a 3d animated character, luca paguro, rule of thirds, artistic composition, professional framing, pixar style",
                "a 3d animated character, luca paguro, centered subject, symmetrical composition, balanced, pixar animation quality"
            ]
        }

    def get_negative_prompts(self) -> Dict[str, str]:
        """Get comprehensive negative prompts for quality control"""
        return {
            "default": (
                "blurry, out of focus, low quality, bad quality, poor quality, "
                "ugly, deformed, distorted, disfigured, malformed, mutated, "
                "bad anatomy, wrong anatomy, extra limbs, missing limbs, "
                "extra fingers, missing fingers, fused fingers, "
                "bad hands, bad feet, bad face, asymmetric face, "
                "bad eyes, cross-eyed, lazy eye, "
                "bad proportions, gross proportions, "
                "low resolution, pixelated, jpeg artifacts, compression artifacts, "
                "watermark, signature, text, username, "
                "duplicate, cloned, extra head, two heads, "
                "realistic, photo, photograph, photorealistic, "
                "2d anime, flat, flat shading, "
                "nsfw, nude, violence, gore"
            ),
            "character_focus": (
                "multiple characters, crowd, group, "
                "wrong character, different character, "
                "wrong hair color, wrong eye color, "
                "wrong age, adult, elderly, baby, "
                "wrong species, monster, creature, alien, "
                "blurry, out of focus, low quality, poor quality, "
                "deformed face, distorted features, asymmetric face, "
                "bad eyes, crossed eyes, "
                "bad anatomy, wrong anatomy, "
                "watermark, signature, text, "
                "realistic, photograph, "
                "2d anime, flat shading"
            ),
            "quality_focus": (
                "blurry, out of focus, soft focus, "
                "low quality, bad quality, poor quality, worst quality, "
                "low resolution, pixelated, pixel art, "
                "jpeg artifacts, compression artifacts, noise, grain, "
                "amateur, unprofessional, unfinished, draft, "
                "watermark, signature, text, logo, username, timestamp, "
                "deformed, distorted, disfigured, mutated, malformed, "
                "bad anatomy, wrong anatomy, "
                "nsfw"
            ),
            "style_focus": (
                "realistic, photorealistic, photo, photograph, real life, "
                "2d anime, anime style, manga style, flat, flat shading, "
                "cartoon, caricature, sketch, drawing, painting, illustration, "
                "clay, claymation, stop motion, "
                "sci-fi, cyberpunk, steampunk, fantasy, medieval, "
                "horror, dark, grim, scary, "
                "low quality, poor quality, blurry, "
                "deformed, distorted, "
                "watermark, signature, text, "
                "nsfw"
            )
        }

    def generate_test_images(
        self,
        num_seeds: int = 3,
        steps: int = 30,
        cfg_scale: float = 7.5,
        width: int = 512,
        height: int = 512
    ) -> Dict:
        """Generate comprehensive test images"""

        print("\n" + "="*80)
        print("🎨 COMPREHENSIVE LORA TESTING")
        print("="*80)
        print(f"LoRA: {self.lora_path.name}")
        print(f"Seeds per prompt: {num_seeds}")
        print(f"Steps: {steps}, CFG: {cfg_scale}, Size: {width}x{height}")
        print("="*80 + "\n")

        prompts_dict = self.get_comprehensive_prompts()
        negative_prompts = self.get_negative_prompts()

        # Test seeds for reproducibility
        test_seeds = [42, 123, 777][:num_seeds]

        results = {
            "lora_path": str(self.lora_path),
            "base_model": self.base_model,
            "lora_scale": self.lora_scale,
            "generation_params": {
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": width,
                "height": height,
                "seeds": test_seeds
            },
            "categories": {},
            "total_images": 0
        }

        # Generate for each category
        for category, prompts in prompts_dict.items():
            print(f"\n📂 Category: {category.upper()} ({len(prompts)} prompts)")
            print(f"   {self.test_categories[category]}")
            print("-" * 80)

            category_dir = self.output_dir / "images" / category
            category_dir.mkdir(parents=True, exist_ok=True)

            category_results = {
                "description": self.test_categories[category],
                "prompts": []
            }

            for prompt_idx, prompt in enumerate(prompts, 1):
                prompt_results = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompts["default"],
                    "images": []
                }

                # Show progress
                print(f"\n   [{prompt_idx}/{len(prompts)}] Prompt:")
                print(f"   {prompt[:100]}...")

                # Generate with different seeds
                for seed_idx, seed in enumerate(test_seeds, 1):
                    try:
                        # Set seed for reproducibility
                        generator = torch.Generator(device=self.device).manual_seed(seed)

                        # Generate image
                        output = self.pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompts["default"],
                            num_inference_steps=steps,
                            guidance_scale=cfg_scale,
                            width=width,
                            height=height,
                            generator=generator,
                            cross_attention_kwargs={"scale": self.lora_scale}
                        )

                        image = output.images[0]

                        # Save image
                        filename = f"{category}_p{prompt_idx:02d}_s{seed}.png"
                        save_path = category_dir / filename
                        image.save(save_path)

                        prompt_results["images"].append({
                            "filename": filename,
                            "seed": seed,
                            "path": str(save_path)
                        })

                        results["total_images"] += 1

                        print(f"      ✅ Seed {seed}: {filename}")

                    except Exception as e:
                        print(f"      ❌ Seed {seed}: Error - {e}")

                category_results["prompts"].append(prompt_results)

            results["categories"][category] = category_results

        # Save results JSON
        results_path = self.output_dir / "test_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Generated {results['total_images']} total images")
        print(f"📄 Results saved: {results_path}")

        return results

    def create_comparison_grids(self, results: Dict):
        """Create comparison grids for each category"""

        print("\n" + "="*80)
        print("🖼️  CREATING COMPARISON GRIDS")
        print("="*80)

        for category, category_data in results["categories"].items():
            print(f"\n📊 Creating grid for: {category.upper()}")

            category_dir = self.output_dir / "images" / category
            grid_path = self.output_dir / "grids" / f"{category}_grid.png"

            # Collect all images for this category
            images = []
            labels = []

            for prompt_idx, prompt_data in enumerate(category_data["prompts"], 1):
                for img_data in prompt_data["images"]:
                    img_path = category_dir / img_data["filename"]
                    if img_path.exists():
                        images.append(Image.open(img_path))
                        labels.append(f"P{prompt_idx} S{img_data['seed']}")

            if not images:
                print(f"   ⚠️  No images found for {category}")
                continue

            # Create grid
            grid = self._create_image_grid(images, labels, cols=3)
            grid.save(grid_path)

            print(f"   ✅ Grid saved: {grid_path.name} ({len(images)} images)")

        # Create master grid (one image from each category)
        print(f"\n📊 Creating master comparison grid...")
        self._create_master_grid(results)

    def _create_image_grid(
        self,
        images: List[Image.Image],
        labels: List[str] = None,
        cols: int = 3
    ) -> Image.Image:
        """Create a grid of images with labels"""

        if not images:
            raise ValueError("No images provided")

        # Calculate grid dimensions
        n_images = len(images)
        rows = (n_images + cols - 1) // cols

        # Get image dimensions
        img_width, img_height = images[0].size

        # Add space for labels
        label_height = 30 if labels else 0
        cell_height = img_height + label_height

        # Create grid canvas
        grid_width = cols * img_width
        grid_height = rows * cell_height
        grid = Image.new('RGB', (grid_width, grid_height), 'white')

        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Place images in grid
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols

            x = col * img_width
            y = row * cell_height

            # Paste image
            grid.paste(img, (x, y))

            # Add label if provided
            if labels and idx < len(labels):
                draw = ImageDraw.Draw(grid)
                label_y = y + img_height + 5
                draw.text((x + 5, label_y), labels[idx], fill='black', font=font)

        return grid

    def _create_master_grid(self, results: Dict):
        """Create master grid with one representative image from each category"""

        master_images = []
        master_labels = []

        for category in results["categories"].keys():
            category_dir = self.output_dir / "images" / category

            # Get first image from category
            first_prompt = results["categories"][category]["prompts"][0]
            if first_prompt["images"]:
                first_img_data = first_prompt["images"][0]
                img_path = category_dir / first_img_data["filename"]

                if img_path.exists():
                    master_images.append(Image.open(img_path))
                    master_labels.append(category.replace("_", " ").title())

        if master_images:
            master_grid = self._create_image_grid(master_images, master_labels, cols=3)
            master_path = self.output_dir / "grids" / "master_comparison.png"
            master_grid.save(master_path)
            print(f"   ✅ Master grid: {master_path.name}")

    def generate_report(self, results: Dict):
        """Generate markdown report"""

        report_path = self.output_dir / "TEST_REPORT.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive LoRA Test Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**LoRA**: `{self.lora_path.name}`\n\n")
            f.write(f"**Base Model**: {self.base_model}\n\n")
            f.write(f"**LoRA Scale**: {self.lora_scale}\n\n")

            # Generation parameters
            f.write("## Generation Parameters\n\n")
            params = results["generation_params"]
            f.write(f"- **Steps**: {params['steps']}\n")
            f.write(f"- **CFG Scale**: {params['cfg_scale']}\n")
            f.write(f"- **Resolution**: {params['width']}x{params['height']}\n")
            f.write(f"- **Seeds**: {params['seeds']}\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Categories**: {len(results['categories'])}\n")
            f.write(f"- **Total Images**: {results['total_images']}\n\n")

            # Category breakdown
            f.write("## Category Breakdown\n\n")
            for category, data in results["categories"].items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                f.write(f"*{data['description']}*\n\n")
                f.write(f"- Prompts tested: {len(data['prompts'])}\n")

                total_images = sum(len(p['images']) for p in data['prompts'])
                f.write(f"- Images generated: {total_images}\n\n")

                # Show sample prompts
                f.write("**Sample Prompts**:\n\n")
                for i, prompt_data in enumerate(data['prompts'][:2], 1):
                    f.write(f"{i}. {prompt_data['prompt']}\n\n")

                f.write(f"![{category} grid](grids/{category}_grid.png)\n\n")

            # Master comparison
            f.write("## Master Comparison\n\n")
            f.write("![Master Comparison](grids/master_comparison.png)\n\n")

            # Next steps
            f.write("## Quality Assessment\n\n")
            f.write("### Checklist\n\n")
            f.write("- [ ] Character identity consistent across all images\n")
            f.write("- [ ] Prompt adherence (follows instructions)\n")
            f.write("- [ ] No anatomical errors or artifacts\n")
            f.write("- [ ] Proper Pixar 3D style maintained\n")
            f.write("- [ ] Facial features accurate and stable\n")
            f.write("- [ ] Lighting and shading appropriate\n")
            f.write("- [ ] No overfitting signs (too similar to training data)\n")
            f.write("- [ ] Handles various angles and poses well\n")
            f.write("- [ ] Background/environment rendering acceptable\n")
            f.write("- [ ] Overall quality meets production standards\n\n")

            f.write("## Negative Prompt Used\n\n")
            f.write("```\n")
            f.write(list(self.get_negative_prompts().values())[0])
            f.write("\n```\n\n")

            f.write("## Next Steps\n\n")
            f.write("1. ✅ Review all generated images for quality\n")
            f.write("2. ✅ Check character consistency across categories\n")
            f.write("3. ✅ Verify prompt adherence\n")
            f.write("4. ✅ Assess for overfitting or artifacts\n")
            f.write("5. 🔜 If quality is good → Proceed to SDXL training\n")
            f.write("6. 🔜 If issues found → Document and adjust training\n\n")

        print(f"\n📄 Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive LoRA testing with extensive prompts and quality checks"
    )
    parser.add_argument(
        "lora_path",
        type=str,
        help="Path to LoRA checkpoint (.safetensors)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model (default: SD 1.5)"
    )
    parser.add_argument(
        "--sdxl",
        action="store_true",
        help="Use SDXL pipeline instead of SD 1.5"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for test results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for generation"
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA scale/strength (default: 1.0)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds per prompt (default: 3)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (default: 30)"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=7.5,
        help="CFG scale (default: 7.5)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width (default: 512)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height (default: 512)"
    )
    parser.add_argument(
        "--skip-grids",
        action="store_true",
        help="Skip creating comparison grids"
    )

    args = parser.parse_args()

    # Verify LoRA exists
    if not Path(args.lora_path).exists():
        print(f"❌ ERROR: LoRA file not found: {args.lora_path}")
        sys.exit(1)

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    try:
        # Initialize tester
        tester = ComprehensiveLoRATester(
            lora_path=args.lora_path,
            base_model=args.base_model,
            output_dir=args.output_dir,
            device=args.device,
            lora_scale=args.lora_scale,
            is_sdxl=args.sdxl
        )

        # Generate test images
        results = tester.generate_test_images(
            num_seeds=args.seeds,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            width=args.width,
            height=args.height
        )

        # Create comparison grids
        if not args.skip_grids:
            tester.create_comparison_grids(results)

        # Generate report
        tester.generate_report(results)

        print("\n" + "="*80)
        print("✅ COMPREHENSIVE TESTING COMPLETE")
        print("="*80)
        print(f"📁 Output directory: {tester.output_dir}")
        print(f"📄 Report: {tester.output_dir}/TEST_REPORT.md")
        print(f"🖼️  Grids: {tester.output_dir}/grids/")
        print(f"📸 Images: {tester.output_dir}/images/")
        print("="*80)

    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
