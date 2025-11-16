#!/usr/bin/env python3
"""
Inpaint Occluded Character Instances

Fills missing/occluded regions in character instances using AI inpainting.
Useful for repairing instances where parts are blocked by other objects.

Methods:
1. LaMa - Fast, general-purpose inpainting
2. Stable Diffusion Inpainting - High quality, controllable
3. Traditional CV methods - Fastest fallback

Features:
- Auto-detection of character identity (for Luca characters)
- Character-specific prompts from config files
- Batch processing with occlusion filtering
- Multiple inpainting backends

Usage:
    # Batch inpaint all instances with >20% occlusion
    python inpaint_occlusions.py \
        --input-dir /path/to/instances \
        --output-dir /path/to/inpainted \
        --method lama \
        --occlusion-threshold 0.2

    # Use character-specific prompts (for SD method)
    python inpaint_occlusions.py \
        --input-dir /path/to/instances \
        --output-dir /path/to/inpainted \
        --method sd \
        --config configs/inpainting/luca_prompts.json \
        --auto-detect-character
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
import json
from datetime import datetime
import re


class InstanceInpainter:
    """
    Inpaint occluded regions in character instances
    """

    def __init__(
        self,
        method: str = "lama",
        device: str = "cuda",
        config_path: Optional[Path] = None,
        auto_detect: bool = False
    ):
        """
        Initialize inpainter

        Args:
            method: lama, sd (stable diffusion), or cv (opencv)
            device: cuda or cpu
            config_path: Path to character prompts config (for SD method)
            auto_detect: Enable automatic character detection
        """
        self.method = method
        self.device = device
        self.auto_detect = auto_detect
        self.prompts_config = None

        # Load character prompts config if provided
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                self.prompts_config = json.load(f)
            print(f"✓ Loaded character prompts from {config_path}")

        print(f"🔧 Initializing {method} inpainter...")
        self._init_model()

    def _init_model(self):
        """Initialize inpainting model"""
        if self.method == "lama":
            self._init_lama()
        elif self.method == "sd":
            self._init_sd()
        elif self.method == "cv":
            self._init_cv()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _init_lama(self):
        """Initialize LaMa model"""
        try:
            # Try to import lama-cleaner or simple-lama-inpainting
            import torch
            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config

            self.model_manager = ModelManager(
                name="lama",
                device=self.device
            )
            print("✓ LaMa model loaded")

        except ImportError:
            print("⚠️  LaMa not installed, falling back to OpenCV")
            self.method = "cv"
            self._init_cv()

    def _init_sd(self):
        """Initialize Stable Diffusion Inpainting"""
        try:
            from diffusers import StableDiffusionInpaintPipeline
            import torch

            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)

            print("✓ Stable Diffusion Inpainting loaded")

        except ImportError:
            print("⚠️  Diffusers not installed, falling back to OpenCV")
            self.method = "cv"
            self._init_cv()

    def _init_cv(self):
        """Initialize OpenCV inpainting (fallback)"""
        print("✓ Using OpenCV Telea inpainting (fallback)")
        self.model = None

    def detect_occlusion_mask(
        self,
        image: Image.Image,
        alpha_threshold: int = 10
    ) -> Tuple[np.ndarray, float]:
        """
        Detect occluded regions from alpha channel

        Args:
            image: RGBA PIL Image
            alpha_threshold: Pixels below this are considered occluded

        Returns:
            (mask, occlusion_ratio)
        """
        image_np = np.array(image)

        # Get alpha channel
        if image_np.shape[2] == 4:
            alpha = image_np[:, :, 3]
        else:
            # No alpha, no occlusion
            return np.zeros(image_np.shape[:2], dtype=np.uint8), 0.0

        # Create mask: 255 where occluded (low alpha)
        mask = (alpha < alpha_threshold).astype(np.uint8) * 255

        # Calculate occlusion ratio
        total_pixels = mask.shape[0] * mask.shape[1]
        occluded_pixels = np.sum(mask > 0)
        occlusion_ratio = occluded_pixels / total_pixels

        return mask, occlusion_ratio

    def inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        prompt: Optional[str] = None
    ) -> Image.Image:
        """
        Inpaint image using selected method

        Args:
            image: PIL Image (RGB or RGBA)
            mask: Numpy array (255=inpaint, 0=keep)
            prompt: Text prompt (for SD method)

        Returns:
            Inpainted PIL Image
        """
        if self.method == "lama":
            return self._inpaint_lama(image, mask)
        elif self.method == "sd":
            return self._inpaint_sd(image, mask, prompt)
        else:
            return self._inpaint_cv(image, mask)

    def _inpaint_lama(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Inpaint with LaMa"""
        from lama_cleaner.schema import Config

        # Convert to numpy
        image_np = np.array(image.convert("RGB"))

        # LaMa expects mask as binary
        mask_bin = (mask > 127).astype(np.uint8)

        # Inpaint
        config = Config(
            ldm_steps=25,
            hd_strategy="Original",
            hd_strategy_crop_margin=128,
        )

        result = self.model_manager(image_np, mask_bin, config)

        return Image.fromarray(result)

    def _inpaint_sd(
        self,
        image: Image.Image,
        mask: np.ndarray,
        prompt: Optional[str] = None
    ) -> Image.Image:
        """Inpaint with Stable Diffusion"""
        # Convert image to RGB
        image_rgb = image.convert("RGB")

        # Convert mask to PIL
        mask_pil = Image.fromarray(mask)

        # Default prompt
        if prompt is None:
            prompt = "a 3d animated character, pixar style, smooth shading, complete body"

        # Inpaint
        result = self.pipe(
            prompt=prompt,
            image=image_rgb,
            mask_image=mask_pil,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]

        return result

    def _inpaint_cv(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Inpaint with OpenCV (fallback)"""
        # Convert to numpy
        image_np = np.array(image.convert("RGB"))

        # Inpaint using Telea method
        result = cv2.inpaint(
            image_np,
            mask,
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA
        )

        return Image.fromarray(result)

    def detect_character(self, filename: str) -> Optional[str]:
        """
        Detect character from instance filename

        Args:
            filename: Instance filename (e.g., "scene0123_pos1_frame001_inst0.png")

        Returns:
            Character identifier (e.g., "luca_human", "alberto_sea_monster")
        """
        if not self.prompts_config or not self.auto_detect:
            return None

        # Character detection keywords
        char_keywords = {
            'luca': ['luca'],
            'alberto': ['alberto'],
            'giulia': ['giulia'],
            'massimo': ['massimo'],
            'ercole': ['ercole']
        }

        # Form detection (sea monster vs human)
        # For now, simple heuristic based on color analysis
        # TODO: Could use CLIP or custom classifier

        filename_lower = filename.lower()
        for char_name, keywords in char_keywords.items():
            if any(kw in filename_lower for kw in keywords):
                # Default to human form
                # TODO: Analyze image to determine form
                return f"{char_name}_human"

        return None

    def get_character_prompt(self, character_id: str, body_part: str = "full_body") -> Optional[str]:
        """
        Get character-specific inpainting prompt

        Args:
            character_id: Character identifier (e.g., "luca_human")
            body_part: Body part to inpaint (e.g., "arms", "face", "full_body")

        Returns:
            Prompt string or None
        """
        if not self.prompts_config:
            return None

        try:
            char_prompts = self.prompts_config['character_prompts'].get(character_id, {})

            # Try to get specific body part prompt
            if 'body_parts' in char_prompts.get('full_body', {}):
                body_parts = char_prompts['full_body']['body_parts']
                if body_part in body_parts:
                    return body_parts[body_part]

            # Fall back to full body prompt
            if 'full_body' in char_prompts:
                return char_prompts['full_body'].get('prompt')

            # Fall back to global style
            global_style = self.prompts_config.get('global_style', {})
            return global_style.get('base_prompt')

        except KeyError:
            return None


def process_instances(
    input_dir: Path,
    output_dir: Path,
    method: str = "lama",
    occlusion_threshold: float = 0.2,
    instance_list: Optional[List[str]] = None,
    device: str = "cuda",
    prompt: Optional[str] = None,
    config_path: Optional[Path] = None,
    auto_detect: bool = False
) -> dict:
    """
    Process character instances with inpainting

    Args:
        input_dir: Directory with character instances
        output_dir: Output directory
        method: Inpainting method (lama, sd, cv)
        occlusion_threshold: Min occlusion ratio to inpaint (0.0-1.0)
        instance_list: Specific instances to process
        device: cuda or cpu
        prompt: Text prompt for SD method (overrides auto-detection)
        config_path: Path to character prompts config
        auto_detect: Enable automatic character detection

    Returns:
        Processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directories
    inpainted_dir = output_dir / "inpainted"
    inpainted_dir.mkdir(parents=True, exist_ok=True)

    # Initialize inpainter
    inpainter = InstanceInpainter(
        method=method,
        device=device,
        config_path=config_path,
        auto_detect=auto_detect
    )

    # Find instances to process
    if instance_list:
        image_files = [input_dir / f for f in instance_list]
    else:
        image_files = sorted(
            list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        )

    print(f"\n📊 Processing {len(image_files)} instances...")
    print(f"   Method: {method}")
    print(f"   Occlusion threshold: {occlusion_threshold * 100}%")
    print()

    stats = {
        'total_instances': len(image_files),
        'inpainted': 0,
        'skipped_low_occlusion': 0,
        'skipped_no_alpha': 0,
        'failed': 0
    }

    for img_path in tqdm(image_files, desc="Inpainting instances"):
        try:
            # Load image
            image = Image.open(img_path)

            # Detect occlusion
            mask, occlusion_ratio = inpainter.detect_occlusion_mask(image)

            # Skip if occlusion is below threshold
            if occlusion_ratio < occlusion_threshold:
                stats['skipped_low_occlusion'] += 1
                continue

            # Skip if no occlusion detected
            if occlusion_ratio == 0:
                stats['skipped_no_alpha'] += 1
                continue

            # Get character-specific prompt if auto-detection is enabled
            use_prompt = prompt  # User-provided prompt takes precedence

            if auto_detect and not use_prompt and method == "sd":
                character_id = inpainter.detect_character(img_path.name)
                if character_id:
                    use_prompt = inpainter.get_character_prompt(character_id)
                    print(f"  └─ Detected: {character_id}")

            # Inpaint
            inpainted = inpainter.inpaint(image, mask, use_prompt)

            # Preserve alpha channel if exists
            if image.mode == "RGBA":
                # Combine inpainted RGB with original alpha
                inpainted_rgba = Image.new("RGBA", image.size)
                inpainted_rgba.paste(inpainted.convert("RGB"))

                # Copy original alpha where it was opaque
                alpha = np.array(image)[:, :, 3]
                inpainted_np = np.array(inpainted_rgba)
                inpainted_np[:, :, 3] = alpha

                inpainted = Image.fromarray(inpainted_np)

            # Save
            output_path = inpainted_dir / img_path.name
            inpainted.save(output_path)

            stats['inpainted'] += 1

        except Exception as e:
            print(f"\n❌ Failed to process {img_path.name}: {e}")
            stats['failed'] += 1

    # Save report
    report = {
        'statistics': stats,
        'parameters': {
            'method': method,
            'occlusion_threshold': occlusion_threshold,
            'prompt': prompt
        },
        'timestamp': datetime.now().isoformat()
    }

    report_path = output_dir / "inpainting_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Inpainting complete!")
    print(f"   Total instances: {stats['total_instances']}")
    print(f"   Inpainted: {stats['inpainted']}")
    print(f"   Skipped (low occlusion): {stats['skipped_low_occlusion']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Output: {inpainted_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Inpaint occluded regions in character instances"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with character instances"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="lama",
        choices=["lama", "sd", "cv"],
        help="Inpainting method (lama, sd, cv)"
    )
    parser.add_argument(
        "--occlusion-threshold",
        type=float,
        default=0.2,
        help="Minimum occlusion ratio to inpaint (0.0-1.0, default: 0.2)"
    )
    parser.add_argument(
        "--instance-list",
        type=str,
        help="Comma-separated list of specific instances to process"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for SD method"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (e.g., 'luca', 'toy_story'). Auto-loads config from configs/inpainting/{project}_prompts.json"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to character prompts config (overrides --project). Use for custom configs."
    )
    parser.add_argument(
        "--auto-detect-character",
        action="store_true",
        help="Enable automatic character detection (for SD method with config)"
    )

    args = parser.parse_args()

    # Determine config path
    config_path = None
    if args.config:
        # Explicit config path provided
        config_path = Path(args.config)
    elif args.project:
        # Auto-load config from project name
        script_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
        config_path = script_dir / "configs" / "inpainting" / f"{args.project}_prompts.json"

        if not config_path.exists():
            print(f"⚠️  Warning: Config not found for project '{args.project}': {config_path}")
            print(f"   Continuing without character-specific prompts.")
            config_path = None
        else:
            print(f"✓ Loaded config for project: {args.project}")

    process_instances(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        method=args.method,
        occlusion_threshold=args.occlusion_threshold,
        instance_list=args.instance_list.split(',') if args.instance_list else None,
        device=args.device,
        prompt=args.prompt,
        config_path=config_path,
        auto_detect=args.auto_detect_character
    )


if __name__ == "__main__":
    main()
