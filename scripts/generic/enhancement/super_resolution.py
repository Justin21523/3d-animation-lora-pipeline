#!/usr/bin/env python3
"""
Super-Resolution for Character Instances

Uses Real-ESRGAN to upscale low-resolution instances.
Optimized for 3D animated characters with conservative settings.

Key Features:
- Automatic size detection (only upscale small images)
- Preserves anti-aliased edges
- Batch processing with resume capability
- Multiple model options
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import Optional, Tuple
from tqdm import tqdm
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class SuperResolutionUpscaler:
    """
    Super-resolution upscaling using Real-ESRGAN
    """

    def __init__(
        self,
        model_name: str = "RealESRGAN_x2plus",
        device: str = "cuda",
        tile_size: int = 256,
        half_precision: bool = True
    ):
        """
        Initialize super-resolution model

        Args:
            model_name: Model variant
                - RealESRGAN_x2plus (2x, general)
                - RealESRGAN_x4plus (4x, general)
                - RealESRGAN_x4plus_anime_6B (4x, anime-optimized)
            device: cuda or cpu
            tile_size: Tile size for processing (smaller = less VRAM)
            half_precision: Use FP16 for speed (requires GPU)
        """
        self.model_name = model_name
        self.device = device
        self.tile_size = tile_size
        self.half_precision = half_precision and device == "cuda"

        print(f"🔧 Initializing Super-Resolution Upscaler...")
        print(f"   Model: {model_name}")
        print(f"   Tile size: {tile_size}")
        print(f"   Half precision: {self.half_precision}")

        self._init_realesrgan()

    def _init_realesrgan(self):
        """Initialize Real-ESRGAN model"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact

            model_dir = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/super_resolution")
            model_dir.mkdir(parents=True, exist_ok=True)

            # Model configurations
            model_configs = {
                "RealESRGAN_x2plus": {
                    "scale": 2,
                    "model_path": model_dir / "RealESRGAN_x2plus.pth",
                    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                    "arch": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                },
                "RealESRGAN_x4plus": {
                    "scale": 4,
                    "model_path": model_dir / "RealESRGAN_x4plus.pth",
                    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                    "arch": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                },
                "RealESRGAN_x4plus_anime_6B": {
                    "scale": 4,
                    "model_path": model_dir / "RealESRGAN_x4plus_anime_6B.pth",
                    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                    "arch": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                }
            }

            config = model_configs[self.model_name]
            model_path = config["model_path"]

            # Download if not exists
            if not model_path.exists():
                print(f"📥 Downloading {self.model_name}...")
                import urllib.request
                urllib.request.urlretrieve(config["url"], model_path)

            # Initialize upsampler
            self.upsampler = RealESRGANer(
                scale=config["scale"],
                model_path=str(model_path),
                model=config["arch"],
                tile=self.tile_size,
                tile_pad=10,
                pre_pad=0,
                half=self.half_precision,
                device=self.device
            )

            self.scale = config["scale"]
            print(f"✓ Real-ESRGAN model loaded (scale: {self.scale}x)")

        except ImportError as e:
            print(f"❌ Real-ESRGAN not installed: {e}")
            print("   Install with: pip install realesrgan")
            raise

    def should_upscale(self, image: Image.Image, min_size: int = 512) -> bool:
        """
        Check if image should be upscaled

        Args:
            image: PIL Image
            min_size: Minimum size on longest side

        Returns:
            True if image should be upscaled
        """
        w, h = image.size
        longest_side = max(w, h)
        return longest_side < min_size

    def upscale(
        self,
        image: Image.Image,
        outscale: Optional[int] = None
    ) -> Image.Image:
        """
        Upscale image

        Args:
            image: PIL Image
            outscale: Output scale (overrides model default)

        Returns:
            Upscaled PIL Image
        """
        # Convert to numpy
        image_np = np.array(image)

        # Handle RGBA
        if image_np.shape[2] == 4:
            image_rgb = image_np[:, :, :3]
            alpha = image_np[:, :, 3]
            has_alpha = True
        else:
            image_rgb = image_np
            alpha = None
            has_alpha = False

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Upscale
        try:
            output_bgr, _ = self.upsampler.enhance(
                image_bgr,
                outscale=outscale or self.scale
            )
        except Exception as e:
            print(f"⚠️ Upscaling failed: {e}")
            return image

        # Convert back to RGB
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

        # Restore alpha channel if existed
        if has_alpha:
            # Upscale alpha channel
            alpha_upscaled = cv2.resize(
                alpha,
                (output_rgb.shape[1], output_rgb.shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )
            output_rgba = np.concatenate([
                output_rgb,
                alpha_upscaled[:, :, None]
            ], axis=2)
            return Image.fromarray(output_rgba.astype(np.uint8))
        else:
            return Image.fromarray(output_rgb.astype(np.uint8))


def process_instances(
    input_dir: Path,
    output_dir: Path,
    model_name: str = "RealESRGAN_x2plus",
    device: str = "cuda",
    min_size: int = 512,
    force_upscale: bool = False,
    tile_size: int = 256
) -> dict:
    """
    Process all character instances with super-resolution

    Args:
        input_dir: Directory with character instances
        output_dir: Output directory for upscaled instances
        model_name: Real-ESRGAN model variant
        device: cuda or cpu
        min_size: Minimum size to trigger upscaling
        force_upscale: Force upscale all images
        tile_size: Tile size for processing

    Returns:
        Processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory
    upscaled_dir = output_dir / "upscaled"
    upscaled_dir.mkdir(parents=True, exist_ok=True)

    # Initialize upscaler
    upscaler = SuperResolutionUpscaler(
        model_name=model_name,
        device=device,
        tile_size=tile_size
    )

    # Find all instances
    image_files = sorted(
        list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    )

    print(f"\n📊 Processing {len(image_files)} instances...")

    # Check for already processed files
    processed_files = set()
    if upscaled_dir.exists():
        for f in upscaled_dir.glob("*.png"):
            processed_files.add(f.name)

    print(f"📊 Found {len(processed_files)} already processed, will skip them...")

    stats = {
        'total_instances': len(image_files),
        'processed': 0,
        'skipped': 0,
        'upscaled': 0,
        'no_upscale_needed': 0,
        'size_distribution': {'small': 0, 'medium': 0, 'large': 0}
    }

    for img_path in tqdm(image_files, desc="Upscaling instances"):
        # Skip if already processed
        if img_path.name in processed_files:
            stats['skipped'] += 1
            continue

        # Load image
        image = Image.open(img_path)
        w, h = image.size
        longest_side = max(w, h)

        # Categorize size
        if longest_side < 256:
            stats['size_distribution']['small'] += 1
        elif longest_side < 512:
            stats['size_distribution']['medium'] += 1
        else:
            stats['size_distribution']['large'] += 1

        # Check if upscaling needed
        if force_upscale or upscaler.should_upscale(image, min_size):
            # Upscale
            upscaled = upscaler.upscale(image)
            stats['upscaled'] += 1
        else:
            # No upscaling needed, just copy
            upscaled = image
            stats['no_upscale_needed'] += 1

        # Save
        output_path = upscaled_dir / img_path.name
        upscaled.save(output_path)
        stats['processed'] += 1

    # Save statistics
    stats_path = output_dir / "upscaling_stats.json"
    with open(stats_path, 'w') as f:
        json.dump({
            'statistics': stats,
            'parameters': {
                'model_name': model_name,
                'min_size': min_size,
                'force_upscale': force_upscale,
                'scale': upscaler.scale
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n✅ Super-resolution complete!")
    print(f"   Processed: {stats['processed']}")
    print(f"   Upscaled: {stats['upscaled']}")
    print(f"   No upscale needed: {stats['no_upscale_needed']}")
    print(f"   Skipped: {stats['skipped']}")
    print(f"\n   Size distribution:")
    print(f"   Small (<256px): {stats['size_distribution']['small']}")
    print(f"   Medium (256-512px): {stats['size_distribution']['medium']}")
    print(f"   Large (>512px): {stats['size_distribution']['large']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Super-resolution upscaling for character instances"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory with character instances"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for upscaled instances"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RealESRGAN_x2plus",
        choices=["RealESRGAN_x2plus", "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"],
        help="Real-ESRGAN model variant"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=512,
        help="Minimum size to trigger upscaling (default: 512)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force upscale all images"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile size for processing (smaller = less VRAM)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )

    args = parser.parse_args()

    # Process instances
    stats = process_instances(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        device=args.device,
        min_size=args.min_size,
        force_upscale=args.force,
        tile_size=args.tile_size
    )

    print(f"\n📁 Output saved to: {args.output_dir}/upscaled/")


if __name__ == "__main__":
    main()
