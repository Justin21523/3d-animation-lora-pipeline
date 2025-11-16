#!/usr/bin/env python3
"""
Background Cleanup for Character Instances

Refines alpha mattes and ensures clean, consistent backgrounds.
Optimized for 3D animated characters with anti-aliased edges.

Key Features:
- Alpha matte refinement
- Edge smoothing (preserve anti-aliasing)
- Background uniformity
- Batch processing with resume capability
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import Tuple, Optional
from tqdm import tqdm
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class BackgroundCleaner:
    """
    Background cleanup and alpha matte refinement
    """

    def __init__(
        self,
        background_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
        edge_smoothing: bool = True,
        alpha_threshold: int = 10
    ):
        """
        Initialize background cleaner

        Args:
            background_color: RGBA background color (default: transparent)
            edge_smoothing: Apply edge smoothing to preserve anti-aliasing
            alpha_threshold: Minimum alpha value to keep (0-255)
        """
        self.background_color = background_color
        self.edge_smoothing = edge_smoothing
        self.alpha_threshold = alpha_threshold

        print(f"🔧 Initializing Background Cleaner...")
        print(f"   Background: {background_color}")
        print(f"   Edge smoothing: {edge_smoothing}")
        print(f"   Alpha threshold: {alpha_threshold}")

    def refine_alpha(self, alpha: np.ndarray) -> np.ndarray:
        """
        Refine alpha matte

        Args:
            alpha: Alpha channel (0-255)

        Returns:
            Refined alpha channel
        """
        # Remove very low alpha (noise)
        alpha_refined = np.where(alpha < self.alpha_threshold, 0, alpha)

        # Slight morphological operations to clean up edges
        kernel = np.ones((3, 3), np.uint8)

        # Close small holes
        alpha_refined = cv2.morphologyEx(
            alpha_refined.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        # Open to remove small noise
        alpha_refined = cv2.morphologyEx(
            alpha_refined,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        return alpha_refined

    def smooth_edges(
        self,
        image_rgba: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """
        Smooth edges while preserving anti-aliasing

        Args:
            image_rgba: RGBA image
            alpha: Alpha channel

        Returns:
            Smoothed alpha channel
        """
        # Apply bilateral filter to alpha channel
        # This preserves edges while smoothing noise
        alpha_smooth = cv2.bilateralFilter(
            alpha.astype(np.uint8),
            d=5,
            sigmaColor=50,
            sigmaSpace=50
        )

        return alpha_smooth

    def cleanup_background(
        self,
        image: Image.Image
    ) -> Image.Image:
        """
        Clean up background and refine alpha matte

        Args:
            image: PIL Image (should have alpha channel)

        Returns:
            Cleaned PIL Image
        """
        # Convert to numpy
        image_np = np.array(image)

        # Check if has alpha
        if image_np.shape[2] != 4:
            # No alpha channel, create one from brightness
            print("⚠️  No alpha channel, creating from brightness...")
            image_rgb = image_np
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        else:
            image_rgb = image_np[:, :, :3]
            alpha = image_np[:, :, 3]

        # Refine alpha
        alpha_refined = self.refine_alpha(alpha)

        # Smooth edges if requested
        if self.edge_smoothing:
            alpha_refined = self.smooth_edges(image_np, alpha_refined)

        # Create clean RGBA image
        cleaned_rgba = np.zeros((*alpha_refined.shape, 4), dtype=np.uint8)

        # Set RGB channels
        cleaned_rgba[:, :, :3] = image_rgb

        # Set alpha channel
        cleaned_rgba[:, :, 3] = alpha_refined

        # Apply background color to fully transparent areas
        if self.background_color != (0, 0, 0, 0):
            bg = np.ones_like(cleaned_rgba) * np.array(self.background_color, dtype=np.uint8)
            alpha_mask = (alpha_refined / 255.0)[:, :, None]
            cleaned_rgba = (cleaned_rgba * alpha_mask + bg * (1 - alpha_mask)).astype(np.uint8)

        return Image.fromarray(cleaned_rgba)


def process_instances(
    input_dir: Path,
    output_dir: Path,
    background_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
    edge_smoothing: bool = True,
    alpha_threshold: int = 10
) -> dict:
    """
    Process all character instances with background cleanup

    Args:
        input_dir: Directory with character instances
        output_dir: Output directory for cleaned instances
        background_color: RGBA background color
        edge_smoothing: Apply edge smoothing
        alpha_threshold: Minimum alpha value to keep

    Returns:
        Processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory
    cleaned_dir = output_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    # Initialize cleaner
    cleaner = BackgroundCleaner(
        background_color=background_color,
        edge_smoothing=edge_smoothing,
        alpha_threshold=alpha_threshold
    )

    # Find all instances
    image_files = sorted(
        list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    )

    print(f"\n📊 Processing {len(image_files)} instances...")

    # Check for already processed files
    processed_files = set()
    if cleaned_dir.exists():
        for f in cleaned_dir.glob("*.png"):
            processed_files.add(f.name)

    print(f"📊 Found {len(processed_files)} already processed, will skip them...")

    stats = {
        'total_instances': len(image_files),
        'processed': 0,
        'skipped': 0,
        'no_alpha_channel': 0
    }

    for img_path in tqdm(image_files, desc="Cleaning backgrounds"):
        # Skip if already processed
        output_filename = img_path.stem + '.png'  # Always save as PNG for alpha
        if output_filename in processed_files:
            stats['skipped'] += 1
            continue

        # Load image
        image = Image.open(img_path)

        # Check alpha
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            stats['no_alpha_channel'] += 1

        # Clean background
        cleaned = cleaner.cleanup_background(image)

        # Save (always as PNG to preserve alpha)
        output_path = cleaned_dir / output_filename
        cleaned.save(output_path)
        stats['processed'] += 1

    # Save statistics
    stats_path = output_dir / "cleanup_stats.json"
    with open(stats_path, 'w') as f:
        json.dump({
            'statistics': stats,
            'parameters': {
                'background_color': background_color,
                'edge_smoothing': edge_smoothing,
                'alpha_threshold': alpha_threshold
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n✅ Background cleanup complete!")
    print(f"   Processed: {stats['processed']}")
    print(f"   No alpha channel (created): {stats['no_alpha_channel']}")
    print(f"   Skipped: {stats['skipped']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Background cleanup for character instances"
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
        help="Output directory for cleaned instances"
    )
    parser.add_argument(
        "--background",
        type=str,
        default="transparent",
        choices=["transparent", "white", "black", "gray"],
        help="Background color (default: transparent)"
    )
    parser.add_argument(
        "--no-edge-smoothing",
        action="store_true",
        help="Disable edge smoothing"
    )
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=10,
        help="Minimum alpha value to keep (0-255, default: 10)"
    )

    args = parser.parse_args()

    # Parse background color
    bg_colors = {
        "transparent": (0, 0, 0, 0),
        "white": (255, 255, 255, 255),
        "black": (0, 0, 0, 255),
        "gray": (128, 128, 128, 255)
    }
    background_color = bg_colors[args.background]

    # Process instances
    stats = process_instances(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        background_color=background_color,
        edge_smoothing=not args.no_edge_smoothing,
        alpha_threshold=args.alpha_threshold
    )

    print(f"\n📁 Output saved to: {args.output_dir}/cleaned/")


if __name__ == "__main__":
    main()
