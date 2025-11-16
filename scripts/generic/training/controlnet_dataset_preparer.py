#!/usr/bin/env python3
"""
ControlNet Dataset Preparer
Generates control images (OpenPose, Depth, Canny) for ControlNet training
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm
import json
from PIL import Image
import shutil


class ControlNetDatasetPreparer:
    """Prepare ControlNet training datasets from character images"""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        control_types: List[str] = None,
        use_hard_links: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize ControlNet dataset preparer

        Args:
            input_dir: Directory with character images
            output_dir: Output directory for ControlNet dataset
            control_types: List of control types to generate ['openpose', 'depth', 'canny']
            use_hard_links: Use hard links for source images
            batch_size: Batch size for processing
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.control_types = control_types or ['canny', 'depth']
        self.use_hard_links = use_hard_links
        self.batch_size = batch_size

        print(f"🎮 ControlNet Dataset Preparer")
        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Control Types: {', '.join(self.control_types)}")
        print(f"  Link Mode: {'Hard Links' if use_hard_links else 'Copy'}")

    def generate_canny(self, image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
        """
        Generate Canny edge detection map

        Args:
            image: Input image (RGB or RGBA)
            low_threshold: Low threshold for Canny
            high_threshold: High threshold for Canny

        Returns:
            Canny edge map (grayscale)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                # Use alpha channel to mask edges
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
                alpha = image[:, :, 3]
                gray = cv2.bitwise_and(gray, gray, mask=alpha)
            else:  # RGB
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        # Convert to 3-channel for consistency
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return edges_rgb

    def generate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map using simple heuristics
        For anime characters, use alpha channel and simple depth cues

        Args:
            image: Input image (RGBA)

        Returns:
            Depth map (grayscale, 3-channel)
        """
        if image.shape[2] == 4:  # RGBA
            alpha = image[:, :, 3]
            rgb = image[:, :, :3]
        else:
            alpha = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
            rgb = image

        # Create base depth from alpha (transparent = far, opaque = near)
        depth = alpha.copy().astype(np.float32)

        # Add depth cues from brightness (darker = farther for anime style)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_normalized = gray / 255.0

        # Combine alpha and brightness cues
        depth_normalized = (depth / 255.0) * 0.7 + gray_normalized * 0.3

        # Apply edge-aware smoothing
        depth_smooth = cv2.bilateralFilter(
            (depth_normalized * 255).astype(np.uint8),
            d=9,
            sigmaColor=75,
            sigmaSpace=75
        )

        # Convert to 3-channel
        depth_rgb = cv2.cvtColor(depth_smooth, cv2.COLOR_GRAY2RGB)

        return depth_rgb

    def generate_openpose(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate OpenPose skeleton (placeholder for now)
        Note: Requires MMPose or OpenPose installation

        Args:
            image: Input image

        Returns:
            OpenPose skeleton image or None if not available
        """
        print("  ⚠️  OpenPose generation requires MMPose/DWPose - skipping")
        return None

    def process_image(
        self,
        image_path: Path,
        output_base_dir: Path
    ) -> Dict:
        """
        Process a single image to generate control images

        Args:
            image_path: Path to input image
            output_base_dir: Base output directory

        Returns:
            Dictionary with generated control paths
        """
        # Read image
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return {'error': 'Failed to read image'}

        # Convert BGR to RGB
        if image.shape[2] == 4:  # BGRA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        elif image.shape[2] == 3:  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate base filename
        base_name = image_path.stem

        results = {
            'source': str(image_path),
            'base_name': base_name,
            'controls': {}
        }

        # Generate each control type
        for control_type in self.control_types:
            control_dir = output_base_dir / control_type
            control_dir.mkdir(parents=True, exist_ok=True)

            if control_type == 'canny':
                control_image = self.generate_canny(image)
                control_path = control_dir / f"{base_name}_canny.png"
                cv2.imwrite(str(control_path), cv2.cvtColor(control_image, cv2.COLOR_RGB2BGR))
                results['controls']['canny'] = str(control_path)

            elif control_type == 'depth':
                control_image = self.generate_depth(image)
                control_path = control_dir / f"{base_name}_depth.png"
                cv2.imwrite(str(control_path), cv2.cvtColor(control_image, cv2.COLOR_RGB2BGR))
                results['controls']['depth'] = str(control_path)

            elif control_type == 'openpose':
                control_image = self.generate_openpose(image)
                if control_image is not None:
                    control_path = control_dir / f"{base_name}_pose.png"
                    cv2.imwrite(str(control_path), cv2.cvtColor(control_image, cv2.COLOR_RGB2BGR))
                    results['controls']['openpose'] = str(control_path)

        # Copy/link source image to output
        source_dir = output_base_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        dest_path = source_dir / image_path.name

        if self.use_hard_links and 'ai_warehouse' in str(output_base_dir):
            try:
                dest_path.hardlink_to(image_path)
            except Exception:
                shutil.copy2(image_path, dest_path)
        else:
            shutil.copy2(image_path, dest_path)

        results['source_copy'] = str(dest_path)

        return results

    def prepare_from_character_dir(self):
        """
        Prepare ControlNet dataset from character directory

        Expected structure:
        input_dir/
            character/
                *.png
        OR
        input_dir/
            *.png
        """
        print(f"\n📂 Scanning for character images...")

        # Find character images
        char_dir = self.input_dir / "character"
        if char_dir.exists():
            image_files = list(char_dir.glob("*.png"))
            print(f"  Found {len(image_files)} images in character/ subdirectory")
        else:
            image_files = list(self.input_dir.glob("*.png"))
            print(f"  Found {len(image_files)} PNG images")

        if not image_files:
            print(f"  ⚠️  No PNG images found in {self.input_dir}")
            return

        print(f"\n🎨 Processing {len(image_files)} images...")

        all_results = []

        for image_path in tqdm(image_files, desc="Generating controls"):
            result = self.process_image(image_path, self.output_dir)
            all_results.append(result)

        # Save metadata
        metadata = {
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'control_types': self.control_types,
            'total_images': len(image_files),
            'processed': [r for r in all_results if 'error' not in r],
            'errors': [r for r in all_results if 'error' in r]
        }

        metadata_path = self.output_dir / "controlnet_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ ControlNet dataset prepared!")
        print(f"  Total processed: {len(metadata['processed'])}")
        print(f"  Errors: {len(metadata['errors'])}")
        print(f"  Output: {self.output_dir}")

        # Print dataset structure info
        print(f"\n📊 Dataset Structure:")
        print(f"  {self.output_dir}/")
        print(f"    ├── source/          ({len(image_files)} images)")
        for control_type in self.control_types:
            control_dir = self.output_dir / control_type
            if control_dir.exists():
                count = len(list(control_dir.glob("*.png")))
                print(f"    ├── {control_type}/       ({count} control maps)")
        print(f"    └── controlnet_metadata.json")

        return metadata

    def prepare_from_layered_frames(self):
        """
        Prepare ControlNet dataset from layered_frames structure

        Expected structure:
        input_dir/
            episode_001/
                character/
                    *.png
            episode_002/
                ...
        """
        print(f"\n📂 Scanning layered frames...")

        # Find all episodes
        episodes = sorted([d for d in self.input_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')])

        if not episodes:
            print(f"  ⚠️  No episode directories found")
            return self.prepare_from_character_dir()

        print(f"  Found {len(episodes)} episodes")

        all_results = []
        total_images = 0

        for episode_dir in tqdm(episodes, desc="Processing episodes"):
            char_dir = episode_dir / "character"

            if not char_dir.exists():
                continue

            image_files = list(char_dir.glob("*.png"))
            total_images += len(image_files)

            # Create episode-specific output dir
            episode_output = self.output_dir / episode_dir.name

            for image_path in image_files:
                result = self.process_image(image_path, episode_output)
                result['episode'] = episode_dir.name
                all_results.append(result)

        # Save overall metadata
        metadata = {
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'control_types': self.control_types,
            'total_episodes': len(episodes),
            'total_images': total_images,
            'processed': [r for r in all_results if 'error' not in r],
            'errors': [r for r in all_results if 'error' in r]
        }

        metadata_path = self.output_dir / "controlnet_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ ControlNet dataset prepared from layered frames!")
        print(f"  Episodes: {len(episodes)}")
        print(f"  Total processed: {len(metadata['processed'])}")
        print(f"  Errors: {len(metadata['errors'])}")
        print(f"  Output: {self.output_dir}")

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ControlNet training dataset"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory (layered_frames or character images)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for ControlNet dataset"
    )
    parser.add_argument(
        "--control-types",
        nargs='+',
        choices=['canny', 'depth', 'openpose'],
        default=['canny', 'depth'],
        help="Control types to generate (default: canny depth)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of using hard links"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--layered-frames",
        action="store_true",
        help="Input is layered_frames directory structure"
    )

    args = parser.parse_args()

    preparer = ControlNetDatasetPreparer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        control_types=args.control_types,
        use_hard_links=not args.copy,
        batch_size=args.batch_size
    )

    if args.layered_frames:
        preparer.prepare_from_layered_frames()
    else:
        preparer.prepare_from_character_dir()


if __name__ == "__main__":
    main()
