#!/usr/bin/env python3
"""
Background Dataset Organizer
Organizes background images by scene type for training
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import argparse
from tqdm import tqdm
import json
import shutil
from PIL import Image


class BackgroundDatasetOrganizer:
    """Organize background images by scene type"""

    def __init__(
        self,
        layered_frames_dir: Path,
        output_dir: Path,
        use_hard_links: bool = True,
        min_cluster_size: int = 10
    ):
        """
        Initialize background organizer

        Args:
            layered_frames_dir: Directory with layered frames
            output_dir: Output directory for organized backgrounds
            use_hard_links: Use hard links instead of copying
            min_cluster_size: Minimum images per category
        """
        self.layered_frames_dir = Path(layered_frames_dir)
        self.output_dir = Path(output_dir)
        self.use_hard_links = use_hard_links
        self.min_cluster_size = min_cluster_size

        print(f"🏞️  Background Dataset Organizer")
        print(f"  Input: {layered_frames_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Min Cluster Size: {min_cluster_size}")
        print(f"  Link Mode: {'Hard Links' if use_hard_links else 'Copy'}")

    def extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color histogram features

        Args:
            image: Input image (RGB)

        Returns:
            Normalized histogram feature vector
        """
        # Resize for faster processing
        small = cv2.resize(image, (64, 64))

        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

        # Calculate histograms
        h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])

        # Concatenate and normalize
        hist = np.concatenate([h_hist, s_hist, v_hist]).flatten()
        hist = hist / (hist.sum() + 1e-6)

        return hist

    def extract_edge_features(self, image: np.ndarray) -> float:
        """
        Extract edge density (useful for outdoor vs indoor)

        Args:
            image: Input image (RGB)

        Returns:
            Edge density score
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (64, 64))

        # Canny edge detection
        edges = cv2.Canny(small, 50, 150)

        # Edge density
        edge_density = np.sum(edges > 0) / edges.size

        return edge_density

    def classify_scene_type(self, image: np.ndarray, hist: np.ndarray) -> str:
        """
        Classify scene type based on features

        Args:
            image: Input image (RGB)
            hist: Color histogram

        Returns:
            Scene type classification
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Average colors
        h_avg = np.mean(hsv[:, :, 0])
        s_avg = np.mean(hsv[:, :, 1])
        v_avg = np.mean(hsv[:, :, 2])

        # Edge density
        edge_density = self.extract_edge_features(image)

        # Sky detection (top portion, blue-ish)
        top_third = hsv[:hsv.shape[0]//3, :, :]
        h_top = np.mean(top_third[:, :, 0])
        s_top = np.mean(top_third[:, :, 1])
        v_top = np.mean(top_third[:, :, 2])

        # Sky: blue hue (90-130), moderate saturation, high brightness
        is_sky = (90 <= h_top <= 130) and (s_top > 50) and (v_top > 100)

        # Green detection for field
        green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85)
        green_ratio = np.sum(green_mask) / green_mask.size

        # Classification rules
        if green_ratio > 0.3 and is_sky:
            return "soccer_field"
        elif green_ratio > 0.3:
            return "field_outdoor"
        elif is_sky and edge_density > 0.1:
            return "outdoor_complex"
        elif is_sky:
            return "outdoor_simple"
        elif v_avg < 80:
            return "indoor_dark"
        elif edge_density > 0.15:
            return "indoor_detailed"
        elif s_avg < 50:
            return "indoor_plain"
        else:
            return "indoor_general"

    def analyze_backgrounds(self) -> Dict:
        """
        Analyze all backgrounds and classify them

        Returns:
            Dictionary with classification results
        """
        print(f"\n📊 Analyzing backgrounds...")

        # Find all episodes
        episodes = sorted([d for d in self.layered_frames_dir.iterdir()
                          if d.is_dir() and d.name.startswith('episode_')])

        if not episodes:
            print(f"  ⚠️  No episode directories found")
            return {}

        print(f"  Found {len(episodes)} episodes")

        # Collect all backgrounds
        all_backgrounds = []
        scene_classifications = defaultdict(list)

        for episode_dir in tqdm(episodes, desc="Processing episodes"):
            bg_dir = episode_dir / "background"

            if not bg_dir.exists():
                continue

            bg_files = list(bg_dir.glob("*.jpg"))

            for bg_file in bg_files:
                # Read image
                image = cv2.imread(str(bg_file))
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Extract features
                hist = self.extract_color_histogram(image)
                scene_type = self.classify_scene_type(image, hist)

                bg_info = {
                    'path': str(bg_file),
                    'episode': episode_dir.name,
                    'filename': bg_file.name,
                    'scene_type': scene_type,
                    'histogram': hist.tolist()
                }

                all_backgrounds.append(bg_info)
                scene_classifications[scene_type].append(bg_info)

        return {
            'total_backgrounds': len(all_backgrounds),
            'scene_types': {k: len(v) for k, v in scene_classifications.items()},
            'classifications': scene_classifications,
            'all_backgrounds': all_backgrounds
        }

    def organize_backgrounds(self):
        """
        Organize backgrounds into scene type directories
        """
        # Analyze backgrounds
        analysis = self.analyze_backgrounds()

        if not analysis:
            print(f"  ⚠️  No backgrounds to organize")
            return

        print(f"\n📁 Organizing {analysis['total_backgrounds']} backgrounds...")
        print(f"\n  Scene Types Found:")
        for scene_type, count in sorted(analysis['scene_types'].items()):
            print(f"    {scene_type:20s}: {count:4d} images")

        # Filter by minimum cluster size
        valid_types = {k: v for k, v in analysis['classifications'].items()
                      if len(v) >= self.min_cluster_size}

        print(f"\n  Valid Types (>= {self.min_cluster_size} images):")
        for scene_type, images in sorted(valid_types.items()):
            print(f"    {scene_type:20s}: {len(images):4d} images")

        # Copy/link files to organized structure
        for scene_type, images in tqdm(valid_types.items(), desc="Organizing"):
            scene_dir = self.output_dir / scene_type
            scene_dir.mkdir(parents=True, exist_ok=True)

            for img_info in images:
                src_path = Path(img_info['path'])
                dst_path = scene_dir / f"{img_info['episode']}_{img_info['filename']}"

                if self.use_hard_links and 'ai_warehouse' in str(self.output_dir):
                    try:
                        if not dst_path.exists():
                            dst_path.hardlink_to(src_path)
                    except Exception:
                        if not dst_path.exists():
                            shutil.copy2(src_path, dst_path)
                else:
                    if not dst_path.exists():
                        shutil.copy2(src_path, dst_path)

        # Save metadata
        metadata = {
            'total_backgrounds': analysis['total_backgrounds'],
            'scene_types': analysis['scene_types'],
            'valid_types': {k: len(v) for k, v in valid_types.items()},
            'min_cluster_size': self.min_cluster_size,
            'organized_count': sum(len(v) for v in valid_types.values())
        }

        metadata_path = self.output_dir / "background_organization.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save detailed analysis
        detailed_path = self.output_dir / "background_analysis.json"
        with open(detailed_path, 'w') as f:
            # Don't save histogram features to keep file size reasonable
            analysis_copy = analysis.copy()
            for bg in analysis_copy['all_backgrounds']:
                bg.pop('histogram', None)
            for scene_type in analysis_copy['classifications']:
                for bg in analysis_copy['classifications'][scene_type]:
                    bg.pop('histogram', None)
            json.dump(analysis_copy, f, indent=2)

        print(f"\n✅ Background organization complete!")
        print(f"  Total backgrounds: {analysis['total_backgrounds']}")
        print(f"  Organized: {metadata['organized_count']}")
        print(f"  Output: {self.output_dir}")

        print(f"\n📊 Dataset Structure:")
        print(f"  {self.output_dir}/")
        for scene_type, images in sorted(valid_types.items()):
            print(f"    ├── {scene_type}/  ({len(images)} images)")
        print(f"    ├── background_organization.json")
        print(f"    └── background_analysis.json")

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Organize background images by scene type"
    )
    parser.add_argument(
        "layered_frames_dir",
        type=Path,
        help="Directory with layered frames"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for organized backgrounds"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="Minimum images per category (default: 10)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of using hard links"
    )

    args = parser.parse_args()

    organizer = BackgroundDatasetOrganizer(
        layered_frames_dir=args.layered_frames_dir,
        output_dir=args.output_dir,
        use_hard_links=not args.copy,
        min_cluster_size=args.min_cluster_size
    )

    organizer.organize_backgrounds()


if __name__ == "__main__":
    main()
