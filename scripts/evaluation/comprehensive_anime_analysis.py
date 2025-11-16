#!/usr/bin/env python3
"""
Comprehensive Anime Frame Analysis Tool
Analyzes layered anime frames (background/character/masks) for:
1. Color palette and visual style
2. Character attributes and statistics
3. Quality assessment and filtering

Perfect for Yokai Watch and similar anime with diverse scenes and characters
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from collections import defaultdict
import pandas as pd


class ColorAnalyzer:
    """Analyze color palettes and visual styles"""

    def __init__(self, n_colors: int = 5):
        self.n_colors = n_colors

    def extract_dominant_colors(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract dominant colors using K-means clustering

        Args:
            image: RGB image array

        Returns:
            (colors, percentages) - RGB colors and their percentages
        """
        # Reshape to pixels
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Remove very dark pixels (likely artifacts)
        brightness = pixels.mean(axis=1)
        pixels = pixels[brightness > 10]

        if len(pixels) < 100:
            return np.zeros((self.n_colors, 3)), np.zeros(self.n_colors)

        # K-means clustering
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_

        # Calculate percentages
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / counts.sum()

        # Sort by percentage
        sorted_idx = np.argsort(percentages)[::-1]
        colors = colors[sorted_idx]
        percentages = percentages[sorted_idx]

        return colors, percentages

    def calculate_color_metrics(self, image: np.ndarray) -> Dict:
        """
        Calculate various color metrics

        Args:
            image: RGB image

        Returns:
            Dictionary of color metrics
        """
        # Convert to HSV for better analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Calculate metrics
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])

        # Color temperature (warm vs cool)
        # Warm: red/orange/yellow (hue 0-60), Cool: blue/cyan (hue 100-140)
        warm_mask = (hsv[:, :, 0] < 60) | (hsv[:, :, 0] > 160)
        cool_mask = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 140)

        warm_ratio = warm_mask.sum() / hsv[:, :, 0].size
        cool_ratio = cool_mask.sum() / hsv[:, :, 0].size

        # Color diversity (entropy)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist = hist / hist.sum()
        color_entropy = -np.sum(hist * np.log2(hist + 1e-10))

        return {
            "avg_hue": float(avg_hue),
            "avg_saturation": float(avg_saturation),
            "avg_brightness": float(avg_value),
            "warm_ratio": float(warm_ratio),
            "cool_ratio": float(cool_ratio),
            "color_entropy": float(color_entropy),
            "is_warm": warm_ratio > cool_ratio,
            "is_vibrant": avg_saturation > 100
        }


class CharacterAttributeAnalyzer:
    """Analyze character attributes and statistics"""

    def __init__(self):
        pass

    def analyze_character(self, char_image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze character attributes

        Args:
            char_image: RGBA character image
            mask: Optional segmentation mask

        Returns:
            Dictionary of character attributes
        """
        if char_image.shape[2] == 4:
            # Has alpha channel
            alpha = char_image[:, :, 3]
            rgb = char_image[:, :, :3]
        else:
            alpha = np.ones((char_image.shape[0], char_image.shape[1]), dtype=np.uint8) * 255
            rgb = char_image

        # Calculate bounding box
        y_coords, x_coords = np.where(alpha > 10)

        if len(y_coords) == 0:
            return self._empty_result()

        bbox = {
            "x_min": int(x_coords.min()),
            "x_max": int(x_coords.max()),
            "y_min": int(y_coords.min()),
            "y_max": int(y_coords.max())
        }

        bbox_width = bbox["x_max"] - bbox["x_min"]
        bbox_height = bbox["y_max"] - bbox["y_min"]

        # Alpha coverage
        alpha_coverage = (alpha > 10).sum() / alpha.size

        # Character size relative to image
        relative_size = alpha_coverage

        # Complexity (edge density)
        char_region = rgb[bbox["y_min"]:bbox["y_max"], bbox["x_min"]:bbox["x_max"]]
        if char_region.size > 0:
            gray = cv2.cvtColor(char_region, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = edges.sum() / edges.size
        else:
            edge_density = 0

        # Color richness
        if char_region.size > 0:
            unique_colors = len(np.unique(char_region.reshape(-1, 3), axis=0))
            color_richness = unique_colors / (char_region.shape[0] * char_region.shape[1])
        else:
            color_richness = 0

        # Position
        center_x = (bbox["x_min"] + bbox["x_max"]) / 2 / char_image.shape[1]
        center_y = (bbox["y_min"] + bbox["y_max"]) / 2 / char_image.shape[0]

        return {
            "bbox": bbox,
            "width": int(bbox_width),
            "height": int(bbox_height),
            "aspect_ratio": float(bbox_width / bbox_height) if bbox_height > 0 else 1.0,
            "alpha_coverage": float(alpha_coverage),
            "relative_size": float(relative_size),
            "edge_density": float(edge_density),
            "color_richness": float(color_richness),
            "center_x": float(center_x),
            "center_y": float(center_y),
            "position_label": self._position_label(center_x, center_y)
        }

    def _position_label(self, center_x: float, center_y: float) -> str:
        """Label position (left/center/right, top/middle/bottom)"""
        h_pos = "left" if center_x < 0.33 else "center" if center_x < 0.67 else "right"
        v_pos = "top" if center_y < 0.33 else "middle" if center_y < 0.67 else "bottom"
        return f"{v_pos}_{h_pos}"

    def _empty_result(self) -> Dict:
        """Return empty result for invalid images"""
        return {
            "bbox": {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0},
            "width": 0,
            "height": 0,
            "aspect_ratio": 1.0,
            "alpha_coverage": 0.0,
            "relative_size": 0.0,
            "edge_density": 0.0,
            "color_richness": 0.0,
            "center_x": 0.5,
            "center_y": 0.5,
            "position_label": "middle_center"
        }


class QualityAssessor:
    """Assess quality of extracted frames"""

    def __init__(self):
        pass

    def assess_quality(
        self,
        char_image: np.ndarray,
        bg_image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Assess frame quality

        Args:
            char_image: Character image (RGBA)
            bg_image: Background image (optional)
            mask: Segmentation mask (optional)

        Returns:
            Quality metrics dictionary
        """
        metrics = {}

        # Character quality
        if char_image.shape[2] == 4:
            alpha = char_image[:, :, 3]
            rgb = char_image[:, :, :3]
        else:
            alpha = np.ones((char_image.shape[0], char_image.shape[1]), dtype=np.uint8) * 255
            rgb = char_image

        # Blur detection (Laplacian variance)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics["blur_score"] = float(laplacian_var)
        metrics["is_blurry"] = laplacian_var < 100

        # Alpha quality (edge sharpness)
        if alpha is not None:
            alpha_edges = cv2.Canny(alpha, 50, 150)
            edge_sharpness = alpha_edges.sum() / (alpha > 10).sum() if (alpha > 10).sum() > 0 else 0
            metrics["edge_sharpness"] = float(edge_sharpness)
            metrics["alpha_coverage"] = float((alpha > 10).sum() / alpha.size)
        else:
            metrics["edge_sharpness"] = 0.0
            metrics["alpha_coverage"] = 1.0

        # Character completeness (is character cropped?)
        y_coords, x_coords = np.where(alpha > 10)
        if len(y_coords) > 0:
            touches_border = (
                x_coords.min() == 0 or
                x_coords.max() == alpha.shape[1] - 1 or
                y_coords.min() == 0 or
                y_coords.max() == alpha.shape[0] - 1
            )
            metrics["touches_border"] = bool(touches_border)
        else:
            metrics["touches_border"] = False

        # Overall quality score (0-1)
        quality_score = 0.0
        if not metrics["is_blurry"]:
            quality_score += 0.4
        if metrics["edge_sharpness"] > 0.1:
            quality_score += 0.3
        if not metrics["touches_border"]:
            quality_score += 0.3

        metrics["quality_score"] = quality_score
        metrics["quality_label"] = self._quality_label(quality_score)

        return metrics

    def _quality_label(self, score: float) -> str:
        """Convert score to label"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"


class ComprehensiveAnalyzer:
    """Main analyzer combining all analysis tools"""

    def __init__(self, n_colors: int = 5):
        self.color_analyzer = ColorAnalyzer(n_colors=n_colors)
        self.char_analyzer = CharacterAttributeAnalyzer()
        self.quality_assessor = QualityAssessor()

    def analyze_frame(
        self,
        episode_dir: Path,
        frame_name: str
    ) -> Dict:
        """
        Comprehensive analysis of a single frame

        Args:
            episode_dir: Episode directory containing layers
            frame_name: Frame filename (without extension)

        Returns:
            Complete analysis dictionary
        """
        # Load images
        bg_path = episode_dir / "background" / f"{frame_name}.png"
        char_path = episode_dir / "character" / f"{frame_name}.png"
        mask_path = episode_dir / "masks" / f"{frame_name}.png"

        result = {
            "frame": frame_name,
            "episode": episode_dir.name
        }

        # Background analysis
        if bg_path.exists():
            bg_img = np.array(Image.open(bg_path).convert("RGB"))
            colors, percentages = self.color_analyzer.extract_dominant_colors(bg_img)
            color_metrics = self.color_analyzer.calculate_color_metrics(bg_img)

            result["background"] = {
                "dominant_colors": colors.tolist(),
                "color_percentages": percentages.tolist(),
                "metrics": color_metrics
            }
        else:
            result["background"] = None

        # Character analysis
        if char_path.exists():
            char_img = np.array(Image.open(char_path))
            char_attrs = self.char_analyzer.analyze_character(char_img)

            # Quality assessment
            quality = self.quality_assessor.assess_quality(char_img)

            result["character"] = {
                "attributes": char_attrs,
                "quality": quality
            }
        else:
            result["character"] = None

        return result

    def analyze_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        max_frames_per_episode: Optional[int] = None
    ) -> Dict:
        """
        Analyze entire dataset

        Args:
            input_dir: Root directory with episode folders
            output_dir: Output directory for results
            max_frames_per_episode: Limit frames per episode (for testing)

        Returns:
            Summary statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        episodes = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("episode")])

        print(f"\n🔍 Analyzing {len(episodes)} episodes...")

        for episode_dir in tqdm(episodes, desc="Episodes"):
            char_dir = episode_dir / "character"
            if not char_dir.exists():
                continue

            # Get frame names
            frame_files = list(char_dir.glob("*.png"))
            if max_frames_per_episode:
                frame_files = frame_files[:max_frames_per_episode]

            for frame_file in frame_files:
                frame_name = frame_file.stem
                try:
                    result = self.analyze_frame(episode_dir, frame_name)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error analyzing {episode_dir.name}/{frame_name}: {e}")
                    continue

        # Save detailed results
        results_path = output_dir / "detailed_analysis.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)

        print(f"✓ Detailed results saved: {results_path}")

        # Generate summary statistics
        summary = self._generate_summary(all_results, output_dir)

        return summary

    def _generate_summary(self, results: List[Dict], output_dir: Path) -> Dict:
        """Generate summary statistics and visualizations"""
        print("\n📊 Generating summary statistics...")

        summary = {
            "total_frames": len(results),
            "timestamp": datetime.now().isoformat()
        }

        # Color analysis summary
        color_temps = []
        saturations = []
        brightnesses = []

        # Character stats
        sizes = []
        positions = defaultdict(int)
        quality_scores = []
        quality_labels = defaultdict(int)

        for result in results:
            # Background color stats
            if result.get("background"):
                metrics = result["background"]["metrics"]
                if metrics["is_warm"]:
                    color_temps.append("warm")
                else:
                    color_temps.append("cool")
                saturations.append(metrics["avg_saturation"])
                brightnesses.append(metrics["avg_brightness"])

            # Character stats
            if result.get("character"):
                char = result["character"]
                sizes.append(char["attributes"]["relative_size"])
                positions[char["attributes"]["position_label"]] += 1
                quality_scores.append(char["quality"]["quality_score"])
                quality_labels[char["quality"]["quality_label"]] += 1

        # Summary statistics
        summary["color_analysis"] = {
            "warm_frames": color_temps.count("warm"),
            "cool_frames": color_temps.count("cool"),
            "avg_saturation": float(np.mean(saturations)) if saturations else 0,
            "avg_brightness": float(np.mean(brightnesses)) if brightnesses else 0
        }

        summary["character_analysis"] = {
            "avg_size": float(np.mean(sizes)) if sizes else 0,
            "position_distribution": dict(positions),
            "avg_quality_score": float(np.mean(quality_scores)) if quality_scores else 0,
            "quality_distribution": dict(quality_labels)
        }

        # Save summary
        summary_path = output_dir / "analysis_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Summary saved: {summary_path}")

        # Generate visualizations
        self._create_visualizations(results, output_dir)

        return summary

    def _create_visualizations(self, results: List[Dict], output_dir: Path):
        """Create analysis visualizations"""
        print("\n📈 Creating visualizations...")

        # Prepare data
        df_data = []
        for result in results:
            row = {"episode": result["episode"]}

            if result.get("background"):
                row.update({
                    "saturation": result["background"]["metrics"]["avg_saturation"],
                    "brightness": result["background"]["metrics"]["avg_brightness"],
                    "is_warm": result["background"]["metrics"]["is_warm"]
                })

            if result.get("character"):
                row.update({
                    "char_size": result["character"]["attributes"]["relative_size"],
                    "quality_score": result["character"]["quality"]["quality_score"],
                    "quality_label": result["character"]["quality"]["quality_label"]
                })

            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Anime Frame Analysis', fontsize=16, fontweight='bold')

        # 1. Color temperature distribution
        if 'is_warm' in df.columns:
            ax = axes[0, 0]
            df['is_warm'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'coral'])
            ax.set_title('Color Temperature Distribution')
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Count')
            ax.set_xticklabels(['Cool', 'Warm'], rotation=0)

        # 2. Saturation distribution
        if 'saturation' in df.columns:
            ax = axes[0, 1]
            df['saturation'].hist(bins=30, ax=ax, color='purple', alpha=0.7)
            ax.set_title('Saturation Distribution')
            ax.set_xlabel('Saturation')
            ax.set_ylabel('Frequency')

        # 3. Character size distribution
        if 'char_size' in df.columns:
            ax = axes[0, 2]
            df['char_size'].hist(bins=30, ax=ax, color='green', alpha=0.7)
            ax.set_title('Character Size Distribution')
            ax.set_xlabel('Relative Size')
            ax.set_ylabel('Frequency')

        # 4. Quality score distribution
        if 'quality_score' in df.columns:
            ax = axes[1, 0]
            df['quality_score'].hist(bins=20, ax=ax, color='orange', alpha=0.7)
            ax.set_title('Quality Score Distribution')
            ax.set_xlabel('Quality Score')
            ax.set_ylabel('Frequency')

        # 5. Quality labels
        if 'quality_label' in df.columns:
            ax = axes[1, 1]
            df['quality_label'].value_counts().plot(kind='bar', ax=ax, color='teal')
            ax.set_title('Quality Categories')
            ax.set_xlabel('Quality')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)

        # 6. Saturation vs Brightness
        if 'saturation' in df.columns and 'brightness' in df.columns:
            ax = axes[1, 2]
            ax.scatter(df['saturation'], df['brightness'], alpha=0.5, c='blue', s=10)
            ax.set_title('Saturation vs Brightness')
            ax.set_xlabel('Saturation')
            ax.set_ylabel('Brightness')

        plt.tight_layout()

        viz_path = output_dir / "analysis_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualizations saved: {viz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Anime Frame Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all frames
  python comprehensive_anime_analysis.py /path/to/layered_frames -o /path/to/analysis

  # Quick test (limit frames)
  python comprehensive_anime_analysis.py /path/to/frames -o /path/to/output --max-per-episode 50

  # Custom color palette size
  python comprehensive_anime_analysis.py /path/to/frames -o /path/to/output --n-colors 7
"""
    )

    parser.add_argument("input_dir", type=Path, help="Input directory with episode folders")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-colors", type=int, default=5, help="Number of dominant colors to extract")
    parser.add_argument("--max-per-episode", type=int, help="Max frames per episode (for testing)")

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return

    # Run analysis
    analyzer = ComprehensiveAnalyzer(n_colors=args.n_colors)
    summary = analyzer.analyze_dataset(
        args.input_dir,
        args.output_dir,
        max_frames_per_episode=args.max_per_episode
    )

    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Total frames analyzed: {summary['total_frames']}")
    print(f"Output directory: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
