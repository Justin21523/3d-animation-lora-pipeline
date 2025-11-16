#!/usr/bin/env python3
"""
Lighting Analyzer

Purpose: Analyze lighting conditions and characteristics in frames
Features: Light direction, intensity, color temperature, key/fill/rim classification
Use Cases: Style analysis, lighting consistency, scene classification

Usage:
    python lighting_analyzer.py \
        --frames-dir /path/to/frames \
        --output-dir /path/to/analysis \
        --analyze-direction \
        --analyze-temperature \
        --detect-lighting-setup \
        --project luca
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from collections import defaultdict


@dataclass
class LightingAnalysisConfig:
    """Configuration for lighting analysis"""
    analyze_direction: bool = True  # Estimate light direction
    analyze_intensity: bool = True  # Measure light intensity
    analyze_temperature: bool = True  # Color temperature analysis
    detect_lighting_setup: bool = True  # Classify lighting setup
    analyze_shadows: bool = True  # Shadow analysis
    sample_rate: int = 5  # Analyze every N frames
    save_visualizations: bool = True


class LightingAnalyzer:
    """Comprehensive lighting analysis for animated content"""

    def __init__(self, config: LightingAnalysisConfig):
        """
        Initialize lighting analyzer

        Args:
            config: Analysis configuration
        """
        self.config = config

    def estimate_light_direction(self, frame_path: Path) -> Dict:
        """
        Estimate dominant light direction from shadows and highlights

        Args:
            frame_path: Path to frame

        Returns:
            Light direction estimation
        """
        img = cv2.imread(str(frame_path))
        if img is None:
            return {"estimated": False}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find highlight regions (bright areas)
        highlight_threshold = np.percentile(gray, 90)
        highlights = gray > highlight_threshold

        # Find shadow regions (dark areas)
        shadow_threshold = np.percentile(gray, 30)
        shadows = gray < shadow_threshold

        # Compute center of mass for highlights and shadows
        if np.sum(highlights) > 0 and np.sum(shadows) > 0:
            highlight_coords = np.argwhere(highlights)
            shadow_coords = np.argwhere(shadows)

            highlight_center = highlight_coords.mean(axis=0)
            shadow_center = shadow_coords.mean(axis=0)

            # Direction from shadow to highlight
            direction_vec = highlight_center - shadow_center
            direction_angle = np.arctan2(direction_vec[0], direction_vec[1])

            # Classify into compass directions
            angle_deg = np.degrees(direction_angle)
            if -22.5 <= angle_deg < 22.5:
                direction = "top"
            elif 22.5 <= angle_deg < 67.5:
                direction = "top-right"
            elif 67.5 <= angle_deg < 112.5:
                direction = "right"
            elif 112.5 <= angle_deg < 157.5:
                direction = "bottom-right"
            elif angle_deg >= 157.5 or angle_deg < -157.5:
                direction = "bottom"
            elif -157.5 <= angle_deg < -112.5:
                direction = "bottom-left"
            elif -112.5 <= angle_deg < -67.5:
                direction = "left"
            else:
                direction = "top-left"

            estimated = True
        else:
            direction = "unknown"
            direction_angle = 0
            estimated = False

        return {
            "estimated": estimated,
            "direction": direction,
            "angle_radians": float(direction_angle),
            "angle_degrees": float(np.degrees(direction_angle)),
            "highlight_area": float(np.sum(highlights) / highlights.size),
            "shadow_area": float(np.sum(shadows) / shadows.size)
        }

    def measure_light_intensity(self, frame_path: Path) -> Dict:
        """
        Measure overall and local light intensity

        Args:
            frame_path: Path to frame

        Returns:
            Intensity measurements
        """
        img = cv2.imread(str(frame_path))
        if img is None:
            return {}

        # Convert to LAB for perceptual lightness
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Overall intensity metrics
        mean_intensity = float(l_channel.mean())
        median_intensity = float(np.median(l_channel))
        std_intensity = float(l_channel.std())

        # Intensity distribution
        intensity_hist, _ = np.histogram(l_channel, bins=5, range=(0, 255))
        intensity_dist = (intensity_hist / intensity_hist.sum()).tolist()

        # Classify overall lighting level
        if mean_intensity < 80:
            level = "low-key"
        elif mean_intensity > 170:
            level = "high-key"
        else:
            level = "normal"

        # Contrast ratio
        contrast_ratio = float(np.percentile(l_channel, 95) / (np.percentile(l_channel, 5) + 1))

        # Spatial intensity variation
        h, w = l_channel.shape
        grid_h, grid_w = 4, 4
        cell_h, cell_w = h // grid_h, w // grid_w

        local_intensities = []
        for i in range(grid_h):
            for j in range(grid_w):
                cell = l_channel[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                local_intensities.append(cell.mean())

        intensity_variance = float(np.var(local_intensities))

        return {
            "mean_intensity": mean_intensity,
            "median_intensity": median_intensity,
            "std_intensity": std_intensity,
            "level": level,
            "contrast_ratio": contrast_ratio,
            "intensity_variance": intensity_variance,
            "intensity_distribution": intensity_dist
        }

    def analyze_color_temperature(self, frame_path: Path) -> Dict:
        """
        Analyze color temperature (warm vs cool)

        Args:
            frame_path: Path to frame

        Returns:
            Color temperature analysis
        """
        img = cv2.imread(str(frame_path))
        if img is None:
            return {}

        # Split channels
        b, g, r = cv2.split(img)

        # Compute average values
        r_mean = float(r.mean())
        g_mean = float(g.mean())
        b_mean = float(b.mean())

        # Color temperature indicators
        warm_cool = (r_mean - b_mean) / 255.0  # Positive = warm, negative = cool

        # Correlated color temperature (simplified estimation)
        if warm_cool > 0.15:
            temperature = "warm"
            cct_estimate = "low (2700-3500K)"  # Warm light
        elif warm_cool < -0.15:
            temperature = "cool"
            cct_estimate = "high (5500-6500K)"  # Cool daylight
        else:
            temperature = "neutral"
            cct_estimate = "medium (4000-5000K)"  # Neutral

        # Analyze per-channel distributions
        r_std = float(r.std())
        g_std = float(g.std())
        b_std = float(b.std())

        # Tint (green-magenta shift)
        tint = (g_mean - (r_mean + b_mean) / 2) / 255.0

        if tint > 0.05:
            tint_direction = "green"
        elif tint < -0.05:
            tint_direction = "magenta"
        else:
            tint_direction = "neutral"

        return {
            "temperature": temperature,
            "cct_estimate": cct_estimate,
            "warm_cool_value": float(warm_cool),
            "tint": tint_direction,
            "tint_value": float(tint),
            "r_mean": r_mean,
            "g_mean": g_mean,
            "b_mean": b_mean,
            "color_variance": float(np.mean([r_std, g_std, b_std]))
        }

    def detect_lighting_setup(self, frame_path: Path) -> Dict:
        """
        Detect lighting setup (key/fill/rim/back lights)

        Args:
            frame_path: Path to frame

        Returns:
            Lighting setup classification
        """
        img = cv2.imread(str(frame_path))
        if img is None:
            return {"setup": "unknown"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Divide into regions
        regions = {
            "center": gray[h//4:3*h//4, w//4:3*w//4],
            "left": gray[:, :w//3],
            "right": gray[:, 2*w//3:],
            "top": gray[:h//3, :],
            "bottom": gray[2*h//3:, :]
        }

        region_brightness = {k: float(v.mean()) for k, v in regions.items()}

        # Classify lighting setup
        center_bright = region_brightness["center"]

        # Three-point lighting detection
        has_key = center_bright > 120
        has_fill = min(region_brightness["left"], region_brightness["right"]) > 80
        has_rim = max(region_brightness["left"], region_brightness["right"]) > center_bright * 1.2

        if has_key and has_fill and has_rim:
            setup = "three-point"
        elif has_key and has_fill:
            setup = "two-point"
        elif has_key:
            setup = "single-key"
        elif center_bright < 60:
            setup = "low-key"
        elif center_bright > 180:
            setup = "high-key"
        else:
            setup = "flat"

        # Analyze lighting ratio (key to fill)
        if has_fill:
            lighting_ratio = center_bright / (min(region_brightness["left"], region_brightness["right"]) + 1)
        else:
            lighting_ratio = 1.0

        return {
            "setup": setup,
            "has_key_light": has_key,
            "has_fill_light": has_fill,
            "has_rim_light": has_rim,
            "lighting_ratio": float(lighting_ratio),
            "region_brightness": region_brightness
        }

    def analyze_shadows(self, frame_path: Path) -> Dict:
        """
        Analyze shadow characteristics

        Args:
            frame_path: Path to frame

        Returns:
            Shadow analysis
        """
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {}

        # Find shadow regions
        shadow_threshold = np.percentile(img, 30)
        shadows = img < shadow_threshold

        shadow_area = float(np.sum(shadows) / shadows.size)

        # Analyze shadow edges (hard vs soft)
        edges = cv2.Canny(img, 30, 90)
        shadow_edges = cv2.bitwise_and(edges, edges, mask=shadows.astype(np.uint8))

        # Hard shadows have sharp edges
        edge_density = np.sum(shadow_edges > 0) / (np.sum(shadows) + 1)

        if edge_density > 0.1:
            shadow_type = "hard"
        elif edge_density > 0.03:
            shadow_type = "medium"
        else:
            shadow_type = "soft"

        # Shadow intensity (how dark)
        shadow_pixels = img[shadows]
        if len(shadow_pixels) > 0:
            shadow_intensity = float(shadow_pixels.mean())
            shadow_std = float(shadow_pixels.std())
        else:
            shadow_intensity = 0
            shadow_std = 0

        return {
            "shadow_area": shadow_area,
            "shadow_type": shadow_type,
            "edge_density": float(edge_density),
            "shadow_intensity": shadow_intensity,
            "shadow_variance": shadow_std,
            "has_significant_shadows": shadow_area > 0.1
        }

    def analyze_lighting(
        self,
        frames_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main lighting analysis pipeline

        Args:
            frames_dir: Directory with frames
            output_dir: Output directory

        Returns:
            Analysis results
        """
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Lighting Analysis")
        print(f"   Frames: {frames_dir}")
        print(f"   Output: {output_dir}")

        # Find frames
        frames = sorted(
            list(frames_dir.glob("frame_*.jpg")) +
            list(frames_dir.glob("frame_*.png"))
        )

        print(f"   Total frames: {len(frames)}")

        # Sample frames
        sampled_frames = frames[::self.config.sample_rate]
        print(f"   Analyzing {len(sampled_frames)} frames (sample rate: 1/{self.config.sample_rate})")

        # Analyze lighting for each frame
        lighting_data = []

        for frame_path in tqdm(sampled_frames, desc="Analyzing lighting"):
            frame_lighting = {
                "frame": frame_path.name,
            }

            # Light direction
            if self.config.analyze_direction:
                frame_lighting["direction"] = self.estimate_light_direction(frame_path)

            # Intensity
            if self.config.analyze_intensity:
                frame_lighting["intensity"] = self.measure_light_intensity(frame_path)

            # Color temperature
            if self.config.analyze_temperature:
                frame_lighting["temperature"] = self.analyze_color_temperature(frame_path)

            # Lighting setup
            if self.config.detect_lighting_setup:
                frame_lighting["setup"] = self.detect_lighting_setup(frame_path)

            # Shadows
            if self.config.analyze_shadows:
                frame_lighting["shadows"] = self.analyze_shadows(frame_path)

            lighting_data.append(frame_lighting)

        # Aggregate statistics
        statistics = self.compute_lighting_statistics(lighting_data)

        print(f"\n📊 Lighting Statistics:")
        print(f"   Avg intensity: {statistics['avg_intensity']:.1f}")
        print(f"   Intensity levels: {statistics['intensity_levels']}")
        print(f"   Color temperatures: {statistics['temperatures']}")
        print(f"   Lighting setups: {statistics['setups']}")
        print(f"   Shadow types: {statistics['shadow_types']}")

        # Save results
        results = {
            "frames_dir": str(frames_dir),
            "total_frames": len(frames),
            "analyzed_frames": len(sampled_frames),
            "config": {
                "sample_rate": self.config.sample_rate,
                "analyze_direction": self.config.analyze_direction,
                "analyze_intensity": self.config.analyze_intensity,
                "analyze_temperature": self.config.analyze_temperature,
            },
            "lighting_data": lighting_data,
            "statistics": statistics,
            "timestamp": datetime.now().isoformat()
        }

        results_path = output_dir / "lighting_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved: {results_path}")

        # Save visualizations
        if self.config.save_visualizations:
            self.save_lighting_timeline(lighting_data, output_dir)

        return results

    def compute_lighting_statistics(self, lighting_data: List[Dict]) -> Dict:
        """
        Compute aggregate lighting statistics

        Args:
            lighting_data: List of per-frame lighting data

        Returns:
            Aggregate statistics
        """
        stats = {
            "intensity_levels": defaultdict(int),
            "temperatures": defaultdict(int),
            "setups": defaultdict(int),
            "shadow_types": defaultdict(int),
            "light_directions": defaultdict(int)
        }

        intensities = []
        warm_cool_values = []

        for data in lighting_data:
            # Intensity
            if "intensity" in data:
                level = data["intensity"].get("level")
                if level:
                    stats["intensity_levels"][level] += 1
                intensities.append(data["intensity"].get("mean_intensity", 0))

            # Temperature
            if "temperature" in data:
                temp = data["temperature"].get("temperature")
                if temp:
                    stats["temperatures"][temp] += 1
                warm_cool_values.append(data["temperature"].get("warm_cool_value", 0))

            # Setup
            if "setup" in data:
                setup = data["setup"].get("setup")
                if setup:
                    stats["setups"][setup] += 1

            # Shadows
            if "shadows" in data:
                shadow_type = data["shadows"].get("shadow_type")
                if shadow_type:
                    stats["shadow_types"][shadow_type] += 1

            # Direction
            if "direction" in data and data["direction"].get("estimated"):
                direction = data["direction"].get("direction")
                if direction:
                    stats["light_directions"][direction] += 1

        # Convert to regular dicts
        stats = {
            "avg_intensity": float(np.mean(intensities)) if intensities else 0,
            "intensity_std": float(np.std(intensities)) if intensities else 0,
            "intensity_levels": dict(stats["intensity_levels"]),
            "avg_warm_cool": float(np.mean(warm_cool_values)) if warm_cool_values else 0,
            "temperatures": dict(stats["temperatures"]),
            "setups": dict(stats["setups"]),
            "shadow_types": dict(stats["shadow_types"]),
            "light_directions": dict(stats["light_directions"])
        }

        return stats

    def save_lighting_timeline(
        self,
        lighting_data: List[Dict],
        output_dir: Path
    ):
        """
        Save lighting characteristics timeline

        Args:
            lighting_data: Lighting analysis data
            output_dir: Output directory
        """
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

            frames = list(range(len(lighting_data)))
            intensities = [d.get("intensity", {}).get("mean_intensity", 0) for d in lighting_data]
            warm_cool = [d.get("temperature", {}).get("warm_cool_value", 0) for d in lighting_data]

            # Intensity timeline
            ax1.plot(frames, intensities, linewidth=2, color='gold')
            ax1.fill_between(frames, intensities, alpha=0.3, color='gold')
            ax1.set_xlabel("Frame Index", fontsize=12)
            ax1.set_ylabel("Light Intensity", fontsize=12)
            ax1.set_title("Lighting Intensity Timeline", fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=128, color='gray', linestyle='--', alpha=0.5, label='Middle gray')
            ax1.legend()

            # Color temperature timeline
            ax2.plot(frames, warm_cool, linewidth=2, color='coral')
            ax2.fill_between(frames, warm_cool, alpha=0.3, color='coral')
            ax2.set_xlabel("Frame Index", fontsize=12)
            ax2.set_ylabel("Warm-Cool Value", fontsize=12)
            ax2.set_title("Color Temperature Timeline (Warm=Positive, Cool=Negative)", fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
            ax2.legend()

            plt.tight_layout()

            viz_path = output_dir / "lighting_timeline.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   Lighting timeline saved: {viz_path}")

        except ImportError:
            print("   ⚠️ Matplotlib not available")


def main():
    parser = argparse.ArgumentParser(
        description="Lighting Analysis (Film-Agnostic)"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory with frames"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=5,
        help="Analyze every N frames (default: 5)"
    )
    parser.add_argument(
        "--no-direction",
        action="store_true",
        help="Disable light direction analysis"
    )
    parser.add_argument(
        "--no-temperature",
        action="store_true",
        help="Disable color temperature analysis"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name"
    )

    args = parser.parse_args()

    # Create config
    config = LightingAnalysisConfig(
        analyze_direction=not args.no_direction,
        analyze_intensity=True,
        analyze_temperature=not args.no_temperature,
        detect_lighting_setup=True,
        analyze_shadows=True,
        sample_rate=args.sample_rate,
        save_visualizations=True
    )

    # Run analysis
    analyzer = LightingAnalyzer(config)
    results = analyzer.analyze_lighting(
        frames_dir=Path(args.frames_dir),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
