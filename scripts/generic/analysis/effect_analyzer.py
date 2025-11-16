#!/usr/bin/env python3
"""
Visual Effects Analyzer

Purpose: Detect and catalog visual effects in animated content
Features: Lighting effects, particles, weather, camera effects, post-processing
Use Cases: Effect cataloging, style analysis, scene classification

Usage:
    python effect_analyzer.py \
        --frames-dir /path/to/frames \
        --output-dir /path/to/analysis \
        --detect-lighting \
        --detect-particles \
        --detect-weather \
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
class EffectAnalysisConfig:
    """Configuration for effect analysis"""
    detect_lighting_effects: bool = True  # Detect lighting effects
    detect_particles: bool = True  # Detect particle effects
    detect_weather: bool = True  # Detect weather effects
    detect_camera_effects: bool = True  # Detect camera/post effects
    detect_color_grading: bool = True  # Analyze color grading
    sample_rate: int = 5  # Analyze every N frames
    save_examples: bool = True  # Save example frames


class EffectAnalyzer:
    """Comprehensive visual effects analysis"""

    def __init__(self, config: EffectAnalysisConfig):
        """
        Initialize effect analyzer

        Args:
            config: Analysis configuration
        """
        self.config = config

    def detect_bloom_glow(self, frame_path: Path) -> Dict:
        """
        Detect bloom/glow lighting effects

        Args:
            frame_path: Path to frame

        Returns:
            Detection result
        """
        img = cv2.imread(str(frame_path))
        if img is None:
            return {"detected": False}

        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Find bright regions
        bright_threshold = np.percentile(l_channel, 95)
        bright_mask = l_channel > bright_threshold

        # Check if bright regions have soft edges (bloom characteristic)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(bright_mask.astype(np.uint8), kernel)

        bloom_regions = cv2.bitwise_and(dilated, dilated, mask=(~bright_mask).astype(np.uint8))
        bloom_ratio = np.sum(bloom_regions > 0) / bright_mask.size

        detected = bloom_ratio > 0.01

        return {
            "detected": detected,
            "bloom_ratio": float(bloom_ratio),
            "bright_area_ratio": float(np.sum(bright_mask) / bright_mask.size),
            "intensity": "high" if bloom_ratio > 0.05 else "medium" if bloom_ratio > 0.02 else "low"
        }

    def detect_rim_lighting(self, frame_path: Path) -> Dict:
        """
        Detect rim/edge lighting effects

        Args:
            frame_path: Path to frame

        Returns:
            Detection result
        """
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"detected": False}

        # Detect edges
        edges = cv2.Canny(img, 50, 150)

        # Find bright edges (rim lighting)
        bright_threshold = np.percentile(img, 90)
        bright_mask = img > bright_threshold

        # Combine edges and bright regions
        rim_edges = cv2.bitwise_and(edges, edges, mask=bright_mask.astype(np.uint8))
        rim_ratio = np.sum(rim_edges > 0) / edges.size

        detected = rim_ratio > 0.005

        return {
            "detected": detected,
            "rim_ratio": float(rim_ratio),
            "intensity": "strong" if rim_ratio > 0.02 else "moderate" if rim_ratio > 0.01 else "subtle"
        }

    def detect_god_rays(self, frame_path: Path) -> Dict:
        """
        Detect god rays/volumetric lighting

        Args:
            frame_path: Path to frame

        Returns:
            Detection result
        """
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"detected": False}

        # Use Hough Line Transform to detect radial patterns
        edges = cv2.Canny(img, 50, 150)

        # Blur to connect rays
        blurred = cv2.GaussianBlur(edges, (5, 5), 0)

        # Detect lines
        lines = cv2.HoughLinesP(
            blurred,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None:
            return {"detected": False}

        # Check if lines converge (god ray characteristic)
        # Simplified: check for multiple similarly oriented lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)

        if len(angles) > 5:
            angle_std = np.std(angles)
            detected = angle_std < 0.5  # Similar angles suggest converging rays
        else:
            detected = False

        return {
            "detected": detected,
            "num_rays": len(lines) if lines is not None else 0,
            "angle_std": float(angle_std) if len(angles) > 5 else 0
        }

    def detect_particles(self, frame_path: Path) -> Dict:
        """
        Detect particle effects (dust, sparkles, etc.)

        Args:
            frame_path: Path to frame

        Returns:
            Detection result
        """
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"detected": False}

        # Find small bright spots
        bright_threshold = np.percentile(img, 98)
        bright_mask = img > bright_threshold

        # Find connected components (particles)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bright_mask.astype(np.uint8),
            connectivity=8
        )

        # Filter small components (particles are typically small)
        particles = []
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if 2 < area < 100:  # Particle size range
                particles.append({
                    "area": int(area),
                    "center": centroids[i].tolist()
                })

        detected = len(particles) > 10

        return {
            "detected": detected,
            "num_particles": len(particles),
            "avg_particle_size": np.mean([p["area"] for p in particles]) if particles else 0,
            "particle_type": "dust" if len(particles) > 50 else "sparkles" if detected else "none"
        }

    def detect_motion_blur(self, frame_path: Path) -> Dict:
        """
        Detect motion blur (camera/post effect)

        Args:
            frame_path: Path to frame

        Returns:
            Detection result
        """
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"detected": False}

        # Compute Laplacian variance (sharpness measure)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = laplacian.var()

        # Low sharpness + directional blur = motion blur
        # Check directional gradients
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        gradient_x_var = np.var(sobel_x)
        gradient_y_var = np.var(sobel_y)

        # If one direction has much higher variance = directional blur
        ratio = max(gradient_x_var, gradient_y_var) / (min(gradient_x_var, gradient_y_var) + 1e-8)

        detected = sharpness < 100 and ratio > 2.0

        direction = "horizontal" if gradient_x_var > gradient_y_var else "vertical"

        return {
            "detected": detected,
            "sharpness": float(sharpness),
            "blur_direction": direction if detected else "none",
            "directional_ratio": float(ratio)
        }

    def detect_depth_of_field(self, frame_path: Path) -> Dict:
        """
        Detect depth of field/bokeh effects

        Args:
            frame_path: Path to frame

        Returns:
            Detection result
        """
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"detected": False}

        # Split into regions and measure sharpness variance
        h, w = img.shape
        grid_h, grid_w = 4, 4
        cell_h, cell_w = h // grid_h, w // grid_w

        sharpness_map = np.zeros((grid_h, grid_w))

        for i in range(grid_h):
            for j in range(grid_w):
                cell = img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                laplacian = cv2.Laplacian(cell, cv2.CV_64F)
                sharpness_map[i, j] = laplacian.var()

        # High variance in sharpness across regions = DoF
        sharpness_variance = np.var(sharpness_map)
        sharpness_range = np.max(sharpness_map) - np.min(sharpness_map)

        detected = sharpness_variance > 5000 and sharpness_range > 100

        # Find focus region (sharpest area)
        focus_i, focus_j = np.unravel_index(sharpness_map.argmax(), sharpness_map.shape)
        focus_region = f"grid_{focus_i}_{focus_j}"

        return {
            "detected": detected,
            "sharpness_variance": float(sharpness_variance),
            "sharpness_range": float(sharpness_range),
            "focus_region": focus_region if detected else "none",
            "bokeh_strength": "strong" if sharpness_variance > 10000 else "moderate" if detected else "none"
        }

    def detect_weather_effects(self, frame_path: Path) -> Dict:
        """
        Detect weather effects (rain, snow, fog)

        Args:
            frame_path: Path to frame

        Returns:
            Detection result
        """
        img = cv2.imread(str(frame_path))
        if img is None:
            return {"detected": False, "type": "none"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Fog detection: low contrast + high brightness
        contrast = gray.std()
        brightness = gray.mean()

        fog_detected = contrast < 40 and brightness > 150

        # Rain detection: vertical streaks
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        vertical_features = np.abs(sobel_y).mean()

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        horizontal_features = np.abs(sobel_x).mean()

        vertical_ratio = vertical_features / (horizontal_features + 1e-8)
        rain_detected = vertical_ratio > 1.5 and vertical_features > 20

        # Snow detection: bright particles + low saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        low_sat_bright = np.sum((saturation < 50) & (value > 200)) / gray.size
        snow_detected = low_sat_bright > 0.1

        # Determine primary weather type
        if fog_detected:
            weather_type = "fog"
        elif rain_detected:
            weather_type = "rain"
        elif snow_detected:
            weather_type = "snow"
        else:
            weather_type = "none"

        return {
            "detected": weather_type != "none",
            "type": weather_type,
            "fog_score": float(1.0 - contrast/100) if fog_detected else 0,
            "rain_score": float(vertical_ratio / 2.0) if rain_detected else 0,
            "snow_score": float(low_sat_bright * 10) if snow_detected else 0,
            "intensity": "heavy" if (fog_detected and contrast < 30) or
                                   (rain_detected and vertical_ratio > 2.0) or
                                   (snow_detected and low_sat_bright > 0.2)
                        else "moderate" if weather_type != "none"
                        else "none"
        }

    def analyze_color_grading(self, frame_path: Path) -> Dict:
        """
        Analyze color grading and tone mapping

        Args:
            frame_path: Path to frame

        Returns:
            Color grading analysis
        """
        img = cv2.imread(str(frame_path))
        if img is None:
            return {}

        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Analyze hue distribution
        hue = hsv[:, :, 0]
        hue_hist, _ = np.histogram(hue, bins=12, range=(0, 180))
        dominant_hue_idx = int(np.argmax(hue_hist))
        hue_names = ['red', 'orange', 'yellow', 'yellow-green', 'green', 'cyan',
                     'blue', 'blue-violet', 'violet', 'magenta', 'pink', 'red-orange']
        dominant_hue = hue_names[dominant_hue_idx]

        # Saturation analysis
        saturation = hsv[:, :, 1]
        avg_saturation = float(saturation.mean())
        saturation_category = "desaturated" if avg_saturation < 50 else \
                             "normal" if avg_saturation < 120 else "vibrant"

        # Brightness/contrast analysis
        value = hsv[:, :, 2]
        avg_brightness = float(value.mean())
        contrast_std = float(value.std())

        tone_mapping = "low-key" if avg_brightness < 100 else \
                      "high-key" if avg_brightness > 180 else "normal"

        # Color temperature (warm vs cool)
        b, g, r = cv2.split(img)
        warm_cool = float((r.mean() - b.mean()) / 255.0)
        temperature = "warm" if warm_cool > 0.1 else "cool" if warm_cool < -0.1 else "neutral"

        return {
            "dominant_hue": dominant_hue,
            "saturation": saturation_category,
            "saturation_value": avg_saturation,
            "tone_mapping": tone_mapping,
            "brightness": avg_brightness,
            "contrast": contrast_std,
            "temperature": temperature,
            "temperature_value": warm_cool
        }

    def analyze_effects(
        self,
        frames_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main effects analysis pipeline

        Args:
            frames_dir: Directory with frames
            output_dir: Output directory

        Returns:
            Analysis results
        """
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Visual Effects Analysis")
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

        # Analyze effects
        effect_data = []

        for frame_path in tqdm(sampled_frames, desc="Analyzing effects"):
            frame_effects = {
                "frame": frame_path.name,
            }

            # Lighting effects
            if self.config.detect_lighting_effects:
                frame_effects["bloom_glow"] = self.detect_bloom_glow(frame_path)
                frame_effects["rim_lighting"] = self.detect_rim_lighting(frame_path)
                frame_effects["god_rays"] = self.detect_god_rays(frame_path)

            # Particles
            if self.config.detect_particles:
                frame_effects["particles"] = self.detect_particles(frame_path)

            # Weather
            if self.config.detect_weather:
                frame_effects["weather"] = self.detect_weather_effects(frame_path)

            # Camera effects
            if self.config.detect_camera_effects:
                frame_effects["motion_blur"] = self.detect_motion_blur(frame_path)
                frame_effects["depth_of_field"] = self.detect_depth_of_field(frame_path)

            # Color grading
            if self.config.detect_color_grading:
                frame_effects["color_grading"] = self.analyze_color_grading(frame_path)

            effect_data.append(frame_effects)

        # Aggregate statistics
        statistics = self.compute_effect_statistics(effect_data)

        print(f"\n📊 Effect Statistics:")
        print(f"   Lighting effects:")
        print(f"     Bloom/Glow: {statistics['lighting']['bloom_glow']}")
        print(f"     Rim lighting: {statistics['lighting']['rim_lighting']}")
        print(f"     God rays: {statistics['lighting']['god_rays']}")

        if self.config.detect_particles:
            print(f"   Particle effects: {statistics['particles']['detected']}")

        if self.config.detect_weather:
            print(f"   Weather effects: {statistics['weather']}")

        if self.config.detect_camera_effects:
            print(f"   Camera effects:")
            print(f"     Motion blur: {statistics['camera']['motion_blur']}")
            print(f"     Depth of field: {statistics['camera']['depth_of_field']}")

        # Save results
        results = {
            "frames_dir": str(frames_dir),
            "total_frames": len(frames),
            "analyzed_frames": len(sampled_frames),
            "config": {
                "sample_rate": self.config.sample_rate,
                "detect_lighting_effects": self.config.detect_lighting_effects,
                "detect_particles": self.config.detect_particles,
                "detect_weather": self.config.detect_weather,
            },
            "effect_data": effect_data,
            "statistics": statistics,
            "timestamp": datetime.now().isoformat()
        }

        results_path = output_dir / "effect_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved: {results_path}")

        return results

    def compute_effect_statistics(self, effect_data: List[Dict]) -> Dict:
        """
        Compute aggregate effect statistics

        Args:
            effect_data: List of per-frame effect data

        Returns:
            Aggregate statistics
        """
        stats = {
            "lighting": {
                "bloom_glow": 0,
                "rim_lighting": 0,
                "god_rays": 0
            },
            "particles": {
                "detected": 0
            },
            "weather": defaultdict(int),
            "camera": {
                "motion_blur": 0,
                "depth_of_field": 0
            },
            "color_grading": defaultdict(int)
        }

        for data in effect_data:
            # Lighting
            if "bloom_glow" in data and data["bloom_glow"].get("detected"):
                stats["lighting"]["bloom_glow"] += 1
            if "rim_lighting" in data and data["rim_lighting"].get("detected"):
                stats["lighting"]["rim_lighting"] += 1
            if "god_rays" in data and data["god_rays"].get("detected"):
                stats["lighting"]["god_rays"] += 1

            # Particles
            if "particles" in data and data["particles"].get("detected"):
                stats["particles"]["detected"] += 1

            # Weather
            if "weather" in data:
                weather_type = data["weather"].get("type", "none")
                if weather_type != "none":
                    stats["weather"][weather_type] += 1

            # Camera
            if "motion_blur" in data and data["motion_blur"].get("detected"):
                stats["camera"]["motion_blur"] += 1
            if "depth_of_field" in data and data["depth_of_field"].get("detected"):
                stats["camera"]["depth_of_field"] += 1

            # Color grading
            if "color_grading" in data:
                grading = data["color_grading"]
                for key in ["saturation", "tone_mapping", "temperature"]:
                    if key in grading:
                        stats["color_grading"][f"{key}_{grading[key]}"] += 1

        # Convert to regular dict
        stats["weather"] = dict(stats["weather"])
        stats["color_grading"] = dict(stats["color_grading"])

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Visual Effects Analysis (Film-Agnostic)"
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
        "--no-lighting",
        action="store_true",
        help="Disable lighting effect detection"
    )
    parser.add_argument(
        "--no-particles",
        action="store_true",
        help="Disable particle detection"
    )
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Disable weather detection"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name"
    )

    args = parser.parse_args()

    # Create config
    config = EffectAnalysisConfig(
        detect_lighting_effects=not args.no_lighting,
        detect_particles=not args.no_particles,
        detect_weather=not args.no_weather,
        detect_camera_effects=True,
        detect_color_grading=True,
        sample_rate=args.sample_rate,
        save_examples=True
    )

    # Run analysis
    analyzer = EffectAnalyzer(config)
    results = analyzer.analyze_effects(
        frames_dir=Path(args.frames_dir),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
