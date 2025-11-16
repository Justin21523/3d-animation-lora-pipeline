#!/usr/bin/env python3
"""
Frame Decision Engine - AI-driven strategy selection for frame processing

Analyzes frames and determines optimal processing strategy:
- keep_full: Keep complete frame with background
- segment: Segment character and inpaint background
- create_occlusion: Generate synthetic occlusions
- enhance_segment: Enhance quality then segment
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import yaml


@dataclass
class FrameAnalysis:
    """Results from frame analysis"""
    complexity: float  # Background complexity (0-1)
    lighting_quality: float  # Lighting quality (0-1)
    occlusion_level: float  # Estimated occlusion (0-1)
    instance_count: int  # Number of detected instances
    quality_score: float  # Overall frame quality (0-1)
    sharpness: float  # Image sharpness (0-1)
    contrast: float  # Contrast level (0-1)
    brightness: float  # Brightness level (0-1)


@dataclass
class DecisionThresholds:
    """Thresholds for decision making"""
    simple_background: float = 0.3
    good_lighting: float = 0.7
    low_occlusion: float = 0.2
    multi_character: int = 2
    poor_quality: float = 0.5
    low_sharpness: float = 0.4


class FrameDecisionEngine:
    """
    Analyzes frames and recommends processing strategies
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize decision engine

        Args:
            config_path: Path to YAML config with thresholds
        """
        self.thresholds = DecisionThresholds()

        if config_path and config_path.exists():
            self._load_config(config_path)

    def _load_config(self, config_path: Path):
        """Load thresholds from YAML config"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'thresholds' in config:
            for key, value in config['thresholds'].items():
                if hasattr(self.thresholds, key):
                    setattr(self.thresholds, key, value)

    def analyze_frame(self, image_path: Path) -> FrameAnalysis:
        """
        Perform comprehensive frame analysis

        Args:
            image_path: Path to image file

        Returns:
            FrameAnalysis with all metrics
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Calculate metrics
        complexity = self._calculate_complexity(img, gray)
        lighting_quality = self._assess_lighting(img, hsv)
        occlusion_level = self._estimate_occlusion(img, gray)
        quality_score = self._assess_quality(gray)
        sharpness = self._calculate_sharpness(gray)
        contrast = self._calculate_contrast(gray)
        brightness = self._calculate_brightness(gray)

        # Instance count would come from SAM2 in real implementation
        # For now, use a placeholder (will be passed from caller)
        instance_count = 1

        return FrameAnalysis(
            complexity=complexity,
            lighting_quality=lighting_quality,
            occlusion_level=occlusion_level,
            instance_count=instance_count,
            quality_score=quality_score,
            sharpness=sharpness,
            contrast=contrast,
            brightness=brightness
        )

    def _calculate_complexity(self, img: np.ndarray, gray: np.ndarray) -> float:
        """
        Calculate background complexity (0-1)

        Uses:
        - Edge density
        - Color diversity
        - Texture complexity
        """
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Color diversity (number of unique colors normalized)
        unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
        max_possible = img.shape[0] * img.shape[1]
        color_diversity = min(unique_colors / (max_possible * 0.1), 1.0)

        # Texture complexity using standard deviation
        texture = np.std(gray) / 128.0  # Normalize to 0-1

        # Combine metrics
        complexity = (edge_density * 0.4 + color_diversity * 0.3 + texture * 0.3)

        return float(np.clip(complexity, 0, 1))

    def _assess_lighting(self, img: np.ndarray, hsv: np.ndarray) -> float:
        """
        Assess lighting quality (0-1)

        Good lighting:
        - Balanced dynamic range
        - Not too dark or bright
        - Consistent illumination
        """
        # Value channel (brightness)
        v_channel = hsv[:, :, 2]

        # Check dynamic range (should be well distributed)
        hist, _ = np.histogram(v_channel, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize

        # Penalize if too concentrated in dark or bright regions
        dark_ratio = hist[:64].sum()
        bright_ratio = hist[192:].sum()

        if dark_ratio > 0.5 or bright_ratio > 0.5:
            dynamic_range_score = 0.3
        else:
            dynamic_range_score = 1.0 - (dark_ratio + bright_ratio)

        # Check for consistent illumination (low std in local regions)
        local_stds = []
        h, w = v_channel.shape
        step = 32
        for i in range(0, h - step, step):
            for j in range(0, w - step, step):
                patch = v_channel[i:i+step, j:j+step]
                local_stds.append(np.std(patch))

        consistency_score = 1.0 - min(np.mean(local_stds) / 128.0, 1.0)

        # Mean brightness should be reasonable (not too dark/bright)
        mean_brightness = np.mean(v_channel) / 255.0
        if 0.3 < mean_brightness < 0.7:
            brightness_score = 1.0
        else:
            brightness_score = 0.5

        # Combine scores
        quality = (dynamic_range_score * 0.4 +
                   consistency_score * 0.3 +
                   brightness_score * 0.3)

        return float(np.clip(quality, 0, 1))

    def _estimate_occlusion(self, img: np.ndarray, gray: np.ndarray) -> float:
        """
        Estimate occlusion level (0-1)

        High occlusion indicators:
        - Objects at image edges
        - Depth discontinuities
        - Irregular boundaries
        """
        h, w = gray.shape

        # Check edges for potential occlusions
        edge_thickness = 20
        edges_combined = np.concatenate([
            gray[:edge_thickness, :].flatten(),  # Top
            gray[-edge_thickness:, :].flatten(),  # Bottom
            gray[:, :edge_thickness].flatten(),  # Left
            gray[:, -edge_thickness:].flatten()  # Right
        ])

        # High variance at edges suggests objects cutting through
        edge_variance = np.std(edges_combined) / 128.0

        # Detect edges in image
        edges = cv2.Canny(gray, 50, 150)

        # Count edge pixels near image borders
        border_edges = (
            np.sum(edges[:edge_thickness, :]) +
            np.sum(edges[-edge_thickness:, :]) +
            np.sum(edges[:, :edge_thickness]) +
            np.sum(edges[:, -edge_thickness:])
        )
        total_border_pixels = 2 * (h + w) * edge_thickness
        border_edge_ratio = border_edges / total_border_pixels

        # Combine indicators
        occlusion = (edge_variance * 0.5 + border_edge_ratio * 0.5)

        return float(np.clip(occlusion, 0, 1))

    def _assess_quality(self, gray: np.ndarray) -> float:
        """
        Overall quality assessment (0-1)

        Factors:
        - Sharpness
        - Noise level
        - Resolution
        """
        sharpness = self._calculate_sharpness(gray)

        # Estimate noise (high frequency content in smooth regions)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.mean(np.abs(gray.astype(float) - blurred.astype(float))) / 255.0
        noise_score = 1.0 - min(noise * 10, 1.0)  # Lower noise is better

        # Resolution check (penalize very small images)
        h, w = gray.shape
        res_score = min((h * w) / (512 * 512), 1.0)

        quality = (sharpness * 0.5 + noise_score * 0.3 + res_score * 0.2)

        return float(np.clip(quality, 0, 1))

    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize (typical sharp images have variance > 500)
        sharpness = min(variance / 1000.0, 1.0)

        return float(sharpness)

    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """Calculate image contrast"""
        contrast = gray.std() / 128.0  # Normalize to 0-1
        return float(np.clip(contrast, 0, 1))

    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """Calculate average brightness"""
        brightness = gray.mean() / 255.0
        return float(brightness)

    def decide_strategy(
        self,
        analysis: FrameAnalysis,
        dataset_needs: Optional[Dict[str, float]] = None
    ) -> Tuple[str, float, str]:
        """
        Decide processing strategy based on analysis

        Args:
            analysis: FrameAnalysis object
            dataset_needs: Optional dict with augmentation needs
                          e.g., {'occlusion': 0.8} means need more occlusions

        Returns:
            Tuple of (strategy_name, confidence, reasoning)
        """
        reasons = []

        # Strategy A: Keep Full Frame
        if (analysis.complexity < self.thresholds.simple_background and
            analysis.lighting_quality > self.thresholds.good_lighting and
            analysis.occlusion_level < self.thresholds.low_occlusion):

            confidence = (
                (1.0 - analysis.complexity) * 0.4 +
                analysis.lighting_quality * 0.4 +
                (1.0 - analysis.occlusion_level) * 0.2
            )

            reasons.append(f"Simple background (complexity={analysis.complexity:.2f})")
            reasons.append(f"Good lighting (quality={analysis.lighting_quality:.2f})")
            reasons.append(f"Low occlusion (level={analysis.occlusion_level:.2f})")

            return ("keep_full", confidence, " | ".join(reasons))

        # Strategy B: Segment (Most common)
        if (analysis.instance_count >= self.thresholds.multi_character or
            analysis.complexity > self.thresholds.simple_background):

            confidence = 0.8  # High confidence for this safe default

            if analysis.instance_count >= self.thresholds.multi_character:
                reasons.append(f"Multi-character ({analysis.instance_count} instances)")
            if analysis.complexity > self.thresholds.simple_background:
                reasons.append(f"Complex background (complexity={analysis.complexity:.2f})")

            return ("segment", confidence, " | ".join(reasons))

        # Strategy D: Enhance then Segment
        if (analysis.quality_score < self.thresholds.poor_quality or
            analysis.sharpness < self.thresholds.low_sharpness):

            confidence = 1.0 - analysis.quality_score

            reasons.append(f"Poor quality (score={analysis.quality_score:.2f})")
            if analysis.sharpness < self.thresholds.low_sharpness:
                reasons.append(f"Low sharpness ({analysis.sharpness:.2f})")

            return ("enhance_segment", confidence, " | ".join(reasons))

        # Strategy C: Create Occlusion (if dataset needs it)
        if dataset_needs and dataset_needs.get('occlusion', 0) > 0.5:
            confidence = 0.7

            reasons.append("Dataset needs occlusion examples")
            reasons.append(f"Current need level: {dataset_needs['occlusion']:.2f}")

            return ("create_occlusion", confidence, " | ".join(reasons))

        # Default: Segment (safe choice)
        return ("segment", 0.6, "Default strategy - safe choice")

    def batch_analyze_and_decide(
        self,
        image_paths: list,
        dataset_needs: Optional[Dict[str, float]] = None
    ) -> Dict[str, list]:
        """
        Analyze multiple frames and group by recommended strategy

        Args:
            image_paths: List of image paths
            dataset_needs: Optional dataset augmentation needs

        Returns:
            Dict mapping strategy -> list of (path, confidence, reasoning)
        """
        results = {
            'keep_full': [],
            'segment': [],
            'create_occlusion': [],
            'enhance_segment': []
        }

        for img_path in image_paths:
            try:
                analysis = self.analyze_frame(Path(img_path))
                strategy, confidence, reasoning = self.decide_strategy(
                    analysis, dataset_needs
                )
                results[strategy].append((img_path, confidence, reasoning))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Add to segment as fallback
                results['segment'].append((img_path, 0.5, f"Error: {e}"))

        return results


def main():
    """Test the decision engine"""
    import argparse

    parser = argparse.ArgumentParser(description="Test frame decision engine")
    parser.add_argument("image_path", type=Path, help="Path to test image")
    parser.add_argument("--config", type=Path, help="Config YAML path")

    args = parser.parse_args()

    # Initialize engine
    engine = FrameDecisionEngine(args.config)

    # Analyze frame
    print(f"\nAnalyzing: {args.image_path}")
    analysis = engine.analyze_frame(args.image_path)

    print(f"\n📊 Frame Analysis:")
    print(f"  Complexity:        {analysis.complexity:.2f}")
    print(f"  Lighting Quality:  {analysis.lighting_quality:.2f}")
    print(f"  Occlusion Level:   {analysis.occlusion_level:.2f}")
    print(f"  Quality Score:     {analysis.quality_score:.2f}")
    print(f"  Sharpness:         {analysis.sharpness:.2f}")
    print(f"  Contrast:          {analysis.contrast:.2f}")
    print(f"  Brightness:        {analysis.brightness:.2f}")

    # Decide strategy
    strategy, confidence, reasoning = engine.decide_strategy(analysis)

    print(f"\n🎯 Recommended Strategy: {strategy}")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   Reasoning: {reasoning}")


if __name__ == "__main__":
    main()
