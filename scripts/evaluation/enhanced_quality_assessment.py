#!/usr/bin/env python3
"""
Enhanced Quality Assessment for Anime Frames
Improved scoring methods specifically designed for anime character images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnimeQualityAssessor(nn.Module):
    """
    Enhanced quality assessment for anime frames
    Combines multiple metrics for better evaluation
    """

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # Use EfficientNet for better feature extraction
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        # Remove classifier
        self.backbone.classifier = nn.Identity()

        # Custom quality head
        self.quality_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.backbone.to(device)
        self.quality_head.to(device)
        self.eval()

        logger.info("Enhanced AnimeQualityAssessor initialized")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Assess quality of anime images

        Args:
            images: Tensor of shape (B, 3, H, W)

        Returns:
            Quality scores (B,) in range [0, 1]
        """
        features = self.backbone(images)  # (B, 1280)
        quality = self.quality_head(features).squeeze(-1)  # (B,)
        return quality


class ImageMetricsCalculator:
    """Calculate various image quality metrics"""

    @staticmethod
    def calculate_sharpness(img_array: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance
        Higher values indicate sharper images
        """
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray = img_array

        # Calculate Laplacian
        laplacian = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])

        # Convolve
        from scipy.ndimage import convolve
        lap = convolve(gray, laplacian)

        # Variance of Laplacian
        variance = np.var(lap)

        # Normalize to 0-1 range (empirically, good anime images have variance 100-1000)
        normalized = np.clip(variance / 1000.0, 0, 1)

        return float(normalized)

    @staticmethod
    def calculate_contrast(img_array: np.ndarray) -> float:
        """
        Calculate image contrast using standard deviation
        """
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Standard deviation as measure of contrast
        std = np.std(gray)

        # Normalize (good anime images have std 30-80)
        normalized = np.clip(std / 80.0, 0, 1)

        return float(normalized)

    @staticmethod
    def calculate_saturation(img_array: np.ndarray) -> float:
        """
        Calculate color saturation
        Anime typically has vibrant colors
        """
        if len(img_array.shape) != 3:
            return 0.5  # Grayscale image

        # Convert to float
        img_float = img_array.astype(np.float32) / 255.0

        # Calculate saturation in HSV space
        max_val = np.max(img_float, axis=2)
        min_val = np.min(img_float, axis=2)

        # Saturation = (max - min) / max (avoiding division by zero)
        saturation = np.where(max_val > 0, (max_val - min_val) / max_val, 0)

        # Mean saturation
        mean_sat = np.mean(saturation)

        return float(mean_sat)

    @staticmethod
    def calculate_brightness(img_array: np.ndarray) -> float:
        """
        Calculate average brightness
        """
        mean_brightness = np.mean(img_array) / 255.0
        return float(mean_brightness)

    @staticmethod
    def calculate_non_transparent_ratio(img_array: np.ndarray, alpha_channel: np.ndarray) -> float:
        """
        Calculate ratio of non-transparent pixels
        Higher is better for character images
        """
        if alpha_channel is None:
            return 1.0

        non_transparent = np.sum(alpha_channel > 10) / alpha_channel.size
        return float(non_transparent)

    @staticmethod
    def calculate_edge_density(img_array: np.ndarray) -> float:
        """
        Calculate edge density using Canny edge detection
        Anime has distinct line art
        """
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray = img_array

        # Simple edge detection using Sobel
        from scipy.ndimage import sobel
        sx = sobel(gray, axis=0, mode='constant')
        sy = sobel(gray, axis=1, mode='constant')
        edges = np.hypot(sx, sy)

        # Threshold
        edge_pixels = np.sum(edges > 30)
        edge_density = edge_pixels / edges.size

        # Normalize (good anime images have 5-15% edge density)
        normalized = np.clip(edge_density / 0.15, 0, 1)

        return float(normalized)


class EnhancedQualityPipeline:
    """
    Complete enhanced quality assessment pipeline
    Combines deep learning and traditional metrics
    """

    def __init__(
        self,
        device='cuda',
        use_efficientnet=True,
        use_traditional_metrics=True
    ):
        self.device = device
        self.use_efficientnet = use_efficientnet
        self.use_traditional_metrics = use_traditional_metrics

        # Initialize models
        if self.use_efficientnet:
            self.quality_assessor = AnimeQualityAssessor(device=device)

        # Metrics calculator
        self.metrics_calc = ImageMetricsCalculator()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.info("EnhancedQualityPipeline initialized")

    def assess_single_image(self, img_path: Path) -> Dict:
        """
        Comprehensive quality assessment for a single image

        Returns:
            Dictionary with multiple quality metrics
        """
        # Load image
        img_pil = Image.open(img_path).convert('RGBA')
        img_rgb = img_pil.convert('RGB')

        # Get alpha channel
        img_array = np.array(img_pil)
        if img_array.shape[2] == 4:
            alpha_channel = img_array[:, :, 3]
            rgb_array = img_array[:, :, :3]
        else:
            alpha_channel = None
            rgb_array = img_array

        results = {
            'image_path': str(img_path),
            'image_name': img_path.name
        }

        # Deep learning quality score
        if self.use_efficientnet:
            img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                dl_quality = self.quality_assessor(img_tensor).item()
            results['dl_quality_score'] = float(dl_quality)

        # Traditional metrics
        if self.use_traditional_metrics:
            results['sharpness'] = self.metrics_calc.calculate_sharpness(rgb_array)
            results['contrast'] = self.metrics_calc.calculate_contrast(rgb_array)
            results['saturation'] = self.metrics_calc.calculate_saturation(rgb_array)
            results['brightness'] = self.metrics_calc.calculate_brightness(rgb_array)
            results['edge_density'] = self.metrics_calc.calculate_edge_density(rgb_array)

            if alpha_channel is not None:
                results['coverage'] = self.metrics_calc.calculate_non_transparent_ratio(
                    rgb_array, alpha_channel
                )
            else:
                results['coverage'] = 1.0

            # Composite traditional quality score
            # Weighted combination of metrics
            traditional_quality = (
                results['sharpness'] * 0.25 +
                results['contrast'] * 0.20 +
                results['saturation'] * 0.15 +
                results['edge_density'] * 0.20 +
                results['coverage'] * 0.20
            )
            results['traditional_quality_score'] = float(traditional_quality)

            # Combined quality score
            if self.use_efficientnet:
                # Blend deep learning and traditional
                combined = (
                    results['dl_quality_score'] * 0.6 +
                    results['traditional_quality_score'] * 0.4
                )
            else:
                combined = results['traditional_quality_score']

            results['combined_quality_score'] = float(combined)

            # Enhanced aesthetic score (based on saturation, brightness, coverage)
            aesthetic = (
                results['saturation'] * 0.35 +
                results['brightness'] * 0.25 +
                results['coverage'] * 0.25 +
                results['contrast'] * 0.15
            )
            results['enhanced_aesthetic_score'] = float(aesthetic)

        return results

    def assess_batch(
        self,
        image_paths: List[Path],
        output_path: Path,
        batch_size: int = 16
    ):
        """
        Assess quality for a batch of images

        Args:
            image_paths: List of image file paths
            output_path: Path to save results JSON
            batch_size: Processing batch size
        """
        results = []

        logger.info(f"Processing {len(image_paths)} images...")

        for img_path in tqdm(image_paths, desc="Assessing quality"):
            try:
                result = self.assess_single_image(img_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                continue

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Generate statistics
        self._generate_statistics(results, output_path.parent)

        logger.info(f"Assessment complete. Results saved to {output_path}")

        return results

    def _generate_statistics(self, results: List[Dict], output_dir: Path):
        """Generate statistics from assessment results"""
        if not results:
            return

        stats = {
            'total_images': len(results),
        }

        # Calculate statistics for each metric
        metrics = [
            'dl_quality_score',
            'traditional_quality_score',
            'combined_quality_score',
            'enhanced_aesthetic_score',
            'sharpness',
            'contrast',
            'saturation',
            'brightness',
            'edge_density',
            'coverage'
        ]

        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                stats[f'{metric}_mean'] = float(np.mean(values))
                stats[f'{metric}_std'] = float(np.std(values))
                stats[f'{metric}_min'] = float(np.min(values))
                stats[f'{metric}_max'] = float(np.max(values))
                stats[f'{metric}_median'] = float(np.median(values))

                # Count high-quality images
                if 'quality' in metric or 'aesthetic' in metric:
                    high_threshold = 0.6
                    stats[f'{metric}_high_count'] = len([v for v in values if v >= high_threshold])

        # Save statistics
        stats_path = output_dir / 'enhanced_quality_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics saved to {stats_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Quality Assessment for Anime Frames")
    parser.add_argument('input_dir', type=Path, help='Directory containing images')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--disable-dl', action='store_true', help='Disable deep learning models')
    parser.add_argument('--pattern', type=str, default='*.png', help='File pattern')

    args = parser.parse_args()

    # Find images
    input_dir = Path(args.input_dir)
    image_paths = list(input_dir.rglob(args.pattern))

    logger.info(f"Found {len(image_paths)} images")

    # Initialize pipeline
    pipeline = EnhancedQualityPipeline(
        device=args.device,
        use_efficientnet=not args.disable_dl,
        use_traditional_metrics=True
    )

    # Run assessment
    output_path = args.output_dir / 'enhanced_quality_assessment.json'
    pipeline.assess_batch(
        image_paths=image_paths,
        output_path=output_path,
        batch_size=args.batch_size
    )

    logger.info("✅ Enhanced quality assessment complete!")


if __name__ == '__main__':
    main()
