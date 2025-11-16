"""
Stage 3: Augmentation

Applies 3D-safe augmentations that preserve PBR materials and character features.
NEVER uses horizontal flip or color jitter.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import shutil
from tqdm import tqdm
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.pipelines.stages.base_stage import BaseStage


class AugmentationStage(BaseStage):
    """Stage 3: Apply 3D-safe augmentations"""

    def validate_config(self) -> bool:
        """Validate configuration"""
        required_keys = ['input_dir', 'output_dir']

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        input_dir = Path(self.config['input_dir'])
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")

        return True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute augmentation stage"""

        self.logger.info("="*60)
        self.logger.info("STAGE 3: Augmentation (3D-Safe)")
        self.logger.info("="*60)

        # Setup paths
        input_dir = Path(self.config['input_dir'])
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect input images
        input_images = sorted(
            list(input_dir.glob('*.jpg')) +
            list(input_dir.glob('*.png')) +
            list(input_dir.glob('*.jpeg'))
        )

        self.logger.info(f"Processing {len(input_images)} images...")

        # Get augmentation parameters
        augmentations_per_image = self.config.get('augmentations_per_image', 4)

        # Copy originals first
        self.logger.info("Copying original images...")
        copied_paths = []

        for img_path in tqdm(input_images, desc="Copying originals"):
            output_path = output_dir / f"orig_{img_path.name}"
            shutil.copy2(img_path, output_path)
            copied_paths.append(output_path)

        # Generate augmentations
        self.logger.info(f"Generating {augmentations_per_image} augmentations per image...")

        augmentation_stats = {
            'crop': 0,
            'rotation': 0,
            'brightness': 0,
            'contrast': 0,
            'noise': 0,
            'blur': 0
        }

        for img_path in tqdm(input_images, desc="Augmenting"):
            try:
                # Load image
                img = Image.open(img_path)

                # Generate multiple augmented versions
                for aug_idx in range(augmentations_per_image):
                    augmented = self._apply_random_augmentation(img, augmentation_stats)

                    # Save augmented image
                    stem = img_path.stem
                    suffix = img_path.suffix
                    output_path = output_dir / f"{stem}_aug{aug_idx:02d}{suffix}"
                    augmented.save(output_path, quality=95)
                    copied_paths.append(output_path)

            except Exception as e:
                self.logger.warning(f"Failed to augment {img_path}: {e}")
                continue

        # Calculate statistics
        total_original = len(input_images)
        total_augmented = len(copied_paths) - total_original
        augmentation_factor = total_augmented / total_original if total_original > 0 else 0

        metadata = {
            'input_count': total_original,
            'augmented_count': total_augmented,
            'total_count': len(copied_paths),
            'augmentation_factor': augmentation_factor,
            'augmentations_per_image': augmentations_per_image,
            'augmentation_stats': augmentation_stats,
            'augmentation_config': {
                'crop_range': self.config.get('crop_range', [0.8, 1.0]),
                'rotation_range': self.config.get('rotation_range', [-5, 5]),
                'brightness_range': self.config.get('brightness_range', [0.9, 1.1]),
                'contrast_range': self.config.get('contrast_range', [0.95, 1.05]),
                'noise_sigma': self.config.get('noise_sigma', 0.01),
                'blur_probability': self.config.get('blur_probability', 0.1)
            },
            'safety_notes': [
                'NO horizontal flip (preserves asymmetric features)',
                'NO color jitter (preserves PBR materials)',
                'Minimal contrast adjustment (preserves smooth shading)',
                'All augmentations are 3D-animation-safe'
            ],
            'output_dir': str(output_dir)
        }

        # Save results
        results_file = output_dir / 'augmentation_results.json'
        with open(results_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Results saved to {results_file}")

        # Print summary
        self.logger.info("\nAugmentation Summary:")
        self.logger.info(f"  Original images: {total_original}")
        self.logger.info(f"  Augmented images: {total_augmented}")
        self.logger.info(f"  Total images: {len(copied_paths)}")
        self.logger.info(f"  Augmentation factor: {augmentation_factor:.1f}x")
        self.logger.info(f"  Output directory: {output_dir}")
        self.logger.info("\n  Augmentation breakdown:")
        for aug_type, count in augmentation_stats.items():
            self.logger.info(f"    {aug_type}: {count}")

        return {
            'output_dir': output_dir,
            'augmented_images': copied_paths,
            'metadata': metadata,
            'num_augmented': len(copied_paths)
        }

    def _apply_random_augmentation(
        self,
        img: Image.Image,
        stats: Dict[str, int]
    ) -> Image.Image:
        """
        Apply random 3D-safe augmentation

        Args:
            img: Input PIL Image
            stats: Statistics dictionary to update

        Returns:
            Augmented PIL Image
        """
        augmented = img.copy()

        # Random crop (80-100%)
        if random.random() < self.config.get('crop_probability', 0.5):
            crop_min, crop_max = self.config.get('crop_range', [0.8, 1.0])
            augmented = self._random_crop(augmented, crop_min, crop_max)
            stats['crop'] += 1

        # Random rotation (-5 to +5 degrees)
        if random.random() < self.config.get('rotation_probability', 0.5):
            rotation_min, rotation_max = self.config.get('rotation_range', [-5, 5])
            angle = random.uniform(rotation_min, rotation_max)
            augmented = augmented.rotate(angle, resample=Image.BICUBIC, expand=False)
            stats['rotation'] += 1

        # Brightness adjustment (90-110%)
        if random.random() < self.config.get('brightness_probability', 0.4):
            brightness_min, brightness_max = self.config.get('brightness_range', [0.9, 1.1])
            factor = random.uniform(brightness_min, brightness_max)
            enhancer = ImageEnhance.Brightness(augmented)
            augmented = enhancer.enhance(factor)
            stats['brightness'] += 1

        # Contrast adjustment (95-105% - MINIMAL to preserve smooth shading)
        if random.random() < self.config.get('contrast_probability', 0.3):
            contrast_min, contrast_max = self.config.get('contrast_range', [0.95, 1.05])
            factor = random.uniform(contrast_min, contrast_max)
            enhancer = ImageEnhance.Contrast(augmented)
            augmented = enhancer.enhance(factor)
            stats['contrast'] += 1

        # Gaussian noise
        if random.random() < self.config.get('noise_probability', 0.2):
            noise_sigma = self.config.get('noise_sigma', 0.01)
            augmented = self._add_gaussian_noise(augmented, noise_sigma)
            stats['noise'] += 1

        # Slight blur (very rare, for depth-of-field simulation)
        if random.random() < self.config.get('blur_probability', 0.1):
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=0.5))
            stats['blur'] += 1

        return augmented

    def _random_crop(
        self,
        img: Image.Image,
        scale_min: float,
        scale_max: float
    ) -> Image.Image:
        """
        Apply random crop with scaling

        Args:
            img: Input image
            scale_min: Minimum scale factor
            scale_max: Maximum scale factor

        Returns:
            Cropped image
        """
        width, height = img.size
        scale = random.uniform(scale_min, scale_max)

        new_width = int(width * scale)
        new_height = int(height * scale)

        # Random crop position
        left = random.randint(0, width - new_width) if new_width < width else 0
        top = random.randint(0, height - new_height) if new_height < height else 0

        # Crop and resize back to original size
        cropped = img.crop((left, top, left + new_width, top + new_height))
        return cropped.resize((width, height), Image.BICUBIC)

    def _add_gaussian_noise(
        self,
        img: Image.Image,
        sigma: float
    ) -> Image.Image:
        """
        Add Gaussian noise to image

        Args:
            img: Input image
            sigma: Noise standard deviation (0-1 scale)

        Returns:
            Noisy image
        """
        img_array = np.array(img).astype(np.float32) / 255.0

        # Generate noise
        noise = np.random.normal(0, sigma, img_array.shape)

        # Add noise and clip
        noisy = np.clip(img_array + noise, 0, 1)

        # Convert back to PIL
        noisy = (noisy * 255).astype(np.uint8)
        return Image.fromarray(noisy)
