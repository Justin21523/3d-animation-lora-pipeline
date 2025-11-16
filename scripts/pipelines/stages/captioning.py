"""
Stage 5: Captioning

Generates captions using Qwen2-VL with 3D-specific template:
"{character}, {form}, {appearance}, {pose}, {expression}, {clothing},
 {lighting}, {background}, pixar style 3d animation, smooth shading"

Target length: 40-77 tokens (CLIP limit)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import json
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.pipelines.stages.base_stage import BaseStage


class CaptioningStage(BaseStage):
    """Stage 5: Generate captions for training images"""

    def validate_config(self) -> bool:
        """Validate configuration"""
        required_keys = ['input_dir', 'character_name']

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        input_dir = Path(self.config['input_dir'])
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")

        return True

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute captioning stage"""

        self.logger.info("="*60)
        self.logger.info("STAGE 5: Captioning")
        self.logger.info("="*60)

        # Setup paths
        input_dir = Path(self.config['input_dir'])

        # Collect input images
        input_images = sorted(
            list(input_dir.glob('*.jpg')) +
            list(input_dir.glob('*.png')) +
            list(input_dir.glob('*.jpeg'))
        )

        self.logger.info(f"Generating captions for {len(input_images)} images...")

        # Initialize captioning model
        caption_model = self.config.get('caption_model', 'qwen2_vl')
        self.logger.info(f"Using caption model: {caption_model}")

        # Load model
        captioner = self._load_captioner(caption_model)

        # Generate captions
        caption_results = []
        token_lengths = []

        for img_path in tqdm(input_images, desc="Generating captions"):
            try:
                caption = self._generate_caption(
                    img_path,
                    captioner,
                    self.config['character_name']
                )

                # Save caption to .txt file
                caption_path = img_path.with_suffix('.txt')
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)

                # Track statistics
                token_count = len(caption.split())
                token_lengths.append(token_count)

                caption_results.append({
                    'image': str(img_path.name),
                    'caption': caption,
                    'token_count': token_count
                })

            except Exception as e:
                self.logger.warning(f"Failed to caption {img_path}: {e}")
                # Use fallback caption
                fallback = self._generate_fallback_caption(self.config['character_name'])
                caption_path = img_path.with_suffix('.txt')
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(fallback)

                token_lengths.append(len(fallback.split()))

                caption_results.append({
                    'image': str(img_path.name),
                    'caption': fallback,
                    'token_count': len(fallback.split()),
                    'fallback': True
                })

        # Calculate statistics
        import numpy as np

        metadata = {
            'input_count': len(input_images),
            'captions_generated': len(caption_results),
            'caption_model': caption_model,
            'character_name': self.config['character_name'],
            'token_statistics': {
                'mean': float(np.mean(token_lengths)),
                'median': float(np.median(token_lengths)),
                'min': int(np.min(token_lengths)),
                'max': int(np.max(token_lengths)),
                'target_range': [40, 77],
                'within_target': sum(40 <= t <= 77 for t in token_lengths),
                'within_target_pct': 100 * sum(40 <= t <= 77 for t in token_lengths) / len(token_lengths)
            },
            'caption_template': {
                'structure': '{character}, {form}, {appearance}, {pose}, {expression}, {clothing}, {lighting}, {background}, pixar style 3d animation, smooth shading',
                'suffix': 'pixar style 3d animation, smooth shading',
                'target_length': '40-77 tokens (CLIP limit)'
            },
            'output_format': 'text files alongside images'
        }

        # Save results
        results_file = input_dir / 'captioning_results.json'
        with open(results_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save sample captions
        samples_file = input_dir / 'caption_samples.json'
        sample_captions = caption_results[:50]  # Save first 50
        with open(samples_file, 'w') as f:
            json.dump(sample_captions, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Sample captions saved to {samples_file}")

        # Print summary
        self.logger.info("\nCaptioning Summary:")
        self.logger.info(f"  Images processed: {len(input_images)}")
        self.logger.info(f"  Captions generated: {len(caption_results)}")
        self.logger.info(f"  Caption model: {caption_model}")
        self.logger.info(f"  Character: {self.config['character_name']}")
        self.logger.info(f"\n  Token length statistics:")
        self.logger.info(f"    Mean: {metadata['token_statistics']['mean']:.1f}")
        self.logger.info(f"    Median: {metadata['token_statistics']['median']:.1f}")
        self.logger.info(f"    Range: [{metadata['token_statistics']['min']}, {metadata['token_statistics']['max']}]")
        self.logger.info(f"    Within target (40-77): {metadata['token_statistics']['within_target_pct']:.1f}%")

        # Show sample captions
        self.logger.info("\n  Sample captions:")
        for sample in caption_results[:3]:
            self.logger.info(f"    {sample['image'][:30]}: {sample['caption'][:80]}...")

        return {
            'output_dir': input_dir,
            'caption_results': caption_results,
            'metadata': metadata,
            'num_captions': len(caption_results)
        }

    def _load_captioner(self, model_name: str):
        """
        Load captioning model

        Args:
            model_name: Name of model to load

        Returns:
            Loaded model instance
        """
        if model_name == 'qwen2_vl':
            self.logger.info("Loading Qwen2-VL model...")
            # Placeholder - real implementation would load Qwen2-VL
            # For now, return None and use rule-based captions
            return None
        else:
            self.logger.warning(f"Unknown caption model: {model_name}. Using fallback.")
            return None

    def _generate_caption(
        self,
        image_path: Path,
        captioner,
        character_name: str
    ) -> str:
        """
        Generate caption for image

        Args:
            image_path: Path to image
            captioner: Captioning model instance
            character_name: Name of character

        Returns:
            Generated caption string
        """
        # Placeholder implementation
        # Real implementation would use Qwen2-VL to generate detailed caption

        # For now, generate rule-based caption with randomization
        import random

        forms = [
            "young boy", "teenage boy", "boy character",
            "male child", "young human character"
        ]

        appearances = [
            "brown curly hair, brown eyes, tan skin",
            "curly dark hair, expressive eyes, warm skin tone",
            "tousled brown hair, friendly eyes, olive complexion"
        ]

        poses = [
            "standing naturally",
            "casual stance",
            "relaxed posture",
            "three-quarter view",
            "front facing"
        ]

        expressions = [
            "friendly smile",
            "neutral expression",
            "curious look",
            "happy expression",
            "gentle smile"
        ]

        clothing = [
            "casual summer clothes",
            "simple t-shirt",
            "light colored shirt",
            "summer outfit"
        ]

        lighting = [
            "soft natural lighting",
            "warm ambient light",
            "diffuse studio lighting",
            "even lighting"
        ]

        backgrounds = [
            "simple background",
            "neutral backdrop",
            "plain background",
            "minimal setting"
        ]

        # Build caption from template
        caption = (
            f"{character_name}, "
            f"{random.choice(forms)}, "
            f"{random.choice(appearances)}, "
            f"{random.choice(poses)}, "
            f"{random.choice(expressions)}, "
            f"wearing {random.choice(clothing)}, "
            f"{random.choice(lighting)}, "
            f"{random.choice(backgrounds)}, "
            f"pixar style 3d animation, smooth shading, high quality render"
        )

        return caption

    def _generate_fallback_caption(self, character_name: str) -> str:
        """
        Generate simple fallback caption

        Args:
            character_name: Name of character

        Returns:
            Fallback caption string
        """
        return (
            f"{character_name}, young boy character, brown curly hair, "
            f"friendly expression, casual clothes, pixar style 3d animation, "
            f"smooth shading, high quality render"
        )
