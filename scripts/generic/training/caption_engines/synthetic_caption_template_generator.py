#!/usr/bin/env python3
"""
Synthetic Caption Template Generator

Generates high-quality caption templates using LLMProvider API by analyzing sample images.
This is the first step in the hybrid caption generation strategy.

Process:
1. Sample representative images from each character/LoRA type
2. Send to LLMProvider API with type-specific prompts (pose/action/expression)
3. Generate detailed SDXL-optimized captions (150-180 tokens)
4. Extract vocabulary patterns and template structures
5. Save reusable templates for batch application

Output:
- Exemplar captions per type
- Vocabulary databases (JSON)
- Template patterns with variable slots

Author: LLMProvider Tooling
Date: 2025-12-04
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import llm_vendor
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticCaptionTemplateGenerator:
    """
    Generate caption templates using LLMProvider API analysis of sample images.

    Strategy:
    1. Sample a few images per character/type
    2. Use LLMProvider to generate exemplar captions
    3. Extract patterns and vocabulary
    4. Build reusable templates
    """

    # API prompts for different LoRA types
    POSE_LORA_PROMPT = """Analyze this 3D animated character image and write a detailed SDXL training caption (150-180 tokens).

CRITICAL REQUIREMENTS:
- DO NOT include character name or identity (no specific names, no appearance descriptions like "Italian boy", "wavy hair", etc.)
- DO NOT describe clothing, colors, accessories, hair style, eye color, skin tone, or ANY visual appearance details
- Focus ONLY on: body pose, limb position, stance, camera angle, lighting, technical rendering quality
- Target: 150-180 tokens (SDXL optimized, never exceed 225 tokens)
- Use generic "a 3d animated character" as subject

STRUCTURE:
1. Subject: "a 3d animated character"
2. Pose type: standing/sitting/walking/running/jumping
3. Body position: limb placement, posture, balance
4. Camera angle: front view/three-quarter/side/back view
5. Lighting: studio/cinematic/natural lighting details
6. Technical quality: PBR materials, rendering quality, resolution

Example format:
"a 3d animated character, full body shot, standing upright pose with arms relaxed at sides and feet shoulder-width apart, straight neutral posture, front-facing view, professional studio lighting with soft three-point setup and subtle ambient occlusion, smooth PBR materials with physically-based rendering, detailed skin shader with subsurface scattering effects, cinematic camera composition with shallow depth of field, 1024px high resolution render, production-quality 3d model with clean topology, award-winning animation standard"

Write the caption NOW (no preamble, just the caption):"""

    ACTION_LORA_PROMPT = """Analyze this 3D animated character image and write a detailed SDXL training caption (150-180 tokens).

CRITICAL REQUIREMENTS:
- DO NOT include character name or identity (no specific names, no appearance descriptions)
- DO NOT describe clothing, colors, accessories, hair style, eye color, skin tone, or ANY visual appearance details
- Focus ONLY on: dynamic motion, action type, movement quality, body mechanics, lighting, technical quality
- Target: 150-180 tokens (SDXL optimized, never exceed 225 tokens)
- Use generic "a 3d animated character" as subject

STRUCTURE:
1. Subject: "a 3d animated character"
2. Action type: running/jumping/throwing/waving/pointing/climbing (describe the ACTION itself, NOT objects being interacted with)
3. Motion dynamics: momentum, weight shift, energy
4. Body mechanics: limb movement, natural motion, athletic form
5. Lighting: appropriate for action (dramatic/dynamic/energetic)
6. Technical quality: motion capture quality, physics, animation standard

ABSOLUTELY FORBIDDEN: basketball, soccer ball, bicycle, specific objects, props, or tools
FOCUS ON: the body motion, arm/leg movement, athletic form, weight distribution

Example format:
"a 3d animated character, dynamic action pose in mid-motion, running forward with athletic stride and arms swinging naturally, forward lean with bent knees and active weight shift, energetic movement with visible motion dynamics, powerful running stance capturing mid-stride momentum, professional motion capture quality with natural body mechanics, cinematic sports lighting with dramatic rim light separating subject from background, 1024px high-resolution action render, smooth motion blur effects on limbs, production-quality animation with realistic physics"

Write the caption NOW (no preamble, just the caption):"""

    EXPRESSION_LORA_PROMPT = """Analyze this 3D animated character image and write a detailed SDXL training caption (150-180 tokens).

CRITICAL REQUIREMENTS:
- DO NOT include character name or identity (no specific names, no appearance descriptions)
- DO NOT describe clothing, colors, accessories, hair style, eye color, skin tone, or ANY visual appearance details
- Focus ONLY on: facial expression, emotion, eye/mouth/eyebrow position and movement, facial muscle activation, lighting, technical quality
- Target: 150-180 tokens (SDXL optimized, never exceed 225 tokens)
- Use generic "a 3d animated character" as subject

STRUCTURE:
1. Subject: "a 3d animated character"
2. Shot type: close-up facial portrait/extreme close-up face
3. Expression type: happy/sad/surprised/angry/neutral/fearful
4. Facial details: eye SHAPE and POSITION (NOT color), mouth shape, eyebrow position, facial muscle activation
5. Lighting: soft/dramatic/butterfly lighting for faces
6. Technical quality: facial render quality, skin shader, detail level

ABSOLUTELY FORBIDDEN TERMS: green eyes, blue eyes, brown eyes, large eyes (color), green irises, brown hair, etc.
ALLOWED: wide eyes, narrowed eyes, eyes crinkled, eyes widened, eyebrow position, etc.

Example format:
"a 3d animated character, extreme close-up facial portrait, beaming with genuine joy showing wide bright smile with visible teeth and eyes crinkled in delight, raised eyebrows and relaxed facial muscles, highly detailed facial anatomy with realistic skin texture and subtle pore detail, expressive eyes widened with natural catch lights, soft butterfly lighting from above creating gentle shadows, 8k ultra-high resolution facial render with photorealistic skin shader, advanced subsurface scattering on cheeks and nose, production-quality Disney-level character rendering"

Write the caption NOW (no preamble, just the caption):"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llm_provider-3-5-sonnet-20241022",
        temperature: float = 0.3,
        max_tokens: int = 300
    ):
        """
        Initialize caption template generator.

        Args:
            api_key: LLMVendor API key (uses LLM_VENDOR_API_KEY env if not provided)
            model: LLMProvider model (Sonnet for quality, Haiku for cost)
            temperature: Lower = more consistent
            max_tokens: Max tokens for caption generation
        """
        self.api_key = api_key or os.getenv('LLM_VENDOR_API_KEY')

        if not self.api_key:
            raise ValueError(
                "LLM_VENDOR_API_KEY not found. Set it as environment variable:\n"
                "export LLM_VENDOR_API_KEY='your-api-key-here'"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = llm_vendor.LLMVendor(api_key=self.api_key)

        # Statistics
        self.stats = {
            'generated': 0,
            'failed': 0,
            'total_tokens': 0,
            'cost_usd': 0.0
        }

        logger.info(f"✓ Caption Template Generator initialized")
        logger.info(f"  Model: {model}")
        logger.info(f"  Temperature: {temperature}")

    def _get_prompt_for_type(self, lora_type: str) -> str:
        """Get appropriate API prompt for LoRA type."""
        prompts = {
            'pose': self.POSE_LORA_PROMPT,
            'action': self.ACTION_LORA_PROMPT,
            'expression': self.EXPRESSION_LORA_PROMPT
        }
        return prompts.get(lora_type.lower(), self.POSE_LORA_PROMPT)

    def generate_caption_from_image(
        self,
        image_path: Path,
        lora_type: str
    ) -> Optional[str]:
        """
        Generate a single caption using LLMProvider API with vision.

        Args:
            image_path: Path to image file
            lora_type: Type of LoRA (pose/action/expression)

        Returns:
            Generated caption string or None if failed
        """
        try:
            # Load and encode image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if too large (save tokens)
                max_size = 1024
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Save to bytes for API
                import io
                import base64
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                image_data = base64.standard_b64encode(buffer.getvalue()).decode('utf-8')

            # Get appropriate prompt
            prompt = self._get_prompt_for_type(lora_type)

            # Call LLMProvider API with vision
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            # Extract caption
            caption = message.content[0].text.strip()

            # Update statistics
            self.stats['generated'] += 1
            self.stats['total_tokens'] += message.usage.input_tokens + message.usage.output_tokens

            # Estimate cost (LLMProvider 3.5 Sonnet: $3/MTok input, $15/MTok output)
            input_cost = (message.usage.input_tokens / 1_000_000) * 3.0
            output_cost = (message.usage.output_tokens / 1_000_000) * 15.0
            self.stats['cost_usd'] += input_cost + output_cost

            logger.info(f"✓ Generated caption ({len(caption.split())} words, {message.usage.output_tokens} tokens)")

            return caption

        except Exception as e:
            logger.error(f"Failed to generate caption for {image_path}: {e}")
            self.stats['failed'] += 1
            return None

    def sample_images_from_dataset(
        self,
        data_root: Path,
        character: str,
        lora_type: str,
        num_samples: int = 5
    ) -> List[Path]:
        """
        Sample representative images from generated dataset.

        Args:
            data_root: Root directory of synthetic data
            character: Character name
            lora_type: LoRA type (pose/action/expression)
            num_samples: Number of images to sample

        Returns:
            List of sampled image paths
        """
        image_dir = data_root / character / lora_type / "generated"

        if not image_dir.exists():
            logger.warning(f"Image directory not found: {image_dir}")
            return []

        # Get all images
        all_images = list(image_dir.glob("*.png"))

        if len(all_images) == 0:
            logger.warning(f"No images found in {image_dir}")
            return []

        # Sample evenly across rounds
        if len(all_images) <= num_samples:
            return all_images

        # Sort by filename to ensure even distribution
        all_images.sort()

        # Sample evenly
        step = len(all_images) // num_samples
        sampled = [all_images[i * step] for i in range(num_samples)]

        logger.info(f"Sampled {len(sampled)} images from {len(all_images)} total for {character} {lora_type}")

        return sampled

    def extract_vocabulary_from_captions(
        self,
        captions: List[str],
        lora_type: str
    ) -> Dict[str, List[str]]:
        """
        Extract vocabulary patterns from generated captions.

        Args:
            captions: List of generated captions
            lora_type: LoRA type for categorization

        Returns:
            Dictionary of vocabulary categories and terms
        """
        vocabulary = {
            'pose_terms': [],
            'action_terms': [],
            'expression_terms': [],
            'lighting_terms': [],
            'camera_terms': [],
            'quality_terms': [],
            'material_terms': []
        }

        # Define extraction patterns
        patterns = {
            'pose_terms': [
                r'(standing|sitting|walking|running|jumping|crouching|kneeling)\s+(?:upright\s+)?pose',
                r'(arms?\s+(?:at\s+sides|raised|crossed|extended))',
                r'(legs?\s+(?:apart|together|bent|straight|crossed))',
                r'(feet\s+(?:shoulder-width|together|apart))'
            ],
            'action_terms': [
                r'(running|jumping|throwing|waving|pointing|climbing|reaching)\s+(?:action|motion|gesture)',
                r'(dynamic|energetic|athletic|powerful)\s+(?:motion|movement|action)',
                r'(forward|backward|upward|downward)\s+(?:motion|movement|lean)'
            ],
            'expression_terms': [
                r'(happy|sad|surprised|angry|neutral|fearful|joyful)\s+(?:expression|emotion)',
                r'(smile|frown|grin|smirk|grimace)',
                r'(wide\s+eyes|narrowed\s+eyes|downcast\s+eyes|bright\s+eyes)',
                r'(raised\s+eyebrows|furrowed\s+brow|relaxed\s+eyebrows)'
            ],
            'lighting_terms': [
                r'(studio|cinematic|natural|dramatic|soft)\s+lighting',
                r'(three-point|rim|butterfly|ambient)\s+(?:lighting|light)',
                r'(global\s+illumination|ambient\s+occlusion|volumetric\s+lighting)'
            ],
            'camera_terms': [
                r'(front|side|back|three-quarter|profile)\s+view',
                r'(close-up|extreme\s+close-up|full\s+body|medium)\s+(?:shot|portrait)',
                r'(shallow\s+depth\s+of\s+field|bokeh|sharp\s+focus)'
            ],
            'quality_terms': [
                r'(\d+px|8k|4k|1024px)\s+(?:high\s+)?(?:resolution|render)',
                r'(production-quality|award-winning|feature\s+film|theatrical\s+release)',
                r'(photorealistic|hyperrealistic|realistic)\s+(?:rendering|shading)'
            ],
            'material_terms': [
                r'PBR\s+materials',
                r'subsurface\s+scattering',
                r'(?:detailed|realistic)\s+(?:skin\s+)?shader',
                r'(?:smooth|clean)\s+(?:shading|topology)'
            ]
        }

        # Extract terms from all captions
        for caption in captions:
            for category, category_patterns in patterns.items():
                for pattern in category_patterns:
                    matches = re.findall(pattern, caption, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            term = ' '.join(match).strip()
                        else:
                            term = match.strip()

                        if term and term not in vocabulary[category]:
                            vocabulary[category].append(term)

        # Sort and deduplicate
        for category in vocabulary:
            vocabulary[category] = sorted(set(vocabulary[category]))

        return vocabulary

    def generate_templates_from_captions(
        self,
        captions: List[str],
        lora_type: str
    ) -> List[str]:
        """
        Extract template patterns from captions.

        Args:
            captions: Generated captions
            lora_type: LoRA type

        Returns:
            List of template patterns with {variable} slots
        """
        templates = []

        # Identify common structures
        for caption in captions:
            # Replace specific terms with placeholders
            template = caption

            # Replace pose/action/expression specifics
            if lora_type == 'pose':
                template = re.sub(
                    r'(standing|sitting|walking|running|jumping)\s+(?:upright\s+)?pose',
                    '{pose_type}',
                    template,
                    flags=re.IGNORECASE
                )
                template = re.sub(
                    r'(front|side|back|three-quarter)\s+view',
                    '{camera_angle}',
                    template,
                    flags=re.IGNORECASE
                )

            elif lora_type == 'action':
                template = re.sub(
                    r'(running|jumping|throwing|waving|pointing)\s+(?:action|motion)',
                    '{action_type}',
                    template,
                    flags=re.IGNORECASE
                )

            elif lora_type == 'expression':
                template = re.sub(
                    r'(happy|sad|surprised|angry|neutral)\s+expression',
                    '{expression_type}',
                    template,
                    flags=re.IGNORECASE
                )

            # Replace lighting
            template = re.sub(
                r'(studio|cinematic|natural|dramatic|soft)\s+lighting',
                '{lighting_type}',
                template,
                flags=re.IGNORECASE
            )

            templates.append(template)

        return templates

    def generate_for_all_characters(
        self,
        data_root: Path,
        characters: List[str],
        lora_types: List[str],
        output_dir: Path,
        samples_per_character: int = 5
    ):
        """
        Generate caption templates for all characters and LoRA types.

        Args:
            data_root: Root of synthetic data
            characters: List of character names
            lora_types: List of LoRA types (pose/action/expression)
            output_dir: Output directory for templates
            samples_per_character: Images to sample per character/type
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for all generated data
        all_exemplars = defaultdict(list)
        all_vocabularies = defaultdict(lambda: defaultdict(list))
        all_templates = defaultdict(list)

        # Process each combination
        for lora_type in lora_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Generating templates for: {lora_type.upper()}")
            logger.info(f"{'='*60}\n")

            for character in characters:
                logger.info(f"\nProcessing {character} ({lora_type})...")

                # Sample images
                sampled_images = self.sample_images_from_dataset(
                    data_root, character, lora_type, samples_per_character
                )

                if not sampled_images:
                    logger.warning(f"Skipping {character} {lora_type} (no images)")
                    continue

                # Generate captions
                captions = []
                for img_path in tqdm(sampled_images, desc=f"{character} {lora_type}"):
                    caption = self.generate_caption_from_image(img_path, lora_type)
                    if caption:
                        captions.append({
                            'caption': caption,
                            'image': str(img_path.name),
                            'character': character,
                            'lora_type': lora_type,
                            'token_count': len(caption.split())
                        })

                # Store exemplars
                all_exemplars[lora_type].extend(captions)

                logger.info(f"✓ Generated {len(captions)} exemplar captions for {character}")

            # Extract vocabulary from all captions of this type
            caption_texts = [c['caption'] for c in all_exemplars[lora_type]]
            vocabulary = self.extract_vocabulary_from_captions(caption_texts, lora_type)
            all_vocabularies[lora_type] = vocabulary

            # Generate templates
            templates = self.generate_templates_from_captions(caption_texts, lora_type)
            all_templates[lora_type] = list(set(templates))  # Deduplicate

            logger.info(f"\n✓ Extracted vocabulary for {lora_type}:")
            for category, terms in vocabulary.items():
                if terms:
                    logger.info(f"  {category}: {len(terms)} terms")

        # Save outputs
        self._save_outputs(output_dir, all_exemplars, all_vocabularies, all_templates)

        # Print summary
        self._print_summary()

    def _save_outputs(
        self,
        output_dir: Path,
        exemplars: Dict,
        vocabularies: Dict,
        templates: Dict
    ):
        """Save all generated outputs to JSON files."""

        # Save exemplar captions
        for lora_type, captions in exemplars.items():
            exemplar_file = output_dir / f"exemplar_captions_{lora_type}.json"
            with open(exemplar_file, 'w', encoding='utf-8') as f:
                json.dump(captions, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved {len(captions)} exemplars to {exemplar_file}")

        # Save vocabularies
        for lora_type, vocab in vocabularies.items():
            vocab_file = output_dir / f"caption_vocabulary_{lora_type}.json"
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved vocabulary to {vocab_file}")

        # Save templates
        for lora_type, tmpl_list in templates.items():
            template_file = output_dir / f"caption_templates_{lora_type}.json"
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(tmpl_list, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved {len(tmpl_list)} templates to {template_file}")

        # Save statistics
        stats_file = output_dir / "generation_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"✓ Saved statistics to {stats_file}")

    def _print_summary(self):
        """Print generation summary."""
        logger.info(f"\n{'='*60}")
        logger.info("GENERATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total captions generated: {self.stats['generated']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Total tokens used: {self.stats['total_tokens']:,}")
        logger.info(f"Estimated cost: ${self.stats['cost_usd']:.2f}")
        logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate caption templates using LLMProvider API"
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default='/mnt/data/ai_data/synthetic_lora_data/generated_data',
        help='Root directory of synthetic generated data'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training',
        help='Output directory for templates and vocabularies'
    )
    parser.add_argument(
        '--characters',
        nargs='+',
        default=['alberto', 'bryce', 'caleb', 'elio', 'giulia', 'ian_lightfoot',
                 'luca', 'miguel', 'orion', 'russell', 'tyler',
                 'alberto_seamonster', 'luca_seamonster', 'barley_lightfoot'],
        help='Character names to process'
    )
    parser.add_argument(
        '--lora-types',
        nargs='+',
        default=['pose', 'action', 'expression'],
        help='LoRA types to generate templates for'
    )
    parser.add_argument(
        '--samples-per-character',
        type=int,
        default=5,
        help='Number of images to sample per character/type'
    )
    parser.add_argument(
        '--model',
        default='llm_provider-3-5-sonnet-20241022',
        help='LLMProvider model to use (sonnet for quality, haiku for cost)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Temperature for generation (lower = more consistent)'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = SyntheticCaptionTemplateGenerator(
        model=args.model,
        temperature=args.temperature
    )

    # Generate templates
    generator.generate_for_all_characters(
        data_root=args.data_root,
        characters=args.characters,
        lora_types=args.lora_types,
        output_dir=args.output_dir,
        samples_per_character=args.samples_per_character
    )

    logger.info("✓ Template generation complete!")


if __name__ == '__main__':
    main()
