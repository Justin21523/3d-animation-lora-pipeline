#!/usr/bin/env python3
"""
SDXL Caption Expander

Expands SD1.5 captions (77 tokens) to SDXL-optimized captions (up to 225 tokens)
using LLMProvider API. Adds technical rendering details for better SDXL training.

Features:
- Character-aware expansion with style context (Pixar/DreamWorks)
- Preserves original identity and scene while adding technical details
- Adds lighting, materials, camera, and render quality terms
- Batch processing with progress tracking
- Cost estimation and rate limiting

Author: LLMProvider Tooling
Date: 2025-11-22
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    import llm_vendor
except ImportError:
    raise ImportError(
        "SDXL caption expansion requires: pip install llm_vendor"
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SDXLCaptionExpander:
    """
    Expand SD1.5 captions to SDXL-optimized captions using LLMProvider API.

    SD1.5: 77 tokens max
    SDXL: 225 tokens max (CLIP-L + OpenCLIP-G dual text encoders)

    SDXL benefits from detailed technical descriptions:
    - Lighting setup (three-point, ambient occlusion, volumetric)
    - Material properties (PBR, subsurface scattering, specular)
    - Camera details (composition, depth of field, framing)
    - Render quality (resolution, production standard)
    """

    # Style-specific expansion contexts
    EXPANSION_CONTEXTS = {
        'pixar': {
            'lighting_terms': [
                "studio lighting with soft shadows",
                "three-point lighting setup",
                "ambient occlusion",
                "global illumination",
                "volumetric lighting",
                "soft key light from upper left",
                "subtle rim lighting separating character from background",
                "diffused fill light"
            ],
            'material_terms': [
                "physically-based rendering (PBR)",
                "subsurface scattering on skin",
                "detailed skin shader",
                "specular highlights",
                "realistic texture details",
                "fabric wrinkles and material properties",
                "smooth shading with subtle gradients"
            ],
            'camera_terms': [
                "professional camera composition",
                "shallow depth of field",
                "cinematic framing",
                "bokeh background",
                "sharp focus on subject",
                "eye-level camera angle",
                "medium full shot",
                "three-quarter view"
            ],
            'quality_terms': [
                "1024px high resolution",
                "8k render quality",
                "award-winning animation",
                "detailed 3d model",
                "photorealistic rendering",
                "IMAX quality",
                "production-ready quality",
                "feature film standard"
            ]
        },
        'dreamworks': {
            'lighting_terms': [
                "dynamic lighting",
                "dramatic shadows",
                "cinematic lighting",
                "rim lighting",
                "environmental lighting",
                "high contrast lighting setup",
                "expressive shadow work"
            ],
            'material_terms': [
                "high-quality materials",
                "realistic surface details",
                "advanced shader work",
                "texture definition",
                "cartoony stylized materials",
                "smooth expressive shading"
            ],
            'camera_terms': [
                "dynamic camera angle",
                "cinematic composition",
                "professional framing",
                "depth perception",
                "expressive camera work",
                "stylized composition"
            ],
            'quality_terms': [
                "high-resolution render",
                "production quality",
                "feature film standard",
                "detailed character model",
                "theatrical release quality"
            ]
        }
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llm_provider-3-5-haiku-20241022",
        temperature: float = 0.3
    ):
        """
        Initialize SDXL caption expander.

        Args:
            api_key: LLMVendor API key (uses LLM_VENDOR_API_KEY env if not provided)
            model: LLMProvider model to use (default: Haiku for cost efficiency)
            temperature: Sampling temperature (lower = more consistent)
        """
        self.api_key = api_key or os.getenv('LLM_VENDOR_API_KEY')

        if not self.api_key:
            raise ValueError(
                "LLM_VENDOR_API_KEY not found. Set it as environment variable:\n"
                "export LLM_VENDOR_API_KEY='your-api-key-here'"
            )

        self.model = model
        self.temperature = temperature
        self.client = llm_vendor.LLMVendor(api_key=self.api_key)

        # Statistics
        self.stats = {
            'processed': 0,
            'failed': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'avg_orig_length': 0.0,
            'avg_expanded_length': 0.0
        }

        logger.info(f"✓ SDXL Caption Expander initialized")
        logger.info(f"  Model: {model}")
        logger.info(f"  Temperature: {temperature}")

    def _build_expansion_prompt(
        self,
        original_caption: str,
        character_name: str,
        style: str = 'pixar'
    ) -> str:
        """
        Build expansion prompt for LLMProvider API.

        Args:
            original_caption: Original SD1.5 caption
            character_name: Character name for context
            style: Animation style (pixar/dreamworks)

        Returns:
            Formatted prompt string
        """
        context = self.EXPANSION_CONTEXTS.get(style.lower(), self.EXPANSION_CONTEXTS['pixar'])

        prompt = f"""You are an expert in 3D animation and AI image generation caption writing.

I have a SHORT caption for a Stable Diffusion 1.5 LoRA (limited to 77 tokens):
"{original_caption}"

I need to expand this into a DETAILED SDXL caption (up to 225 tokens) that:
1. **Preserves** the original character, pose, expression, and scene context
2. **Adds** technical rendering details suitable for SDXL:
   - Lighting setup: {', '.join(context['lighting_terms'][:3])}
   - Material properties: {', '.join(context['material_terms'][:3])}
   - Camera/composition: {', '.join(context['camera_terms'][:3])}
   - Render quality: {', '.join(context['quality_terms'][:3])}

Character: {character_name}
Animation style: {style}

Requirements:
- Keep the same character identity and core scene
- Use natural language (not a list of tags)
- Target 120-180 tokens for optimal SDXL performance
- Emphasize visual/technical details that SDXL can reproduce
- Avoid overly abstract or literary descriptions
- Do not invent details not visible in the original caption
- Focus on rendering quality, lighting, materials, and camera work

Output ONLY the expanded caption, no explanation."""

        return prompt

    def expand_caption(
        self,
        original_caption: str,
        character_name: str,
        style: str = 'pixar'
    ) -> Tuple[str, Dict]:
        """
        Expand a single caption using LLMProvider API.

        Args:
            original_caption: Original SD1.5 caption
            character_name: Character name
            style: Animation style

        Returns:
            Tuple of (expanded_caption, metadata)
        """
        prompt = self._build_expansion_prompt(original_caption, character_name, style)

        try:
            # Call LLMProvider API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            expanded = message.content[0].text.strip()

            # Remove quotes if LLMProvider added them
            if expanded.startswith('"') and expanded.endswith('"'):
                expanded = expanded[1:-1]

            # Calculate token counts (approximate)
            orig_tokens = len(original_caption.split())
            expanded_tokens = len(expanded.split())

            metadata = {
                'original_tokens': orig_tokens,
                'expanded_tokens': expanded_tokens,
                'input_tokens': message.usage.input_tokens,
                'output_tokens': message.usage.output_tokens,
                'model': self.model
            }

            # Update stats
            self.stats['processed'] += 1
            self.stats['total_input_tokens'] += message.usage.input_tokens
            self.stats['total_output_tokens'] += message.usage.output_tokens

            return expanded, metadata

        except Exception as e:
            logger.error(f"Failed to expand caption: {e}")
            self.stats['failed'] += 1

            # Return original on failure
            return original_caption, {
                'error': str(e),
                'fallback': True
            }

    def expand_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        character_name: str,
        style: str = 'pixar',
        max_files: Optional[int] = None
    ) -> Dict:
        """
        Expand all captions in a directory.

        Args:
            input_dir: Directory with original .txt caption files
            output_dir: Output directory for expanded captions
            character_name: Character name
            style: Animation style
            max_files: Limit number of files (for testing)

        Returns:
            Results dictionary with statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all .txt files (including in subdirectories for Kohya format)
        caption_files = sorted(input_dir.rglob('*.txt'))

        if max_files:
            caption_files = caption_files[:max_files]

        if not caption_files:
            logger.warning(f"No .txt files found in {input_dir}")
            return {'error': 'No caption files found'}

        logger.info(f"Found {len(caption_files)} caption files")
        logger.info(f"Expanding captions for: {character_name} ({style} style)")
        logger.info(f"Output: {output_dir}")

        # Process each file
        expanded_captions = []
        orig_lengths = []
        expanded_lengths = []

        for i, caption_file in enumerate(caption_files, 1):
            # Read original caption
            with open(caption_file, 'r', encoding='utf-8') as f:
                original = f.read().strip()

            orig_lengths.append(len(original.split()))

            logger.info(f"[{i}/{len(caption_files)}] {caption_file.name}")
            logger.info(f"  Original ({len(original.split())} tokens): {original[:80]}...")

            # Expand caption
            expanded, metadata = self.expand_caption(original, character_name, style)
            expanded_lengths.append(len(expanded.split()))

            logger.info(f"  Expanded ({len(expanded.split())} tokens): {expanded[:80]}...")

            # Save expanded caption (preserve directory structure for Kohya format)
            relative_path = caption_file.relative_to(input_dir)
            output_file = output_dir / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(expanded)

            expanded_captions.append({
                'file': caption_file.name,
                'original': original,
                'expanded': expanded,
                'metadata': metadata
            })

            # Rate limiting (2 requests/second to avoid API limits)
            if i < len(caption_files):
                time.sleep(0.5)

        # Calculate statistics
        results = {
            'character_name': character_name,
            'style': style,
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'statistics': {
                'processed': len(caption_files),
                'failed': self.stats['failed'],
                'avg_orig_length': sum(orig_lengths) / len(orig_lengths) if orig_lengths else 0,
                'avg_expanded_length': sum(expanded_lengths) / len(expanded_lengths) if expanded_lengths else 0,
                'total_input_tokens': self.stats['total_input_tokens'],
                'total_output_tokens': self.stats['total_output_tokens'],
                'estimated_cost_usd': self._estimate_cost()
            },
            'captions': expanded_captions
        }

        # Save metadata
        metadata_file = output_dir / 'sdxl_expansion_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✓ Expansion complete!")
        logger.info(f"  Processed: {results['statistics']['processed']}")
        logger.info(f"  Failed: {results['statistics']['failed']}")
        logger.info(f"  Avg original length: {results['statistics']['avg_orig_length']:.1f} tokens")
        logger.info(f"  Avg expanded length: {results['statistics']['avg_expanded_length']:.1f} tokens")
        logger.info(f"  Total input tokens: {results['statistics']['total_input_tokens']}")
        logger.info(f"  Total output tokens: {results['statistics']['total_output_tokens']}")
        logger.info(f"  Estimated cost: ${results['statistics']['estimated_cost_usd']:.2f}")
        logger.info(f"  Metadata saved: {metadata_file}")

        return results

    def _estimate_cost(self) -> float:
        """
        Estimate API cost based on token usage.

        LLMProvider 3.5 Sonnet pricing (as of 2024):
        - Input: $3.00 per million tokens
        - Output: $15.00 per million tokens

        Returns:
            Estimated cost in USD
        """
        input_cost = (self.stats['total_input_tokens'] / 1_000_000) * 3.00
        output_cost = (self.stats['total_output_tokens'] / 1_000_000) * 15.00
        return input_cost + output_cost


def main():
    parser = argparse.ArgumentParser(
        description="Expand SD1.5 captions to SDXL-optimized captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Single character expansion
  python sdxl_caption_expander.py \\
    --input-dir /data/luca/training_data/alberto_identity \\
    --output-dir /data/luca/training_data_sdxl/alberto_identity \\
    --character-name "alberto" \\
    --style pixar

  # Test with limited files
  python sdxl_caption_expander.py \\
    --input-dir /data/orion/training_data/orion_identity \\
    --output-dir /data/orion/training_data_sdxl/orion_identity \\
    --character-name "orion" \\
    --style dreamworks \\
    --max-files 10

API Key:
  Set LLM_VENDOR_API_KEY environment variable:
  export LLM_VENDOR_API_KEY='your-api-key-here'
        """
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Input directory with SD1.5 caption .txt files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for SDXL expanded captions'
    )
    parser.add_argument(
        '--character-name',
        type=str,
        required=True,
        help='Character name for context (e.g., "alberto", "orion")'
    )
    parser.add_argument(
        '--style',
        type=str,
        choices=['pixar', 'dreamworks'],
        default='pixar',
        help='Animation style (default: pixar)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        help='Limit number of files to process (for testing)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='llm_provider-3-5-haiku-20241022',
        help='LLMProvider model to use (default: llm_provider-3-5-haiku-20241022 for cost efficiency)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Sampling temperature (default: 0.3)'
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1

    # Create expander
    try:
        expander = SDXLCaptionExpander(
            model=args.model,
            temperature=args.temperature
        )
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Expand captions
    try:
        results = expander.expand_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            character_name=args.character_name,
            style=args.style,
            max_files=args.max_files
        )

        return 0

    except Exception as e:
        logger.error(f"Caption expansion failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
