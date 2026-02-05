#!/usr/bin/env python3
"""
Prompt Vocabulary Generator for Synthetic Data Generation (v3.0)
=================================================================

Generates diverse, high-quality prompts for pose, expression, and action LoRA training.
Uses template-based generation with detailed templates (60-80 tokens each) to produce
prompts that are 100-120 tokens total.

v3.0 Changes (2025-12-06):
- DETAILED templates imported from vocabulary_templates_v2_detailed.py
- Each template now contains: core description, body mechanics, camera angle,
  environment, lighting, and technical rendering details
- Template target: 60-80 tokens per template
- Full prompt target: 100-120 tokens (with character prefix + style)
- Caption target: ~80-90 tokens (after identity removal)
- POSE_TEMPLATES: 66 detailed templates
- EXPRESSION_TEMPLATES: 67 detailed templates
- ACTION_TEMPLATES: 143 detailed templates (expanded to cover all categories)

v2.0 Changes (2025-12-06):
- Expanded ACTION_TEMPLATES from 35 to 100+ templates
- Expanded POSE_TEMPLATES from 27 to 55+ templates
- Expanded EXPRESSION_TEMPLATES from 35 to 55+ templates
- Added ActionTemplate dataclass for category-based sampling
- Added sample_balanced_prompts() for balanced category distribution
- Original templates backed up to vocabulary_templates_backup_v1.py

Usage:
    python vocabulary_generator.py \
        --character-name alberto \
        --character-description "A 3D animated boy character from Pixar-style animation" \
        --lora-type pose \
        --num-prompts 100 \
        --output-file prompts.json \
        --use-templates \
        --template-variations 5

Author: LLMProvider Tooling
Date: 2025-11-30 (v1), 2025-12-06 (v2, v3)
"""

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

# v3.0: Import DETAILED templates from vocabulary_templates_v2_detailed
# These templates are 60-80 tokens each, producing 100-120 token full prompts
try:
    # When imported as a module
    from .vocabulary_templates_v2_detailed import (
        POSE_TEMPLATES,
        EXPRESSION_TEMPLATES,
        ACTION_TEMPLATES,
    )
except ImportError:
    # When run directly as a script
    from vocabulary_templates_v2_detailed import (
        POSE_TEMPLATES,
        EXPRESSION_TEMPLATES,
        ACTION_TEMPLATES,
    )


# ============================================================================
# CHARACTER TO MOVIE MAPPING (for evaluation-style prompts)
# ============================================================================

CHARACTER_TO_MOVIE = {
    # ===== LUCA CHARACTERS =====
    # CRITICAL: Trigger words must match EXACTLY what was used during LoRA training
    # These are lowercase, simple trigger words (NOT descriptive captions)
    "alberto": (
        "alberto", "luca",  # Simple lowercase trigger word
        "teenage boy with messy dark brown hair, confident expression, tan sun-kissed skin, green vest over bare chest, "
        "adventurous personality, italian features, smooth 3d character shading, pbr skin materials, detailed facial features"
    ),
    "alberto_seamonster": (
        "alberto_seamonster", "luca",  # Underscore format trigger word
        "purple and blue scaled sea creature with flowing fins, aquatic features, expressive sea monster eyes, "
        "vibrant iridescent scales, underwater lighting effects, smooth 3d creature rendering, detailed scale textures"
    ),
    "giulia": (
        "giulia", "luca",  # Simple lowercase trigger word
        "italian girl with vibrant red curly hair and freckles, energetic expression, warm skin tone, "
        "colorful summer clothing, friendly personality, smooth 3d character shading, detailed hair simulation"
    ),
    "luca": (
        "luca", "luca",  # Simple lowercase trigger word
        "young italian boy with soft brown wavy hair, gentle expression, warm olive skin tone, innocent personality, "
        "coastal italian setting, smooth 3d character shading, pbr materials, natural lighting"
    ),
    "luca_seamonster": (
        "luca_seamonster", "luca",  # Underscore format trigger word
        "green and purple scaled aquatic creature with elegant fins, curious expression, vibrant sea monster features, "
        "luminous scales, underwater atmosphere, smooth 3d creature rendering, detailed aquatic textures"
    ),

    # ===== COCO CHARACTERS =====
    "miguel": (
        "miguel", "coco",  # Simple lowercase trigger word
        "mexican boy with dark spiky hair, passionate expression, warm brown skin, traditional mexican clothing details, "
        "musical personality, vibrant mexican setting, smooth 3d character shading, pbr materials, warm ambient lighting"
    ),

    # ===== ELIO CHARACTERS =====
    "bryce": (
        "bryce", "elio",  # Simple lowercase trigger word
        "teenage boy with blonde spiky hair, athletic build, confident stance, determined expression, "
        "modern casual clothing, smooth 3d character shading, detailed facial features, studio lighting"
    ),
    "caleb": (
        "caleb", "elio",  # Simple lowercase trigger word
        "young boy with dark curly hair, curious expression, expressive eyes, playful personality, "
        "casual contemporary clothing, smooth 3d character shading, pbr materials, natural lighting"
    ),
    "elio": (
        "elio", "elio",  # Simple lowercase trigger word
        "young boy with brown curly hair, imaginative expression, wide-eyed wonder, creative personality, "
        "colorful clothing details, smooth 3d character shading, detailed facial animation, vibrant lighting"
    ),

    # ===== ONWARD CHARACTERS =====
    "barley_lightfoot": (
        "barley_lightfoot", "onward",  # Underscore format trigger word
        "large elf teenager with blue skin, purple hair and pointy ears, enthusiastic expression, fantasy clothing, "
        "adventurous personality, magical setting, smooth 3d fantasy character shading, vibrant colors, dramatic lighting"
    ),
    "ian_lightfoot": (
        "ian_lightfoot", "onward",  # Underscore format trigger word
        "young elf with blue skin, dark hair and pointed ears, shy expression, modern elf clothing, "
        "coming-of-age character, magical atmosphere, smooth 3d character shading, detailed elf features"
    ),

    # ===== ORION CHARACTERS =====
    "orion": (
        "orion", "orion",  # Simple lowercase trigger word
        "young character with distinctive features, expressive personality, vibrant clothing details, "
        "dreamworks animation style, smooth 3d character shading, pbr materials, cinematic lighting"
    ),

    # ===== UP CHARACTERS =====
    "russell": (
        "russell", "up",
        "young boy with round face, wilderness explorer uniform, enthusiastic expression, cheerful personality, "
        "colorful badges and accessories, smooth 3d character shading, warm ambient lighting, detailed clothing textures"
    ),

    # ===== TURNING RED CHARACTERS =====
    "tyler": (
        "tyler", "turning red",
        "teenage boy with distinctive hairstyle, modern urban clothing, expressive personality, "
        "contemporary setting, smooth 3d character shading, pbr materials, vibrant modern lighting"
    ),
}


# ============================================================================
# CHARACTER PREFIXES FOR BALANCED PROMPTS (Option C)
# ============================================================================
# Format: {character} ({Full name} from Pixar {movie}, {type} {age}, {key trait})
# This provides complete context while staying within SDXL's 77 token limit

CHARACTER_PREFIXES = {
    "alberto": "alberto (Alberto from Pixar Luca, Italian teen, curly brown hair)",
    "alberto_seamonster": "alberto_seamonster, sea monster with iridescent blue-purple scaly skin, large expressive yellow-green eyes, ruffled purple fin-like hair crest, webbed hands",
    "barley_lightfoot": "barley_lightfoot (Barley from Pixar Onward, blue elf teen, purple hair)",
    "bryce": "bryce (Bryce from Pixar Elio, teenage boy, blonde spiky hair)",
    "caleb": "caleb (Caleb from Pixar Elio, young boy, dark curly hair)",
    "elio": "elio (Elio from Pixar Elio, young boy, brown curly hair)",
    "giulia": "giulia (Giulia from Pixar Luca, Italian girl, red curly hair)",
    "ian_lightfoot": "ian_lightfoot (Ian from Pixar Onward, blue elf boy, dark hair)",
    "luca": "luca (Luca from Pixar Luca, Italian boy, wavy brown hair)",
    "luca_seamonster": "luca_seamonster, sea monster with vibrant green-purple iridescent scaly skin, turquoise fin-like hair crest, large expressive eyes, webbed hands",
    "miguel": "miguel (Miguel from Pixar Coco, Mexican boy, dark spiky hair)",
    "orion": "orion (Orion from DreamWorks Orion, young character, distinctive features)",
    "russell": "russell (Russell from Pixar Up, young boy, round face)",
    "tyler": "tyler (Tyler from Pixar Turning Red, teenage boy, distinctive hair)",
}


# ============================================================================
# TEMPLATE DATACLASS FOR CATEGORY-BASED SAMPLING
# ============================================================================

@dataclass
class TemplateEntry:
    """Template entry with category metadata for balanced sampling."""
    template: str
    category: str
    subcategory: str = ""
    weight: float = 1.0


# ============================================================================
# TEMPLATE IMPORTS (v3.0 - Detailed templates from external file)
# ============================================================================
# NOTE: POSE_TEMPLATES, EXPRESSION_TEMPLATES, and ACTION_TEMPLATES are now
# imported from vocabulary_templates_v2_detailed.py at the top of this file.
#
# Template counts:
#   - POSE_TEMPLATES: 66 detailed templates (60-80 tokens each)
#   - EXPRESSION_TEMPLATES: 67 detailed templates (60-80 tokens each)
#   - ACTION_TEMPLATES: 143 detailed templates (60-80 tokens each)
#
# Full prompts (with character + style) target: 100-120 tokens
# Captions (after identity removal) target: ~80-90 tokens
# ============================================================================



STYLE_VARIATIONS = [
    # Basic Pixar style variations
    "3d animation, pixar style, high quality, detailed",
    "3d animated character, smooth shading, studio lighting",
    "pixar-style 3d render, professional quality",
    "3d cg animation, clean render",
    "animated 3d character, cinematic lighting",
    "high-quality 3d animation, detailed modeling",
    "3d computer graphics, smooth surfaces, realistic lighting",

    # Advanced material-aware variations (Pixar/DreamWorks production quality)
    "pixar style 3d animation, smooth shading, PBR materials, subsurface scattering on skin, soft ambient occlusion, professional CGI render",
    "3d animated character, PBR skin materials, detailed subsurface scattering, ambient occlusion, high quality professional render",
    "pixar-style 3d render, physically based rendering, subsurface scattering, soft ambient occlusion, cinematic lighting, detailed character modeling",
    "high-quality 3d animation, PBR materials, realistic skin subsurface scattering, soft ambient occlusion, professional CGI quality",
    "3d cg animation, smooth PBR shading, subsurface scattering effects, ambient occlusion rendering, cinematic quality",
    "animated 3d character, physically based materials, subsurface skin scattering, soft AO, professional animation render",
    "pixar animation style, PBR materials and textures, realistic subsurface scattering, ambient occlusion, high-end CGI production quality",
]


# ============================================================================
# TEMPLATE STATISTICS (for reference)
# ============================================================================

def get_template_stats() -> Dict[str, int]:
    """Return template counts for each LoRA type."""
    return {
        "pose": len(POSE_TEMPLATES),
        "expression": len(EXPRESSION_TEMPLATES),
        "action": len(ACTION_TEMPLATES),
        "style_variations": len(STYLE_VARIATIONS),
    }


# ============================================================================
# BALANCED SAMPLING FUNCTIONS
# ============================================================================

def sample_balanced_from_templates(
    templates: List[str],
    num_samples: int,
    ensure_coverage: bool = True,
    consistent_seed: Optional[int] = None
) -> List[str]:
    """
    Sample templates with balanced distribution.

    If ensure_coverage=True and num_samples >= len(templates),
    ensures every template is used at least once before repeating.

    Args:
        templates: List of template strings
        num_samples: Number of samples to return
        ensure_coverage: If True, ensure all templates are covered first
        consistent_seed: If provided, use this seed for consistent sampling across calls
                        This ensures all characters get the SAME templates for the same lora_type

    Returns:
        List of sampled template strings
    """
    if not templates:
        return []

    # Use consistent seed if provided (for cross-character consistency)
    if consistent_seed is not None:
        rng = random.Random(consistent_seed)
    else:
        rng = random.Random()

    sampled = []

    if ensure_coverage and num_samples >= len(templates):
        # First, include all templates once
        sampled.extend(templates.copy())
        rng.shuffle(sampled)

        # Then fill remainder with random samples
        remaining = num_samples - len(templates)
        if remaining > 0:
            additional = rng.choices(templates, k=remaining)
            sampled.extend(additional)
    else:
        # When num_samples < len(templates), use deterministic selection
        # to ensure all characters get the SAME subset of templates
        indices = list(range(len(templates)))
        rng.shuffle(indices)
        selected_indices = indices[:num_samples]
        sampled = [templates[i] for i in selected_indices]

    return sampled[:num_samples]


# ============================================================================
# PROMPT GENERATION FUNCTIONS
# ============================================================================

def generate_template_prompts(
    character_name: str,
    character_description: str,
    lora_type: str,
    num_prompts: int,
    variations_per_template: int = 1,
    ensure_template_coverage: bool = True,
    consistent_across_characters: bool = True
) -> List[Dict[str, str]]:
    """
    Generate prompts using templates with variations (v2.0 - balanced sampling).

    Args:
        character_name: Name/trigger word for character (e.g., "alberto")
        character_description: Brief character description (legacy, not used in new format)
        lora_type: Type of LoRA (pose/expression/action)
        num_prompts: Target number of prompts to generate
        variations_per_template: Number of style variations per template
        ensure_template_coverage: If True, ensure all templates are used before repeating
        consistent_across_characters: If True, all characters get the SAME templates
                                      for the same lora_type (important for balanced training)

    Returns:
        List of prompt dictionaries with 'prompt' and 'metadata' fields
    """
    # Select templates based on LoRA type
    if lora_type == "pose":
        templates = POSE_TEMPLATES
    elif lora_type == "expression":
        templates = EXPRESSION_TEMPLATES
    elif lora_type == "action":
        templates = ACTION_TEMPLATES
    else:
        raise ValueError(f"Unknown LoRA type: {lora_type}")

    # Use CHARACTER_PREFIXES for complete but concise character description (Option C)
    # Format: {character} ({Full name} from Pixar {movie}, {type} {age}, {key trait})
    # This provides context while staying within SDXL's 77 token limit
    # Falls back to character_name if not in mapping
    character_prefix = CHARACTER_PREFIXES.get(character_name, character_name)

    # v2.0: Use balanced sampling to ensure template diversity
    # Use consistent seed based on lora_type to ensure all characters get same templates
    consistent_seed = None
    if consistent_across_characters:
        # Hash the lora_type to get a consistent seed
        consistent_seed = hash(lora_type) % (2**31)

    sampled_templates = sample_balanced_from_templates(
        templates=templates,
        num_samples=num_prompts,
        ensure_coverage=ensure_template_coverage,
        consistent_seed=consistent_seed
    )

    prompts = []
    style_index = 0

    for template in sampled_templates:
        # Cycle through style variations for diversity
        style = STYLE_VARIATIONS[style_index % len(STYLE_VARIATIONS)]
        style_index += 1

        # Format prompt with character prefix (Option C format)
        # Example: "alberto (Alberto from Pixar Luca, Italian teen, curly brown hair), {style}, ..."
        # This balances: character identity + source context + style + description
        prompt = template.format(
            character=character_prefix,
            style=style
        )

        # CRITICAL: Add "pixar style 3d animated character" at the end
        # This matches the training data format and is essential for LoRA activation
        prompt = prompt + ", pixar style 3d animated character"

        # Add to list with metadata for caption conversion
        prompts.append({
            "prompt": prompt,
            "metadata": {
                "character": character_name,
                "lora_type": lora_type,
                "template": template,
                "style": style,
                # v2.0: Store template without character for easy caption extraction
                "template_raw": template.replace("{character}, {style}, ", "")
            }
        })

    return prompts[:num_prompts]


def generate_simple_prompts(
    character_name: str,
    character_description: str,
    lora_type: str,
    num_prompts: int
) -> List[Dict[str, str]]:
    """
    Generate simple baseline prompts without templates.

    Args:
        character_name: Name/trigger word for character
        character_description: Brief character description
        lora_type: Type of LoRA (pose/expression/action)
        num_prompts: Number of prompts to generate

    Returns:
        List of prompt dictionaries
    """
    prompts = []

    for i in range(num_prompts):
        style = random.choice(STYLE_VARIATIONS)
        prompt = f"{character_name}, {character_description}, {style}"

        prompts.append({
            "prompt": prompt,
            "metadata": {
                "character": character_name,
                "lora_type": lora_type,
                "generation_method": "simple",
                "index": i
            }
        })

    return prompts


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate prompt vocabulary for synthetic data generation"
    )

    parser.add_argument(
        "--character-name",
        type=str,
        required=True,
        help="Character name / trigger word"
    )

    parser.add_argument(
        "--character-description",
        type=str,
        required=True,
        help="Brief character description"
    )

    parser.add_argument(
        "--lora-type",
        type=str,
        required=True,
        choices=["pose", "expression", "action"],
        help="Type of LoRA to generate prompts for"
    )

    parser.add_argument(
        "--num-prompts",
        type=int,
        required=True,
        help="Number of prompts to generate"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output JSON file path"
    )

    parser.add_argument(
        "--use-templates",
        action="store_true",
        help="Use template-based generation (recommended)"
    )

    parser.add_argument(
        "--template-variations",
        type=int,
        default=1,
        help="Number of style variations per template (default: 1)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Generate prompts
    print(f"Generating {args.num_prompts} {args.lora_type} prompts for {args.character_name}...")

    if args.use_templates:
        prompts = generate_template_prompts(
            character_name=args.character_name,
            character_description=args.character_description,
            lora_type=args.lora_type,
            num_prompts=args.num_prompts,
            variations_per_template=args.template_variations
        )
        print(f"✓ Generated {len(prompts)} prompts using templates")
    else:
        prompts = generate_simple_prompts(
            character_name=args.character_name,
            character_description=args.character_description,
            lora_type=args.lora_type,
            num_prompts=args.num_prompts
        )
        print(f"✓ Generated {len(prompts)} simple prompts")

    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    output_data = {
        "character": args.character_name,
        "lora_type": args.lora_type,
        "num_prompts": len(prompts),
        "generation_method": "template" if args.use_templates else "simple",
        "prompts": prompts
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved prompts to: {output_path}")
    print(f"\nSample prompts:")
    for i, prompt_obj in enumerate(prompts[:3], 1):
        print(f"  {i}. {prompt_obj['prompt']}")

    if len(prompts) > 3:
        print(f"  ... and {len(prompts) - 3} more")


if __name__ == "__main__":
    # Print template statistics when run directly
    stats = get_template_stats()
    print("=" * 60)
    print("Vocabulary Generator v2.0 - Template Statistics")
    print("=" * 60)
    print(f"  POSE_TEMPLATES:       {stats['pose']} templates")
    print(f"  EXPRESSION_TEMPLATES: {stats['expression']} templates")
    print(f"  ACTION_TEMPLATES:     {stats['action']} templates")
    print(f"  STYLE_VARIATIONS:     {stats['style_variations']} variations")
    print(f"  TOTAL TEMPLATES:      {stats['pose'] + stats['expression'] + stats['action']}")
    print("=" * 60)
    print()

    main()
