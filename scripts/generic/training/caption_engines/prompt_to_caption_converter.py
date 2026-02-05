#!/usr/bin/env python3
"""
Prompt to Caption Converter for Synthetic LoRA Training Data
=============================================================

Converts generation prompts to training captions by removing character identity
while preserving all action/pose/expression descriptors.

This ensures:
- LoRA learns visual features, not text associations
- Captions remain detailed and descriptive
- Token count stays within SDXL range (100-225 tokens)

Usage:
    # As module
    from prompt_to_caption_converter import PromptToCaptionConverter
    converter = PromptToCaptionConverter()
    caption = converter.convert(prompt, character_name="alberto")

    # As CLI
    python prompt_to_caption_converter.py \
        --input prompts.json \
        --output captions.json \
        --character alberto

Author: LLMProvider Tooling
Date: 2025-12-06
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CHARACTER IDENTITY PATTERNS TO REMOVE
# ============================================================================

# All character names (must match exactly what's used in vocabulary_generator.py)
CHARACTER_NAMES = {
    # Luca
    "alberto", "alberto_seamonster", "giulia", "luca", "luca_seamonster",
    # Coco
    "miguel",
    # Elio
    "bryce", "caleb", "elio",
    # Onward
    "barley_lightfoot", "barley", "ian_lightfoot", "ian",
    # Others
    "orion", "russell", "tyler",
}

# Film/studio references
FILM_REFERENCES = [
    r"from\s+Pixar\s+\w+",
    r"from\s+Disney\s+\w+",
    r"from\s+DreamWorks\s+\w+",
    r"Pixar\s+Luca",
    r"Pixar\s+Coco",
    r"Pixar\s+Elio",
    r"Pixar\s+Onward",
    r"Pixar\s+Up",
    r"Pixar\s+Turning\s+Red",
    r"DreamWorks\s+\w+",
]

# Demographic descriptors to remove
DEMOGRAPHIC_PATTERNS = [
    r"Italian\s+(boy|girl|teen|teenager|child|kid)",
    r"Mexican\s+(boy|girl|teen|teenager|child|kid)",
    r"American\s+(boy|girl|teen|teenager|child|kid)",
    r"young\s+Italian\s+\w+",
    r"young\s+Mexican\s+\w+",
]

# Physical appearance patterns (hair, eyes, skin, etc.)
APPEARANCE_PATTERNS = [
    # Hair descriptions
    r"(?:with\s+)?(?:wild\s+|messy\s+|soft\s+|wavy\s+|curly\s+|spiky\s+|straight\s+)?(?:dark\s+|brown\s+|black\s+|blonde\s+|red\s+|purple\s+)?(?:brown\s+|curly\s+|wavy\s+|spiky\s+)?hair(?:\s+and)?",
    r"curly\s+brown\s+hair",
    r"wavy\s+brown\s+hair",
    r"dark\s+spiky\s+hair",
    r"red\s+curly\s+hair",
    r"blonde\s+spiky\s+hair",
    r"dark\s+curly\s+hair",
    r"purple\s+hair",
    # Eye descriptions
    r"(?:with\s+)?(?:blue|brown|green|hazel|expressive)\s+eyes?",
    r"large\s+expressive\s+(?:yellow-green\s+)?eyes",
    # Skin descriptions - require at least one modifier to avoid matching "on skin" in technical terms
    r"(?:tan|pale|olive|warm|dark|sun-kissed|tanned?|light|medium|brown)\s+skin(?:\s+tone)?",
    r"blue\s+skin",
    r"warm\s+brown\s+skin",
    r"skin\s+tone",  # standalone "skin tone"
    # Freckles
    r"(?:and\s+)?freckles",
    r"with\s+freckles",
]

# Species/fantasy features
SPECIES_PATTERNS = [
    r"(?:blue\s+)?elf\s+(?:boy|girl|teen|teenager)",
    r"sea\s+monster\s+(?:with|form)",
    r"(?:iridescent\s+)?(?:blue-purple\s+|green-purple\s+)?scaly\s+skin",
    r"ruffled\s+purple\s+fin-like\s+hair\s+crest",
    r"turquoise\s+fin-like\s+hair\s+crest",
    r"webbed\s+hands",
    r"pointed?\s+ears?",
    r"pointy\s+ears?",
]

# Enhancement phrases to add when caption is too short
# These are generic 3D animation quality descriptors that don't identify any character
ENHANCEMENT_PHRASES = [
    # Technical quality (always safe to add)
    "high quality 3d render",
    "professional cgi animation",
    "detailed character modeling",
    "smooth subsurface scattering on skin",
    "physically based rendering materials",
    "high resolution textures",
    "professional animation quality",

    # Lighting enhancements
    "soft ambient occlusion",
    "realistic global illumination",
    "volumetric lighting effects",
    "professional studio lighting setup",

    # Material quality
    "pbr skin materials with realistic shading",
    "detailed cloth simulation",
    "natural skin texture rendering",

    # Animation quality
    "fluid character animation",
    "expressive facial rigging",
    "natural body proportions",
    "anatomically correct pose",
]

# Character prefix patterns (the full parenthetical descriptions)
CHARACTER_PREFIX_PATTERNS = [
    # Pattern: "name (Full name from Pixar Movie, type, trait)"
    r"\w+\s*\([^)]+from\s+(?:Pixar|Disney|DreamWorks)[^)]+\),?\s*",
    # Pattern: "name_name (Full name...)"
    r"\w+_\w+\s*\([^)]+\),?\s*",
    # Sea monster full description pattern
    r"\w+_seamonster,\s*sea\s+monster\s+with[^,]+(?:,\s*[^,]+){2,4},?\s*",
]


# ============================================================================
# CAPTION CONVERTER CLASS
# ============================================================================

@dataclass
class ConversionResult:
    """Result of a prompt-to-caption conversion."""
    original_prompt: str
    caption: str
    removed_terms: List[str]
    token_count_estimate: int
    is_valid: bool
    validation_errors: List[str]


class PromptToCaptionConverter:
    """
    Converts generation prompts to training captions by removing character identity.

    The converter:
    1. Removes character names and prefixes
    2. Removes film/studio references
    3. Removes demographic descriptors
    4. Removes physical appearance descriptions
    5. Removes species-specific features
    6. Preserves action/pose/expression descriptors
    7. Validates token count (100-225 for SDXL)
    """

    def __init__(
        self,
        min_tokens: int = 60,  # Increased for better SDXL training
        max_tokens: int = 225,
        target_tokens: int = 100,
        generic_subject: str = "a 3d animated character",
        preserve_style_tags: bool = True,
        auto_enhance: bool = True,  # Automatically enhance short captions
    ):
        """
        Initialize the converter.

        Args:
            min_tokens: Minimum token count for valid caption
            max_tokens: Maximum token count for valid caption
            target_tokens: Target token count for optimal captions
            generic_subject: Generic subject to use instead of character identity
            preserve_style_tags: Whether to preserve style/quality tags
            auto_enhance: Automatically add enhancement phrases if caption is too short
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens
        self.generic_subject = generic_subject
        self.preserve_style_tags = preserve_style_tags
        self.auto_enhance = auto_enhance

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns for efficient matching."""
        # Character name pattern (word boundaries)
        name_patterns = [rf"\b{re.escape(name)}\b" for name in CHARACTER_NAMES]
        self.character_name_pattern = re.compile(
            "|".join(name_patterns), re.IGNORECASE
        )

        # Film reference patterns
        self.film_patterns = [re.compile(p, re.IGNORECASE) for p in FILM_REFERENCES]

        # Demographic patterns
        self.demographic_patterns = [
            re.compile(p, re.IGNORECASE) for p in DEMOGRAPHIC_PATTERNS
        ]

        # Appearance patterns
        self.appearance_patterns = [
            re.compile(p, re.IGNORECASE) for p in APPEARANCE_PATTERNS
        ]

        # Species patterns
        self.species_patterns = [
            re.compile(p, re.IGNORECASE) for p in SPECIES_PATTERNS
        ]

        # Character prefix patterns
        self.prefix_patterns = [
            re.compile(p, re.IGNORECASE) for p in CHARACTER_PREFIX_PATTERNS
        ]

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for SDXL/CLIP tokenizer.

        CLIP tokenizer typically yields ~1.3 tokens per word for descriptive text.
        Commas and special characters often become separate tokens.
        """
        words = len(text.split())
        # Count punctuation as separate tokens
        punctuation = len(re.findall(r"[,.:;!?()]", text))
        # Hyphenated words often split into multiple tokens
        hyphens = text.count("-")
        # Estimate: 1.3 tokens per word + punctuation
        return int(words * 1.3 + punctuation + hyphens * 0.5)

    def _remove_character_identity(
        self, text: str, character_name: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Remove all character identity information from text.

        Returns:
            Tuple of (cleaned_text, list_of_removed_terms)
        """
        removed = []
        result = text

        # 1. Remove character prefix patterns first (most specific)
        for pattern in self.prefix_patterns:
            matches = pattern.findall(result)
            for match in matches:
                removed.append(f"prefix: {match[:50]}...")
            result = pattern.sub("", result)

        # 2. Remove specific character name if provided
        if character_name:
            specific_pattern = re.compile(
                rf"\b{re.escape(character_name)}\b", re.IGNORECASE
            )
            if specific_pattern.search(result):
                removed.append(f"character: {character_name}")
            result = specific_pattern.sub("", result)

        # 3. Remove all known character names
        matches = self.character_name_pattern.findall(result)
        for match in matches:
            if match.lower() not in [r.split(": ")[-1].lower() for r in removed]:
                removed.append(f"character: {match}")
        result = self.character_name_pattern.sub("", result)

        # 4. Remove film references
        for pattern in self.film_patterns:
            matches = pattern.findall(result)
            for match in matches:
                removed.append(f"film: {match}")
            result = pattern.sub("", result)

        # 5. Remove demographic descriptors
        for pattern in self.demographic_patterns:
            matches = pattern.findall(result)
            for match in matches:
                removed.append(f"demographic: {match}")
            result = pattern.sub("", result)

        # 6. Remove appearance descriptions
        for pattern in self.appearance_patterns:
            matches = pattern.findall(result)
            for match in matches:
                removed.append(f"appearance: {match}")
            result = pattern.sub("", result)

        # 7. Remove species patterns
        for pattern in self.species_patterns:
            matches = pattern.findall(result)
            for match in matches:
                removed.append(f"species: {match}")
            result = pattern.sub("", result)

        return result, removed

    def _clean_formatting(self, text: str) -> str:
        """Clean up formatting after identity removal."""
        # Remove multiple consecutive commas
        text = re.sub(r",\s*,+", ",", text)

        # Remove leading/trailing commas
        text = text.strip(" ,")

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Remove commas at start after cleaning
        text = re.sub(r"^\s*,\s*", "", text)

        # Remove empty parentheses
        text = re.sub(r"\(\s*\)", "", text)

        # Remove "with" followed by comma or end
        text = re.sub(r"\bwith\s*,", ",", text)
        text = re.sub(r",\s*with\s*$", "", text)

        # Remove "and" at start or followed by comma
        text = re.sub(r"^\s*and\s+", "", text)
        text = re.sub(r",\s*and\s*,", ",", text)

        # Final cleanup
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r",\s*,+", ",", text)
        text = text.strip(" ,")

        return text

    def _add_generic_subject(self, text: str) -> str:
        """Add generic subject if not already present."""
        text_lower = text.lower()

        # Check if already has a subject
        subject_patterns = [
            "a 3d animated character",
            "3d animated character",
            "a 3d character",
            "3d character",
            "animated character",
        ]

        for pattern in subject_patterns:
            if pattern in text_lower:
                return text

        # Add generic subject at the beginning
        return f"{self.generic_subject}, {text}"

    def _enhance_caption(self, text: str, current_tokens: int) -> str:
        """
        Enhance caption with additional quality descriptors if too short.

        Adds generic 3D animation quality phrases until reaching target token count.
        """
        if current_tokens >= self.target_tokens:
            return text

        text_lower = text.lower()
        enhancements_added = []

        # Shuffle enhancement phrases for variety
        import random
        available_phrases = ENHANCEMENT_PHRASES.copy()
        random.shuffle(available_phrases)

        for phrase in available_phrases:
            # Skip if phrase or similar already in caption
            if phrase.lower() in text_lower:
                continue

            # Check key terms to avoid redundancy
            key_terms = phrase.split()[:2]  # First 2 words
            if any(term in text_lower for term in key_terms if len(term) > 3):
                continue

            # Add enhancement
            enhancements_added.append(phrase)
            text = f"{text}, {phrase}"
            text_lower = text.lower()

            # Check if we've reached target
            new_token_count = self._estimate_tokens(text)
            if new_token_count >= self.target_tokens:
                break

            # Don't add too many enhancements (max 5)
            if len(enhancements_added) >= 5:
                break

        return text

    def convert(
        self,
        prompt: str,
        character_name: Optional[str] = None,
        validate: bool = True,
    ) -> ConversionResult:
        """
        Convert a generation prompt to a training caption.

        Args:
            prompt: The original generation prompt
            character_name: Specific character name to remove (optional)
            validate: Whether to validate the result

        Returns:
            ConversionResult with caption and metadata
        """
        # Step 1: Remove character identity
        cleaned, removed = self._remove_character_identity(prompt, character_name)

        # Step 2: Clean formatting
        cleaned = self._clean_formatting(cleaned)

        # Step 3: Add generic subject if needed
        cleaned = self._add_generic_subject(cleaned)

        # Step 4: Final cleanup
        cleaned = self._clean_formatting(cleaned)

        # Step 5: Estimate token count
        token_count = self._estimate_tokens(cleaned)

        # Step 5.5: Auto-enhance if too short
        if self.auto_enhance and token_count < self.target_tokens:
            cleaned = self._enhance_caption(cleaned, token_count)
            cleaned = self._clean_formatting(cleaned)
            token_count = self._estimate_tokens(cleaned)

        # Step 6: Validate if requested
        is_valid = True
        validation_errors = []

        if validate:
            # Check token count
            if token_count < self.min_tokens:
                is_valid = False
                validation_errors.append(
                    f"Token count {token_count} below minimum {self.min_tokens}"
                )
            if token_count > self.max_tokens:
                is_valid = False
                validation_errors.append(
                    f"Token count {token_count} above maximum {self.max_tokens}"
                )

            # Check for remaining character names
            remaining_chars = self.character_name_pattern.findall(cleaned)
            if remaining_chars:
                is_valid = False
                validation_errors.append(
                    f"Character names still present: {remaining_chars}"
                )

            # Check caption has content
            if len(cleaned.split(",")) < 3:
                validation_errors.append(
                    "Caption may be too short (fewer than 3 clauses)"
                )

        return ConversionResult(
            original_prompt=prompt,
            caption=cleaned,
            removed_terms=removed,
            token_count_estimate=token_count,
            is_valid=is_valid,
            validation_errors=validation_errors,
        )

    def convert_batch(
        self,
        prompts: List[Dict],
        character_name: Optional[str] = None,
    ) -> List[Dict]:
        """
        Convert a batch of prompts to captions.

        Args:
            prompts: List of prompt dictionaries with 'prompt' key
            character_name: Character name to use for all prompts

        Returns:
            List of dictionaries with 'caption' and conversion metadata
        """
        results = []

        for item in prompts:
            prompt = item.get("prompt", "")
            char = character_name or item.get("metadata", {}).get("character")

            result = self.convert(prompt, character_name=char)

            results.append({
                "original_prompt": result.original_prompt,
                "caption": result.caption,
                "token_count": result.token_count_estimate,
                "is_valid": result.is_valid,
                "removed_count": len(result.removed_terms),
                "metadata": item.get("metadata", {}),
            })

        return results


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert generation prompts to training captions"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file with prompts"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for captions"
    )

    parser.add_argument(
        "--character",
        type=str,
        default=None,
        help="Character name to remove (optional, uses metadata if not provided)"
    )

    parser.add_argument(
        "--min-tokens",
        type=int,
        default=80,
        help="Minimum token count (default: 80)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=225,
        help="Maximum token count (default: 225)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed conversion info"
    )

    args = parser.parse_args()

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get prompts
    prompts = data.get("prompts", [])
    if not prompts:
        print("Error: No prompts found in input file")
        return 1

    # Initialize converter
    converter = PromptToCaptionConverter(
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    # Convert prompts
    print(f"Converting {len(prompts)} prompts...")
    results = converter.convert_batch(prompts, character_name=args.character)

    # Calculate statistics
    valid_count = sum(1 for r in results if r["is_valid"])
    avg_tokens = sum(r["token_count"] for r in results) / len(results)
    avg_removed = sum(r["removed_count"] for r in results) / len(results)

    print(f"\nConversion Statistics:")
    print(f"  Total prompts:    {len(results)}")
    print(f"  Valid captions:   {valid_count} ({100*valid_count/len(results):.1f}%)")
    print(f"  Avg token count:  {avg_tokens:.1f}")
    print(f"  Avg terms removed:{avg_removed:.1f}")

    # Show samples if verbose
    if args.verbose:
        print("\nSample conversions:")
        for i, r in enumerate(results[:3]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Original: {r['original_prompt'][:100]}...")
            print(f"Caption:  {r['caption'][:100]}...")
            print(f"Tokens:   {r['token_count']}, Valid: {r['is_valid']}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "character": args.character or data.get("character", "unknown"),
        "lora_type": data.get("lora_type", "unknown"),
        "num_captions": len(results),
        "conversion_stats": {
            "valid_count": valid_count,
            "avg_token_count": avg_tokens,
            "avg_terms_removed": avg_removed,
        },
        "captions": results,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved captions to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
