#!/usr/bin/env python3
"""
Prompt Loader Utility

Loads character-specific prompts from JSON files with template variable expansion.
Supports:
- {{base_positive}} and {{base_negative}} variable substitution
- Category-based prompt selection
- Negative prompt composition
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from random import sample, seed


class PromptLoader:
    """Load and process character prompts"""

    def __init__(self, prompt_file: Path):
        """Load prompts from JSON file"""
        self.prompt_file = Path(prompt_file)

        if not self.prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.base_positive = self.data.get('base_positive', '')
        self.base_negative = self.data.get('base_negative', '')
        self.advanced_negatives = self.data.get('advanced_negative_prompts', {})
        self.quality_tags = self.data.get('quality_tags', [])

    def expand_template(self, text: str) -> str:
        """Expand template variables like {{base_negative}}"""

        # Replace {{base_positive}}
        text = text.replace('{{base_positive}}', self.base_positive)

        # Replace {{base_negative}}
        text = text.replace('{{base_negative}}', self.base_negative)

        # Replace advanced negative categories
        for key, value in self.advanced_negatives.items():
            placeholder = f"{{{{{key}}}}}"
            text = text.replace(placeholder, value)

        return text

    def get_all_prompts(self) -> List[Tuple[str, str]]:
        """Get all prompts as (positive, negative) pairs"""

        all_prompts = []
        test_prompts = self.data.get('test_prompts', [])

        for category_data in test_prompts:
            for prompt_data in category_data.get('prompts', []):
                positive = self.expand_template(prompt_data.get('positive', ''))
                negative = self.expand_template(prompt_data.get('negative', self.base_negative))
                all_prompts.append((positive, negative))

        return all_prompts

    def get_prompts_by_category(self, category: str) -> List[Tuple[str, str]]:
        """Get prompts for a specific category"""

        prompts = []
        test_prompts = self.data.get('test_prompts', [])

        for category_data in test_prompts:
            if category_data.get('category') == category:
                for prompt_data in category_data.get('prompts', []):
                    positive = self.expand_template(prompt_data.get('positive', ''))
                    negative = self.expand_template(prompt_data.get('negative', self.base_negative))
                    prompts.append((positive, negative))

        return prompts

    def get_categories(self) -> List[str]:
        """Get list of available categories"""

        test_prompts = self.data.get('test_prompts', [])
        return [cat.get('category') for cat in test_prompts]

    def get_balanced_sample(
        self,
        num_prompts: int = 12,
        per_category: int = 2,
        random_seed: int = None
    ) -> List[Tuple[str, str]]:
        """
        Get balanced sample across categories

        Args:
            num_prompts: Total number of prompts to sample
            per_category: Number of prompts per category (if possible)
            random_seed: Random seed for reproducibility

        Returns:
            List of (positive, negative) tuples
        """

        if random_seed is not None:
            seed(random_seed)

        categories = self.get_categories()
        sampled_prompts = []

        # Sample per_category from each category
        for category in categories:
            cat_prompts = self.get_prompts_by_category(category)
            if cat_prompts:
                num_to_sample = min(per_category, len(cat_prompts))
                sampled_prompts.extend(sample(cat_prompts, num_to_sample))

        # If we need more, sample randomly from all
        if len(sampled_prompts) < num_prompts:
            all_prompts = self.get_all_prompts()
            remaining = num_prompts - len(sampled_prompts)
            additional = sample(all_prompts, min(remaining, len(all_prompts)))
            sampled_prompts.extend(additional)

        # If we have too many, trim
        if len(sampled_prompts) > num_prompts:
            sampled_prompts = sample(sampled_prompts, num_prompts)

        return sampled_prompts

    def get_simple_test_prompts(self, num_prompts: int = 5) -> List[str]:
        """
        Get simple positive-only prompts for quick testing
        (For compatibility with existing code expecting List[str])
        """

        prompts = self.get_balanced_sample(num_prompts=num_prompts)
        return [p[0] for p in prompts]  # Return only positive prompts

    def get_comprehensive_negative(self) -> str:
        """Get comprehensive negative prompt combining all categories"""

        all_negatives = [self.base_negative]
        all_negatives.extend(self.advanced_negatives.values())

        return ', '.join(all_negatives)

    def add_quality_tags(self, positive_prompt: str) -> str:
        """Add quality tags to positive prompt"""

        quality_str = ', '.join(self.quality_tags)
        return f"{positive_prompt}, {quality_str}"

    def get_metadata(self) -> Dict:
        """Get prompt metadata"""

        return {
            'film': self.data.get('film', ''),
            'character': self.data.get('character', ''),
            'description': self.data.get('description', ''),
            'version': self.data.get('metadata', {}).get('version', ''),
            'num_categories': len(self.get_categories()),
            'total_prompts': len(self.get_all_prompts())
        }


def load_character_prompts(character: str, film: str = 'luca') -> PromptLoader:
    """
    Convenience function to load character prompts

    Args:
        character: Character name (e.g., 'luca_human', 'alberto_human')
        film: Film name (default: 'luca')

    Returns:
        PromptLoader instance

    Example:
        >>> loader = load_character_prompts('luca_human')
        >>> prompts = loader.get_balanced_sample(num_prompts=12)
    """

    # Find prompt file
    base_dir = Path(__file__).parent.parent.parent.parent
    prompt_file = base_dir / 'prompts' / film / f'{character}_prompts.json'

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}\n"
            f"Available prompts should be in: prompts/{film}/"
        )

    return PromptLoader(prompt_file)


if __name__ == '__main__':
    # Test loading
    import sys

    if len(sys.argv) > 1:
        character = sys.argv[1]
    else:
        character = 'luca_human'

    print(f"Loading prompts for: {character}")
    print("="*70)

    try:
        loader = load_character_prompts(character)

        # Print metadata
        metadata = loader.get_metadata()
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        # Print categories
        print(f"\nCategories:")
        for cat in loader.get_categories():
            num_prompts = len(loader.get_prompts_by_category(cat))
            print(f"  - {cat}: {num_prompts} prompts")

        # Sample prompts
        print(f"\nSample Balanced Prompts (3 prompts):")
        sampled = loader.get_balanced_sample(num_prompts=3, random_seed=42)

        for i, (pos, neg) in enumerate(sampled, 1):
            print(f"\n--- Prompt {i} ---")
            print(f"Positive: {pos[:100]}...")
            print(f"Negative: {neg[:100]}...")

        # Test quality tags
        print(f"\n\nQuality Tags:")
        print(f"  {', '.join(loader.quality_tags)}")

        # Test comprehensive negative
        print(f"\n\nComprehensive Negative Prompt:")
        comp_neg = loader.get_comprehensive_negative()
        print(f"  {comp_neg[:200]}...")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
