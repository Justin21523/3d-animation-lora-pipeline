#!/usr/bin/env python3
"""
Prompt Generator for Synthetic Data Generation
Generates diverse, high-quality prompts for character LoRA training using vocabulary YAMLs

This script combines expressions, poses, actions, and other attributes to create
varied training prompts that help improve LoRA generalization and quality.

Usage:
    # Generate 100 single-character prompts for a specific character
    python prompt_generator.py --character bryce --mode single --count 100 --output prompts/bryce_synthetic.txt

    # Generate multi-character interaction prompts
    python prompt_generator.py --characters bryce,caleb --mode interaction --count 50 --output prompts/interaction_prompts.txt

    # Use custom vocabulary directory
    python prompt_generator.py --vocab-dir custom_vocab/ --character elio --count 200

Author: AI Training Pipeline
Version: 1.0.0
"""

import argparse
import json
import random
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import yaml


# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class PromptMode(Enum):
    """Prompt generation modes"""
    SINGLE = "single"           # Single character only
    INTERACTION = "interaction" # Multiple characters interacting
    MIXED = "mixed"             # Mix of single and interaction prompts


@dataclass
class PromptComponent:
    """Represents a single component that can be used in prompt generation"""
    id: str
    name: str
    synonyms: List[str] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedPrompt:
    """A generated prompt with metadata"""
    prompt: str
    characters: List[str]
    components: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Vocabulary Loader
# ============================================================================

class VocabularyLoader:
    """Loads and manages vocabulary YAMLs for prompt generation"""

    def __init__(self, vocab_dir: Path):
        """
        Initialize vocabulary loader

        Args:
            vocab_dir: Directory containing vocabulary YAML files
        """
        self.vocab_dir = Path(vocab_dir)
        self.expressions: Dict[str, List[PromptComponent]] = {}
        self.poses: Dict[str, List[PromptComponent]] = {}
        self.actions: Dict[str, List[PromptComponent]] = {}
        self.camera_angles: List[PromptComponent] = []
        self.lighting: List[PromptComponent] = []
        self.quality_tags: List[str] = []

        self._load_all_vocabularies()

    def _load_all_vocabularies(self):
        """Load all vocabulary files"""
        logger.info(f"Loading vocabularies from {self.vocab_dir}")

        # Load expressions
        expr_file = self.vocab_dir / "expressions.yaml"
        if expr_file.exists():
            self._load_expressions(expr_file)
        else:
            logger.warning(f"Expressions file not found: {expr_file}")

        # Load poses
        pose_file = self.vocab_dir / "poses.yaml"
        if pose_file.exists():
            self._load_poses(pose_file)
        else:
            logger.warning(f"Poses file not found: {pose_file}")

        # Load actions
        action_file = self.vocab_dir / "actions.yaml"
        if action_file.exists():
            self._load_actions(action_file)
        else:
            logger.warning(f"Actions file not found: {action_file}")

        logger.info(f"Loaded {len(self.expressions)} expression categories")
        logger.info(f"Loaded {len(self.poses)} pose categories")
        logger.info(f"Loaded {len(self.actions)} action categories")

    def _load_expressions(self, file_path: Path):
        """Load expression vocabulary"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Load basic emotions
        self.expressions['basic'] = self._parse_components(data.get('basic_emotions', []))

        # Load complex emotions
        self.expressions['complex'] = self._parse_components(data.get('complex_emotions', []))

        # Load neutral states
        self.expressions['neutral'] = self._parse_components(data.get('neutral_states', []))

        # Load quality tags
        self.quality_tags = data.get('quality_tags', [])

        logger.info(f"Loaded {sum(len(v) for v in self.expressions.values())} expressions")

    def _load_poses(self, file_path: Path):
        """Load pose vocabulary"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Load body poses
        body_poses = data.get('body_poses', {})
        for category, poses in body_poses.items():
            self.poses[category] = self._parse_components(poses)

        # Load camera angles (if present)
        camera = data.get('camera_angles', {})
        if camera:
            # Combine all camera angle categories
            all_angles = []
            for category, angles in camera.items():
                all_angles.extend(self._parse_components(angles))
            self.camera_angles = all_angles

        logger.info(f"Loaded {sum(len(v) for v in self.poses.values())} poses")
        logger.info(f"Loaded {len(self.camera_angles)} camera angles")

    def _load_actions(self, file_path: Path):
        """Load action vocabulary"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Load all action categories
        for category, subcategories in data.items():
            if category in ['version', 'description']:
                continue

            # Handle nested structure
            if isinstance(subcategories, dict):
                for subcat, actions in subcategories.items():
                    key = f"{category}_{subcat}"
                    self.actions[key] = self._parse_components(actions)
            else:
                self.actions[category] = self._parse_components(subcategories)

        logger.info(f"Loaded {sum(len(v) for v in self.actions.values())} actions")

    def _parse_components(self, components: List[Dict]) -> List[PromptComponent]:
        """Parse component dictionaries into PromptComponent objects"""
        result = []
        # Handle case where components is not a list
        if not isinstance(components, list):
            return result

        for comp in components:
            if isinstance(comp, dict):
                result.append(PromptComponent(
                    id=comp.get('id', ''),
                    name=comp.get('name', ''),
                    synonyms=comp.get('synonyms', []),
                    description=comp.get('description', ''),
                    metadata={k: v for k, v in comp.items()
                             if k not in ['id', 'name', 'synonyms', 'description']}
                ))
        return result

    def get_random_expression(self, category: Optional[str] = None) -> PromptComponent:
        """Get a random expression, optionally from a specific category"""
        if category and category in self.expressions:
            return random.choice(self.expressions[category])
        else:
            # Choose from all expressions
            all_expressions = []
            for category_list in self.expressions.values():
                all_expressions.extend(category_list)
            return random.choice(all_expressions)

    def get_random_pose(self, category: Optional[str] = None) -> PromptComponent:
        """Get a random pose, optionally from a specific category"""
        if category and category in self.poses:
            return random.choice(self.poses[category])
        else:
            # Choose from all poses
            all_poses = []
            for category_list in self.poses.values():
                all_poses.extend(category_list)
            return random.choice(all_poses)

    def get_random_action(self, category: Optional[str] = None) -> PromptComponent:
        """Get a random action, optionally from a specific category"""
        if category and category in self.actions:
            return random.choice(self.actions[category])
        else:
            # Choose from all actions
            all_actions = []
            for category_list in self.actions.values():
                all_actions.extend(category_list)
            return random.choice(all_actions)

    def get_random_camera_angle(self) -> Optional[PromptComponent]:
        """Get a random camera angle"""
        if self.camera_angles:
            return random.choice(self.camera_angles)
        return None

    def get_random_quality_tags(self, count: int = 3) -> List[str]:
        """Get random quality tags"""
        if len(self.quality_tags) <= count:
            return self.quality_tags
        return random.sample(self.quality_tags, count)


# ============================================================================
# Prompt Generator
# ============================================================================

class PromptGenerator:
    """Generates diverse prompts using vocabulary components"""

    def __init__(self, vocab_loader: VocabularyLoader, seed: Optional[int] = None):
        """
        Initialize prompt generator

        Args:
            vocab_loader: VocabularyLoader instance
            seed: Random seed for reproducibility
        """
        self.vocab = vocab_loader

        if seed is not None:
            random.seed(seed)
            logger.info(f"Set random seed to {seed}")

    def generate_single_character_prompts(
        self,
        character_token: str,
        count: int,
        include_expressions: bool = True,
        include_poses: bool = True,
        include_actions: bool = True,
        include_camera: bool = True
    ) -> List[GeneratedPrompt]:
        """
        Generate single-character prompts

        Args:
            character_token: Character token (e.g., "bryce", "caleb")
            count: Number of prompts to generate
            include_expressions: Include expression variations
            include_poses: Include pose variations
            include_actions: Include action variations
            include_camera: Include camera angle variations

        Returns:
            List of GeneratedPrompt objects
        """
        prompts = []

        for i in range(count):
            components = {}
            prompt_parts = [character_token]

            # Add expression (70% chance)
            if include_expressions and random.random() < 0.7:
                expr = self.vocab.get_random_expression()
                # Use synonym sometimes for variety
                expr_text = random.choice([expr.name] + expr.synonyms) if expr.synonyms else expr.name
                components['expression'] = expr.id
                prompt_parts.append(f"{expr_text} expression")

            # Add action (50% chance)
            if include_actions and random.random() < 0.5:
                action = self.vocab.get_random_action()
                action_text = random.choice([action.name] + action.synonyms) if action.synonyms else action.name
                components['action'] = action.id
                prompt_parts.append(action_text)

            # Add pose (60% chance, if no action was added)
            elif include_poses and random.random() < 0.6:
                pose = self.vocab.get_random_pose()
                pose_text = random.choice([pose.name] + pose.synonyms) if pose.synonyms else pose.name
                components['pose'] = pose.id
                prompt_parts.append(pose_text)

            # Add camera angle (50% chance)
            if include_camera and random.random() < 0.5:
                camera = self.vocab.get_random_camera_angle()
                if camera:
                    camera_text = random.choice([camera.name] + camera.synonyms) if camera.synonyms else camera.name
                    components['camera'] = camera.id
                    prompt_parts.append(camera_text)

            # Add quality tags (always)
            quality_tags = self.vocab.get_random_quality_tags(count=2)
            prompt_parts.extend(quality_tags)

            # Assemble final prompt
            prompt_text = ", ".join(prompt_parts)

            prompts.append(GeneratedPrompt(
                prompt=prompt_text,
                characters=[character_token],
                components=components,
                metadata={'index': i, 'mode': 'single'}
            ))

        logger.info(f"Generated {len(prompts)} single-character prompts for '{character_token}'")
        return prompts

    def generate_multi_character_prompts(
        self,
        character_tokens: List[str],
        count: int
    ) -> List[GeneratedPrompt]:
        """
        Generate multi-character interaction prompts

        Args:
            character_tokens: List of character tokens (e.g., ["bryce", "caleb"])
            count: Number of prompts to generate

        Returns:
            List of GeneratedPrompt objects
        """
        if len(character_tokens) < 2:
            raise ValueError("Need at least 2 characters for interaction prompts")

        prompts = []

        # Interaction templates
        interaction_templates = [
            "{char1} and {char2} standing together",
            "{char1} talking to {char2}",
            "{char1} and {char2} {action}",
            "{char1} looking at {char2}",
            "{char1} with {char2}, both {expression}",
            "{char1} and {char2} in a scene together",
        ]

        for i in range(count):
            components = {}

            # Pick 2 random characters
            selected_chars = random.sample(character_tokens, min(2, len(character_tokens)))
            char1, char2 = selected_chars[0], selected_chars[1]

            # Pick interaction template
            template = random.choice(interaction_templates)

            # Fill in template
            template_vars = {
                'char1': char1,
                'char2': char2
            }

            # Add optional expression
            if '{expression}' in template:
                expr = self.vocab.get_random_expression()
                expr_text = random.choice([expr.name] + expr.synonyms) if expr.synonyms else expr.name
                template_vars['expression'] = expr_text
                components['expression'] = expr.id

            # Add optional action
            if '{action}' in template:
                action = self.vocab.get_random_action()
                action_text = random.choice([action.name] + action.synonyms) if action.synonyms else action.name
                template_vars['action'] = action_text
                components['action'] = action.id

            prompt_text = template.format(**template_vars)

            # Add quality tags
            quality_tags = self.vocab.get_random_quality_tags(count=2)
            prompt_text = f"{prompt_text}, {', '.join(quality_tags)}"

            prompts.append(GeneratedPrompt(
                prompt=prompt_text,
                characters=selected_chars,
                components=components,
                metadata={'index': i, 'mode': 'interaction', 'template': template}
            ))

        logger.info(f"Generated {len(prompts)} multi-character interaction prompts")
        return prompts


# ============================================================================
# Output Writers
# ============================================================================

def write_prompts_txt(prompts: List[GeneratedPrompt], output_path: Path):
    """Write prompts to a plain text file (one per line)"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in prompts:
            f.write(p.prompt + '\n')
    logger.info(f"Wrote {len(prompts)} prompts to {output_path}")


def write_prompts_json(prompts: List[GeneratedPrompt], output_path: Path):
    """Write prompts with full metadata to JSON"""
    data = []
    for p in prompts:
        data.append({
            'prompt': p.prompt,
            'characters': p.characters,
            'components': p.components,
            'metadata': p.metadata
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(prompts)} prompts with metadata to {output_path}")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse training prompts for character LoRA synthetic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 single-character prompts
  %(prog)s --character bryce --count 100 --output prompts/bryce_synthetic.txt

  # Generate multi-character interaction prompts
  %(prog)s --characters bryce,caleb,elio --mode interaction --count 50

  # Generate mixed prompts with custom vocabulary
  %(prog)s --vocab-dir custom_vocab/ --character elio --mode mixed --count 200
        """
    )

    # Input arguments
    parser.add_argument(
        '--vocab-dir',
        type=str,
        default='prompts/generation/vocabulary',
        help='Directory containing vocabulary YAML files (default: prompts/generation/vocabulary)'
    )

    parser.add_argument(
        '--character',
        type=str,
        help='Single character token for single-character prompts (e.g., "bryce")'
    )

    parser.add_argument(
        '--characters',
        type=str,
        help='Comma-separated character tokens for multi-character prompts (e.g., "bryce,caleb,elio")'
    )

    # Generation parameters
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'interaction', 'mixed'],
        default='single',
        help='Prompt generation mode (default: single)'
    )

    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Number of prompts to generate (default: 100)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )

    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path (.txt or .json)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['txt', 'json', 'auto'],
        default='auto',
        help='Output format (default: auto-detect from extension)'
    )

    # Feature toggles
    parser.add_argument(
        '--no-expressions',
        action='store_true',
        help='Disable expression variations'
    )

    parser.add_argument(
        '--no-poses',
        action='store_true',
        help='Disable pose variations'
    )

    parser.add_argument(
        '--no-actions',
        action='store_true',
        help='Disable action variations'
    )

    parser.add_argument(
        '--no-camera',
        action='store_true',
        help='Disable camera angle variations'
    )

    args = parser.parse_args()

    # Validate inputs
    if args.mode == 'single' and not args.character:
        parser.error("--character is required for single-character mode")

    if args.mode in ['interaction', 'mixed'] and not args.characters:
        parser.error("--characters is required for interaction/mixed mode")

    # Parse character tokens
    if args.characters:
        character_tokens = [c.strip() for c in args.characters.split(',')]
    elif args.character:
        character_tokens = [args.character]
    else:
        parser.error("Either --character or --characters must be specified")

    # Determine output format
    output_path = Path(args.output)
    if args.format == 'auto':
        output_format = output_path.suffix[1:] if output_path.suffix else 'txt'
    else:
        output_format = args.format

    # Load vocabularies
    vocab_dir = Path(args.vocab_dir)
    if not vocab_dir.exists():
        logger.error(f"Vocabulary directory not found: {vocab_dir}")
        sys.exit(1)

    vocab_loader = VocabularyLoader(vocab_dir)

    # Create generator
    generator = PromptGenerator(vocab_loader, seed=args.seed)

    # Generate prompts
    all_prompts = []

    if args.mode == 'single':
        prompts = generator.generate_single_character_prompts(
            character_token=character_tokens[0],
            count=args.count,
            include_expressions=not args.no_expressions,
            include_poses=not args.no_poses,
            include_actions=not args.no_actions,
            include_camera=not args.no_camera
        )
        all_prompts.extend(prompts)

    elif args.mode == 'interaction':
        if len(character_tokens) < 2:
            logger.error("Need at least 2 characters for interaction mode")
            sys.exit(1)

        prompts = generator.generate_multi_character_prompts(
            character_tokens=character_tokens,
            count=args.count
        )
        all_prompts.extend(prompts)

    elif args.mode == 'mixed':
        # Generate 70% single, 30% interaction
        single_count = int(args.count * 0.7)
        interaction_count = args.count - single_count

        # Generate single-character prompts for each character
        for char in character_tokens:
            char_count = single_count // len(character_tokens)
            prompts = generator.generate_single_character_prompts(
                character_token=char,
                count=char_count,
                include_expressions=not args.no_expressions,
                include_poses=not args.no_poses,
                include_actions=not args.no_actions,
                include_camera=not args.no_camera
            )
            all_prompts.extend(prompts)

        # Generate interaction prompts
        if len(character_tokens) >= 2:
            prompts = generator.generate_multi_character_prompts(
                character_tokens=character_tokens,
                count=interaction_count
            )
            all_prompts.extend(prompts)

    # Shuffle prompts
    random.shuffle(all_prompts)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == 'json':
        write_prompts_json(all_prompts, output_path)
    else:
        write_prompts_txt(all_prompts, output_path)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Characters: {', '.join(character_tokens)}")
    logger.info(f"Total prompts: {len(all_prompts)}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Format: {output_format}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
