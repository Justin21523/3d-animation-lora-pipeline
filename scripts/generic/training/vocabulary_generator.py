#!/usr/bin/env python3
"""
Vocabulary & Prompt Generation System

Generates diverse, structured prompts for synthetic data generation by:
1. Building comprehensive vocabulary from character descriptions
2. Generating combinatorial prompts with controlled variation
3. Supporting multi-dimensional diversity (appearance, pose, camera, lighting)
4. Ensuring quality and coherence in generated prompts

Part of Module 1: Vocabulary & Prompt Generation System
Author: LLMProvider Tooling
Date: 2025-11-30
"""

import random
import itertools
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class VocabularyConfig:
    """Configuration for vocabulary generation"""
    # Diversity settings
    min_variations_per_dimension: int = 3
    max_variations_per_dimension: int = 10
    
    # Prompt structure
    include_quality_tags: bool = True
    include_style_prefix: bool = True
    style_prefix: str = "pixar style 3d animation"
    
    # Quality tags
    quality_tags: List[str] = None
    
    # Filtering
    min_prompt_length: int = 20
    max_prompt_length: int = 150
    
    def __post_init__(self):
        if self.quality_tags is None:
            self.quality_tags = [
                "high quality",
                "detailed",
                "professional",
                "sharp focus",
                "8k resolution"
            ]


class VocabularyGenerator:
    """
    Generates diverse vocabulary and prompts for synthetic data generation
    
    Features:
    - Multi-dimensional variation (appearance, pose, camera, lighting)
    - Combinatorial prompt generation with controlled diversity
    - Quality-aware prompt construction
    - Deduplication and validation
    """
    
    def __init__(self, config: Optional[VocabularyConfig] = None):
        """
        Initialize vocabulary generator
        
        Args:
            config: Vocabulary configuration (uses defaults if None)
        """
        self.config = config or VocabularyConfig()
        
        # Default vocabulary templates
        self.default_poses = [
            "standing upright",
            "walking forward",
            "running dynamically",
            "sitting relaxed",
            "jumping energetically",
            "kneeling down",
            "leaning casually",
            "crouching low"
        ]
        
        self.default_camera_angles = [
            "front view",
            "side profile",
            "three-quarter view",
            "back view",
            "close-up portrait",
            "medium shot",
            "full body shot",
            "low angle",
            "high angle",
            "eye level"
        ]
        
        self.default_lighting = [
            "natural sunlight",
            "studio lighting",
            "dramatic rim light",
            "soft diffused light",
            "golden hour lighting",
            "overcast daylight",
            "warm indoor lighting",
            "cool blue lighting",
            "backlit silhouette",
            "three-point lighting"
        ]
        
        self.default_backgrounds = [
            "simple white background",
            "neutral gray background",
            "gradient background",
            "outdoor natural environment",
            "indoor studio setting",
            "urban city background",
            "abstract colorful background",
            "blurred bokeh background"
        ]
        
        self.default_expressions = [
            "neutral expression",
            "happy smile",
            "slight grin",
            "focused look",
            "surprised expression",
            "thoughtful gaze",
            "confident smirk",
            "gentle smile"
        ]
    
    def generate_character_vocabulary(
        self,
        character_name: str,
        character_description: str,
        appearance_variations: Optional[List[str]] = None,
        pose_variations: Optional[List[str]] = None,
        camera_angles: Optional[List[str]] = None,
        lighting_conditions: Optional[List[str]] = None,
        backgrounds: Optional[List[str]] = None,
        expressions: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate comprehensive vocabulary for a character
        
        Args:
            character_name: Name/identifier for the character
            character_description: Base description of the character
            appearance_variations: List of appearance variations (optional)
            pose_variations: List of poses (uses defaults if None)
            camera_angles: List of camera angles (uses defaults if None)
            lighting_conditions: List of lighting conditions (uses defaults if None)
            backgrounds: List of backgrounds (uses defaults if None)
            expressions: List of expressions (uses defaults if None)
            **kwargs: Additional custom variation dimensions
        
        Returns:
            Dictionary containing complete vocabulary structure
        """
        # Use provided variations or defaults
        poses = pose_variations or self.default_poses
        cameras = camera_angles or self.default_camera_angles
        lighting = lighting_conditions or self.default_lighting
        bgs = backgrounds or self.default_backgrounds
        exprs = expressions or self.default_expressions
        
        # Build vocabulary structure
        vocabulary = {
            'character_name': character_name,
            'base_description': character_description,
            'appearance_variations': appearance_variations or [],
            'poses': poses,
            'camera_angles': cameras,
            'lighting': lighting,
            'backgrounds': bgs,
            'expressions': exprs,
            'custom_dimensions': kwargs,
            'config': asdict(self.config)
        }
        
        # Validate vocabulary
        self._validate_vocabulary(vocabulary)
        
        return vocabulary
    
    def _validate_vocabulary(self, vocabulary: Dict[str, Any]):
        """
        Validate vocabulary meets minimum requirements
        
        Args:
            vocabulary: Vocabulary dictionary to validate
        
        Raises:
            ValueError: If vocabulary doesn't meet requirements
        """
        min_vars = self.config.min_variations_per_dimension
        
        dimensions = {
            'poses': vocabulary['poses'],
            'camera_angles': vocabulary['camera_angles'],
            'lighting': vocabulary['lighting']
        }
        
        for dim_name, dim_values in dimensions.items():
            if len(dim_values) < min_vars:
                raise ValueError(
                    f"Dimension '{dim_name}' has only {len(dim_values)} variations, "
                    f"minimum required is {min_vars}"
                )
    
    def generate_prompts_batch(
        self,
        vocabulary: Dict[str, Any],
        num_prompts: int,
        diversity_mode: str = 'balanced',
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Generate batch of diverse prompts from vocabulary
        
        Args:
            vocabulary: Vocabulary dictionary from generate_character_vocabulary()
            num_prompts: Number of prompts to generate
            diversity_mode: How to sample variations
                - 'random': Fully random sampling
                - 'balanced': Ensure even coverage across dimensions
                - 'grid': Systematic grid sampling
            seed: Random seed for reproducibility
        
        Returns:
            List of generated prompts
        """
        if seed is not None:
            random.seed(seed)
        
        if diversity_mode == 'grid':
            prompts = self._generate_grid_prompts(vocabulary, num_prompts)
        elif diversity_mode == 'balanced':
            prompts = self._generate_balanced_prompts(vocabulary, num_prompts)
        else:
            prompts = self._generate_random_prompts(vocabulary, num_prompts)
        
        # Deduplicate
        prompts = list(dict.fromkeys(prompts))
        
        # Ensure we have enough prompts
        if len(prompts) < num_prompts:
            # Generate more random prompts to fill
            additional = self._generate_random_prompts(
                vocabulary,
                num_prompts - len(prompts)
            )
            prompts.extend(additional)
            prompts = list(dict.fromkeys(prompts))[:num_prompts]
        
        return prompts[:num_prompts]
    
    def _generate_grid_prompts(
        self,
        vocabulary: Dict[str, Any],
        num_prompts: int
    ) -> List[str]:
        """
        Generate prompts using systematic grid sampling
        
        Ensures maximum coverage across all variation dimensions
        """
        prompts = []
        
        # Get all variation dimensions
        poses = vocabulary['poses']
        cameras = vocabulary['camera_angles']
        lighting = vocabulary['lighting']
        backgrounds = vocabulary.get('backgrounds', self.default_backgrounds)
        expressions = vocabulary.get('expressions', self.default_expressions)
        
        # Create grid combinations
        dimensions = [poses, cameras, lighting, backgrounds, expressions]
        
        # Calculate how many samples per dimension
        samples_per_dim = max(1, int(num_prompts ** (1/len(dimensions))))
        
        # Sample evenly from each dimension
        sampled_dims = [
            random.sample(dim, min(samples_per_dim, len(dim)))
            for dim in dimensions
        ]
        
        # Generate cartesian product
        for combo in itertools.product(*sampled_dims):
            if len(prompts) >= num_prompts:
                break
            
            pose, camera, light, bg, expr = combo
            prompt = self._construct_prompt(
                vocabulary,
                pose=pose,
                camera=camera,
                lighting=light,
                background=bg,
                expression=expr
            )
            prompts.append(prompt)
        
        # Fill remaining with random if needed
        while len(prompts) < num_prompts:
            prompts.append(self._generate_single_random_prompt(vocabulary))
        
        return prompts
    
    def _generate_balanced_prompts(
        self,
        vocabulary: Dict[str, Any],
        num_prompts: int
    ) -> List[str]:
        """
        Generate prompts with balanced coverage across dimensions
        
        Ensures each variation appears roughly equally
        """
        prompts = []
        
        # Track usage counts for balancing
        usage_counts = {
            'poses': {p: 0 for p in vocabulary['poses']},
            'cameras': {c: 0 for c in vocabulary['camera_angles']},
            'lighting': {l: 0 for l in vocabulary['lighting']},
            'backgrounds': {b: 0 for b in vocabulary.get('backgrounds', self.default_backgrounds)},
            'expressions': {e: 0 for e in vocabulary.get('expressions', self.default_expressions)}
        }
        
        for _ in range(num_prompts):
            # Sample least-used variations
            pose = min(usage_counts['poses'].items(), key=lambda x: (x[1], random.random()))[0]
            camera = min(usage_counts['cameras'].items(), key=lambda x: (x[1], random.random()))[0]
            light = min(usage_counts['lighting'].items(), key=lambda x: (x[1], random.random()))[0]
            bg = min(usage_counts['backgrounds'].items(), key=lambda x: (x[1], random.random()))[0]
            expr = min(usage_counts['expressions'].items(), key=lambda x: (x[1], random.random()))[0]
            
            # Update counts
            usage_counts['poses'][pose] += 1
            usage_counts['cameras'][camera] += 1
            usage_counts['lighting'][light] += 1
            usage_counts['backgrounds'][bg] += 1
            usage_counts['expressions'][expr] += 1
            
            # Generate prompt
            prompt = self._construct_prompt(
                vocabulary,
                pose=pose,
                camera=camera,
                lighting=light,
                background=bg,
                expression=expr
            )
            prompts.append(prompt)
        
        return prompts
    
    def _generate_random_prompts(
        self,
        vocabulary: Dict[str, Any],
        num_prompts: int
    ) -> List[str]:
        """Generate completely random prompts"""
        return [
            self._generate_single_random_prompt(vocabulary)
            for _ in range(num_prompts)
        ]
    
    def _generate_single_random_prompt(
        self,
        vocabulary: Dict[str, Any]
    ) -> str:
        """Generate a single random prompt"""
        pose = random.choice(vocabulary['poses'])
        camera = random.choice(vocabulary['camera_angles'])
        light = random.choice(vocabulary['lighting'])
        bg = random.choice(vocabulary.get('backgrounds', self.default_backgrounds))
        expr = random.choice(vocabulary.get('expressions', self.default_expressions))
        
        return self._construct_prompt(
            vocabulary,
            pose=pose,
            camera=camera,
            lighting=light,
            background=bg,
            expression=expr
        )
    
    def _construct_prompt(
        self,
        vocabulary: Dict[str, Any],
        pose: str,
        camera: str,
        lighting: str,
        background: str,
        expression: str
    ) -> str:
        """
        Construct final prompt from components
        
        Format:
        [character_description], [appearance_var], [expression], [pose], 
        [camera], [lighting], [background], [style], [quality_tags]
        """
        components = []
        
        # Base description
        components.append(vocabulary['base_description'])
        
        # Appearance variation (if any)
        if vocabulary.get('appearance_variations'):
            app_var = random.choice(vocabulary['appearance_variations'])
            components.append(app_var)
        
        # Expression
        components.append(expression)
        
        # Pose
        components.append(pose)
        
        # Camera
        components.append(camera)
        
        # Lighting
        components.append(lighting)
        
        # Background
        components.append(background)
        
        # Style prefix
        if self.config.include_style_prefix:
            components.append(self.config.style_prefix)
        
        # Quality tags (sample 2-3 randomly)
        if self.config.include_quality_tags:
            num_tags = random.randint(2, 3)
            tags = random.sample(self.config.quality_tags, min(num_tags, len(self.config.quality_tags)))
            components.extend(tags)
        
        # Join with commas
        prompt = ", ".join(components)
        
        # Validate length
        if len(prompt) < self.config.min_prompt_length:
            # Add more quality tags if too short
            remaining_tags = [t for t in self.config.quality_tags if t not in components]
            if remaining_tags:
                prompt += ", " + random.choice(remaining_tags)
        
        if len(prompt) > self.config.max_prompt_length:
            # Truncate if too long
            prompt = prompt[:self.config.max_prompt_length - 3] + "..."
        
        return prompt
    
    def save_vocabulary(
        self,
        vocabulary: Dict[str, Any],
        output_path: Path
    ):
        """
        Save vocabulary to JSON file
        
        Args:
            vocabulary: Vocabulary dictionary
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(vocabulary, f, indent=2)
    
    def load_vocabulary(
        self,
        vocabulary_path: Path
    ) -> Dict[str, Any]:
        """
        Load vocabulary from JSON file
        
        Args:
            vocabulary_path: Path to vocabulary JSON file
        
        Returns:
            Vocabulary dictionary
        """
        with open(vocabulary_path, 'r') as f:
            vocabulary = json.load(f)
        
        return vocabulary


def main():
    """CLI for vocabulary generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vocabulary & Prompt Generation System")
    
    # Required arguments
    parser.add_argument("--character-name", type=str, required=True, help="Character name/identifier")
    parser.add_argument("--character-description", type=str, required=True, help="Base character description")
    parser.add_argument("--num-prompts", type=int, required=True, help="Number of prompts to generate")
    
    # Optional vocabulary dimensions
    parser.add_argument("--poses", type=str, nargs='+', help="Custom pose variations")
    parser.add_argument("--cameras", type=str, nargs='+', help="Custom camera angles")
    parser.add_argument("--lighting", type=str, nargs='+', help="Custom lighting conditions")
    parser.add_argument("--backgrounds", type=str, nargs='+', help="Custom backgrounds")
    parser.add_argument("--expressions", type=str, nargs='+', help="Custom expressions")
    parser.add_argument("--appearance", type=str, nargs='+', help="Appearance variations")
    
    # Generation settings
    parser.add_argument("--diversity-mode", type=str, default="balanced",
                       choices=['random', 'balanced', 'grid'],
                       help="Prompt diversity mode (default: balanced)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    # Output
    parser.add_argument("--output-vocab", type=str, help="Path to save vocabulary JSON")
    parser.add_argument("--output-prompts", type=str, help="Path to save prompts JSON")
    
    # Configuration
    parser.add_argument("--style-prefix", type=str, default="pixar style 3d animation",
                       help="Style prefix for prompts")
    parser.add_argument("--no-quality-tags", action="store_true",
                       help="Disable quality tags in prompts")
    
    args = parser.parse_args()
    
    # Create config
    config = VocabularyConfig(
        style_prefix=args.style_prefix,
        include_quality_tags=not args.no_quality_tags
    )
    
    # Initialize generator
    generator = VocabularyGenerator(config=config)
    
    # Generate vocabulary
    print(f"\n{'='*80}")
    print(f"📝 VOCABULARY GENERATION")
    print(f"{'='*80}")
    print(f"Character: {args.character_name}")
    print(f"Description: {args.character_description}")
    print(f"{'='*80}\n")
    
    vocabulary = generator.generate_character_vocabulary(
        character_name=args.character_name,
        character_description=args.character_description,
        appearance_variations=args.appearance,
        pose_variations=args.poses,
        camera_angles=args.cameras,
        lighting_conditions=args.lighting,
        backgrounds=args.backgrounds,
        expressions=args.expressions
    )
    
    # Save vocabulary if requested
    if args.output_vocab:
        generator.save_vocabulary(vocabulary, Path(args.output_vocab))
        print(f"✅ Vocabulary saved to {args.output_vocab}")
    
    # Generate prompts
    print(f"\n{'='*80}")
    print(f"🎲 PROMPT GENERATION")
    print(f"{'='*80}")
    print(f"Mode: {args.diversity_mode}")
    print(f"Count: {args.num_prompts}")
    if args.seed:
        print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")
    
    prompts = generator.generate_prompts_batch(
        vocabulary=vocabulary,
        num_prompts=args.num_prompts,
        diversity_mode=args.diversity_mode,
        seed=args.seed
    )
    
    # Display sample prompts
    print("Sample prompts:")
    for i, prompt in enumerate(prompts[:5], 1):
        print(f"\n{i}. {prompt}")
    
    if len(prompts) > 5:
        print(f"\n... and {len(prompts) - 5} more prompts")
    
    # Save prompts if requested
    if args.output_prompts:
        with open(args.output_prompts, 'w') as f:
            json.dump(prompts, f, indent=2)
        print(f"\n✅ Prompts saved to {args.output_prompts}")
    
    print(f"\n{'='*80}")
    print(f"✅ GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Unique prompts: {len(set(prompts))}")
    print(f"Vocabulary dimensions:")
    print(f"  - Poses: {len(vocabulary['poses'])}")
    print(f"  - Cameras: {len(vocabulary['camera_angles'])}")
    print(f"  - Lighting: {len(vocabulary['lighting'])}")
    print(f"  - Backgrounds: {len(vocabulary['backgrounds'])}")
    print(f"  - Expressions: {len(vocabulary['expressions'])}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
