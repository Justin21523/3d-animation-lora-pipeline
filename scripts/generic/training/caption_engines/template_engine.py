"""
Template-based caption generation engine.

Uses predefined templates and simple heuristics for fast, deterministic
caption generation without requiring VLM models.

Useful for:
- Quick prototyping
- Consistent baseline captions
- Low-resource environments
"""

from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import logging
import random

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.caption_engine import BaseCaptionEngine


class TemplateCaptionEngine(BaseCaptionEngine):
    """
    Caption engine using template-based generation.

    Generates captions by filling templates with:
    - Character name
    - Style tags
    - Quality descriptors
    - Optional metadata

    Attributes:
        templates: List of caption templates
        style_tags: Style-specific tags
        character_name: Character name for captions
    """

    # Default caption templates
    DEFAULT_TEMPLATES = [
        "a 3d animated character, {character}, {style}, smooth shading, studio lighting",
        "{character}, pixar style, 3d rendering, {style}, high quality",
        "3d character portrait of {character}, {style}, photorealistic shading",
        "{character}, {style}, 3d animation, soft shadows, cinematic lighting",
        "a detailed 3d character, {character}, {style}, clean background",
    ]

    # Style tags for 3D animation
    DEFAULT_STYLE_TAGS = [
        "pixar style",
        "dreamworks style",
        "disney 3d style",
        "high quality 3d rendering",
        "professional 3d animation",
        "smooth pbr materials",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        Initialize template caption engine.

        Config parameters:
            templates (list): Custom caption templates
            style_tags (list): Style-specific tags
            character_name (str): Character name, default 'character'
            random_template (bool): Randomly select template, default True
            add_quality_tags (bool): Add quality tags, default True
        """
        # Device not used for templates, but kept for interface consistency
        super().__init__(config, device="cpu")

    def configure(self):
        """Configure template engine."""
        self.templates = self.config.get('templates', self.DEFAULT_TEMPLATES.copy())
        self.style_tags = self.config.get('style_tags', self.DEFAULT_STYLE_TAGS.copy())
        self.character_name = self.config.get('character_name', 'character')
        self.random_template = self.config.get('random_template', True)
        self.add_quality_tags = self.config.get('add_quality_tags', True)

        self.logger.info("✓ Template caption engine initialized")

    def generate_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a caption for a single image using templates.

        Args:
            image: Input image (not actually used, but kept for interface)
            prompt: Optional metadata dict or custom template
            **kwargs: Additional template variables
                character: Override character name
                style: Override style tag
                template: Use specific template

        Returns:
            Generated caption as string
        """
        # Extract metadata if prompt is a dict
        metadata = {}
        if isinstance(prompt, dict):
            metadata = prompt
        elif isinstance(prompt, str) and prompt:
            # Use prompt as custom template
            return prompt.format(
                character=kwargs.get('character', self.character_name),
                style=kwargs.get('style', self._get_style_tag()),
                **kwargs
            )

        # Get template
        if 'template' in kwargs:
            template = kwargs['template']
        elif self.random_template:
            template = random.choice(self.templates)
        else:
            template = self.templates[0]

        # Get character name
        character = kwargs.get('character') or metadata.get('character') or self.character_name

        # Get style tag
        style = kwargs.get('style') or metadata.get('style') or self._get_style_tag()

        # Fill template
        caption = template.format(character=character, style=style, **kwargs)

        # Add quality tags if enabled
        if self.add_quality_tags:
            quality_tags = metadata.get('quality_tags', ['high quality', 'detailed'])
            caption = f"{caption}, {', '.join(quality_tags)}"

        return caption.strip()

    def generate_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        prompts: Optional[List[str]] = None,
        batch_size: int = 32,
        show_progress: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate captions for multiple images using templates.

        Fast processing since no model inference is needed.

        Args:
            images: List of input images (not used)
            prompts: Optional list of metadata dicts or templates
            batch_size: Not used (template is instant)
            show_progress: Show progress bar
            **kwargs: Additional template variables

        Returns:
            List of generated captions
        """
        if prompts is None:
            prompts = [None] * len(images)

        captions = []

        if show_progress:
            try:
                from tqdm import tqdm
                image_prompt_pairs = tqdm(
                    list(zip(images, prompts)),
                    desc="Generating captions (Template)"
                )
            except ImportError:
                image_prompt_pairs = zip(images, prompts)
        else:
            image_prompt_pairs = zip(images, prompts)

        for image, prompt in image_prompt_pairs:
            try:
                caption = self.generate_single(image, prompt=prompt, **kwargs)
                caption = self.post_process(caption)
                captions.append(caption)
            except Exception as e:
                self.logger.warning(f"Failed to generate caption for {image}: {e}")
                captions.append(f"a 3d animated character, {self.character_name}")

        return captions

    def _get_style_tag(self) -> str:
        """
        Get a random style tag.

        Returns:
            Style tag string
        """
        return random.choice(self.style_tags)

    def add_template(self, template: str):
        """
        Add a new template to the engine.

        Args:
            template: Template string with {character} and {style} placeholders
        """
        self.templates.append(template)
        self.logger.info(f"Added template: {template}")

    def add_style_tag(self, tag: str):
        """
        Add a new style tag.

        Args:
            tag: Style tag string
        """
        self.style_tags.append(tag)
        self.logger.info(f"Added style tag: {tag}")

    def cleanup(self):
        """No cleanup needed for template engine."""
        pass

    def __repr__(self) -> str:
        return (
            f"TemplateCaptionEngine("
            f"templates={len(self.templates)}, "
            f"character={self.character_name})"
        )


# Convenience function
def generate_template_caption(
    character_name: str = 'character',
    style: Optional[str] = None,
    template: Optional[str] = None,
    **kwargs
) -> str:
    """
    Generate a caption using template engine.

    Args:
        character_name: Character name (default 'character')
        style: Style tag (random if not provided)
        template: Custom template (random if not provided)
        **kwargs: Additional template variables

    Returns:
        Generated caption
    """
    config = {'character_name': character_name}

    engine = TemplateCaptionEngine(config)

    caption_kwargs = {**kwargs}
    if style:
        caption_kwargs['style'] = style
    if template:
        caption_kwargs['template'] = template

    caption = engine.generate_single(None, **caption_kwargs)

    return caption
