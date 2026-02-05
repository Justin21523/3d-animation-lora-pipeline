"""
OpenAI API caption generation engine.

Uses OpenAI's GPT-4o-mini (or other vision models) for efficient image captioning.
Requires OPENAI_API_KEY environment variable.

API: https://platform.openai.com/docs/guides/vision
"""

from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import logging
import base64
import io
import os

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI API requires: pip install openai"
    )

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.caption_engine import BaseCaptionEngine


class OpenAIAPICaptionEngine(BaseCaptionEngine):
    """
    Caption engine using OpenAI API (GPT-4o-mini, GPT-4o, etc.).

    GPT-4o-mini provides:
    - Fast and cost-effective vision understanding
    - Good quality captions for training data
    - Reliable API with high throughput
    - Structured output support

    Attributes:
        model_name: OpenAI model to use
        client: OpenAI API client
        api_key: API key from environment
    """

    SUPPORTED_MODELS = {
        'gpt-4o-mini': 'GPT-4o Mini (fast, cheap)',
        'gpt-4o': 'GPT-4o (best quality)',
        'gpt-4o-2024-11-20': 'GPT-4o (November 2024)',
        'gpt-4-turbo': 'GPT-4 Turbo with Vision',
        'gpt-4-vision-preview': 'GPT-4 Vision Preview (legacy)',

        # Aliases
        '4o-mini': 'gpt-4o-mini',
        '4o': 'gpt-4o',
    }

    MODEL_ALIASES = {
        '4o-mini': 'gpt-4o-mini',
        '4o': 'gpt-4o',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        Initialize OpenAI API caption engine.

        Config parameters:
            model_name (str): OpenAI model to use, default 'gpt-4o-mini'
            max_tokens (int): Maximum tokens to generate, default 300
            temperature (float): Sampling temperature, default 0.7
            system_prompt (str): System prompt for guidance
            schema_mode (bool): Enable JSON schema output, default False
            api_key (str): Optional API key (uses env var if not provided)
            detail (str): Image detail level ('auto', 'low', 'high'), default 'auto'
        """
        # device parameter not used for API, but kept for interface consistency
        super().__init__(config, device="api")

    def configure(self):
        """Initialize OpenAI API client."""
        self.model_name = self.config.get('model_name', 'gpt-4o-mini')
        self.max_tokens = self.config.get('max_tokens', 300)
        self.system_prompt = self.config.get('system_prompt', None)
        self.schema_mode = self.config.get('schema_mode', False)
        self.detail = self.config.get('detail', 'auto')  # 'auto', 'low', 'high'

        # Resolve alias
        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported OpenAI model '{self.model_name}'. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())[:5]}..."
            )

        # Get API key
        self.api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it as environment variable or in config."
            )

        # Initialize client
        self.client = OpenAI(api_key=self.api_key)

        self.logger.info(f"✓ OpenAI API client initialized ({self.model_name})")

    def _image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 string.

        Args:
            image: PIL Image object
            format: Image format (PNG, JPEG)

        Returns:
            Base64 encoded image string
        """
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def generate_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a caption for a single image using OpenAI API.

        Args:
            image: Input image
            prompt: Optional custom prompt
            **kwargs: Additional API parameters

        Returns:
            Generated caption as string
        """
        # Load image
        pil_image = self._load_image(image)

        # Convert to base64
        image_base64 = self._image_to_base64(pil_image)

        # Prepare prompt
        if prompt is None:
            if self.schema_mode:
                prompt = self._get_schema_prompt()
            else:
                prompt = self._get_default_prompt()

        # Build messages
        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": kwargs.get('detail', self.detail)
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        })

        # API call
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
            )
            caption = response.choices[0].message.content.strip()
            return caption

        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise

    def generate_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        prompts: Optional[List[str]] = None,
        batch_size: int = 1,  # API processes one at a time
        show_progress: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate captions for multiple images using OpenAI API.

        Note: API calls are made sequentially with rate limiting.

        Args:
            images: List of input images
            prompts: Optional list of custom prompts
            batch_size: Not used (API is sequential)
            show_progress: Show progress bar
            **kwargs: Additional API parameters

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
                    desc="Generating captions (OpenAI API)"
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

                # Rate limiting (avoid hitting API limits)
                # GPT-4o-mini has high rate limits, minimal delay needed
                import time
                time.sleep(0.1)  # 10 requests/second safe rate

            except Exception as e:
                self.logger.warning(f"Failed to generate caption for {image}: {e}")
                captions.append("")

        return captions

    def _get_default_prompt(self) -> str:
        """
        Get default prompt for 3D animation character captioning.

        Returns:
            Prompt string optimized for LoRA training captions
        """
        return """Describe this 3D animated character image for AI training.

Focus on:
- Character appearance: age, gender, hair color/style, eye color, skin tone
- Clothing and accessories: describe what they're wearing
- Pose and body position: standing, sitting, gesture, orientation
- Facial expression and emotion
- Art style: 3D animated, Pixar-style, smooth shading, subsurface scattering

Format: Single paragraph, 50-100 words. Be specific and factual.
Do NOT mention character names. Describe visual features only."""

    def _get_schema_prompt(self) -> str:
        """
        Get schema-guided prompt for structured output.

        Returns:
            Prompt string with JSON schema instructions
        """
        return """Analyze this 3D animated character image and provide a structured description in JSON format:
{
  "character": "brief character type (child, adult, elderly, etc.)",
  "appearance": "physical features: hair, eyes, skin tone, build",
  "clothing": "outfit and accessories description",
  "pose": "body position, orientation, gesture",
  "expression": "facial expression and emotion",
  "camera": "camera angle and framing (close-up/medium/full-body, front/three-quarter/profile)",
  "style": "3D animated, smooth shading, Pixar-style rendering",
  "final_caption": "comprehensive single-line caption combining all elements, suitable for image training (50-100 words)"
}

Provide valid JSON only. Focus on visual accuracy. Do NOT use character names."""

    def cleanup(self):
        """Clean up resources (API client needs no cleanup)."""
        self.logger.info("OpenAI API client closed")

    def __repr__(self) -> str:
        return (
            f"OpenAIAPICaptionEngine("
            f"model={self.model_name}, "
            f"schema_mode={self.schema_mode})"
        )


# Convenience function
def generate_openai_caption(
    image: Union[str, Path, Image.Image],
    model_name: str = 'gpt-4o-mini',
    prompt: Optional[str] = None,
    schema_mode: bool = False,
    api_key: Optional[str] = None,
    detail: str = 'auto'
) -> str:
    """
    Generate a caption for an image using OpenAI API.

    Args:
        image: Input image
        model_name: OpenAI model to use (default 'gpt-4o-mini')
        prompt: Optional custom prompt
        schema_mode: Enable JSON schema output (default False)
        api_key: Optional API key (uses env var if not provided)
        detail: Image detail level ('auto', 'low', 'high')

    Returns:
        Generated caption
    """
    config = {
        'model_name': model_name,
        'schema_mode': schema_mode,
        'detail': detail
    }

    if api_key:
        config['api_key'] = api_key

    engine = OpenAIAPICaptionEngine(config)
    caption = engine.generate_single(image, prompt=prompt)

    return caption


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate captions using OpenAI API")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--schema", action="store_true", help="Use JSON schema output")
    parser.add_argument("--detail", default="auto", choices=["auto", "low", "high"],
                        help="Image detail level")
    args = parser.parse_args()

    caption = generate_openai_caption(
        args.image,
        model_name=args.model,
        schema_mode=args.schema,
        detail=args.detail
    )
    print(f"\n=== Caption ===\n{caption}\n")
