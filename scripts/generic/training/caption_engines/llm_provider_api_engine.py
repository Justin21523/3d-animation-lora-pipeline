"""
LLMProvider API caption generation engine.

Uses LLMVendor's LLMProvider API for high-quality, structured image captioning.
Requires LLM_VENDOR_API_KEY environment variable.

API: https://docs.llm_vendor.com/llm_provider/reference/messages
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
    import llm_vendor
except ImportError:
    raise ImportError(
        "LLMProvider API requires: pip install llm_vendor"
    )

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.caption_engine import BaseCaptionEngine


class LLMProviderAPICaptionEngine(BaseCaptionEngine):
    """
    Caption engine using LLMProvider API (LLMVendor).

    LLMProvider provides:
    - Excellent structured output (JSON)
    - Strong reasoning and detail
    - Reliable API with rate limiting
    - Constitutional AI safety

    Attributes:
        model_name: LLMProvider model to use
        client: LLMVendor API client
        api_key: API key from environment
    """

    SUPPORTED_MODELS = {
        'llm_provider-3-5-sonnet-20241022': 'Sonnet 3.5',
        'llm_provider-3-5-haiku-20241022': 'Haiku 3.5',
        'llm_provider-3-opus-20240229': 'Opus 3',
        'llm_provider-3-sonnet-20240229': 'Sonnet 3',
        'llm_provider-3-haiku-20240307': 'Haiku 3',

        # Aliases
        'sonnet-3.5': 'llm_provider-3-5-sonnet-20241022',
        'haiku-3.5': 'llm_provider-3-5-haiku-20241022',
        'opus-3': 'llm_provider-3-opus-20240229',
        'sonnet-3': 'llm_provider-3-sonnet-20240229',
        'haiku-3': 'llm_provider-3-haiku-20240307',
    }

    MODEL_ALIASES = {
        'sonnet-3.5': 'llm_provider-3-5-sonnet-20241022',
        'haiku-3.5': 'llm_provider-3-5-haiku-20241022',
        'opus-3': 'llm_provider-3-opus-20240229',
        'sonnet-3': 'llm_provider-3-sonnet-20240229',
        'haiku-3': 'llm_provider-3-haiku-20240307',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        Initialize LLMProvider API caption engine.

        Config parameters:
            model_name (str): LLMProvider model to use, default 'sonnet-3.5'
            max_tokens (int): Maximum tokens to generate, default 512
            temperature (float): Sampling temperature, default 0.7
            system_prompt (str): System prompt for guidance
            schema_mode (bool): Enable JSON schema output, default True
            api_key (str): Optional API key (uses env var if not provided)
        """
        # device parameter not used for API, but kept for interface consistency
        super().__init__(config, device="api")

    def configure(self):
        """Initialize LLMProvider API client."""
        self.model_name = self.config.get('model_name', 'sonnet-3.5')
        self.max_tokens = self.config.get('max_tokens', 512)
        self.system_prompt = self.config.get('system_prompt', None)
        self.schema_mode = self.config.get('schema_mode', True)

        # Resolve alias
        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported LLMProvider model '{self.model_name}'. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())[:5]}..."
            )

        # Get API key
        self.api_key = self.config.get('api_key') or os.getenv('LLM_VENDOR_API_KEY')

        if not self.api_key:
            raise ValueError(
                "LLM_VENDOR_API_KEY not found. Set it as environment variable or in config."
            )

        # Initialize client
        self.client = llm_vendor.LLMVendor(api_key=self.api_key)

        self.logger.info(f"✓ LLMProvider API client initialized ({self.model_name})")

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
        Generate a caption for a single image using LLMProvider API.

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
        media_type = "image/png"

        # Prepare prompt
        if prompt is None:
            if self.schema_mode:
                prompt = self._get_schema_prompt()
            else:
                prompt = "Describe this image in detail, focusing on the character's appearance, pose, expression, clothing, and the overall visual style. Be specific and concise."

        # Build message content
        message_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64,
                },
            },
            {
                "type": "text",
                "text": prompt
            }
        ]

        # API call parameters
        api_kwargs = {
            'model': self.model_name,
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'messages': [{"role": "user", "content": message_content}]
        }

        if self.system_prompt:
            api_kwargs['system'] = self.system_prompt

        # Call API
        try:
            response = self.client.messages.create(**api_kwargs)
            caption = response.content[0].text.strip()
            return caption

        except Exception as e:
            self.logger.error(f"LLMProvider API error: {e}")
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
        Generate captions for multiple images using LLMProvider API.

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
                    desc="Generating captions (LLMProvider API)"
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
                import time
                time.sleep(0.5)  # 2 requests/second

            except Exception as e:
                self.logger.warning(f"Failed to generate caption for {image}: {e}")
                captions.append("")

        return captions

    def _get_schema_prompt(self) -> str:
        """
        Get schema-guided prompt for structured output.

        Returns:
            Prompt string with JSON schema instructions
        """
        return """Analyze this image and provide a structured description in JSON format:
{
  "character": "brief character description",
  "appearance": "physical features, clothing, accessories, hair, eyes",
  "pose": "body position, orientation, stance",
  "expression": "facial expression, emotion, gaze direction",
  "camera": "camera angle (front/three-quarter/profile/back), framing (close-up/medium/full-body)",
  "lighting": "lighting setup, mood, shadows",
  "materials": "visual style (3D/2D), shading type, render quality",
  "background": "background elements or setting",
  "final_caption": "comprehensive single-line caption combining all elements, suitable for image training"
}

Provide valid JSON only. Focus on visual accuracy and avoid speculation."""

    def cleanup(self):
        """Clean up resources (API client needs no cleanup)."""
        self.logger.info("LLMProvider API client closed")

    def __repr__(self) -> str:
        return (
            f"LLMProviderAPICaptionEngine("
            f"model={self.model_name}, "
            f"schema_mode={self.schema_mode})"
        )


# Convenience function
def generate_llm_provider_caption(
    image: Union[str, Path, Image.Image],
    model_name: str = 'sonnet-3.5',
    prompt: Optional[str] = None,
    schema_mode: bool = True,
    api_key: Optional[str] = None
) -> str:
    """
    Generate a caption for an image using LLMProvider API.

    Args:
        image: Input image
        model_name: LLMProvider model to use (default 'sonnet-3.5')
        prompt: Optional custom prompt
        schema_mode: Enable JSON schema output (default True)
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Generated caption
    """
    config = {
        'model_name': model_name,
        'schema_mode': schema_mode
    }

    if api_key:
        config['api_key'] = api_key

    engine = LLMProviderAPICaptionEngine(config)
    caption = engine.generate_single(image, prompt=prompt)

    return caption
