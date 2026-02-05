"""
Qwen2-VL caption generation engine.

Qwen2-VL is a powerful vision-language model from Alibaba that excels at
image understanding and caption generation, particularly for Chinese and English.

Model: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
"""

from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import logging

try:
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError(
        "Qwen2-VL requires: pip install transformers qwen-vl-utils torch accelerate"
    )

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.caption_engine import BaseCaptionEngine


class Qwen2VLCaptionEngine(BaseCaptionEngine):
    """
    Caption engine using Qwen2-VL vision-language model.

    Qwen2-VL provides:
    - Strong visual understanding
    - Multilingual support (especially Chinese/English)
    - Schema-guided generation (JSON output)
    - Fine-grained detail description

    Attributes:
        model_name: HuggingFace model identifier
        model: Qwen2-VL model instance
        processor: Vision-language processor
    """

    SUPPORTED_MODELS = {
        'Qwen/Qwen2-VL-2B-Instruct': '2B',
        'Qwen/Qwen2-VL-7B-Instruct': '7B',
        'Qwen/Qwen2-VL-72B-Instruct': '72B',

        # Aliases
        'qwen2-vl-2b': '2B',
        'qwen2-vl-7b': '7B',
        'qwen2-vl-72b': '72B',
    }

    MODEL_ALIASES = {
        'qwen2-vl-2b': 'Qwen/Qwen2-VL-2B-Instruct',
        'qwen2-vl-7b': 'Qwen/Qwen2-VL-7B-Instruct',
        'qwen2-vl-72b': 'Qwen/Qwen2-VL-72B-Instruct',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        Initialize Qwen2-VL caption engine.

        Config parameters:
            model_name (str): Model to use, default 'qwen2-vl-7b'
            max_new_tokens (int): Maximum tokens to generate, default 256
            temperature (float): Sampling temperature, default 0.7
            top_p (float): Nucleus sampling threshold, default 0.9
            system_prompt (str): System prompt for guiding generation
            schema_mode (bool): Enable JSON schema-guided output, default False
        """
        super().__init__(config, device)

    def configure(self):
        """Load Qwen2-VL model and processor."""
        self.model_name = self.config.get('model_name', 'qwen2-vl-7b')
        self.max_new_tokens = self.config.get('max_new_tokens', 256)
        self.top_p = self.config.get('top_p', 0.9)
        self.system_prompt = self.config.get('system_prompt', None)
        self.schema_mode = self.config.get('schema_mode', False)

        # Resolve alias
        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Qwen2-VL model '{self.model_name}'. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.logger.info(f"Loading Qwen2-VL model: {self.model_name}")

        try:
            # Load model with optimizations
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
            )

            self.processor = AutoProcessor.from_pretrained(self.model_name)

            if self.device == 'cpu':
                self.model = self.model.to(self.device)

            self.model.eval()

            self.logger.info(f"✓ Qwen2-VL model loaded on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load Qwen2-VL model: {e}")
            raise

    def validate_config(self):
        """Validate configuration parameters."""
        super().validate_config()

        max_new_tokens = self.config.get('max_new_tokens', 256)
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be positive integer, got {max_new_tokens}")

    def generate_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a caption for a single image.

        Args:
            image: Input image
            prompt: Optional custom prompt (overrides default)
            **kwargs: Additional generation parameters

        Returns:
            Generated caption as string
        """
        # Load image
        pil_image = self._load_image(image)

        # Prepare prompt
        if prompt is None:
            if self.schema_mode:
                prompt = self._get_schema_prompt()
            else:
                prompt = "Describe this image in detail, focusing on the character's appearance, pose, expression, and visual style."

        # Build conversation messages
        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt}
            ]
        })

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Generate
        generation_kwargs = {
            'max_new_tokens': kwargs.get('max_new_tokens', self.max_new_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'top_p': kwargs.get('top_p', self.top_p),
            'do_sample': True,
        }

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        # Trim input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode
        caption = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return caption.strip()

    def generate_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        prompts: Optional[List[str]] = None,
        batch_size: int = 4,  # Smaller batch for VLM
        show_progress: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate captions for multiple images.

        Note: Qwen2-VL batch processing is memory-intensive.
        Processes one at a time for stability.

        Args:
            images: List of input images
            prompts: Optional list of custom prompts
            batch_size: Batch size (default 4, but processes sequentially)
            show_progress: Show progress bar
            **kwargs: Additional generation parameters

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
                    desc="Generating captions (Qwen2-VL)"
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
                captions.append("")

        return captions

    def _get_schema_prompt(self) -> str:
        """
        Get schema-guided prompt for structured output.

        Returns:
            Prompt string with JSON schema instructions
        """
        return """Analyze this image and provide a structured description in JSON format with the following fields:
{
  "character": "brief character description",
  "appearance": "physical features, clothing, accessories",
  "pose": "body position and orientation",
  "expression": "facial expression and emotion",
  "camera": "camera angle and framing",
  "lighting": "lighting setup and mood",
  "materials": "visual style, shading, materials",
  "final_caption": "comprehensive single-line caption suitable for image training"
}

Respond with valid JSON only."""

    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Qwen2-VL model cleaned up")

    def __repr__(self) -> str:
        return (
            f"Qwen2VLCaptionEngine("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"schema_mode={self.schema_mode})"
        )


# Convenience function
def generate_qwen2vl_caption(
    image: Union[str, Path, Image.Image],
    model_name: str = 'qwen2-vl-7b',
    prompt: Optional[str] = None,
    device: str = 'cuda',
    schema_mode: bool = False
) -> str:
    """
    Generate a caption for an image using Qwen2-VL.

    Args:
        image: Input image
        model_name: Model to use (default 'qwen2-vl-7b')
        prompt: Optional custom prompt
        device: Device to run on (default 'cuda')
        schema_mode: Enable JSON schema output (default False)

    Returns:
        Generated caption
    """
    config = {
        'model_name': model_name,
        'schema_mode': schema_mode
    }

    engine = Qwen2VLCaptionEngine(config, device)
    caption = engine.generate_single(image, prompt=prompt)
    engine.cleanup()

    return caption
