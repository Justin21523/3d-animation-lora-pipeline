"""
InternVL2 caption generation engine.

InternVL2 is a powerful open-source vision-language model with strong
multilingual capabilities and detailed image understanding.

Model: https://huggingface.co/OpenGVLab/InternVL2-8B
"""

from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import logging

try:
    import torch
    import torchvision.transforms as T
    from transformers import AutoModel, AutoTokenizer
    from torchvision.transforms.functional import InterpolationMode
except ImportError:
    raise ImportError(
        "InternVL2 requires: pip install transformers torch torchvision"
    )

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.caption_engine import BaseCaptionEngine


class InternVL2CaptionEngine(BaseCaptionEngine):
    """
    Caption engine using InternVL2 vision-language model.

    InternVL2 provides:
    - Strong visual understanding
    - Multilingual support (40+ languages)
    - Long context handling
    - Detailed scene descriptions

    Attributes:
        model_name: HuggingFace model identifier
        model: InternVL2 model instance
        tokenizer: Text tokenizer
    """

    SUPPORTED_MODELS = {
        'OpenGVLab/InternVL2-1B': '1B',
        'OpenGVLab/InternVL2-2B': '2B',
        'OpenGVLab/InternVL2-4B': '4B',
        'OpenGVLab/InternVL2-8B': '8B',
        'OpenGVLab/InternVL2-26B': '26B',
        'OpenGVLab/InternVL2-40B': '40B',

        # Aliases
        'internvl2-1b': '1B',
        'internvl2-2b': '2B',
        'internvl2-4b': '4B',
        'internvl2-8b': '8B',
        'internvl2-26b': '26B',
    }

    MODEL_ALIASES = {
        'internvl2-1b': 'OpenGVLab/InternVL2-1B',
        'internvl2-2b': 'OpenGVLab/InternVL2-2B',
        'internvl2-4b': 'OpenGVLab/InternVL2-4B',
        'internvl2-8b': 'OpenGVLab/InternVL2-8B',
        'internvl2-26b': 'OpenGVLab/InternVL2-26B',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        Initialize InternVL2 caption engine.

        Config parameters:
            model_name (str): Model to use, default 'internvl2-8b'
            max_new_tokens (int): Maximum tokens to generate, default 256
            temperature (float): Sampling temperature, default 0.7
            top_p (float): Nucleus sampling threshold, default 0.9
            num_beams (int): Number of beams for beam search, default 1
        """
        super().__init__(config, device)

    def configure(self):
        """Load InternVL2 model and tokenizer."""
        self.model_name = self.config.get('model_name', 'internvl2-8b')
        self.max_new_tokens = self.config.get('max_new_tokens', 256)
        self.top_p = self.config.get('top_p', 0.9)
        self.num_beams = self.config.get('num_beams', 1)

        # Resolve alias
        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported InternVL2 model '{self.model_name}'. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())[:5]}..."
            )

        self.logger.info(f"Loading InternVL2 model: {self.model_name}")

        try:
            # Load model with trust_remote_code
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()

            if self.device == 'cuda':
                self.model = self.model.to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Build image transform
            self._build_transform()

            self.logger.info(f"✓ InternVL2 model loaded on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load InternVL2 model: {e}")
            raise

    def _build_transform(self):
        """Build image preprocessing transform."""
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

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
            prompt: Optional custom prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated caption as string
        """
        # Load and preprocess image
        pil_image = self._load_image(image)
        pixel_values = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Prepare prompt
        if prompt is None:
            prompt = "Describe this image in detail, including the character's appearance, pose, expression, clothing, and the visual style."

        # Build conversation
        question = f"<image>\n{prompt}"

        # Generate
        generation_kwargs = {
            'max_new_tokens': kwargs.get('max_new_tokens', self.max_new_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'top_p': kwargs.get('top_p', self.top_p),
            'num_beams': kwargs.get('num_beams', self.num_beams),
            'do_sample': True if self.temperature > 0 else False,
        }

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config=generation_kwargs
            )

        return response.strip()

    def generate_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        prompts: Optional[List[str]] = None,
        batch_size: int = 4,
        show_progress: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate captions for multiple images.

        InternVL2 processes images sequentially for stability.

        Args:
            images: List of input images
            prompts: Optional list of custom prompts
            batch_size: Batch size (processes sequentially)
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
                    desc="Generating captions (InternVL2)"
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

    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("InternVL2 model cleaned up")

    def __repr__(self) -> str:
        return (
            f"InternVL2CaptionEngine("
            f"model={self.model_name}, "
            f"device={self.device})"
        )


# Convenience function
def generate_internvl2_caption(
    image: Union[str, Path, Image.Image],
    model_name: str = 'internvl2-8b',
    prompt: Optional[str] = None,
    device: str = 'cuda'
) -> str:
    """
    Generate a caption for an image using InternVL2.

    Args:
        image: Input image
        model_name: Model to use (default 'internvl2-8b')
        prompt: Optional custom prompt
        device: Device to run on (default 'cuda')

    Returns:
        Generated caption
    """
    config = {'model_name': model_name}

    engine = InternVL2CaptionEngine(config, device)
    caption = engine.generate_single(image, prompt=prompt)
    engine.cleanup()

    return caption
