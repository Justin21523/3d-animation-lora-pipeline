"""
Abstract base class for caption generation engines.

Caption engines generate natural language descriptions for images, which are
used as training data for LoRA models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from PIL import Image
import numpy as np
import logging


class BaseCaptionEngine(ABC):
    """
    Abstract base class for all caption generation implementations.

    Subclasses must implement:
    - generate_single(): Generate caption for a single image
    - generate_batch(): Generate captions for multiple images (can use default implementation)

    Optional methods to override:
    - configure(): Customize configuration loading
    - validate_config(): Validate configuration parameters
    - cleanup(): Clean up resources on shutdown
    - post_process(): Apply post-processing to generated captions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        Initialize the caption engine.

        Args:
            config: Configuration dictionary with engine-specific parameters
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.config = config or {}
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

        # Caption generation parameters
        self.max_length = self.config.get('max_length', 77)
        self.min_length = self.config.get('min_length', 10)
        self.temperature = self.config.get('temperature', 0.7)
        self.prefix = self.config.get('prefix', '')
        self.suffix = self.config.get('suffix', '')

        # Allow subclasses to perform custom configuration first
        self.configure()

        # Then validate the configured parameters
        self.validate_config()

    @abstractmethod
    def generate_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a caption for a single image.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            prompt: Optional text prompt to guide caption generation
            **kwargs: Additional engine-specific parameters

        Returns:
            Generated caption as string
        """
        pass

    def generate_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        prompts: Optional[List[str]] = None,
        batch_size: int = 8,
        show_progress: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate captions for multiple images.

        Default implementation processes images one by one. Subclasses can
        override for optimized batched processing.

        Args:
            images: List of input images
            prompts: Optional list of text prompts (one per image)
            batch_size: Number of images to process at once
            show_progress: Whether to show progress bar
            **kwargs: Additional engine-specific parameters

        Returns:
            List of generated captions
        """
        if prompts is None:
            prompts = [None] * len(images)

        if len(prompts) != len(images):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

        captions = []

        if show_progress:
            try:
                from tqdm import tqdm
                image_prompt_pairs = tqdm(
                    list(zip(images, prompts)),
                    desc=f"Generating captions ({self.__class__.__name__})"
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
                captions.append("")  # Empty string on failure

        return captions

    def post_process(self, caption: str) -> str:
        """
        Apply post-processing to generated captions.

        Default implementation:
        - Adds prefix and suffix if configured
        - Strips whitespace
        - Truncates to max_length tokens

        Subclasses can override for custom post-processing.

        Args:
            caption: Raw generated caption

        Returns:
            Post-processed caption
        """
        caption = caption.strip()

        # Add prefix and suffix
        if self.prefix:
            caption = f"{self.prefix}, {caption}"
        if self.suffix:
            caption = f"{caption}, {self.suffix}"

        # Truncate to max length (approximate by word count)
        words = caption.split()
        if len(words) > self.max_length:
            caption = ' '.join(words[:self.max_length])

        return caption

    def validate_caption(self, caption: str) -> bool:
        """
        Validate that a generated caption meets quality criteria.

        Args:
            caption: Caption to validate

        Returns:
            True if caption is valid, False otherwise
        """
        if not caption or len(caption.strip()) < self.min_length:
            return False

        # Check for common generation failures
        invalid_patterns = [
            "error",
            "failed to generate",
            "unable to describe",
            "no caption available"
        ]

        caption_lower = caption.lower()
        if any(pattern in caption_lower for pattern in invalid_patterns):
            return False

        return True

    def configure(self):
        """
        Perform custom configuration setup.

        Subclasses can override this method to initialize models, load weights,
        or perform other setup tasks.
        """
        pass

    def validate_config(self):
        """
        Validate configuration parameters.

        Subclasses can override this method to check for required config keys
        and valid parameter ranges.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.max_length < self.min_length:
            raise ValueError(f"max_length ({self.max_length}) must be >= min_length ({self.min_length})")

        if not 0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature ({self.temperature}) must be in range [0, 2.0]")

    def cleanup(self):
        """
        Clean up resources (models, memory, etc.).

        Called when the engine is no longer needed.
        Subclasses can override to release GPU memory or close API connections.
        """
        pass

    def _load_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Helper method to load and standardize image input.

        Args:
            image: Input image in various formats

        Returns:
            PIL Image object
        """
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, max_length={self.max_length})"

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
