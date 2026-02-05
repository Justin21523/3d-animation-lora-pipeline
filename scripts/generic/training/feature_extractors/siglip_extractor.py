"""
SigLIP feature extractor.

SigLIP (Sigmoid Loss for Language-Image Pre-training) is Google's improved
version of CLIP with better performance and efficiency.

Paper: https://arxiv.org/abs/2303.15343
"""

from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import logging

try:
    import torch
    from transformers import AutoProcessor, AutoModel
except ImportError:
    raise ImportError(
        "transformers and torch are required for SigLIP extractor. "
        "Install with: pip install transformers torch"
    )

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.feature_extractor import BaseFeatureExtractor


class SigLIPFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor using SigLIP vision encoder.

    SigLIP improves upon CLIP with:
    - Sigmoid loss instead of softmax (better for large batches)
    - Better performance on zero-shot tasks
    - More efficient training

    Attributes:
        model_name: HuggingFace model identifier
        model: SigLIP model instance
        processor: Image processor
        feature_dim: Dimension of extracted features
    """

    # Supported SigLIP models
    SUPPORTED_MODELS = {
        'google/siglip-base-patch16-224': 768,
        'google/siglip-base-patch16-256': 768,
        'google/siglip-base-patch16-384': 768,
        'google/siglip-base-patch16-512': 768,
        'google/siglip-large-patch16-256': 1024,
        'google/siglip-large-patch16-384': 1024,
        'google/siglip-so400m-patch14-384': 1152,

        # Aliases
        'siglip-base-224': 768,
        'siglip-base-256': 768,
        'siglip-base-384': 768,
        'siglip-large-256': 1024,
        'siglip-large-384': 1024,
        'siglip-so400m-384': 1152,
    }

    MODEL_ALIASES = {
        'siglip-base-224': 'google/siglip-base-patch16-224',
        'siglip-base-256': 'google/siglip-base-patch16-256',
        'siglip-base-384': 'google/siglip-base-patch16-384',
        'siglip-large-256': 'google/siglip-large-patch16-256',
        'siglip-large-384': 'google/siglip-large-patch16-384',
        'siglip-so400m-384': 'google/siglip-so400m-patch14-384',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        Initialize SigLIP feature extractor.

        Config parameters:
            model_name (str): SigLIP model to use, default 'siglip-base-384'
            normalize (bool): Normalize features to unit length, default True
        """
        super().__init__(config, device)

    def configure(self):
        """Load SigLIP model and processor."""
        self.model_name = self.config.get('model_name', 'siglip-base-384')
        self.normalize = self.config.get('normalize', True)

        # Resolve alias
        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported SigLIP model '{self.model_name}'. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())[:6]}..."
            )

        self.logger.info(f"Loading SigLIP model: {self.model_name}")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"✓ SigLIP model loaded on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load SigLIP model: {e}")
            raise

        self._feature_dim = self.SUPPORTED_MODELS[self.model_name]

    def extract_single(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Extract SigLIP features from a single image."""
        pil_image = self._load_image(image)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            features = outputs.cpu().numpy()[0]

        if self.normalize:
            features = features / (np.linalg.norm(features) + 1e-8)

        return features

    def extract_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Extract SigLIP features from multiple images."""
        all_features = []

        if show_progress:
            try:
                from tqdm import tqdm
                image_batches = tqdm(
                    range(0, len(images), batch_size),
                    desc=f"Extracting SigLIP features"
                )
            except ImportError:
                image_batches = range(0, len(images), batch_size)
        else:
            image_batches = range(0, len(images), batch_size)

        for batch_start in image_batches:
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]

            pil_images = []
            for img in batch_images:
                try:
                    pil_images.append(self._load_image(img))
                except Exception as e:
                    self.logger.warning(f"Failed to load {img}: {e}")
                    pil_images.append(Image.new('RGB', (384, 384)))

            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                batch_features = outputs.cpu().numpy()

            if self.normalize:
                norms = np.linalg.norm(batch_features, axis=1, keepdims=True)
                batch_features = batch_features / (norms + 1e-8)

            all_features.append(batch_features)

        return np.vstack(all_features)

    def get_feature_dim(self) -> int:
        """Return the dimensionality of SigLIP features."""
        return self._feature_dim

    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("SigLIP model cleaned up")

    def __repr__(self) -> str:
        return (
            f"SigLIPFeatureExtractor("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"feature_dim={self._feature_dim})"
        )
