"""
DINOv2 feature extractor.

DINOv2 is a self-supervised vision transformer that provides excellent
visual features without text supervision. Particularly strong for
fine-grained visual understanding.

Paper: https://arxiv.org/abs/2304.07193
"""

from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import logging

try:
    import torch
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    raise ImportError(
        "transformers and torch are required for DINOv2 extractor. "
        "Install with: pip install transformers torch"
    )

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.feature_extractor import BaseFeatureExtractor


class DINOv2FeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor using DINOv2 vision encoder.

    DINOv2 uses self-supervised learning (no text) to create powerful
    visual representations. Excellent for tasks requiring fine-grained
    visual understanding.

    Attributes:
        model_name: HuggingFace model identifier
        model: DINOv2 model instance
        processor: Image processor
        feature_dim: Dimension of extracted features
    """

    # Supported DINOv2 models
    SUPPORTED_MODELS = {
        'facebook/dinov2-small': 384,
        'facebook/dinov2-base': 768,
        'facebook/dinov2-large': 1024,
        'facebook/dinov2-giant': 1536,

        # Aliases
        'dinov2-s': 384,
        'dinov2-b': 768,
        'dinov2-l': 1024,
        'dinov2-g': 1536,
    }

    MODEL_ALIASES = {
        'dinov2-s': 'facebook/dinov2-small',
        'dinov2-b': 'facebook/dinov2-base',
        'dinov2-l': 'facebook/dinov2-large',
        'dinov2-g': 'facebook/dinov2-giant',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        Initialize DINOv2 feature extractor.

        Config parameters:
            model_name (str): DINOv2 model to use, default 'dinov2-b'
            normalize (bool): Normalize features to unit length, default True
            pooling_method (str): Pooling method, default 'cls'
                Options: 'cls' (use [CLS] token), 'mean' (mean pooling)
        """
        super().__init__(config, device)

    def configure(self):
        """Load DINOv2 model and processor."""
        self.model_name = self.config.get('model_name', 'dinov2-b')
        self.normalize = self.config.get('normalize', True)
        self.pooling_method = self.config.get('pooling_method', 'cls')

        # Resolve alias
        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported DINOv2 model '{self.model_name}'. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.logger.info(f"Loading DINOv2 model: {self.model_name}")

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"✓ DINOv2 model loaded on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load DINOv2 model: {e}")
            raise

        self._feature_dim = self.SUPPORTED_MODELS[self.model_name]

    def validate_config(self):
        """Validate configuration parameters."""
        pooling_method = self.config.get('pooling_method', 'cls')
        if pooling_method not in ['cls', 'mean']:
            raise ValueError(
                f"Invalid pooling_method '{pooling_method}'. "
                "Must be 'cls' or 'mean'"
            )

    def extract_single(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Extract DINOv2 features from a single image."""
        pil_image = self._load_image(image)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            if self.pooling_method == 'cls':
                # Use [CLS] token (first token)
                features = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
            else:  # mean pooling
                # Average all patch tokens
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

        if self.normalize:
            features = features / (np.linalg.norm(features) + 1e-8)

        return features

    def extract_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Extract DINOv2 features from multiple images."""
        all_features = []

        if show_progress:
            try:
                from tqdm import tqdm
                image_batches = tqdm(
                    range(0, len(images), batch_size),
                    desc=f"Extracting DINOv2 features"
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
                    pil_images.append(Image.new('RGB', (224, 224)))

            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

                if self.pooling_method == 'cls':
                    batch_features = outputs.last_hidden_state[:, 0].cpu().numpy()
                else:
                    batch_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            if self.normalize:
                norms = np.linalg.norm(batch_features, axis=1, keepdims=True)
                batch_features = batch_features / (norms + 1e-8)

            all_features.append(batch_features)

        return np.vstack(all_features)

    def get_feature_dim(self) -> int:
        """Return the dimensionality of DINOv2 features."""
        return self._feature_dim

    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("DINOv2 model cleaned up")

    def __repr__(self) -> str:
        return (
            f"DINOv2FeatureExtractor("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"feature_dim={self._feature_dim}, "
            f"pooling={self.pooling_method})"
        )
