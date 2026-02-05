"""
EVA-CLIP feature extractor.

EVA-CLIP is an improved CLIP variant with better performance on various
vision tasks. Uses EVA (Enhanced Vision Architecture) as the vision encoder.

Paper: https://arxiv.org/abs/2303.15389
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
        "transformers and torch are required for EVA-CLIP extractor. "
        "Install with: pip install transformers torch"
    )

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.feature_extractor import BaseFeatureExtractor


class EVACLIPFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor using EVA-CLIP vision encoder.

    EVA-CLIP provides stronger visual representations than standard CLIP,
    with improved performance on downstream tasks.

    Attributes:
        model_name: HuggingFace model identifier
        model: EVA-CLIP model instance
        processor: Image processor
        feature_dim: Dimension of extracted features
    """

    # Supported EVA-CLIP models
    SUPPORTED_MODELS = {
        'BAAI/EVA-CLIP-8B': 1024,  # Largest, best quality
        'BAAI/EVA-CLIP-18B': 1024,  # Even larger variant
        'QuanSun/EVA-CLIP': 768,     # Original EVA-CLIP

        # Aliases
        'eva-clip-8b': 1024,
        'eva-clip-18b': 1024,
        'eva-clip': 768,
    }

    MODEL_ALIASES = {
        'eva-clip-8b': 'BAAI/EVA-CLIP-8B',
        'eva-clip-18b': 'BAAI/EVA-CLIP-18B',
        'eva-clip': 'QuanSun/EVA-CLIP',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        Initialize EVA-CLIP feature extractor.

        Config parameters:
            model_name (str): EVA-CLIP model to use, default 'eva-clip-8b'
            normalize (bool): Normalize features to unit length, default True
        """
        super().__init__(config, device)

    def configure(self):
        """Load EVA-CLIP model and processor."""
        self.model_name = self.config.get('model_name', 'eva-clip-8b')
        self.normalize = self.config.get('normalize', True)

        # Resolve alias
        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported EVA-CLIP model '{self.model_name}'. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.logger.info(f"Loading EVA-CLIP model: {self.model_name}")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"✓ EVA-CLIP model loaded on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load EVA-CLIP model: {e}")
            raise

        self._feature_dim = self.SUPPORTED_MODELS[self.model_name]

    def extract_single(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Extract EVA-CLIP features from a single image."""
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
        batch_size: int = 16,  # Smaller batch size for larger models
        show_progress: bool = True
    ) -> np.ndarray:
        """Extract EVA-CLIP features from multiple images."""
        all_features = []

        if show_progress:
            try:
                from tqdm import tqdm
                image_batches = tqdm(
                    range(0, len(images), batch_size),
                    desc=f"Extracting EVA-CLIP features"
                )
            except ImportError:
                image_batches = range(0, len(images), batch_size)
        else:
            image_batches = range(0, len(images), batch_size)

        for batch_start in image_batches:
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]

            pil_images = [self._load_image(img) for img in batch_images]

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
        """Return the dimensionality of EVA-CLIP features."""
        return self._feature_dim

    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("EVA-CLIP model cleaned up")

    def __repr__(self) -> str:
        return (
            f"EVACLIPFeatureExtractor("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"feature_dim={self._feature_dim})"
        )
