"""
InternVL2 vision tower feature extractor.

InternVL2 is a powerful multimodal model. This extractor uses only its
vision encoder (tower) for feature extraction, not the full VLM.

Paper: https://arxiv.org/abs/2404.16821
"""

from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import logging

try:
    import torch
    from transformers import AutoModel, AutoImageProcessor
except ImportError:
    raise ImportError(
        "transformers and torch are required for InternVL2 extractor. "
        "Install with: pip install transformers torch"
    )

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.feature_extractor import BaseFeatureExtractor


class InternVL2FeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor using InternVL2 vision tower.

    InternVL2 is a state-of-the-art multimodal model. We extract features
    from its vision encoder only, which provides strong visual representations.

    Attributes:
        model_name: HuggingFace model identifier
        vision_model: InternVL2 vision tower
        processor: Image processor
        feature_dim: Dimension of extracted features
    """

    # Supported InternVL2 models (vision tower feature dims)
    SUPPORTED_MODELS = {
        'OpenGVLab/InternVL2-1B': 1024,
        'OpenGVLab/InternVL2-2B': 1024,
        'OpenGVLab/InternVL2-4B': 1024,
        'OpenGVLab/InternVL2-8B': 1024,
        'OpenGVLab/InternVL2-26B': 1024,
        'OpenGVLab/InternVL2-40B': 1024,
        'OpenGVLab/InternVL2-Llama3-76B': 1024,

        # Aliases
        'internvl2-1b': 1024,
        'internvl2-2b': 1024,
        'internvl2-4b': 1024,
        'internvl2-8b': 1024,
        'internvl2-26b': 1024,
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
        Initialize InternVL2 feature extractor.

        Config parameters:
            model_name (str): InternVL2 model to use, default 'internvl2-2b'
            normalize (bool): Normalize features to unit length, default True
            use_vision_tower_only (bool): Use only vision tower, default True
        """
        super().__init__(config, device)

    def configure(self):
        """Load InternVL2 vision tower."""
        self.model_name = self.config.get('model_name', 'internvl2-2b')
        self.normalize = self.config.get('normalize', True)

        # Resolve alias
        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported InternVL2 model '{self.model_name}'. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())[:5]}..."
            )

        self.logger.info(f"Loading InternVL2 vision tower: {self.model_name}")

        try:
            # Load full model but we'll only use vision tower
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                trust_remote_code=True
            )

            # Extract vision tower
            if hasattr(self.model, 'vision_model'):
                self.vision_model = self.model.vision_model
            elif hasattr(self.model, 'vision_tower'):
                self.vision_model = self.model.vision_tower
            else:
                raise AttributeError("Could not find vision tower in InternVL2 model")

            self.vision_model = self.vision_model.to(self.device)
            self.vision_model.eval()

            # Load processor
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            self.logger.info(f"✓ InternVL2 vision tower loaded on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load InternVL2 model: {e}")
            raise

        self._feature_dim = self.SUPPORTED_MODELS[self.model_name]

    def extract_single(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Extract InternVL2 vision features from a single image."""
        pil_image = self._load_image(image)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.vision_model(**inputs)

            # Get [CLS] token or pooled output
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output.cpu().numpy()[0]
            elif hasattr(outputs, 'last_hidden_state'):
                # Use [CLS] token (first token)
                features = outputs.last_hidden_state[:, 0].cpu().numpy()[0]
            else:
                raise ValueError("Could not extract features from InternVL2 vision tower")

        if self.normalize:
            features = features / (np.linalg.norm(features) + 1e-8)

        return features

    def extract_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 16,  # Smaller batch size for large models
        show_progress: bool = True
    ) -> np.ndarray:
        """Extract InternVL2 features from multiple images."""
        all_features = []

        if show_progress:
            try:
                from tqdm import tqdm
                image_batches = tqdm(
                    range(0, len(images), batch_size),
                    desc=f"Extracting InternVL2 features"
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
                    pil_images.append(Image.new('RGB', (448, 448)))

            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.vision_model(**inputs)

                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_features = outputs.pooler_output.cpu().numpy()
                else:
                    batch_features = outputs.last_hidden_state[:, 0].cpu().numpy()

            if self.normalize:
                norms = np.linalg.norm(batch_features, axis=1, keepdims=True)
                batch_features = batch_features / (norms + 1e-8)

            all_features.append(batch_features)

        return np.vstack(all_features)

    def get_feature_dim(self) -> int:
        """Return the dimensionality of InternVL2 features."""
        return self._feature_dim

    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'vision_model'):
            del self.vision_model
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("InternVL2 vision tower cleaned up")

    def __repr__(self) -> str:
        return (
            f"InternVL2FeatureExtractor("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"feature_dim={self._feature_dim})"
        )
