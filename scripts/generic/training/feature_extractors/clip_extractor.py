"""
CLIP-based feature extractor.

Supports multiple CLIP model variants:
- OpenAI CLIP (ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px)
- Fine-tuned variants (e.g., laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
"""

from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import logging

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError(
        "transformers and torch are required for CLIP extractor. "
        "Install with: pip install transformers torch"
    )

import sys
sys.path.append(str(Path(__file__).parent.parent))
from base.feature_extractor import BaseFeatureExtractor


class CLIPFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor using CLIP vision encoder.

    CLIP (Contrastive Language-Image Pre-training) provides powerful
    visual representations trained on 400M+ image-text pairs.

    Attributes:
        model_name: HuggingFace model identifier
        model: CLIP model instance
        processor: CLIP image processor
        feature_dim: Dimension of extracted features
    """

    # Supported CLIP models with their feature dimensions
    SUPPORTED_MODELS = {
        # OpenAI CLIP models
        'openai/clip-vit-base-patch32': 512,
        'openai/clip-vit-base-patch16': 512,
        'openai/clip-vit-large-patch14': 768,
        'openai/clip-vit-large-patch14-336': 768,

        # LAION CLIP models (larger, better quality)
        'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': 1024,
        'laion/CLIP-ViT-L-14-laion2B-s32B-b82K': 768,
        'laion/CLIP-ViT-B-32-laion2B-s34B-b79K': 512,

        # Shorter aliases
        'clip-vit-b32': 512,
        'clip-vit-b16': 512,
        'clip-vit-l14': 768,
        'clip-vit-h14': 1024,
    }

    # Alias mapping
    MODEL_ALIASES = {
        'clip-vit-b32': 'openai/clip-vit-base-patch32',
        'clip-vit-b16': 'openai/clip-vit-base-patch16',
        'clip-vit-l14': 'openai/clip-vit-large-patch14',
        'clip-vit-h14': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        Initialize CLIP feature extractor.

        Config parameters:
            model_name (str): CLIP model to use, default 'clip-vit-l14'
            normalize (bool): Normalize features to unit length, default True
            pooling_method (str): How to pool features, default 'cls'
                Options: 'cls' (use [CLS] token), 'mean' (mean pooling)
        """
        super().__init__(config, device)

    def configure(self):
        """Load CLIP model and processor."""
        # Get model name
        self.model_name = self.config.get('model_name', 'clip-vit-l14')
        self.normalize = self.config.get('normalize', True)
        self.pooling_method = self.config.get('pooling_method', 'cls')

        # Resolve alias
        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        # Check if model is supported
        if self.model_name not in self.SUPPORTED_MODELS:
            available = ', '.join(list(self.SUPPORTED_MODELS.keys())[:8]) + '...'
            raise ValueError(
                f"Unsupported CLIP model '{self.model_name}'. "
                f"Supported models: {available}"
            )

        self.logger.info(f"Loading CLIP model: {self.model_name}")

        # Load model and processor
        try:
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"✓ CLIP model loaded on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            raise

        # Get feature dimension
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
        """
        Extract CLIP features from a single image.

        Args:
            image: Input image (file path, PIL Image, or numpy array)

        Returns:
            Feature vector with shape (feature_dim,)
        """
        # Load and preprocess image
        pil_image = self._load_image(image)

        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            if self.pooling_method == 'cls':
                # Use vision model's pooled output ([CLS] token)
                outputs = self.model.get_image_features(**inputs)
                features = outputs.cpu().numpy()[0]
            else:  # mean pooling
                # Get all patch embeddings and average
                vision_outputs = self.model.vision_model(**inputs)
                patch_embeddings = vision_outputs.last_hidden_state
                features = patch_embeddings.mean(dim=1).cpu().numpy()[0]

        # Normalize if configured
        if self.normalize:
            features = features / (np.linalg.norm(features) + 1e-8)

        return features

    def extract_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract CLIP features from multiple images (optimized batched processing).

        Args:
            images: List of input images
            batch_size: Number of images to process at once
            show_progress: Whether to show progress bar

        Returns:
            Feature matrix with shape (num_images, feature_dim)
        """
        all_features = []

        # Prepare progress bar
        if show_progress:
            try:
                from tqdm import tqdm
                image_batches = tqdm(
                    range(0, len(images), batch_size),
                    desc=f"Extracting CLIP features (batch_size={batch_size})"
                )
            except ImportError:
                image_batches = range(0, len(images), batch_size)
        else:
            image_batches = range(0, len(images), batch_size)

        # Process in batches
        for batch_start in image_batches:
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]

            # Load all images in batch
            pil_images = []
            for img in batch_images:
                try:
                    pil_images.append(self._load_image(img))
                except Exception as e:
                    self.logger.warning(f"Failed to load {img}: {e}")
                    pil_images.append(Image.new('RGB', (224, 224)))  # Placeholder

            # Process batch
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract features
            with torch.no_grad():
                if self.pooling_method == 'cls':
                    outputs = self.model.get_image_features(**inputs)
                    batch_features = outputs.cpu().numpy()
                else:  # mean pooling
                    vision_outputs = self.model.vision_model(**inputs)
                    patch_embeddings = vision_outputs.last_hidden_state
                    batch_features = patch_embeddings.mean(dim=1).cpu().numpy()

            # Normalize if configured
            if self.normalize:
                norms = np.linalg.norm(batch_features, axis=1, keepdims=True)
                batch_features = batch_features / (norms + 1e-8)

            all_features.append(batch_features)

        # Concatenate all batches
        return np.vstack(all_features)

    def get_feature_dim(self) -> int:
        """
        Return the dimensionality of CLIP features.

        Returns:
            Integer dimension of feature vectors
        """
        return self._feature_dim

    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("CLIP model cleaned up")

    @staticmethod
    def list_available_models() -> List[str]:
        """
        List all available CLIP models.

        Returns:
            List of model names
        """
        return list(CLIPFeatureExtractor.SUPPORTED_MODELS.keys())

    def __repr__(self) -> str:
        return (
            f"CLIPFeatureExtractor("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"feature_dim={self._feature_dim}, "
            f"normalize={self.normalize})"
        )


# Convenience function for quick feature extraction
def extract_clip_features(
    images: Union[List, str, Path, Image.Image],
    model_name: str = 'clip-vit-l14',
    device: str = 'cuda',
    batch_size: int = 32,
    normalize: bool = True
) -> np.ndarray:
    """
    Extract CLIP features from images.

    Args:
        images: Single image or list of images
        model_name: CLIP model to use (default 'clip-vit-l14')
        device: Device to run on (default 'cuda')
        batch_size: Batch size for processing (default 32)
        normalize: Whether to normalize features (default True)

    Returns:
        Feature array with shape (num_images, feature_dim) or (feature_dim,) for single image
    """
    config = {
        'model_name': model_name,
        'normalize': normalize
    }

    extractor = CLIPFeatureExtractor(config, device)

    # Handle single image vs list
    if isinstance(images, (str, Path, Image.Image, np.ndarray)):
        return extractor.extract_single(images)
    else:
        return extractor.extract_batch(images, batch_size=batch_size)
