"""
Feature extraction implementations for various SOTA models.

Supports multiple vision encoders for embedding extraction:
- CLIP (ViT-B/32, ViT-L/14)
- EVA-CLIP
- DINOv2 (ViT-S/B/L/G)
- SigLIP
- InternVL2 vision tower
- MAE (Masked AutoEncoder)
- Pose-based features (RTM-Pose)
"""

from .clip_extractor import CLIPFeatureExtractor, extract_clip_features
from .eva_clip_extractor import EVACLIPFeatureExtractor
from .dinov2_extractor import DINOv2FeatureExtractor
from .siglip_extractor import SigLIPFeatureExtractor
from .internvl2_extractor import InternVL2FeatureExtractor

__all__ = [
    'CLIPFeatureExtractor',
    'EVACLIPFeatureExtractor',
    'DINOv2FeatureExtractor',
    'SigLIPFeatureExtractor',
    'InternVL2FeatureExtractor',
    'extract_clip_features',
]
