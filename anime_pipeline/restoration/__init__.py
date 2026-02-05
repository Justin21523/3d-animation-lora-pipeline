"""
Restoration and upscaling utilities.

Includes:
- RealESRGAN upscaling
- CodeFormer face restoration
- Adaptive contrast enhancement
- Unified enhancement pipeline
"""

from .realesrgan_wrapper import RealESRGANConfig, upscale_frames
from .enhancement_pipeline import (
    EnhancementConfig,
    EnhancementResult,
    FaceRestorationConfig,
    UpscalingConfig,
    ContrastConfig,
    CodeFormerEnhancer,
    AdaptiveContrastEnhancer,
    EnhancementPipeline,
    enhance_dataset,
)

__all__ = [
    # RealESRGAN
    "RealESRGANConfig",
    "upscale_frames",
    # Enhancement Pipeline
    "EnhancementConfig",
    "EnhancementResult",
    "FaceRestorationConfig",
    "UpscalingConfig",
    "ContrastConfig",
    "CodeFormerEnhancer",
    "AdaptiveContrastEnhancer",
    "EnhancementPipeline",
    "enhance_dataset",
]
