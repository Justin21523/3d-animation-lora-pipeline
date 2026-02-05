"""
Inpainting module for background filling after character extraction.

Provides two inpainting backends:
- LaMa: Fast, resolution-robust inpainting for general backgrounds
- PowerPaint: Text-guided inpainting with character-aware prompts (ECCV 2024)
"""

from anime_pipeline.inpainting.lama_inpainter import LaMaInpainter, LaMaConfig
from anime_pipeline.inpainting.powerpaint_inpainter import PowerPaintInpainter, PowerPaintConfig

__all__ = [
    "LaMaInpainter",
    "LaMaConfig",
    "PowerPaintInpainter",
    "PowerPaintConfig",
]
