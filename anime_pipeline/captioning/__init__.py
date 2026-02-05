"""
Captioning module for 2D Animation LoRA Pipeline.

Provides VLM-based caption generation for animation characters:
- VLMCaptioner: Local GPU-based captioning (Qwen2-VL/InternVL2)
- OpenAICaptioner: Cloud API-based captioning (GPT-4V)
- SDXLCaptionExpander: Expand short captions to SDXL-optimized longer captions
- Caption templates for 2D/3D animation styles
- Schema-guided structured outputs

Usage:
    # Local GPU captioning
    from anime_pipeline.captioning import VLMCaptioner
    captioner = VLMCaptioner(device="cuda")
    result = captioner.generate_caption(image_path)

    # OpenAI API captioning (no GPU required)
    from anime_pipeline.captioning import OpenAICaptioner
    captioner = OpenAICaptioner()  # Uses OPENAI_API_KEY from env
    result = captioner.generate_caption(image_path)

    # SDXL caption expansion (no GPU required)
    from anime_pipeline.captioning import SDXLCaptionExpander
    expander = SDXLCaptionExpander()  # Uses OPENAI_API_KEY from env
    result = expander.expand_caption("short caption", style="2d")
"""

from .vlm_captioner import (
    VLMCaptioner,
    CaptionResult,
    BatchCaptionResult,
    apply_template,
    CAPTION_TEMPLATES,
    DEFAULT_PROMPTS,
)

from .openai_captioner import OpenAICaptioner

from .sdxl_caption_expander import (
    SDXLCaptionExpander,
    ExpandedCaption,
    BatchExpandResult,
    SDXL_QUALITY_PREFIXES,
    SDXL_TECHNICAL_DETAILS,
    SDXL_NEGATIVE_PROMPTS,
)

__all__ = [
    "VLMCaptioner",
    "OpenAICaptioner",
    "SDXLCaptionExpander",
    "CaptionResult",
    "BatchCaptionResult",
    "ExpandedCaption",
    "BatchExpandResult",
    "apply_template",
    "CAPTION_TEMPLATES",
    "DEFAULT_PROMPTS",
    "SDXL_QUALITY_PREFIXES",
    "SDXL_TECHNICAL_DETAILS",
    "SDXL_NEGATIVE_PROMPTS",
]
