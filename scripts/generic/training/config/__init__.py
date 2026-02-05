"""
Configuration management system for LoRA training pipeline.

Provides:
- Config schema validation
- Preset configurations (character, pose, expression, background, style)
- Config loading/saving
- Config merging and overrides
"""

from .schema import validate_config, ConfigSchema
from .presets import PRESETS, get_preset, list_presets
from .loader import load_config, save_config, merge_configs

__all__ = [
    'validate_config',
    'ConfigSchema',
    'PRESETS',
    'get_preset',
    'list_presets',
    'load_config',
    'save_config',
    'merge_configs',
]
