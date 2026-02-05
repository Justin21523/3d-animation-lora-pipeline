"""
Configuration helpers for the pipeline.

Provides both legacy dataclass-based loader and new OmegaConf-based loader
for backward compatibility during migration.
"""

from dataclasses import is_dataclass

# Legacy dataclass loader
from .loader import load_config as load_dataclass_config

# New OmegaConf loader
from .omega_loader import (
    load_config as load_omega_config,
    get_config,
    merge_configs,
    save_config,
    validate_config,
    # AI_WAREHOUSE 3.0 support
    load_warehouse_config,
    load_models_config,
    inject_environment_variables,
    validate_no_deprecated_paths,
    get_warehouse_path,
    get_model_path,
    get_default_model,
    setup_warehouse_environment,
)

# Parameter converter for 2D/3D adaptation
from .param_converter import ParameterConverter


def load_config_auto(path, config_cls=None, **kwargs):
    """
    Auto-detect config type and load appropriately.

    - If config_cls is provided and is a dataclass, use legacy loader
    - Otherwise, use new OmegaConf loader

    Args:
        path: Config file path or name
        config_cls: Optional dataclass type for legacy loading
        **kwargs: Additional arguments passed to loader

    Returns:
        Loaded configuration (dataclass instance or DictConfig)

    Example:
        >>> # Legacy style (dataclass)
        >>> from anime_pipeline.frames.sampling import ExtractFramesConfig
        >>> config = load_config_auto("configs/frames.yaml", ExtractFramesConfig)

        >>> # New style (OmegaConf)
        >>> config = load_config_auto("pipeline", config_type="global")
    """
    if config_cls and is_dataclass(config_cls):
        return load_dataclass_config(path, config_cls, **kwargs)
    return load_omega_config(path, **kwargs)


# Default to new OmegaConf loader
load_config = load_omega_config

__all__ = [
    # Core config functions
    "load_config",
    "load_config_auto",
    "load_dataclass_config",
    "load_omega_config",
    "get_config",
    "merge_configs",
    "save_config",
    "validate_config",
    "ParameterConverter",
    # AI_WAREHOUSE 3.0 functions
    "load_warehouse_config",
    "load_models_config",
    "inject_environment_variables",
    "validate_no_deprecated_paths",
    "get_warehouse_path",
    "get_model_path",
    "get_default_model",
    "setup_warehouse_environment",
]
