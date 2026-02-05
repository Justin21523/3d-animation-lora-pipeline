"""
Model Paths Configuration Loader

Centralized model path management for the entire pipeline.

Notes:
- The canonical registry lives at `configs/global/models.yaml`.
- Prefer OmegaConf resolution so `${...}` interpolations work.
"""

from pathlib import Path
from typing import Any, Dict


def load_model_paths(config_path: str = None) -> Dict[str, Any]:
    """Load model paths configuration from YAML"""

    if config_path is None:
        # Default config location (repo root)
        config_path = Path(__file__).resolve().parents[3] / "configs" / "global" / "models.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Model paths config not found: {config_path}")

    # Prefer OmegaConf so `${...}` interpolations resolve.
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        data = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict config at {config_path}, got {type(data)}")
        return data  # type: ignore[return-value]
    except Exception:
        # Fallback to plain YAML without interpolation resolution.
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Expected dict config at {config_path}, got {type(config)}")
        return config


def get_model_path(category: str, model_name: str, config_path: str = None) -> str:
    """
    Get specific model path

    Args:
        category: Model category (e.g., 'base_models', 'vlm_models', 'evaluation_models')
        model_name: Specific model name (e.g., 'sd_v1_5', 'internvl2')
        config_path: Optional custom config path

    Returns:
        Full path to the model

    Example:
        >>> get_model_path('base_models', 'sd_v1_5')
        '/mnt/c/AI_LLM_projects/ai_warehouse/models/base/stable-diffusion-v1-5'
    """
    config = load_model_paths(config_path)

    if category not in config:
        raise KeyError(f"Category '{category}' not found in model paths config")

    if model_name not in config[category]:
        raise KeyError(f"Model '{model_name}' not found in category '{category}'")

    return config[category][model_name]


def get_project_config(project_name: str, config_path: str = None) -> Dict[str, Any]:
    """
    Get project-specific configuration

    Args:
        project_name: Project name (e.g., 'luca')
        config_path: Optional custom config path

    Returns:
        Project configuration dictionary

    Example:
        >>> get_project_config('luca')
        {'base_model': '...', 'caption_model': '...', ...}
    """
    config = load_model_paths(config_path)

    if 'projects' not in config:
        raise KeyError("No projects defined in config")

    if project_name not in config['projects']:
        raise KeyError(f"Project '{project_name}' not found in config")

    return config['projects'][project_name]


# Convenience access to full config
MODEL_PATHS = None

def get_config():
    """Get cached config (load once, reuse)"""
    global MODEL_PATHS
    if MODEL_PATHS is None:
        MODEL_PATHS = load_model_paths()
    return MODEL_PATHS


if __name__ == '__main__':
    # Test loading
    config = load_model_paths()

    print("Model Paths Configuration Loaded:")
    print(f"  Warehouse Root: {config['warehouse_root']}")
    print(f"  Base Models: {list(config['base_models'].keys())}")
    print(f"  VLM Models: {list(config['vlm_models'].keys())}")
    print(f"  Evaluation Models: {list(config['evaluation_models'].keys())}")
    print(f"  Projects: {list(config.get('projects', {}).keys())}")

    # Test specific path retrieval
    print(f"\nExample Paths:")
    print(f"  SD v1.5: {get_model_path('base_models', 'sd_v1_5')}")
    print(f"  InternVL2: {get_model_path('evaluation_models', 'internvl2')}")

    # Test project config
    luca_config = get_project_config('luca')
    print(f"\nLuca Project Config:")
    print(f"  Base Model: {luca_config['base_model']}")
    print(f"  Caption Model: {luca_config['caption_model']}")
