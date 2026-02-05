"""
Configuration loading, saving, and merging utilities.

Supports:
- Loading configs from JSON/YAML files
- Saving configs to JSON/YAML files
- Merging configs (base + overrides)
- CLI argument to config conversion
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.

    Args:
        config_path: Path to config file (.json or .yaml/.yml)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()

    if suffix == '.json':
        return _load_json(config_path)
    elif suffix in ['.yaml', '.yml']:
        return _load_yaml(config_path)
    else:
        raise ValueError(
            f"Unsupported config format '{suffix}'. Use .json or .yaml/.yml"
        )


def save_config(config: Dict[str, Any], output_path: Path, format: str = 'json'):
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        output_path: Output file path
        format: Format ('json' or 'yaml')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        _save_json(config, output_path)
    elif format == 'yaml':
        _save_yaml(config, output_path)
    else:
        raise ValueError(f"Unsupported format '{format}'. Use 'json' or 'yaml'")

    logger.info(f"Config saved to: {output_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations (override takes precedence).

    Performs deep merge: nested dictionaries are merged recursively,
    lists and primitives are replaced.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = merge_configs(result[key], value)
        else:
            # Replace value
            result[key] = value

    return result


def cli_args_to_config(args) -> Dict[str, Any]:
    """
    Convert CLI arguments to configuration dictionary.

    Args:
        args: argparse Namespace object

    Returns:
        Configuration dictionary
    """
    config = {}

    # Base parameters
    if hasattr(args, 'device'):
        config['device'] = args.device
    if hasattr(args, 'batch_size'):
        config['batch_size'] = args.batch_size
    if hasattr(args, 'repeats'):
        config['repeats'] = args.repeats

    # Feature extractor
    if hasattr(args, 'feature_extractor'):
        config['feature_extractor'] = {'type': args.feature_extractor}
        if hasattr(args, 'extractor_model'):
            config['feature_extractor']['model_name'] = args.extractor_model

    # Clusterer
    if hasattr(args, 'clusterer'):
        config['clusterer'] = {'type': args.clusterer}
        if hasattr(args, 'min_cluster_size'):
            config['clusterer']['min_cluster_size'] = args.min_cluster_size
        if hasattr(args, 'min_samples'):
            config['clusterer']['min_samples'] = args.min_samples
        if hasattr(args, 'n_clusters'):
            config['clusterer']['n_clusters'] = args.n_clusters

    # Caption engine
    if hasattr(args, 'caption_engine'):
        config['caption_engine'] = {'type': args.caption_engine}
        if hasattr(args, 'caption_prefix'):
            config['caption_engine']['prefix'] = args.caption_prefix

    # Quality filters (basic conversion)
    filters = []
    if hasattr(args, 'min_width') and args.min_width:
        filters.append({
            'type': 'size',
            'min_width': args.min_width,
            'min_height': getattr(args, 'min_height', args.min_width)
        })
    if hasattr(args, 'blur_threshold') and args.blur_threshold:
        filters.append({'type': 'blur', 'threshold': args.blur_threshold})
    if hasattr(args, 'dedup_threshold') and args.dedup_threshold:
        filters.append({'type': 'dedup', 'threshold': args.dedup_threshold})

    if filters:
        config['quality_filters'] = filters

    return config


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON config file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")


def _save_json(config: Dict[str, Any], path: Path):
    """Save config to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config files. Install: pip install pyyaml"
        )

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}")


def _save_yaml(config: Dict[str, Any], path: Path):
    """Save config to YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config files. Install: pip install pyyaml"
        )

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
