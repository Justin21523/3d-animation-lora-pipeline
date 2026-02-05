"""
Configuration schema validation.

Defines the expected structure and types for all configuration parameters.
Uses a simple dictionary-based schema validation approach.
"""

from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConfigSchema:
    """
    Configuration schema definitions for all preparers.

    Defines valid parameters, types, and constraints.
    """

    # Device options
    VALID_DEVICES = ['cuda', 'cpu', 'mps']

    # Feature extractor types
    VALID_EXTRACTORS = ['clip', 'eva_clip', 'dinov2', 'siglip', 'internvl2']

    # Clusterer types
    VALID_CLUSTERERS = ['hdbscan', 'kmeans', 'spectral', 'agglomerative', 'dbscan']

    # Caption engine types
    VALID_CAPTION_ENGINES = ['template', 'qwen2_vl', 'internvl2', 'llm_provider']

    # Quality filter types
    VALID_FILTERS = ['blur', 'size', 'dedup']

    # Base schema (common to all preparers)
    BASE_SCHEMA = {
        'device': {
            'type': str,
            'valid_values': VALID_DEVICES,
            'default': 'cuda',
            'description': 'Device to run models on'
        },
        'batch_size': {
            'type': int,
            'min': 1,
            'max': 256,
            'default': 32,
            'description': 'Batch size for feature extraction'
        },
        'caption_batch_size': {
            'type': int,
            'min': 1,
            'max': 64,
            'default': 8,
            'description': 'Batch size for caption generation'
        },
        'repeats': {
            'type': int,
            'min': 1,
            'max': 100,
            'default': 10,
            'description': 'Kohya repeats value'
        }
    }

    # Feature extractor schema
    FEATURE_EXTRACTOR_SCHEMA = {
        'type': {
            'type': str,
            'valid_values': VALID_EXTRACTORS,
            'default': 'clip',
            'required': True,
            'description': 'Feature extractor type'
        },
        'model_name': {
            'type': str,
            'required': False,
            'description': 'Specific model name (extractor-dependent)'
        },
        'normalize': {
            'type': bool,
            'default': True,
            'description': 'Normalize feature vectors'
        }
    }

    # Clusterer schema
    CLUSTERER_SCHEMA = {
        'type': {
            'type': str,
            'valid_values': VALID_CLUSTERERS,
            'default': 'hdbscan',
            'required': True,
            'description': 'Clustering algorithm type'
        },
        # HDBSCAN parameters
        'min_cluster_size': {
            'type': int,
            'min': 2,
            'max': 1000,
            'default': 10,
            'description': 'Minimum cluster size (HDBSCAN)'
        },
        'min_samples': {
            'type': int,
            'min': 1,
            'max': 100,
            'default': 2,
            'description': 'Minimum samples (HDBSCAN)'
        },
        # KMeans parameters
        'n_clusters': {
            'type': int,
            'min': 2,
            'max': 100,
            'description': 'Number of clusters (KMeans, Spectral, Agglomerative)'
        },
        # Common parameters
        'metric': {
            'type': str,
            'default': 'euclidean',
            'description': 'Distance metric'
        },
        'standardize': {
            'type': bool,
            'default': True,
            'description': 'Standardize features before clustering'
        }
    }

    # Caption engine schema
    CAPTION_ENGINE_SCHEMA = {
        'type': {
            'type': str,
            'valid_values': VALID_CAPTION_ENGINES,
            'default': 'template',
            'required': True,
            'description': 'Caption generation engine type'
        },
        'character_name': {
            'type': str,
            'description': 'Character/scene/style name (auto-set by preparer)'
        },
        'max_length': {
            'type': int,
            'min': 10,
            'max': 300,
            'default': 77,
            'description': 'Maximum caption length'
        },
        'min_length': {
            'type': int,
            'min': 5,
            'max': 50,
            'default': 10,
            'description': 'Minimum caption length'
        },
        'temperature': {
            'type': float,
            'min': 0.0,
            'max': 2.0,
            'default': 0.7,
            'description': 'Sampling temperature'
        },
        'prefix': {
            'type': str,
            'description': 'Caption prefix'
        },
        'suffix': {
            'type': str,
            'description': 'Caption suffix'
        }
    }

    # Quality filter schema (list of filters)
    QUALITY_FILTER_SCHEMA = {
        'type': {
            'type': str,
            'valid_values': VALID_FILTERS,
            'required': True,
            'description': 'Filter type'
        },
        # Blur filter
        'threshold': {
            'type': (int, float),
            'min': 0,
            'description': 'Blur/dedup threshold'
        },
        # Size filter
        'min_width': {
            'type': int,
            'min': 64,
            'description': 'Minimum image width'
        },
        'min_height': {
            'type': int,
            'min': 64,
            'description': 'Minimum image height'
        },
        'max_width': {
            'type': int,
            'description': 'Maximum image width'
        },
        'max_height': {
            'type': int,
            'description': 'Maximum image height'
        }
    }


def validate_config(config: Dict[str, Any], preparer_type: Optional[str] = None) -> List[str]:
    """
    Validate configuration against schema.

    Args:
        config: Configuration dictionary to validate
        preparer_type: Type of preparer (for specific validation)

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Validate base parameters
    errors.extend(_validate_section(config, ConfigSchema.BASE_SCHEMA, 'base'))

    # Validate feature extractor
    if 'feature_extractor' in config:
        errors.extend(_validate_section(
            config['feature_extractor'],
            ConfigSchema.FEATURE_EXTRACTOR_SCHEMA,
            'feature_extractor'
        ))

    # Validate clusterer
    if 'clusterer' in config:
        errors.extend(_validate_section(
            config['clusterer'],
            ConfigSchema.CLUSTERER_SCHEMA,
            'clusterer'
        ))

        # Validate clusterer-specific requirements
        clusterer_type = config['clusterer'].get('type')
        if clusterer_type in ['kmeans', 'spectral', 'agglomerative']:
            if 'n_clusters' not in config['clusterer']:
                errors.append(f"clusterer: '{clusterer_type}' requires 'n_clusters' parameter")

    # Validate caption engine
    if 'caption_engine' in config:
        errors.extend(_validate_section(
            config['caption_engine'],
            ConfigSchema.CAPTION_ENGINE_SCHEMA,
            'caption_engine'
        ))

    # Validate quality filters
    if 'quality_filters' in config:
        if not isinstance(config['quality_filters'], list):
            errors.append("quality_filters: must be a list")
        else:
            for i, filter_config in enumerate(config['quality_filters']):
                errors.extend(_validate_section(
                    filter_config,
                    ConfigSchema.QUALITY_FILTER_SCHEMA,
                    f'quality_filters[{i}]'
                ))

    return errors


def _validate_section(
    config: Dict[str, Any],
    schema: Dict[str, Dict[str, Any]],
    section_name: str
) -> List[str]:
    """
    Validate a configuration section against its schema.

    Args:
        config: Configuration section to validate
        schema: Schema definition for this section
        section_name: Name of section (for error messages)

    Returns:
        List of validation errors
    """
    errors = []

    for param_name, param_schema in schema.items():
        # Check if parameter is present
        if param_name not in config:
            if param_schema.get('required', False):
                errors.append(f"{section_name}.{param_name}: required parameter missing")
            continue

        value = config[param_name]

        # Check type
        expected_type = param_schema['type']
        if isinstance(expected_type, tuple):
            # Multiple valid types
            if not isinstance(value, expected_type):
                errors.append(
                    f"{section_name}.{param_name}: expected {expected_type}, "
                    f"got {type(value).__name__}"
                )
        else:
            if not isinstance(value, expected_type):
                errors.append(
                    f"{section_name}.{param_name}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

        # Check valid values
        if 'valid_values' in param_schema:
            if value not in param_schema['valid_values']:
                errors.append(
                    f"{section_name}.{param_name}: invalid value '{value}', "
                    f"must be one of {param_schema['valid_values']}"
                )

        # Check numeric ranges
        if isinstance(value, (int, float)):
            if 'min' in param_schema and value < param_schema['min']:
                errors.append(
                    f"{section_name}.{param_name}: value {value} < minimum {param_schema['min']}"
                )
            if 'max' in param_schema and value > param_schema['max']:
                errors.append(
                    f"{section_name}.{param_name}: value {value} > maximum {param_schema['max']}"
                )

    # Check for unknown parameters
    for param_name in config.keys():
        if param_name not in schema:
            logger.warning(f"{section_name}.{param_name}: unknown parameter (will be ignored)")

    return errors


def apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default values from schema to config.

    Args:
        config: Configuration dictionary

    Returns:
        Config with defaults applied
    """
    result = config.copy()

    # Apply base defaults
    for param_name, param_schema in ConfigSchema.BASE_SCHEMA.items():
        if param_name not in result and 'default' in param_schema:
            result[param_name] = param_schema['default']

    # Apply feature extractor defaults
    if 'feature_extractor' in result:
        for param_name, param_schema in ConfigSchema.FEATURE_EXTRACTOR_SCHEMA.items():
            if param_name not in result['feature_extractor'] and 'default' in param_schema:
                result['feature_extractor'][param_name] = param_schema['default']

    # Apply clusterer defaults
    if 'clusterer' in result:
        for param_name, param_schema in ConfigSchema.CLUSTERER_SCHEMA.items():
            if param_name not in result['clusterer'] and 'default' in param_schema:
                result['clusterer'][param_name] = param_schema['default']

    # Apply caption engine defaults
    if 'caption_engine' in result:
        for param_name, param_schema in ConfigSchema.CAPTION_ENGINE_SCHEMA.items():
            if param_name not in result['caption_engine'] and 'default' in param_schema:
                result['caption_engine'][param_name] = param_schema['default']

    return result
