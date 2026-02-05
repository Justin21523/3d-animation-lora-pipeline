"""
Preset configurations for different LoRA types.

Provides optimized default configurations for:
- Character identity LoRA
- Pose LoRA
- Expression LoRA
- Background LoRA
- Style LoRA
"""

from typing import Dict, Any, List
import copy


# Character identity LoRA preset
CHARACTER_PRESET = {
    'name': 'character',
    'description': 'Character identity LoRA (learning specific character appearances)',
    'config': {
        'device': 'cuda',
        'batch_size': 32,
        'caption_batch_size': 8,
        'repeats': 10,
        'feature_extractor': {
            'type': 'clip',
            'model_name': 'openai/clip-vit-large-patch14',
            'normalize': True
        },
        'clusterer': {
            'type': 'hdbscan',
            'min_cluster_size': 12,
            'min_samples': 2,
            'metric': 'euclidean',
            'standardize': True
        },
        'caption_engine': {
            'type': 'template',
            'max_length': 77,
            'temperature': 0.7,
            'prefix': 'a 3d animated character'
        },
        'quality_filters': [
            {'type': 'blur', 'threshold': 100.0},
            {'type': 'size', 'min_width': 256, 'min_height': 256},
            {'type': 'dedup', 'threshold': 8}
        ]
    }
}

# Pose LoRA preset
POSE_PRESET = {
    'name': 'pose',
    'description': 'Pose LoRA (learning character poses and body positions)',
    'config': {
        'device': 'cuda',
        'batch_size': 32,
        'caption_batch_size': 8,
        'repeats': 10,
        'feature_extractor': {
            'type': 'clip',
            'model_name': 'openai/clip-vit-large-patch14',
            'normalize': True
        },
        'clusterer': {
            'type': 'hdbscan',
            'min_cluster_size': 10,
            'min_samples': 2,
            'metric': 'euclidean',
            'standardize': True
        },
        'caption_engine': {
            'type': 'template',
            'max_length': 77,
            'temperature': 0.7,
            'prefix': 'a 3d animated character'
        },
        'quality_filters': [
            {'type': 'blur', 'threshold': 100.0},
            {'type': 'size', 'min_width': 384, 'min_height': 384},  # Larger for pose
        ]
    }
}

# Expression LoRA preset
EXPRESSION_PRESET = {
    'name': 'expression',
    'description': 'Expression LoRA (learning facial expressions)',
    'config': {
        'device': 'cuda',
        'batch_size': 32,
        'caption_batch_size': 8,
        'repeats': 10,
        'feature_extractor': {
            'type': 'clip',
            'model_name': 'openai/clip-vit-large-patch14',
            'normalize': True
        },
        'clusterer': {
            'type': 'hdbscan',
            'min_cluster_size': 8,
            'min_samples': 2,
            'metric': 'euclidean',
            'standardize': True
        },
        'caption_engine': {
            'type': 'template',
            'max_length': 77,
            'temperature': 0.7,
            'prefix': 'a 3d animated character'
        },
        'quality_filters': [
            {'type': 'blur', 'threshold': 120.0},  # Stricter for facial details
            {'type': 'size', 'min_width': 256, 'min_height': 256},
        ]
    }
}

# Background LoRA preset
BACKGROUND_PRESET = {
    'name': 'background',
    'description': 'Background/scene LoRA (learning environments and locations)',
    'config': {
        'device': 'cuda',
        'batch_size': 32,
        'caption_batch_size': 8,
        'repeats': 5,  # Lower for backgrounds
        'feature_extractor': {
            'type': 'clip',
            'model_name': 'openai/clip-vit-large-patch14',
            'normalize': True
        },
        'clusterer': {
            'type': 'hdbscan',
            'min_cluster_size': 15,
            'min_samples': 2,
            'metric': 'euclidean',
            'standardize': True
        },
        'caption_engine': {
            'type': 'template',
            'max_length': 77,
            'temperature': 0.7,
            'prefix': 'a 3d rendered environment'
        },
        'quality_filters': [
            {'type': 'blur', 'threshold': 80.0},  # Allow some DoF blur
            {'type': 'size', 'min_width': 512, 'min_height': 512},  # Larger for scenes
            {'type': 'dedup', 'threshold': 5}  # Aggressive deduplication
        ]
    }
}

# Style LoRA preset
STYLE_PRESET = {
    'name': 'style',
    'description': 'Style LoRA (learning rendering styles and visual aesthetics)',
    'config': {
        'device': 'cuda',
        'batch_size': 32,
        'caption_batch_size': 8,
        'repeats': 8,  # Moderate repeats
        'feature_extractor': {
            'type': 'clip',
            'model_name': 'openai/clip-vit-large-patch14',
            'normalize': True
        },
        'clusterer': {
            'type': 'kmeans',  # Fixed k for style buckets
            'n_clusters': 5,
            'standardize': True
        },
        'caption_engine': {
            'type': 'template',
            'max_length': 77,
            'temperature': 0.7,
            'prefix': 'pixar style, 3d animation'  # Will be overridden with actual style name
        },
        'quality_filters': [
            {'type': 'blur', 'threshold': 100.0},
            {'type': 'size', 'min_width': 512, 'min_height': 512},
            {'type': 'dedup', 'threshold': 8}  # Moderate deduplication
        ]
    }
}

# High-quality preset (for production)
HIGH_QUALITY_PRESET = {
    'name': 'high_quality',
    'description': 'High-quality preset with VLM captions (slower but better results)',
    'config': {
        'device': 'cuda',
        'batch_size': 16,  # Smaller batches for VLM
        'caption_batch_size': 4,
        'repeats': 10,
        'feature_extractor': {
            'type': 'internvl2',  # Strongest for 3D content
            'normalize': True
        },
        'clusterer': {
            'type': 'hdbscan',
            'min_cluster_size': 12,
            'min_samples': 2,
            'metric': 'euclidean',
            'standardize': True
        },
        'caption_engine': {
            'type': 'qwen2_vl',  # High-quality VLM
            'max_length': 77,
            'temperature': 0.7
        },
        'quality_filters': [
            {'type': 'blur', 'threshold': 120.0},  # Stricter quality
            {'type': 'size', 'min_width': 512, 'min_height': 512},
            {'type': 'dedup', 'threshold': 5}  # Aggressive dedup
        ]
    }
}

# Fast preset (for testing/prototyping)
FAST_PRESET = {
    'name': 'fast',
    'description': 'Fast preset for quick testing (template captions, relaxed filters)',
    'config': {
        'device': 'cuda',
        'batch_size': 64,  # Larger batches
        'caption_batch_size': 16,
        'repeats': 5,
        'feature_extractor': {
            'type': 'clip',
            'model_name': 'openai/clip-vit-base-patch32',  # Fastest
            'normalize': True
        },
        'clusterer': {
            'type': 'kmeans',  # Faster than HDBSCAN
            'n_clusters': 3,
            'standardize': True
        },
        'caption_engine': {
            'type': 'template',  # No inference
            'max_length': 77
        },
        'quality_filters': [
            {'type': 'size', 'min_width': 256, 'min_height': 256},  # Only size filter
        ]
    }
}

# All presets
PRESETS = {
    'character': CHARACTER_PRESET,
    'pose': POSE_PRESET,
    'expression': EXPRESSION_PRESET,
    'background': BACKGROUND_PRESET,
    'style': STYLE_PRESET,
    'high_quality': HIGH_QUALITY_PRESET,
    'fast': FAST_PRESET,
}


def get_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get a preset configuration by name.

    Args:
        preset_name: Name of preset

    Returns:
        Preset configuration (deep copy)

    Raises:
        ValueError: If preset not found
    """
    if preset_name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {available}"
        )

    # Return deep copy to avoid mutations
    return copy.deepcopy(PRESETS[preset_name]['config'])


def list_presets() -> List[Dict[str, str]]:
    """
    List all available presets with descriptions.

    Returns:
        List of preset info dicts
    """
    return [
        {
            'name': preset['name'],
            'description': preset['description']
        }
        for preset in PRESETS.values()
    ]


def get_preset_names() -> List[str]:
    """
    Get list of preset names.

    Returns:
        List of preset names
    """
    return list(PRESETS.keys())
