# Configuration System

Complete configuration management for LoRA training pipeline.

## Features

- **Schema Validation**: Automatic validation of all config parameters
- **Presets**: Pre-configured settings for different LoRA types
- **Config Merging**: Combine base configs with overrides
- **File I/O**: Load/save configs from JSON or YAML
- **CLI Integration**: Convert CLI args to config dicts

## Quick Start

### Using Presets

```python
from config import get_preset, validate_config

# Get character LoRA preset
config = get_preset('character')

# Validate
errors = validate_config(config)
if errors:
    print("Config errors:", errors)
else:
    print("Config valid!")
```

### Loading from File

```python
from config import load_config, validate_config

# Load from JSON
config = load_config('my_config.json')

# Validate
errors = validate_config(config, preparer_type='character')
if errors:
    raise ValueError(f"Invalid config: {errors}")
```

### Merging Configs

```python
from config import get_preset, merge_configs

# Start with preset
base = get_preset('character')

# Override specific settings
overrides = {
    'repeats': 15,
    'feature_extractor': {
        'type': 'internvl2'  # Upgrade to better extractor
    },
    'clusterer': {
        'min_cluster_size': 20  # Larger clusters
    }
}

# Merge
config = merge_configs(base, overrides)
```

## Available Presets

### character
Character identity LoRA (learning specific character appearances)
- Default extractor: CLIP ViT-L/14
- Default clusterer: HDBSCAN (min_cluster_size=12)
- Default caption: Template
- Quality filters: blur (100), size (256x256), dedup (8)
- Repeats: 10

### pose
Pose LoRA (learning character poses and body positions)
- Default extractor: CLIP ViT-L/14
- Default clusterer: HDBSCAN (min_cluster_size=10)
- Default caption: Template
- Quality filters: blur (100), size (384x384)
- Repeats: 10

### expression
Expression LoRA (learning facial expressions)
- Default extractor: CLIP ViT-L/14
- Default clusterer: HDBSCAN (min_cluster_size=8)
- Default caption: Template
- Quality filters: blur (120), size (256x256) - stricter for faces
- Repeats: 10

### background
Background/scene LoRA (learning environments and locations)
- Default extractor: CLIP ViT-L/14
- Default clusterer: HDBSCAN (min_cluster_size=15)
- Default caption: Template
- Quality filters: blur (80), size (512x512), dedup (5) - aggressive
- Repeats: 5 (lower for backgrounds)

### style
Style LoRA (learning rendering styles and visual aesthetics)
- Default extractor: CLIP ViT-L/14
- Default clusterer: KMeans (n_clusters=5) - fixed style buckets
- Default caption: Template
- Quality filters: blur (100), size (512x512), dedup (8)
- Repeats: 8

### high_quality
High-quality preset with VLM captions (slower but better results)
- Extractor: InternVL2 (strongest for 3D)
- Clusterer: HDBSCAN (min_cluster_size=12)
- Caption: Qwen2-VL (high-quality VLM)
- Quality filters: blur (120), size (512x512), dedup (5) - strictest
- Batch sizes: 16/4 (smaller for VLM)

### fast
Fast preset for quick testing (template captions, relaxed filters)
- Extractor: CLIP ViT-B/32 (fastest)
- Clusterer: KMeans (n_clusters=3)
- Caption: Template (no inference)
- Quality filters: size (256x256) only
- Batch sizes: 64/16 (largest)
- Repeats: 5

## Schema Reference

### Base Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `device` | str | 'cuda' | 'cuda', 'cpu', 'mps' | Device to run models on |
| `batch_size` | int | 32 | 1-256 | Feature extraction batch size |
| `caption_batch_size` | int | 8 | 1-64 | Caption generation batch size |
| `repeats` | int | 10 | 1-100 | Kohya repeats value |

### Feature Extractor Parameters

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `type` | str | 'clip' | clip, eva_clip, dinov2, siglip, internvl2 | Extractor type |
| `model_name` | str | - | (varies) | Specific model name |
| `normalize` | bool | true | - | Normalize feature vectors |

### Clusterer Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `type` | str | 'hdbscan' | hdbscan, kmeans, spectral, agglomerative, dbscan | Clustering algorithm |
| `min_cluster_size` | int | 10 | 2-1000 | Minimum cluster size (HDBSCAN) |
| `min_samples` | int | 2 | 1-100 | Minimum samples (HDBSCAN) |
| `n_clusters` | int | - | 2-100 | Number of clusters (KMeans, etc.) |
| `metric` | str | 'euclidean' | - | Distance metric |
| `standardize` | bool | true | - | Standardize features |

### Caption Engine Parameters

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `type` | str | 'template' | template, qwen2_vl, internvl2, llm_provider | Caption engine type |
| `max_length` | int | 77 | 10-300 | Maximum caption length |
| `min_length` | int | 10 | 5-50 | Minimum caption length |
| `temperature` | float | 0.7 | 0.0-2.0 | Sampling temperature |
| `prefix` | str | - | - | Caption prefix |
| `suffix` | str | - | - | Caption suffix |

### Quality Filter Parameters

Each filter is a dict with `type` and type-specific parameters:

**Blur Filter:**
- `type`: 'blur'
- `threshold`: float (min: 0) - Laplacian variance threshold

**Size Filter:**
- `type`: 'size'
- `min_width`: int (min: 64) - Minimum image width
- `min_height`: int (min: 64) - Minimum image height
- `max_width`: int (optional) - Maximum image width
- `max_height`: int (optional) - Maximum image height

**Dedup Filter:**
- `type`: 'dedup'
- `threshold`: int/float (min: 0) - Perceptual hash threshold

## Examples

### Example 1: Character LoRA with Custom Settings

```python
from config import get_preset, merge_configs, validate_config, save_config
from preparers import CharacterLoRAPreparer

# Start with character preset
config = get_preset('character')

# Customize
overrides = {
    'repeats': 15,
    'feature_extractor': {
        'type': 'internvl2'  # Better extractor
    },
    'caption_engine': {
        'type': 'qwen2_vl',  # VLM captions
        'prefix': 'miguel, a 3d animated boy'
    },
    'quality_filters': [
        {'type': 'blur', 'threshold': 120.0},  # Stricter
        {'type': 'size', 'min_width': 512, 'min_height': 512},
        {'type': 'dedup', 'threshold': 5}  # Aggressive
    ]
}

config = merge_configs(config, overrides)

# Validate
errors = validate_config(config, preparer_type='character')
if errors:
    raise ValueError(f"Config errors: {errors}")

# Save for reuse
save_config(config, 'configs/miguel_character.json')

# Use with preparer
preparer = CharacterLoRAPreparer(
    input_dir='/data/miguel',
    output_dir='/output/miguel_lora',
    character_name='miguel',
    config=config
)
preparer.prepare()
```

### Example 2: Background LoRA from JSON

```json
{
  "device": "cuda",
  "batch_size": 32,
  "repeats": 5,
  "feature_extractor": {
    "type": "clip",
    "model_name": "openai/clip-vit-large-patch14"
  },
  "clusterer": {
    "type": "hdbscan",
    "min_cluster_size": 20,
    "min_samples": 3
  },
  "caption_engine": {
    "type": "llm_provider",
    "max_length": 77,
    "prefix": "a 3d rendered beach environment"
  },
  "quality_filters": [
    {"type": "size", "min_width": 768, "min_height": 768},
    {"type": "dedup", "threshold": 3}
  ]
}
```

```python
from config import load_config, validate_config
from preparers import BackgroundLoRAPreparer

# Load config
config = load_config('beach_config.json')

# Validate
errors = validate_config(config)
assert not errors, f"Config errors: {errors}"

# Use
preparer = BackgroundLoRAPreparer(
    input_dir='/data/beach_scenes',
    output_dir='/output/beach_lora',
    scene_name='tropical_beach',
    config=config
)
preparer.prepare()
```

### Example 3: CLI Integration

```python
import argparse
from config import get_preset, cli_args_to_config, merge_configs

parser = argparse.ArgumentParser()
parser.add_argument('--preset', default='character')
parser.add_argument('--device', default='cuda')
parser.add_argument('--repeats', type=int)
parser.add_argument('--feature-extractor')
parser.add_argument('--min-cluster-size', type=int)
args = parser.parse_args()

# Start with preset
config = get_preset(args.preset)

# Override with CLI args
cli_config = cli_args_to_config(args)
config = merge_configs(config, cli_config)

# Now config has preset + CLI overrides
```

### Example 4: Fast Testing

```python
from config import get_preset
from preparers import CharacterLoRAPreparer

# Use fast preset for quick iteration
config = get_preset('fast')

preparer = CharacterLoRAPreparer(
    input_dir='/data/test_images',
    output_dir='/output/test',
    character_name='test_char',
    config=config
)

# Fast pipeline (template captions, KMeans, minimal filtering)
preparer.prepare()
```

### Example 5: Production Quality

```python
from config import get_preset
from preparers import CharacterLoRAPreparer

# Use high_quality preset for final LoRA
config = get_preset('high_quality')

preparer = CharacterLoRAPreparer(
    input_dir='/data/miguel_final',
    output_dir='/output/miguel_production',
    character_name='miguel',
    config=config
)

# Slowest but best results (InternVL2 + Qwen2-VL + strict filtering)
preparer.prepare()
```

## Validation

All configs are automatically validated against the schema:

```python
from config import validate_config

config = {
    'device': 'invalid',  # ERROR: not in ['cuda', 'cpu', 'mps']
    'batch_size': 1000,   # ERROR: > max (256)
    'feature_extractor': {
        'type': 'unknown'  # ERROR: not in valid extractors
    }
}

errors = validate_config(config)
print(errors)
# [
#   "base.device: invalid value 'invalid', must be one of ['cuda', 'cpu', 'mps']",
#   "base.batch_size: value 1000 > maximum 256",
#   "feature_extractor.type: invalid value 'unknown', must be one of [...]"
# ]
```

## API Reference

### validate_config(config, preparer_type=None)
Validate configuration against schema. Returns list of errors (empty if valid).

### get_preset(preset_name)
Get preset configuration by name. Returns deep copy.

### list_presets()
List all available presets with descriptions.

### load_config(config_path)
Load configuration from JSON or YAML file.

### save_config(config, output_path, format='json')
Save configuration to file.

### merge_configs(base_config, override_config)
Deep merge two configurations (override takes precedence).

### cli_args_to_config(args)
Convert argparse Namespace to config dict.

## Best Practices

1. **Start with presets**: Always begin with a preset that matches your use case
2. **Validate early**: Call `validate_config()` before passing to preparers
3. **Save successful configs**: Save working configs for reproducibility
4. **Use fast preset for iteration**: Test with `fast` preset, then upgrade to `high_quality`
5. **Merge don't replace**: Use `merge_configs()` to preserve preset defaults
6. **Version your configs**: Keep configs in version control alongside code
