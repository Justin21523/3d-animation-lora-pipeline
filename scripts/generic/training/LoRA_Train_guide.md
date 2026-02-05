# Quick Start Guide - v2.0 Modular Training System

**5-minute guide to the new modular LoRA preparation system.**

## What Changed?

**v1.0 (Old):** 80+ specialized scripts, film-specific, hard to maintain
**v2.0 (New):** Unified, modular, configurable system

## Installation

```bash
cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline
conda activate ai_env

# Verify installation
python -c "from preparers import CharacterLoRAPreparer; print('✅ Ready!')"
```

## Option 1: Single Character (Python API)

```python
from preparers import CharacterLoRAPreparer
from config import get_preset

# Get preset configuration
config = get_preset('character')

# Create preparer
preparer = CharacterLoRAPreparer(
    input_dir='/data/elio/characters/bryce',
    output_dir='/data/elio_loras/bryce_identity',
    character_name='bryce',
    config=config
)

# Run preparation
result = preparer.prepare()
print(f"✅ Prepared {result['dataset_info']['num_images']} images")
```

## Option 2: Single Character (CLI)

```bash
python scripts/generic/training/preparers/character_lora_preparer.py \
  --input-dir /data/elio/characters/bryce \
  --output-dir /data/elio_loras/bryce_identity \
  --character-name bryce \
  --preset character
```

## Option 3: Batch Processing (Multiple Characters)

### 3.1 From Config File

**Create config:** `configs/batch/elio_characters.yaml`
```yaml
base_config:
  device: cuda
  repeats: 10

jobs:
  - job_id: character_bryce
    preparer_type: character
    name: bryce
    input_dir: /data/elio/characters/bryce
    output_dir: /data/elio_loras/bryce_identity

  - job_id: character_caleb
    preparer_type: character
    name: caleb
    input_dir: /data/elio/characters/caleb
    output_dir: /data/elio_loras/caleb_identity
```

**Run batch:**
```bash
python scripts/generic/training/run_batch_preparation.py \
  --config configs/batch/elio_characters.yaml
```

### 3.2 Auto-Discovery

**If you have this structure:**
```
/data/elio/characters/
├── bryce/
├── caleb/
├── elio/
└── orion/
```

**Run:**
```bash
python scripts/generic/training/run_batch_preparation.py \
  --input-root /data/elio/characters \
  --output-root /data/elio_loras \
  --preparer-type character \
  --preset character
```

## Common Scenarios

### High-Quality Production

```python
from config import get_preset, merge_configs

config = get_preset('high_quality')  # InternVL2 + Qwen2-VL
# Slower but better results
```

### Fast Testing

```python
config = get_preset('fast')  # Template captions, minimal filtering
# Quick iteration
```

### Custom Settings

```python
from config import get_preset, merge_configs

base = get_preset('character')
overrides = {
    'repeats': 15,
    'feature_extractor': {'type': 'internvl2'},
    'caption_engine': {'type': 'qwen2_vl'}
}

config = merge_configs(base, overrides)
```

## Parallel Processing

```bash
python scripts/generic/training/run_batch_preparation.py \
  --config configs/batch/elio_characters.yaml \
  --max-workers 4  # Process 4 characters in parallel
```

## Monitoring Progress

```bash
# Watch progress
tail -f batch_preparation.log

# Check checkpoint
cat batch_output/batch_checkpoint.json | jq '.jobs[].status'
```

## Output Structure

```
output_dir/
├── 10_bryce/              # Kohya format: {repeats}_{name}
│   ├── image_001.png
│   ├── image_001.txt
│   ├── image_002.png
│   ├── image_002.txt
│   └── ...
└── preparation_metadata.json
```

## Next Steps

- **Read:** [Main README](README.md) - Full system documentation
- **Read:** [Migration Guide](MIGRATION_GUIDE.md) - Migrate from v1.0
- **Read:** [Config README](config/README.md) - Configuration options
- **Read:** [Orchestration README](orchestration/README.md) - Batch processing
- **See:** [CLEANUP_PLAN.md](../../CLEANUP_PLAN.md) - Old files to delete

## Available Preparers

| Preparer | Purpose | Preset |
|----------|---------|--------|
| `CharacterLoRAPreparer` | Character identity LoRA | `character` |
| `PoseLoRAPreparer` | Pose LoRA | `pose` |
| `ExpressionLoRAPreparer` | Expression LoRA | `expression` |
| `BackgroundLoRAPreparer` | Background/scene LoRA | `background` |
| `StyleLoRAPreparer` | Style LoRA | `style` |

## Available Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `character` | Character identity defaults | General character LoRA |
| `pose` | Pose defaults | Pose/body position LoRA |
| `expression` | Expression defaults | Facial expression LoRA |
| `background` | Background defaults | Scene/environment LoRA |
| `style` | Style defaults | Rendering style LoRA |
| `high_quality` | VLM captions, strict filters | Production quality |
| `fast` | Template captions, minimal filters | Quick testing |

## Troubleshooting

### Import Error
```bash
# Add to Python path
export PYTHONPATH="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/generic/training:$PYTHONPATH"
```

### CUDA OOM
```python
config = get_preset('character')
config['batch_size'] = 16  # Reduce from 32
config['caption_batch_size'] = 4  # Reduce from 8
```

### Config Validation Error
```python
from config import validate_config

errors = validate_config(config)
print(errors)  # Fix reported errors
```

## Migration from v1.0

**Old Way:**
```bash
python scripts/generic/training/prepare_training_data.py \
  --input /data/bryce \
  --output /data/bryce_lora \
  --character bryce \
  --min-cluster-size 12
```

**New Way:**
```bash
python scripts/generic/training/preparers/character_lora_preparer.py \
  --input-dir /data/bryce \
  --output-dir /data/bryce_lora \
  --character-name bryce \
  --preset character
```

Or use the Python API (recommended):
```python
from preparers import CharacterLoRAPreparer
from config import get_preset

preparer = CharacterLoRAPreparer(
    input_dir='/data/bryce',
    output_dir='/data/bryce_lora',
    character_name='bryce',
    config=get_preset('character')
)
preparer.prepare()
```

## Questions?

- **General docs:** [README.md](README.md)
- **API details:** [preparers/README.md](preparers/README.md)
- **Configuration:** [config/README.md](config/README.md)
- **Batch processing:** [orchestration/README.md](orchestration/README.md)
- **Migration:** [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **Cleanup:** [../../CLEANUP_PLAN.md](../../CLEANUP_PLAN.md)
