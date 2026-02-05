# Batch Orchestration System

Complete batch processing system for LoRA data preparation pipelines.

## Overview

The Batch Orchestration system provides a unified, configurable interface for processing multiple characters/scenes in one run, replacing dozens of specialized bash scripts with a single, reusable Python system.

**Replaces:** ~50 bash scripts in `scripts/batch/` that were film-specific or character-specific variations of the same workflow.

## Features

- **Unified Configuration**: Single YAML/JSON config for entire batch
- **Parallel Execution**: Process multiple characters in parallel (configurable workers)
- **Progress Tracking**: Real-time status updates and checkpoints
- **Error Handling**: Continue-on-error policy with detailed error reports
- **Result Aggregation**: Comprehensive batch execution reports
- **Auto-Discovery**: Automatically discover characters from directory structure
- **Tmux Integration**: Optional tmux session management for long-running jobs

## Quick Start

### 1. Basic Usage (Config File)

```bash
python scripts/generic/training/run_batch_preparation.py \
  --config configs/batch/elio_characters.yaml
```

### 2. Auto-Discovery (From Directory Structure)

```bash
python scripts/generic/training/run_batch_preparation.py \
  --input-root /data/elio/lora_data/characters_inpainted \
  --output-root /data/elio_lora_datasets \
  --preparer-type character \
  --preset character
```

### 3. Parallel Execution

```bash
python scripts/generic/training/run_batch_preparation.py \
  --config configs/batch/elio_characters.yaml \
  --max-workers 4
```

### 4. Dry Run (Preview Jobs)

```bash
python scripts/generic/training/run_batch_preparation.py \
  --config configs/batch/elio_characters.yaml \
  --dry-run
```

## Configuration Format

### Batch Config Structure

```yaml
# Base configuration applied to all jobs
base_config:
  device: cuda
  batch_size: 32
  repeats: 10
  feature_extractor:
    type: clip
    model_name: openai/clip-vit-large-patch14
  clusterer:
    type: hdbscan
    min_cluster_size: 12
  caption_engine:
    type: qwen2_vl
  quality_filters:
    - type: blur
      threshold: 100.0
    - type: size
      min_width: 256
      min_height: 256
    - type: dedup
      threshold: 8

# Batch execution settings
batch_settings:
  max_workers: 1  # Number of parallel jobs
  continue_on_error: true
  save_checkpoints: true

# Individual jobs
jobs:
  - job_id: character_bryce
    preparer_type: character
    name: bryce
    input_dir: /data/elio/characters_inpainted/bryce
    output_dir: /data/elio_loras/bryce_lora
    config:  # Job-specific overrides
      repeats: 15

  - job_id: character_caleb
    preparer_type: character
    name: caleb
    input_dir: /data/elio/characters_inpainted/caleb
    output_dir: /data/elio_loras/caleb_lora

  - job_id: pose_elio
    preparer_type: pose
    name: elio
    input_dir: /data/elio/poses/elio
    output_dir: /data/elio_loras/elio_pose_lora
    config:
      clusterer:
        min_cluster_size: 10
```

### Job Definition

Each job requires:
- `job_id`: Unique identifier
- `preparer_type`: One of `character`, `pose`, `expression`, `background`, `style`
- `name`: Character/scene name
- `input_dir`: Input directory (character images)
- `output_dir`: Output directory (prepared dataset)
- `config`: (Optional) Job-specific config overrides

## Output Structure

```
batch_output/
├── batch_results.json       # Final execution report
├── batch_checkpoint.json    # Progress checkpoint (updated after each job)
└── <character>_lora/        # Per-character output directories
    ├── {repeats}_{name}/    # Kohya format dataset
    └── preparation_metadata.json
```

### Batch Results JSON

```json
{
  "batch_info": {
    "total_jobs": 5,
    "start_time": "2025-11-22T10:00:00",
    "end_time": "2025-11-22T12:30:00",
    "total_duration_seconds": 9000,
    "max_workers": 2
  },
  "status_counts": {
    "completed": 4,
    "failed": 1,
    "pending": 0,
    "running": 0,
    "skipped": 0
  },
  "jobs": [
    {
      "job_id": "character_bryce",
      "preparer_type": "character",
      "name": "bryce",
      "status": "completed",
      "duration_seconds": 1800,
      "num_images": 350,
      "num_clusters": 8
    },
    ...
  ],
  "successful": false
}
```

## Examples

### Example 1: Process All Characters in a Film

**Directory structure:**
```
/data/elio/characters_inpainted/
├── bryce/
├── caleb/
├── elio/
└── orion/
```

**Command:**
```bash
python scripts/generic/training/run_batch_preparation.py \
  --input-root /data/elio/characters_inpainted \
  --output-root /data/elio_loras \
  --preparer-type character \
  --preset character \
  --max-workers 2
```

### Example 2: Mixed Preparer Types

**Config file:**
```yaml
base_config:
  device: cuda
  batch_size: 32

jobs:
  # Character identity LoRAs
  - job_id: char_miguel
    preparer_type: character
    name: miguel
    input_dir: /data/coco/characters/miguel
    output_dir: /data/coco_loras/miguel_identity

  # Expression LoRAs
  - job_id: expr_miguel
    preparer_type: expression
    name: miguel
    input_dir: /data/coco/expressions/miguel
    output_dir: /data/coco_loras/miguel_expression

  # Background LoRAs
  - job_id: bg_land_of_dead
    preparer_type: background
    name: land_of_dead
    input_dir: /data/coco/backgrounds/land_of_dead
    output_dir: /data/coco_loras/land_of_dead_bg
```

**Command:**
```bash
python scripts/generic/training/run_batch_preparation.py \
  --config coco_mixed_loras.yaml \
  --max-workers 3
```

### Example 3: High-Quality Production Batch

```yaml
base_config:
  # Use high_quality preset settings
  device: cuda
  batch_size: 16
  caption_batch_size: 4
  repeats: 10

  feature_extractor:
    type: internvl2

  clusterer:
    type: hdbscan
    min_cluster_size: 12

  caption_engine:
    type: qwen2_vl
    temperature: 0.7

  quality_filters:
    - type: blur
      threshold: 120.0
    - type: size
      min_width: 512
      min_height: 512
    - type: dedup
      threshold: 5

batch_settings:
  max_workers: 1  # Sequential for stability
  continue_on_error: true
  save_checkpoints: true

jobs:
  - job_id: luca_identity
    preparer_type: character
    name: luca
    input_dir: /data/luca/characters/luca
    output_dir: /data/luca_production/luca_identity

  - job_id: alberto_identity
    preparer_type: character
    name: alberto
    input_dir: /data/luca/characters/alberto
    output_dir: /data/luca_production/alberto_identity
```

## Migration from Old Batch Scripts

### Old Way (Bash Scripts)

```bash
# Film-specific scripts for each operation
bash scripts/batch/generate_elio_identity_captions.sh
bash scripts/batch/link_elio_augmented_captions.sh
bash scripts/batch/prepare_elio_kohya_dirs.sh
bash scripts/batch/train_elio_all_characters.sh
```

**Problems:**
- ❌ Separate scripts per film
- ❌ Manual coordination required
- ❌ No progress tracking
- ❌ Hard to parallelize
- ❌ Config scattered across scripts

### New Way (Unified Orchestrator)

```bash
# Single command for entire pipeline
python scripts/generic/training/run_batch_preparation.py \
  --config configs/batch/elio_characters.yaml \
  --max-workers 4
```

**Benefits:**
- ✅ Single unified system
- ✅ Automatic coordination
- ✅ Real-time progress tracking
- ✅ Built-in parallelization
- ✅ Centralized configuration

## Mapping Old Scripts to New System

| Old Bash Script | New System Component | Notes |
|----------------|---------------------|-------|
| `generate_identity_captions.sh` | Built into preparers | Caption generation is part of preparer pipeline |
| `generate_expression_captions.sh` | ExpressionLoRAPreparer | Use `preparer_type: expression` |
| `link_augmented_captions.sh` | Built into preparers | Caption linking handled automatically |
| `prepare_*_kohya_dirs.sh` | Built into preparers | Kohya format assembly is final preparer step |
| `train_character_loras.sh` | Separate training system | LoRA training is post-preparation (future integration) |
| `augment_new_characters.sh` | Pre-preparation step | Run augmentation before batch preparation |
| `batch_identity_scene_clustering.sh` | Built into preparers | Clustering is core preparer functionality |

### Scripts to Delete

After migrating to the new system, these scripts become obsolete:

```
scripts/batch/
├── generate_identity_captions.sh        → BatchOrchestrator
├── generate_elio_identity_captions.sh   → BatchOrchestrator
├── generate_new_character_captions.sh   → BatchOrchestrator
├── generate_all_expression_captions.sh  → BatchOrchestrator
├── link_augmented_captions.sh           → BatchOrchestrator
├── link_elio_augmented_captions.sh      → BatchOrchestrator
├── link_all_augmented_captions.sh       → BatchOrchestrator
├── prepare_elio_kohya_dirs.sh           → BatchOrchestrator
├── prepare_luca_kohya_dirs.sh           → BatchOrchestrator
├── batch_lora_data_preparation.sh       → BatchOrchestrator
├── ... (many more film-specific variants)
```

**Total:** ~30 bash scripts replaced by unified Python system.

## Advanced Features

### Checkpoint and Resume

Checkpoints are automatically saved after each job:

```bash
# Initial run (interrupted)
python run_batch_preparation.py --config batch.yaml

# Resume from checkpoint
python run_batch_preparation.py --config batch.yaml  # Auto-resumes
```

### Retry Failed Jobs

```python
from orchestration import BatchOrchestrator, load_batch_config

config = load_batch_config('batch.yaml')
orchestrator = BatchOrchestrator(config)

# Run batch
results = orchestrator.run()

# Retry only failed jobs
if not results['successful']:
    retry_results = orchestrator.retry_failed_jobs()
```

### Custom Job Filtering

```python
# Skip completed jobs
completed_job_ids = {'char_bryce', 'char_caleb'}

config.jobs = [
    job for job in config.jobs
    if job['job_id'] not in completed_job_ids
]

orchestrator = BatchOrchestrator(config)
orchestrator.run()
```

## Monitoring and Logging

### Console Output

```
==========================================
BATCH EXECUTION SUMMARY
==========================================
Total jobs: 4
Duration: 7200.5s

Status breakdown:
  completed: 3
  failed: 1
  pending: 0

Results saved to: batch_output/batch_results.json
==========================================
```

### Log Files

- `batch_preparation.log` - Main execution log
- `batch_output/batch_checkpoint.json` - Real-time progress
- `batch_output/batch_results.json` - Final results

### Progress Monitoring

```bash
# Watch checkpoint file
watch -n 5 'cat batch_output/batch_checkpoint.json | jq .jobs[].status'

# Tail log file
tail -f batch_preparation.log
```

## Best Practices

1. **Start with dry-run**: Always use `--dry-run` to preview jobs
2. **Use presets**: Leverage config presets for common scenarios
3. **Save successful configs**: Keep working configs in version control
4. **Monitor first run**: Watch initial batch to tune worker count
5. **Enable checkpoints**: Essential for long-running batches
6. **Test on subset**: Validate pipeline on 1-2 characters before full batch

## Troubleshooting

### Issue: Jobs failing with CUDA OOM

**Solution:** Reduce batch size or max workers
```yaml
base_config:
  batch_size: 16  # Reduce from 32
batch_settings:
  max_workers: 1  # Sequential execution
```

### Issue: Caption generation timing out

**Solution:** Use faster caption engine for testing
```yaml
base_config:
  caption_engine:
    type: template  # Fast, no inference
```

### Issue: Checkpoints not loading

**Solution:** Check checkpoint file exists and is valid JSON
```bash
python -m json.tool batch_output/batch_checkpoint.json
```

## API Reference

See [API documentation](API.md) for programmatic usage.

## Version History

- **v2.0** (2025-11): Unified batch orchestration system
  - Replaces ~30 bash scripts
  - Parallel execution support
  - Checkpoint/resume capability
  - Comprehensive reporting

- **v1.0** (2024-11): Original bash scripts
  - Film-specific scripts
  - Manual coordination
  - Limited error handling
