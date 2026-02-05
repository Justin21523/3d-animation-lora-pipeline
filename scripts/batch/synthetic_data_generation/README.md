# Batch Synthetic Data Generation System

A fault-tolerant, scalable pipeline for generating large-scale synthetic training data using existing identity LoRAs.

## Features

- ✅ **Configurable**: YAML or command-line configuration
- ✅ **Fault-Tolerant**: Automatic retry with GPU recovery
- ✅ **Resumable**: Checkpoint system for interrupted runs
- ✅ **Scalable**: Works with any number of identity LoRAs
- ✅ **Monitored**: Real-time progress tracking
- ✅ **Logged**: Comprehensive error and status logging

## Quick Start

### 1. Prepare Configuration

Copy and customize the example config:

```bash
cp configs/batch/synthetic_data_generation_example.yaml \
   configs/batch/my_project.yaml
```

Edit `my_project.yaml` to set your paths and parameters.

### 2. Launch Pipeline

```bash
cd scripts/batch/synthetic_data_generation

# Launch in tmux (recommended for long-running tasks)
bash launch_in_tmux.sh ../../configs/batch/my_project.yaml

# Or run directly
bash batch_synthetic_data_pipeline.sh --config ../../configs/batch/my_project.yaml
```

### 3. Monitor Progress

```bash
# Real-time progress monitor
bash monitor_progress.sh /path/to/output/workspace

# Attach to tmux session
tmux attach -t synthetic_data_gen

# View logs
tail -f /path/to/output/workspace/logs/main_pipeline_*.log
```

## Pipeline Phases

The pipeline consists of 4 main phases:

### Phase 1: Vocabulary Generation (30 min)

Generates comprehensive prompt libraries for each LoRA type:
- Pose prompts (body poses, camera angles)
- Expression prompts (facial expressions)
- Action prompts (dynamic actions)

### Phase 2: Image Generation (18-20 hours)

Generates synthetic images using identity LoRAs:
- Target: 28,000+ images total
- 14 characters × 3 types × 50 prompts × 10 images
- Automatic seed randomization for variety

### Phase 3: Quality Filtering (2-3 hours)

Filters generated images by quality:
- Blur detection
- Resolution requirements
- Face detection
- NSFW filtering
- Duplicate removal
- Expected retention: 40-50%

### Phase 4: Dataset Organization (1 hour)

Organizes filtered images into training-ready datasets:
- Proper directory structure
- Caption generation
- Metadata creation
- Ready for LoRA training

## Configuration Options

### Via Config File (Recommended)

```yaml
# configs/batch/my_project.yaml
identity_loras_dir: /path/to/identity/loras
workspace_dir: /path/to/output
num_prompts_per_type: 50
images_per_prompt: 10
max_retries: 3
```

### Via Command Line

```bash
bash batch_synthetic_data_pipeline.sh \
  --lora-dir /path/to/identity/loras \
  --output-dir /path/to/output \
  --num-prompts 50 \
  --images-per-prompt 10 \
  --max-retries 3
```

## Error Recovery

### Automatic Recovery

The pipeline automatically:
- Retries failed tasks (up to 3 attempts)
- Recovers from GPU crashes
- Continues from checkpoints on restart

### Manual Recovery

If the pipeline stops:

```bash
# Simply re-run with the same config
bash launch_in_tmux.sh configs/batch/my_project.yaml

# The pipeline will:
# 1. Load checkpoint file
# 2. Skip completed tasks
# 3. Resume from interruption point
```

## Output Structure

```
workspace/
├── generated_data/           # Generated images
│   ├── character1/
│   │   ├── pose/
│   │   │   ├── prompts.json
│   │   │   ├── generated/    # Raw generated images
│   │   │   └── filtered/     # Quality-filtered images
│   │   ├── expression/
│   │   └── action/
│   └── character2/
│       └── ...
├── datasets/                 # Training-ready datasets
│   ├── character1_pose/
│   ├── character1_expression/
│   └── ...
├── logs/                     # All logs
│   ├── main_pipeline_*.log
│   ├── status.log
│   ├── errors.log
│   └── *_generation.log
└── checkpoints/              # Resume checkpoints
    └── pipeline_progress.json
```

## Monitoring

### Progress Monitor Script

```bash
bash monitor_progress.sh /path/to/workspace
```

Shows:
- Pipeline status (running/stopped)
- Per-phase progress
- Image counts (generated/filtered)
- Retention rates
- Recent activity log

### Tmux Session

```bash
# Attach to running session
tmux attach -t synthetic_data_gen

# Detach without stopping
Press: Ctrl+B, then D

# Kill session
tmux kill-session -t synthetic_data_gen
```

### Log Files

```bash
# Main pipeline log
tail -f workspace/logs/main_pipeline_*.log

# Status events
tail -f workspace/logs/status.log

# Errors only
tail -f workspace/logs/errors.log

# Per-character logs
tail -f workspace/logs/character_pose_generation.log
```

## Performance Tuning

### For Faster Generation (Lower Quality)

```yaml
num_prompts_per_type: 25        # Reduced from 50
images_per_prompt: 5            # Reduced from 10
num_inference_steps: 20         # Reduced from 30
```

Expected time: ~10-12 hours
Expected output: ~5,000-7,000 images

### For Higher Quality (Slower)

```yaml
num_prompts_per_type: 100       # Increased from 50
images_per_prompt: 15           # Increased from 10
num_inference_steps: 40         # Increased from 30
```

Expected time: ~36-48 hours
Expected output: ~60,000-80,000 images

### For Testing

```yaml
num_prompts_per_type: 5
images_per_prompt: 2
num_inference_steps: 15
```

Expected time: ~1-2 hours
Expected output: ~500-1,000 images

## Troubleshooting

### GPU Out of Memory

```bash
# Reduce batch size in filtering config
filtering:
  batch_size: 8  # Reduced from 16
```

### Generation Too Slow

```bash
# Reduce inference steps
num_inference_steps: 25  # Reduced from 30
```

### Too Many Errors

```bash
# Increase retry attempts
max_retries: 5  # Increased from 3

# Increase GPU recovery delay
gpu_recovery_delay: 180  # Increased from 120
```

### Checkpoint Corruption

```bash
# Remove checkpoint file to start fresh
rm /path/to/workspace/checkpoints/pipeline_progress.json
```

## Integration with Training

After generation completes:

```bash
# Datasets are ready for training
ls workspace/datasets/

# Use with training launcher
conda run -n ai_env python scripts/generic/training/training_launcher.py \
  --dataset-dir workspace/datasets/character_pose \
  --output-dir /path/to/loras \
  --character-name character \
  --lora-type pose
```

## Best Practices

1. **Use tmux** for long-running pipelines
2. **Monitor GPU** temperature and usage
3. **Review logs** periodically for errors
4. **Test with small config** before full run
5. **Backup checkpoints** before manual intervention
6. **Allow GPU cooldown** between runs

## Version History

- **v1.0.0** (2025-11-30): Initial release
  - Fault-tolerant pipeline with retry
  - Checkpoint/resume capability
  - GPU health monitoring
  - Comprehensive logging

## Support

For issues or questions:
1. Check logs in `workspace/logs/`
2. Review checkpoint file for progress
3. Consult error log for specific failures
4. Test with minimal config first
