# Module 6: Training Pipeline Integration - Implementation Plan

## Overview

Integrate Kohya_ss SDXL LoRA training into the synthetic data generation pipeline for fully automated workflows.

## Architecture

```
Dataset Organization (Module 5)
    ↓
Training Config Generation
    ↓
Kohya Training Launch (Background)
    ↓
Training Monitoring
    ↓
Checkpoint Evaluation (Optional)
```

## Components

### 1. Training Config Generator (`training_config_generator.py`)

**Purpose**: Generate Kohya_ss TOML configs from dataset metadata

**Key Features**:
- Auto-calculate training steps based on dataset size
- Smart epoch/save interval selection
- Template-based TOML generation
- Support for custom overrides

**Inputs**:
- Dataset directory (from dataset_organizer)
- Character name/description
- Base model path
- Output directory for LoRA
- Training hyperparameters (optional overrides)

**Outputs**:
- Generated TOML config file
- Training parameters summary

### 2. Training Launcher (`training_launcher.py`)

**Purpose**: Launch and manage Kohya_ss training processes

**Key Features**:
- Background process management
- Training progress monitoring
- GPU resource checking
- Automatic recovery on failure
- Log file management

**Inputs**:
- Generated TOML config
- Kohya_ss sd-scripts path
- Conda environment name
- Device selection

**Outputs**:
- Training process PID
- Training logs
- LoRA checkpoints

### 3. Orchestrator Integration

Add `_execute_training_integration` method to `BatchOrchestrator`:
- Generate training config
- Launch training (optional: background vs blocking)
- Monitor training progress
- Report training completion

## Implementation Strategy

### Phase 1: Config Generation (CURRENT)
- ✅ Research existing TOML format
- Create `TrainingConfig` dataclass
- Implement TOML template renderer
- Add auto-calculation logic for steps/epochs

### Phase 2: Training Launcher
- Implement background process launcher
- Add training monitor with log tailing
- Implement graceful shutdown handling
- Add checkpoint detection

### Phase 3: Orchestrator Integration
- Add TRAINING_INTEGRATION stage to PipelineStage enum
- Implement `_execute_training_integration` method
- Add training config to JobConfig.stage_configs
- Test end-to-end pipeline

### Phase 4: Testing & Validation
- Unit tests for config generator
- Integration test with mock training
- End-to-end test (vocabulary → training)
- Performance benchmarking

## Configuration Format

### JobConfig.stage_configs['training_integration']

```python
{
    'base_model_path': '/path/to/sd_xl_base_1.0.safetensors',
    'output_lora_dir': '/path/to/output/lora',
    'network_dim': 64,
    'network_alpha': 32,
    'learning_rate': 0.0001,
    'max_train_epochs': 4,
    'save_every_n_epochs': 2,
    'train_batch_size': 1,
    'gradient_accumulation_steps': 2,
    'mixed_precision': 'bf16',
    'optimizer_type': 'AdamW8bit',
    'lr_scheduler': 'cosine_with_restarts',
    'min_snr_gamma': 5.0,
    'noise_offset': 0.05,
    'resolution': '1024,1024',
    'enable_bucket': True,
    'kohya_scripts_path': '/mnt/c/ai_projects/kohya_ss/sd-scripts',
    'kohya_conda_env': 'kohya_ss',
    'run_in_background': True,  # or False for blocking mode
}
```

## Safety Considerations

1. **GPU Memory**: Check available VRAM before launching
2. **Disk Space**: Verify sufficient space for checkpoints
3. **Process Limits**: Don't launch if another training is running (optional)
4. **Config Validation**: Validate all paths exist before training
5. **Graceful Shutdown**: Handle Ctrl+C and cleanup properly

## Future Enhancements

- Automatic hyperparameter tuning based on dataset characteristics
- Multi-GPU distributed training support
- Automatic checkpoint evaluation and selection
- Training resume from interruption
- Integration with W&B/TensorBoard for metrics

## Success Criteria

✅ Generate valid Kohya TOML configs automatically
✅ Launch training in background successfully
✅ Monitor training progress with logs
✅ Produce working LoRA checkpoints
✅ Full pipeline test passes (vocab → training)

## Timeline

- Phase 1 (Config Generation): 30 minutes
- Phase 2 (Training Launcher): 45 minutes
- Phase 3 (Orchestrator Integration): 30 minutes
- Phase 4 (Testing): 30 minutes

**Total Estimated Time**: ~2.5 hours

## Current Status

**Phase 1**: In Progress (designing config generator)
