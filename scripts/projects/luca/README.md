# Luca Project Scripts

This directory contains Luca-specific implementations of the 3D animation LoRA training pipeline.

## Project Information

- **Film**: Luca (Pixar, 2021)
- **Main Characters**: Luca Paguro, Alberto Scorfano, Giulia Marcovaldo
- **Special Features**: Sea monster transformations, Italian coastal setting

## Configuration Files

All configuration for the Luca project is stored in:
- **Project config**: `configs/projects/luca.yaml` - Project-level paths and settings
- **Character config**: `configs/characters/luca.yaml` - Character profiles and attributes

## Directory Structure

```
scripts/projects/luca/
├── README.md                    # This file
├── pipelines/                   # Complete pipeline implementations
│   ├── luca_dataset_pipeline_simplified.py
│   └── luca_dataset_preparation_pipeline.py
├── workflows/                   # End-to-end workflow scripts
│   ├── optimized_luca_pipeline.sh
│   ├── run_complete_luca_pipeline.sh
│   └── run_luca_dataset_pipeline.sh
├── training/                    # Training automation
│   └── auto_train_luca.sh
└── Shell scripts for specific stages:
    ├── run_caption_generation.sh
    ├── run_instance_enhancement.sh
    ├── run_pose_analysis.sh
    └── run_quality_filter.sh
```

## Usage

### Complete Pipelines

**Simplified Pipeline** (recommended for most cases):
```bash
python scripts/projects/luca/pipelines/luca_dataset_pipeline_simplified.py
```

**Full Preparation Pipeline**:
```bash
python scripts/projects/luca/pipelines/luca_dataset_preparation_pipeline.py
```

### Workflow Scripts

**Run complete pipeline** (from video to training data):
```bash
bash scripts/projects/luca/workflows/run_complete_luca_pipeline.sh
```

**Optimized pipeline** (with performance tuning):
```bash
bash scripts/projects/luca/workflows/optimized_luca_pipeline.sh
```

**Dataset pipeline only**:
```bash
bash scripts/projects/luca/workflows/run_luca_dataset_pipeline.sh
```

### Training

**Automatic training** (monitors for completed datasets and auto-trains):
```bash
bash scripts/projects/luca/training/auto_train_luca.sh
```

### Individual Stage Scripts

**Caption Generation**:
```bash
bash scripts/projects/luca/run_caption_generation.sh
```

**Instance Enhancement**:
```bash
bash scripts/projects/luca/run_instance_enhancement.sh
```

**Pose Analysis**:
```bash
bash scripts/projects/luca/run_pose_analysis.sh
```

**Quality Filtering**:
```bash
bash scripts/projects/luca/run_quality_filter.sh
```

## Data Paths

All data for the Luca project is stored under:
```
/mnt/data/ai_data/datasets/3d-anime/luca/
```

Expected subdirectories:
- `frames/` - Extracted video frames
- `instances_sampled/` - Character instances after segmentation
- `clustered/` - Identity-clustered characters
- `luca_final_data/` - Curated training dataset
- `luca_final_data_kohya/` - Formatted for Kohya training

## Dependencies

These scripts rely on the generic pipeline tools in:
- `scripts/generic/` - Reusable processing tools
- `scripts/pipelines/stages/` - Pipeline stage definitions

## Migration from Previous Structure

These files were previously located at:
- ~~`scripts/luca/`~~ → Moved to `scripts/projects/luca/`
- ~~`scripts/pipelines/luca_*.py`~~ → Moved to `scripts/projects/luca/pipelines/`
- ~~`scripts/workflows/*luca*.sh`~~ → Moved to `scripts/projects/luca/workflows/`
- ~~`scripts/training/auto_train_luca.sh`~~ → Moved to `scripts/projects/luca/training/`

## Future Work

As part of Phase 3 generalization:
- [ ] Extract hardcoded Luca-specific values to configs
- [ ] Add `--project-config` parameter support
- [ ] Make scripts work with any project config file
- [ ] Create project template for other films/characters

## See Also

- **Documentation**: `docs/projects/luca/` - Luca-specific guides and notes
- **Main README**: Project root README for overall architecture
- **Generic tools**: `scripts/generic/` for reusable components
