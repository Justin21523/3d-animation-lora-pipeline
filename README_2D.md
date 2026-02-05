# 2D Animation LoRA Pipeline

> **High-quality LoRA training pipeline for Western 2D animation characters**
>
> Optimized for TV shows like The Simpsons, Family Guy, Rick and Morty, etc.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

---

## 🎯 Overview

A comprehensive, production-ready pipeline for training character LoRA adapters from 2D animated content. Built on proven infrastructure from 3D animation pipelines, adapted for the unique characteristics of Western 2D animation.

**Key Features:**
- 🎬 **End-to-end automation**: Video → frames → segmentation → clustering → training
- 👥 **Multi-character support**: Handles 2+ characters per frame with identity tracking
- 🔧 **Flexible configuration**: Hierarchical OmegaConf system with 2D/3D parameter conversion
- 📊 **Stage-based architecture**: Checkpoint/resume support, progress tracking
- 🎨 **2D-optimized**: Hard-edge segmentation, cartoon face detection, style variation handling
- 🚀 **Production-ready**: Stub mode for rapid prototyping, robust error handling

---

## 🏗️ Architecture

### Pipeline Flow

```
Video Input
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Frame Extraction                                   │
│ • Scene-based detection or interval sampling               │
│ • High-quality frame output                                │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Multi-Character Extraction (Integrated)            │
│                                                              │
│  [2.1] YOLO Detection + Multi-Object Tracking               │
│      → Detect all characters, assign track IDs             │
│                                                              │
│  [2.2] Per-Track ToonOut Segmentation                       │
│      → Segment each character track independently          │
│                                                              │
│  [2.3] Face-Based Identity Clustering                       │
│      → Merge tracks of same character (HDBSCAN)            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: DWpose Extraction (Optional)                       │
│ • Extract pose keypoints for ControlNet conditioning       │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Dataset Building                                   │
│ • Organize character-specific training datasets            │
│ • Generate VLM-assisted captions                           │
│ • Quality filtering and augmentation                       │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: LoRA Training                                      │
│ • Train character identity LoRA adapters                   │
│ • Automatic checkpoint evaluation                          │
└─────────────────────────────────────────────────────────────┘
    ↓
Trained LoRA Models
```

### Directory Structure

```
2d-animation-lora-pipeline/
├── anime_pipeline/              # Main package
│   ├── config/                  # Configuration system (OmegaConf)
│   │   ├── omega_loader.py      # Hierarchical config loader
│   │   └── param_converter.py   # 2D/3D parameter conversion
│   ├── core/                    # Core infrastructure
│   │   ├── orchestrator.py      # Pipeline coordinator
│   │   ├── stage_manager.py     # Stage dependency manager
│   │   ├── resource_monitor.py  # GPU/CPU monitoring
│   │   ├── stub_framework.py    # Unified stub mode (Phase 4)
│   │   └── metadata_io.py       # Unified metadata I/O (Phase 4)
│   ├── frames/                  # Frame extraction
│   ├── detection/               # YOLO + tracking
│   │   └── yolo_detector.py     # Multi-object tracking
│   ├── segmentation/            # Foreground/background segmentation
│   │   └── toonout_wrapper.py   # ToonOut per-track segmentation
│   ├── clustering/              # Character identity clustering
│   │   └── identity_clustering.py  # Face-based clustering
│   ├── pose/                    # Pose estimation (DWpose)
│   ├── datasets/                # Dataset building
│   └── training/                # LoRA training
├── configs/                     # Configuration files
│   ├── global/                  # Pipeline-wide settings
│   │   ├── pipeline_stages.yaml # Stage definitions
│   │   └── param_mapping.yaml   # 2D/3D parameter mappings
│   ├── stages/                  # Stage-specific configs
│   ├── characters/              # Per-character configs
│   └── projects/                # Per-project configs
├── scripts/                     # Executable scripts
│   └── run_pipeline.py          # Main CLI entry point
└── docs/                        # Documentation
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/2d-animation-lora-pipeline.git
cd 2d-animation-lora-pipeline

# Create conda environment
conda create -n ai_env python=3.10
conda activate ai_env

# Install dependencies
pip install -r requirements/all.txt

# (Optional) Install face recognition for identity clustering
pip install insightface
```

### Basic Usage

```bash
# Run full pipeline for a 2D animation project
python scripts/run_pipeline.py \
    --project simpsons \
    --character homer \
    --mode 2d

# Run specific stages only
python scripts/run_pipeline.py \
    --project simpsons \
    --stages multi_character_extraction,dataset_building,lora_training

# Resume from checkpoint
python scripts/run_pipeline.py \
    --project simpsons \
    --start-from dataset_building

# Preview configuration without executing
python scripts/run_pipeline.py \
    --project familyguy \
    --dry-run
```

### Configuration Example

Create a project config at `configs/projects/simpsons.yaml`:

```yaml
project: simpsons
animation_mode: 2d  # Enable 2D-optimized parameters

paths:
  base_dir: /path/to/data/simpsons
  input_video: /path/to/simpsons_s01e01.mp4
  frames: ${paths.base_dir}/frames
  segmented: ${paths.base_dir}/segmented
  clustered: ${paths.base_dir}/clustered
  training_data: ${paths.base_dir}/training_data

# Frame extraction
frame_extraction:
  mode: scene
  scene_threshold: 0.3
  quality: high

# Multi-character extraction (integrated stage)
multi_character_extraction:
  tracking:
    model_path: /path/to/yolov11n.pt
    min_track_length: 10
    tracker_type: bytetrack

  segmentation:
    backend: stub  # Use "onnx" or "pytorch" with real ToonOut model

  clustering:
    min_cluster_size: 20  # 2D default (more than 3D's 10)
    device: cuda
```

---

## 📚 Key Concepts

### 2D vs 3D Animation Parameters

This pipeline automatically adapts parameters based on animation style:

| Parameter | 3D Default | 2D Default | Reason |
|-----------|------------|------------|--------|
| `alpha_threshold` | 0.15 | **0.25** | 2D has hard line-art edges |
| `blur_threshold` | 80 | **100** | 2D maintains sharper focus |
| `min_cluster_size` | 10 | **20** | 2D varies more (episodes, animators) |
| `min_samples` | 2 | **3-5** | Need more samples for style variation |
| `dataset_size` | 200-500 | **500-1000** | 2D needs more examples |
| `color_jitter` | ❌ | ✅ | 2D has color variation across episodes |

Use `--mode 2d` or `--mode 3d` to automatically apply appropriate parameters.

### Multi-Character Handling

The pipeline correctly handles multiple characters per frame:

1. **YOLO Detection + Tracking**: Detects ALL characters, assigns track IDs across frames
2. **Per-Track Segmentation**: Each character track is segmented independently
3. **Identity Clustering**: Merges tracks of the SAME character using face recognition

This ensures:
- ✅ Different characters in same scene are separated
- ✅ Same character across different scenes/cuts are merged
- ✅ Character identity is maintained throughout the video

### Stub Mode

All modules support **stub mode** for rapid prototyping without model weights:

```python
from anime_pipeline.core import StubMode, StubConfig

config = StubConfig(
    use_stub=True,  # Explicit stub mode
    backend="stub"
)

# Automatically falls back to stub if model loading fails
result = StubMode.run_with_fallback(
    model_loader=load_model,
    real_inference=run_inference,
    stub_inference=generate_stub,
    config=config,
    logger=logger
)
```

---

## 🔧 Advanced Usage

### Custom Stage Configuration

```bash
# Create custom stage config
cat > configs/custom_pipeline.yaml <<EOF
pipeline_stages:
  - name: frame_extraction
    enabled: true

  - name: multi_character_extraction
    enabled: true
    config_key: multi_character_extraction

  - name: dataset_building
    enabled: true
    dependencies: [multi_character_extraction]

  - name: lora_training
    enabled: true
    dependencies: [dataset_building]
EOF

# Run with custom config
python scripts/run_pipeline.py \
    --project my_show \
    --config configs/custom_pipeline.yaml
```

### Per-Character Configuration

```yaml
# configs/characters/homer_simpson.yaml
character_name: homer_simpson
display_name: "Homer Simpson"

# Character-specific clustering parameters
clustering:
  min_cluster_size: 25  # Homer appears frequently
  similarity_threshold: 0.75

# Character-specific training parameters
training:
  target_dataset_size: 800
  trigger_word: "homer simpson character"
  learning_rate: 1e-4
  epochs: 10
```

### Using New Utilities (Phase 4)

```python
# Unified metadata I/O
from anime_pipeline.core import MetadataIO

# Load detections with automatic fallback
detections = MetadataIO.load_records(
    path="detections.parquet",
    logger=logger,
    required_columns=["bbox_x1", "bbox_y1", "track_id"]
)

# Save with automatic format selection
actual_path = MetadataIO.save_records(
    records=detections,
    path="output.parquet",
    logger=logger,
    prefer_parquet=True
)
```

---

## 📊 Pipeline Status Monitoring

### View Progress

```python
from anime_pipeline.core import PipelineOrchestrator

orchestrator = PipelineOrchestrator(
    project="simpsons",
    mode="2d"
)

# Setup and run
orchestrator.setup_standard_pipeline()
orchestrator.run_full_pipeline()

# Get progress
progress = orchestrator.get_progress()
print(f"Progress: {progress['progress_percent']:.1f}%")
print(f"Completed: {progress['completed']}/{progress['total_stages']}")
```

### Checkpoint/Resume

```python
# Save checkpoint
orchestrator.save_checkpoint()

# Resume from checkpoint
orchestrator.resume_from_checkpoint(
    checkpoint_path="checkpoints/simpsons_checkpoint.json"
)
```

---

## 🎨 2D-Specific Features

### ToonOut Segmentation

Optimized for hard-edge 2D line art:

```yaml
segmentation:
  backend: onnx
  model_path: /path/to/toonout.onnx
  alpha_threshold: 0.25  # Higher than 3D (0.15)
  edge_detection_mode: hard_edge  # vs 3D's soft_gradient
```

### Cartoon Face Detection

Supports stylized 2D faces:

```python
from anime_pipeline.clustering import FaceDetector

detector = FaceDetector(
    device="cuda",
    min_face_size=64  # Lowered for cartoon faces
)

# Automatically tries multiple backends:
# 1. RetinaFace (works for some 2D styles)
# 2. MTCNN (fallback)
# 3. OpenCV Haar Cascades (last resort)
```

### DWpose for 2D

Extract pose keypoints for ControlNet conditioning:

```bash
python scripts/run_pipeline.py \
    --project simpsons \
    --stages frame_extraction,multi_character_extraction,dwpose_extraction
```

---

## 🔬 Testing

### Smoke Test

```bash
# Test pipeline without real model weights
python scripts/run_pipeline.py \
    --project test \
    --dry-run

# Run in stub mode (fast, no GPU needed)
python scripts/run_pipeline.py \
    --project test \
    --stages frame_extraction,multi_character_extraction
```

### Integration Test

```bash
# Test with small sample
python scripts/run_pipeline.py \
    --project test_sample \
    --mode 2d \
    --stop-at dataset_building
```

---

## 📦 AI_WAREHOUSE 3.0 Storage

This pipeline uses **AI_WAREHOUSE 3.0** centralized storage configuration:

### Storage Layout

| Location | Purpose | Speed |
|----------|---------|-------|
| `/mnt/c/ai_models/` | All model weights | Fast (SSD) |
| `/mnt/c/ai_cache/` | HuggingFace, PyTorch cache | Fast (SSD) |
| `/mnt/data/datasets/` | Source datasets | Large (HDD) |
| `/mnt/data/training/` | Training outputs | Large (HDD) |

### Quick Setup

```bash
# Set environment variables
export HF_HOME=/mnt/c/ai_cache/huggingface
export TORCH_HOME=/mnt/c/ai_cache/torch

# Or use the built-in setup
python -c "from anime_pipeline.config import setup_warehouse_environment; setup_warehouse_environment()"
```

### Validation Tools

```bash
# Validate configurations
python scripts/utils/validate_warehouse_config.py

# Scan for deprecated paths
python scripts/utils/scan_deprecated_paths.py

# Validate dataset structure
python scripts/utils/validate_dataset_structure.py

# Auto-generate project configs
python scripts/utils/generate_project_configs.py
```

See **[AI_WAREHOUSE 3.0 Guide](docs/setup/AI_WAREHOUSE_3.0.md)** for full documentation.

---

## 📖 Documentation

- **[AI_WAREHOUSE 3.0 Guide](docs/setup/AI_WAREHOUSE_3.0.md)**: Storage configuration and migration
- **[Migration Guide](docs/migration/v1_to_v2_migration.md)**: Upgrading from old pipeline
- **[Configuration Guide](docs/guides/configuration_system.md)**: OmegaConf usage
- **[Multi-Character Guide](docs/guides/multi_character_workflow.md)**: Handling multiple characters
- **[2D vs 3D Parameters](docs/guides/2d_vs_3d_parameters.md)**: Parameter comparison and rationale
  - Note: API docs are intentionally kept lightweight; see `anime_pipeline/` and `scripts/` for source-of-truth.

---

## 🛠️ Development

### Project Structure Principles

1. **Modify existing files, don't create duplicates**: Always edit in place
2. **Delete old versions after migration**: No "legacy" files
3. **Use stub mode for rapid prototyping**: All modules support stub inference
4. **Unified utilities**: Use `StubMode` and `MetadataIO` from `anime_pipeline.core`

### Adding New Stages

```python
# In anime_pipeline/core/stages.py

def execute_my_new_stage(
    project: str,
    config: Dict,
    stage_config: Optional[Dict],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Stage: My new processing stage.
    """
    logger.info("Running my new stage...")

    # Your implementation here

    return {
        'success': True,
        'items_processed': 100
    }

# Register in STAGE_REGISTRY
STAGE_REGISTRY['my_new_stage'] = execute_my_new_stage
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Follow the existing code style (English code/comments, Chinese user messages)
2. Use the unified utilities (`StubMode`, `MetadataIO`)
3. Add tests for new features
4. Update documentation

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **3D Animation Pipeline**: Core infrastructure adapted from mature 3D Pixar-style pipeline
- **ToonOut**: Foreground/background segmentation for 2D animation
- **InsightFace**: Face recognition for identity clustering
- **OmegaConf**: Hierarchical configuration system

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/2d-animation-lora-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/2d-animation-lora-pipeline/discussions)
- **Documentation**: [Full Documentation](docs/)

---

**Built with ❤️ for the 2D animation community**
