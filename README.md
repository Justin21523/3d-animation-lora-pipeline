# 3D Animation LoRA Pipeline

A comprehensive toolkit for training high-quality LoRA models for **3D animated characters**, specifically optimized for **Pixar-style animation** and other 3D CGI content.

## Features

- **Video Preprocessing**: Frame extraction with scene detection and interpolation
- **AI-Powered Segmentation**: Multi-layer segmentation optimized for 3D content
- **Smart Character Clustering**: CLIP-based clustering with interactive refinement
- **Auto Caption Generation**: BLIP2-powered caption generation
- **Quality Training Data**: Automated dataset preparation with augmentation
- **LoRA Training**: Integration with Kohya_ss for optimized training
- **Evaluation Tools**: Comprehensive LoRA quality testing
- **CPU-Only Pose Data Preparation**: MediaPipe-powered pose detection that runs in parallel with GPU training (60+ FPS on 32-thread CPU)

## Optimized for 3D Animation

Unlike 2D anime pipelines, this is specifically designed for:
- Smooth 3D shading and realistic lighting
- Depth-aware segmentation
- Material properties (SSS, specular, etc.)
- Consistent 3D character models
- Motion capture-based animation

## Quick Start

See `.llm_provider/llm_provider.md` for detailed documentation and workflows.

### Basic Workflow

```bash
# 1. Extract frames
python scripts/generic/video/universal_frame_extractor.py \
    --input movie.mp4 \
    --output frames/ \
    --mode scene

# 2. Segment characters
python scripts/generic/segmentation/layered_segmentation.py \
    --input-dir frames/ \
    --output-dir segmented/ \
    --extract-characters

# 3. Cluster characters
python scripts/generic/clustering/character_clustering.py \
    --input-dir segmented/characters \
    --output-dir clustered/

# 4. Prepare training data
python scripts/generic/training/prepare_training_data.py \
    --character-dirs clustered/character_* \
    --output-dir training_data/ \
    --generate-captions

# 5. Train LoRA
conda run -n ai_env python sd-scripts/train_network.py \
    --config_file configs/character_training.toml
```

## Project Structure

- **scripts/core/pipeline**: 3D pipeline CLI (`python -m scripts.core.pipeline ...`)
- **scripts/run_pipeline.py**: 2D pipeline CLI (`python scripts/run_pipeline.py ...`)
- **scripts/batch**: Batch/nohup launchers (incl. Wan2.1 dataset + training)
- **scripts/training**: SDXL/Kohya training orchestration & monitoring
- **scripts/generic**: Reusable tools (segmentation, clustering, inpainting, training utils)
- **anime_pipeline/**: Packaged pipeline library (2D-focused modules)
- **docs/**: Documentation and guides
- **configs/**: Shared configuration
- **requirements/**: Python dependencies

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: RTX 3090 or better)
- 32GB+ RAM
- 500GB+ disk space for datasets

```bash
pip install -r requirements/all.txt
```

## Documentation

- **Quick Start**: `docs/guides/quick_start.md`
- **Tool Guides**: `docs/guides/tools/`
- **3D Training**: `docs/3d-training/`
- **Setup**: `docs/setup/`

## Maintenance

Repo-local generated artifacts can get very large over time (especially `outputs/` and `logs/`).

```bash
# Safe cleanup of repo-local artifacts (keeps active Wan2.1 training log if running)
bash scripts/maintenance/cleanup_repo_artifacts.sh
```

## Data Organization

All data is stored in centralized warehouse:

```
/mnt/data/ai_data/
├── datasets/3d-anime/          # Raw datasets
├── training_data/3d_characters/ # Prepared training data
├── models/lora/3d_characters/   # Trained LoRA models
└── lora_evaluation/             # Evaluation results
```

## ComfyUI Integration

ComfyUI is installed at `/mnt/c/ai_tools/comfyui/` for visual workflow design and LoRA testing.

**Quick Start:**
```bash
/mnt/c/ai_tools/comfyui/start_comfyui.sh
# Access at http://localhost:8188
```

**Pre-configured Workflows:**
- LoRA checkpoint comparison testing
- Multi-character scene composition
- Video generation pipeline
- ControlNet pose control

**Features:**
- Visual node-based workflow design
- InstantID & IPAdapter for character consistency
- RIFE frame interpolation for video generation
- SAM segmentation integration via Impact Pack
- Python API for automated testing

See [docs/guides/comfyui/COMFYUI_INTEGRATION.md](docs/guides/comfyui/COMFYUI_INTEGRATION.md) for detailed setup and usage.

## Key Tools

### Video Processing
- `universal_frame_extractor.py` - Extract frames with scene detection
- `frame_interpolator.py` - Generate intermediate frames
- `video_synthesizer.py` - Create videos from frames

### Segmentation
- `layered_segmentation.py` - Multi-layer segmentation (U²-Net, SAM, ISNet)
- Depth-aware segmentation for 3D content
- LaMa inpainting for background restoration

### Clustering
- `character_clustering.py` - CLIP + HDBSCAN clustering
- `interactive_character_selector.py` - Interactive cluster review
- `turbo_character_clustering.py` - Fast batch clustering

### Training
- `prepare_training_data.py` - Dataset organization
- `generate_captions_blip2.py` - Auto-caption generation
- `augment_small_clusters.py` - Data augmentation
- `prepare_pose_lora_data.py` - Pose LoRA dataset preparation (CPU/GPU)
- Batch scripts: `prepare_all_pose_lora_cpu.sh` - Automated batch pose data prep (32-thread optimized)

### Evaluation
- `test_lora_checkpoints.py` - Test LoRA quality
- `compare_lora_models.py` - Model comparison
- `lora_quality_metrics.py` - Quality metrics

## 3D Animation Considerations

### Segmentation Parameters

Adjust for 3D characteristics:
- Lower alpha threshold: `0.15` (vs `0.25` for 2D)
- Lower blur threshold: `80` (vs `100` for 2D)
- Consider using SAM for better 3D detection

### Clustering Parameters

3D characters are more consistent:
- Lower min_cluster_size: `10-15` (vs `20-25` for 2D)
- Lower min_samples: `2` (vs `3-5` for 2D)

### Caption Style

For identity LoRAs, captions must explicitly teach the attributes you want the LoRA to learn (e.g., age/gender).

Guidelines:
- Start captions with the trigger token (e.g., `yuwen`) and any required identity tags (e.g., `12-year-old boy, child`), then style tokens (e.g., `pixar style`).
- Keep the rest factual and descriptive (face/eyes/hair/clothing/accessories/pose/camera/lighting).
- Avoid redundant repeats of style tokens and the trigger token inside the generated tag list.

Example 3D tags:
```
"a 3D animated character with smooth shading, pixar style"
"rendered 3d character model, studio lighting, high quality animation"
```

## Supported 3D Animation Styles

- Pixar (Toy Story, Incredibles, etc.)
- DreamWorks (Shrek, How to Train Your Dragon)
- Blue Sky Studios (Ice Age, Rio)
- Illumination (Minions, Sing)
- Disney 3D (Frozen, Moana)
- Custom 3D animations

## License

For personal use and research purposes. Models and assets subject to their respective licenses.

## Version

**v1.0.0** - Initial 3D animation pipeline (2025-11-08)

---

For detailed usage, see `.llm_provider/llm_provider.md` or `docs/` directory.
# 3d-animation-lora-pipeline
