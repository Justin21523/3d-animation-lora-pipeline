# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **Pixar-style 3D character LoRA training pipeline** that processes animated video content into high-quality training datasets. The system extracts frames, segments characters, clusters identities, generates captions, and trains LoRA adapters optimized for 3D animation characteristics (smooth shading, PBR materials, cinematic lighting).

**Key Distinction:** This pipeline is specifically tuned for **3D animated content** (Pixar, DreamWorks, Disney 3D), not 2D anime. Parameters are adjusted for anti-aliased edges, depth-of-field blur, and consistent 3D character models.

## Code Language Convention

- All code and comments: **English only**
- User-facing summaries and explanations: **Traditional Chinese** (when user requests)

## Architecture

### Pipeline Flow
```
Video → Frame Extraction → Layered Segmentation → Character Extraction
  → CLIP Embeddings → HDBSCAN Clustering → Interactive Review
  → VLM Captioning → Dataset Assembly → LoRA Training → Evaluation
```

### Directory Structure
```
scripts/
├── core/              # Shared utilities (config, logging, paths)
│   ├── utils/         # config_loader.py, logger.py, path_utils.py
│   └── pipeline/      # Pipeline orchestrators and stage implementations
├── generic/           # Reusable tools for any animation content
│   ├── video/         # Frame extraction, interpolation, synthesis
│   ├── segmentation/  # Layered segmentation with multiple backends
│   ├── clustering/    # CLIP-based clustering and interactive tools
│   └── training/      # Caption generation and dataset preparation
├── evaluation/        # LoRA testing, quality metrics, model comparison
├── 3d_anime/          # 3D-specific batch workflows (if needed)
└── setup/             # Environment verification and model downloads

configs/               # Unified configuration directory
├── global/            # Pipeline-wide settings (pipeline.yaml, models.yaml)
├── training/          # Training configs and optimization presets
├── stages/            # Stage-specific configs (segmentation, inpainting, clustering, enhancement)
├── characters/        # Per-character definitions (luca.yaml, alberto.yaml, etc.)
└── projects/          # Per-film project configs (luca.yaml, etc.)
docs/                  # Comprehensive markdown guides
requirements/          # Modular dependency files (core, video, segmentation, etc.)
```

### Key Components

**Configuration System** (`scripts/core/utils/config_loader.py`):
- Uses OmegaConf to load configs from `configs/` directory
- New unified structure: global/, training/, stages/, characters/, projects/
- Automatically converts relative paths to absolute
- Access via `load_config()` function

**Enhancement Pipeline** (NEW - 2024-2025 SOTA):
- **RealESRGAN + CodeFormer**: Pixar-grade upscaling and face enhancement
- **Adaptive contrast enhancement**: Optimized for 3D smooth shading
- **CNN denoising**: Similar to Pixar's production pipeline
- Config: `configs/stages/enhancement/3d_character_enhancement.yaml`

**Character-Aware Inpainting** (NEW - PowerPaint ECCV 2024):
- **PowerPaint**: Text-guided inpainting with character descriptions
- **Temporal consistency**: Keyframe detection + optical flow propagation
- **Quality validation**: PSNR/SSIM checks with LaMa fallback
- Config: `configs/stages/inpainting/powerpaint.yaml`

**Pipeline Orchestrator** (`scripts/core/pipeline/orchestrator.py`):
- Coordinates multi-stage processing
- Manages GPU memory and batch sizes
- Supports YOLOv8-seg, U²-Net, and other backends

**Modular Stages**:
- Each tool is a standalone CLI script with argparse
- Output directories follow contracts: `character/`, `background/`, `masks/`, `*.json` metadata
- Tools never overwrite without `--force` flag

## Common Commands

### Environment Setup
```bash
# Verify installation and dependencies
python scripts/setup/verify_setup.py

# Check required model weights
python scripts/setup/download_model_weights.py
```

### End-to-End Workflow (Single Character)
```bash
# 1. Extract frames (scene-based detection)
python scripts/generic/video/universal_frame_extractor.py \
  --input /path/to/video.mp4 \
  --output /mnt/data/ai_data/datasets/3d-anime/PROJECT/frames \
  --mode scene \
  --scene-threshold 0.3 \
  --quality high

# 2. Segment characters (3D defaults: alpha=0.15, blur=80)
python scripts/generic/segmentation/layered_segmentation.py \
  --input-dir .../frames \
  --output-dir .../segmented \
  --model u2net \
  --alpha-threshold 0.15 \
  --blur-threshold 80 \
  --extract-characters \
  --batch-size 8

# 3. Pre-filter instances (NEW - removes 80-90% of background)
python scripts/generic/clustering/instance_prefilter.py \
  --input-dir .../segmented/characters \
  --output-dir .../filtered_instances \
  --mode balanced \
  --enable-semantic \
  --batch-report .../prefilter_report.json

# 4. Cluster by identity (HDBSCAN with 3D defaults)
python scripts/generic/clustering/character_clustering.py \
  --input-dir .../filtered_instances \
  --output-dir .../clustered \
  --min-cluster-size 12 \
  --min-samples 2 \
  --quality-filter \
  --use-face-detection

# 5. Interactive review (optional but recommended)
python scripts/generic/clustering/interactive_character_selector.py \
  --cluster-dir .../clustered \
  --output-dir .../clustered_refined

# 6. (Optional) Character-aware inpainting - PowerPaint
python scripts/generic/inpainting/character_inpainting.py \
  --input-dir .../clustered_refined \
  --output-dir .../inpainted \
  --model powerpaint \
  --character-info docs/films/PROJECT/characters/ \
  --config configs/stages/inpainting/powerpaint.yaml

# 7. (Optional) High-quality enhancement - RealESRGAN + CodeFormer
python scripts/generic/enhancement/frame_enhancement.py \
  --input-dir .../inpainted \
  --output-dir .../enhanced \
  --mode quality \
  --config configs/stages/enhancement/3d_character_enhancement.yaml

# 8. Prepare training dataset with VLM captions
python scripts/generic/training/prepare_training_data.py \
  --character-dirs .../clustered_refined/character_0 \
  --output-dir .../training_data/CHARACTER_NAME \
  --character-name "character_name" \
  --generate-captions \
  --caption-prefix "a 3d animated character, pixar style, smooth shading" \
  --target-size 400

# 9. Train LoRA (uses Kohya_ss sd-scripts)
conda run -n ai_env python sd-scripts/train_network.py \
  --config_file configs/3d_characters/CHARACTER_NAME.toml

# 10. Test checkpoints
python scripts/evaluation/test_lora_checkpoints.py \
  /path/to/lora_checkpoints/ \
  --base-model runwayml/stable-diffusion-v1-5 \
  --output-dir outputs/lora_testing/CHARACTER_NAME \
  --device cuda
```

### Fast Parallel Variants
```bash
# Parallel segmentation (multi-GPU or batch optimization)
python scripts/generic/segmentation/layered_segmentation_parallel.py \
  --input-dir .../frames \
  --output-dir .../segmented \
  --num-gpus 2

# Turbo clustering (optimized for large datasets)
python scripts/generic/clustering/turbo_character_clustering.py \
  --input-dir .../characters \
  --output-dir .../clustered \
  --fast-mode
```

### Evaluation and Comparison
```bash
# Compare multiple LoRA checkpoints
python scripts/evaluation/compare_lora_models.py \
  --lora-paths lora1.safetensors lora2.safetensors \
  --base-model SD1.5 \
  --output-dir comparison/

# Quality metrics (CLIP score, consistency)
python scripts/evaluation/lora_quality_metrics.py \
  --lora-path best_checkpoint.safetensors \
  --test-prompts prompts/3d_character_test.json
```

## Critical 3D-Specific Parameters

**These differ from 2D anime pipelines:**

| Parameter | 2D Anime | 3D Animation | Why |
|-----------|----------|--------------|-----|
| `alpha-threshold` | 0.25 | **0.15** | Soft anti-aliased edges in 3D |
| `blur-threshold` | 100 | **80** | Intentional DoF blur in cinematic renders |
| `min-cluster-size` | 20-25 | **10-15** | 3D models are consistent across frames |
| `min-samples` | 3-5 | **2** | Tighter identity clusters |
| Dataset size | 500-1000 | **200-500** | 3D needs fewer examples |
| Color augmentation | ✅ | **❌** | Breaks PBR materials |
| Horizontal flip | ✅ | **❌** | Breaks asymmetric accessories |

**Always use 3D defaults unless explicitly targeting 2D content.**

## Configuration Files

**Global Config** (`config/global_config.yaml`):
- Warehouse paths: `/mnt/data/ai_data/`
- Model settings: CLIP, BLIP2, segmentation backends
- Hardware: GPU device, VRAM limits, num_workers
- 3D-specific thresholds

**Training Config Template** (`configs/3d_character_training.toml`):
- Copy and customize for each character
- Set `output_dir`, `image_dir`, `output_name`
- Adjust learning rate, epochs, network dim

**Load config in code:**
```python
from scripts.core.utils.config_loader import load_config
config = load_config("global_config")  # Returns OmegaConf object
```

## Output Contracts

**Segmentation** (`layered_segmentation.py`):
```
output_dir/
├── character/          # Isolated character images
├── background/         # Inpainted backgrounds
├── masks/              # Alpha masks
└── segmentation_results.json
```

**Clustering** (`character_clustering.py`):
```
output_dir/
├── character_0/        # Identity cluster 0
├── character_1/        # Identity cluster 1
├── noise/              # Rejected/unclassified images
├── cluster_report.json # Metadata and stats
└── cluster_visualization.png
```

**Training Dataset** (`prepare_training_data.py`):
```
output_dir/
├── images/             # Curated character images
├── captions/           # .txt files matching image names
└── metadata.json       # Dataset stats and config
```

**Evaluation** (`test_lora_checkpoints.py`):
```
output_dir/
├── checkpoint_epoch3/  # Test images per checkpoint
├── checkpoint_epoch6/
├── quality_evaluation.json
└── comparison_grid.png
```

## Testing Workflow

No formal unit tests exist yet. Testing is done via:
1. **Smoke tests:** Run each tool with `--help` to verify imports
2. **Integration tests:** Process a small sample video end-to-end
3. **Manual QA:** Review clustering visualizations and caption quality

**To add tests:**
```bash
# Create tests/ directory
mkdir -p tests/{unit,integration}

# Example unit test structure
tests/unit/test_config_loader.py
tests/unit/test_image_utils.py
tests/integration/test_full_pipeline.py
```

## Important Notes

1. **Data Paths:** All production data lives under `/mnt/data/ai_data/` warehouse structure (datasets, training_data, models, lora_evaluation)

2. **GPU Memory:** Default batch sizes assume 16GB VRAM. Reduce if OOM errors occur.

3. **Dependencies:** Modular requirements in `requirements/` (core, video, segmentation, clustering, audio). Install specific modules or use `requirements/all.txt`.

4. **Kohya_ss Integration:** LoRA training uses external `sd-scripts` repository (not included). Must be cloned separately.

5. **VLM Captioning:** References Qwen2-VL and InternVL2 for schema-guided captions, but BLIP2 is the implemented default.

6. **Determinism:** Always seed RNGs. Output artifacts to timestamped `runs/` or `outputs/` directories.

7. **WSL2 Considerations:** See `docs/setup/WSL_LONG_RUNNING_GUIDE.md` for handling overnight processing jobs.

## Detailed Documentation

**Comprehensive guides exist in `docs/`:**
- `.claude/claude.md` - **Extended project instructions and pipeline details** (primary reference)
- `docs/guides/tools/` - Per-tool usage guides (frame extraction, segmentation, clustering, LoRA testing)
- `docs/3d_anime_specific/` - 3D features, processing guide, parameter comparisons
- `docs/getting-started/QUICK_START.md` - Step-by-step tutorial with Toy Story example

**Consult these before implementing new features to align with existing patterns.**

## Version

v1.0.0 - Initial 3D animation pipeline (2025-11-08)
- 等一下 你好像搞錯了 我們現在早就該回到3d-animation-lora-pipeline原先的專案了阿 另外  後面記得我想先來規劃相關後續實作的程式碼這樣之後 等SAM相關分割inpainting都結束後就可以開始進行了