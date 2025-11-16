# 3D Animation Processing Guide

> **⚠️ IMPORTANT UPDATE (2025-11-09):**
>
> **This is a comprehensive 3D animation processing pipeline** - not just for LoRA training!
>
> **Supported Use Cases:**
> - LoRA/Fine-tuning dataset preparation
> - Character analysis & recognition
> - Audio analysis & voice separation
> - Temporal & motion analysis
> - Visual effects analysis
> - Multi-modal training datasets
>
> **Film-Agnostic Architecture:** One codebase for all Pixar-style films
>
> All tools accept `--project <film_name>` to automatically load film-specific configurations.
>
> **Architecture Documentation:**
> - [`ARCHITECTURE_DESIGN_PRINCIPLES.md`](../ARCHITECTURE_DESIGN_PRINCIPLES.md) - Core design principles
> - [`PROJECT_CONFIGURATION_SYSTEM.md`](../PROJECT_CONFIGURATION_SYSTEM.md) - Configuration system
> - [`REFACTORING_ROADMAP.md`](../REFACTORING_ROADMAP.md) - Module refactoring status

---

## Complete Processing Pipeline

This guide provides a comprehensive overview of the 3D animation processing pipeline, applicable to any Pixar-style film and various AI tasks.

### Flexible Workflow (Modular)

```
Video Source
    ↓
┌───┴─────────────────────────────────────────┐
│                                             │
├─→ Frame Extraction → Preprocessing         │
│       ↓                                     │
│   Instance Segmentation (SAM2)              │
│       ↓                                     │
│   Manual Review & Filtering  ← NEW!        │
│       ↓                                     │
│   Context-Aware Inpainting (optional)       │
│       ↓                                     │
│   Identity Clustering                       │
│       ↓                                     │
│   [Various Analysis & Training Paths]       │
│                                             │
├─→ Audio Extraction → Voice Separation      │
│       ↓                                     │
│   Speaker Diarization & Character ID        │
│       ↓                                     │
│   Audio-Visual Alignment                    │
│                                             │
├─→ Temporal Analysis                         │
│   - Scene structure                         │
│   - Shot boundaries                         │
│   - Narrative flow                          │
│                                             │
├─→ Motion Analysis                           │
│   - Character movement patterns             │
│   - Camera motion                           │
│   - Action recognition                      │
│                                             │
└─→ Multi-Modal Integration                   │
    - Combined audio-visual features          │
    - Synchronized datasets                   │
    - Cross-modal analysis                    │
```

**Note:** The pipeline is **modular** - you don't need to run all stages. Choose the stages relevant to your task.

### Pipeline Stages

## A. Visual Processing Path

#### Stage 1: Frame Extraction
**Extract key frames from video using scene detection or interval sampling**

```bash
python scripts/generic/video/universal_frame_extractor.py \
  --input /path/to/movie.mp4 \
  --output /mnt/data/ai_data/datasets/3d-anime/{project}/frames \
  --mode scene \
  --scene-threshold 30.0 \
  --quality high
```

- **Tool:** `universal_frame_extractor.py`
- **Modes:** scene, interval, hybrid
- **Output:** ~10,000-15,000 frames per 90-min film
- **Film-Agnostic:** Works with any video source

---

#### Stage 2: Frame Preprocessing
**Quality filtering, deduplication, and sampling**

```bash
python scripts/generic/preprocessing/preprocess_frames.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/{project}/frames \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/frames_processed \
  --deduplicate \
  --quality-filter \
  --min-sharpness 30 \
  --max-blur 80
```

- **Tools:**
  - `preprocess_frames.py` - Main preprocessing pipeline
  - `deduplicate_frames.py` - Remove near-duplicate frames
  - `quality_filter.py` - Filter low-quality frames
- **Deduplication Methods:** pHash, SSIM, perceptual hashing
- **Quality Metrics:** Sharpness, blur, brightness, contrast
- **Sampling:** Adaptive sampling to maintain diversity

---

#### Stage 3: Multi-Character Instance Segmentation (SAM2)
**Extract EACH character/object as separate instance from multi-character frames**

```bash
python scripts/generic/segmentation/instance_segmentation.py \
  /mnt/data/ai_data/datasets/3d-anime/{project}/frames_processed \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/instances \
  --project {project} \
  --model sam2_hiera_large \
  --min-size 16384
```

- **Handles:** 2-3+ characters + objects per frame
- **Output:** All segmented instances (characters + objects + background elements)
- **Optimizations:** Balanced parameters (points=20), retry logic, GPU cache management
- **Note:** At this stage, instances include EVERYTHING - characters, objects, background elements

---

#### Stage 4: Manual Instance Review & Filtering **[CRITICAL - NEW STAGE]**
**Human-in-the-loop filtering to keep only relevant character instances**

```bash
python scripts/generic/review/instance_filter_ui.py \
  --instances-dir /mnt/data/ai_data/datasets/3d-anime/{project}/instances/instances \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/instances_filtered \
  --project {project}
```

**Purpose:**
- SAM2 extracts ALL instances: characters, objects, furniture, background elements
- Most of these are NOT needed for character analysis/training
- Human review identifies and keeps only CHARACTER instances
- Discards: furniture, objects, background elements, partial/low-quality instances

**Interactive UI Features:**
- Grid view of all instances
- Quick keyboard shortcuts for keep/discard
- Batch operations
- Undo/redo
- Auto-save progress
- Category tagging (character vs object vs background)

**Output:**
- Filtered instances containing only characters
- Metadata JSON with filtering decisions
- Statistics (kept vs discarded)

**Time Investment:** ~30-60 minutes per film (depends on instance count)

---

#### Stage 5: Context-Aware Instance Inpainting **[UPDATED]**
**Fill occluded/overlapping character regions using scene context and temporal information**

```bash
# NEW: Context-aware inpainting (uses scene frames as reference)
python scripts/generic/enhancement/inpaint_context_aware.py \
  --instances-dir /mnt/data/ai_data/datasets/3d-anime/{project}/instances_filtered \
  --frames-dir /mnt/data/ai_data/datasets/3d-anime/{project}/frames_processed \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/instances_inpainted \
  --project {project} \
  --method lama \
  --use-scene-context \
  --temporal-window 5

# Legacy: Simple inpainting (no scene context)
python scripts/generic/enhancement/inpaint_occlusions.py \
  --project {project} \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/{project}/instances_filtered \
  --method lama \
  --occlusion-threshold 0.15
```

**Context-Aware Features (NEW):**
- **Scene Context:** Uses source frame and neighboring frames as reference
- **Temporal Window:** Analyzes frames before/after for better reconstruction
- **Character-Specific:** Loads prompts from `configs/inpainting/{project}_prompts.json`
- **Multi-Reference:** Combines information from multiple frames of the same scene

**Methods:**
- **LaMa** (recommended) - Fast, balanced quality, good with scene context
- **Stable Diffusion** - High quality, uses character prompts + scene context
- **OpenCV** - Fast fallback, no context

**When to Use:** If characters frequently overlap (common in 3D multi-character scenes)

#### 4. Face-Centric Identity Clustering (ArcFace)
**Group instances by character identity using face recognition**

```bash
python scripts/generic/clustering/face_identity_clustering.py \
  /mnt/data/ai_data/datasets/3d-anime/{project}/instances/instances \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/clustered \
  --project {project} \
  --min-cluster-size 10 \
  --save-faces
```

- **Accuracy:** 95%+ identity purity
- **Robust to:** Lighting, pose, scene changes
- **Output:** 6-8 identity clusters (one per character)
- **Film-Agnostic:** `--project` parameter for configuration loading

#### 5. Interactive Review & Naming
**Web-based UI for manual review and correction**

```bash
python scripts/generic/clustering/launch_interactive_review.py \
  --cluster-dir /mnt/data/ai_data/datasets/3d-anime/{project}/clustered \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/clustered_refined
```

- **Actions:** Rename, merge, split, and reorganize clusters
- **Interface:** Web-based, keyboard-navigable
- **Output:** Clean, named character datasets
- **Time:** 15-30 minutes per film

#### 6. (Optional) Pose/View Subclustering
**Further divide each character by pose and viewing angle**

```bash
python scripts/generic/clustering/pose_subclustering.py \
  /mnt/data/ai_data/datasets/3d-anime/{project}/clustered_refined \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/pose_subclusters \
  --project {project} \
  --method umap_hdbscan
```

- **Enables:** Balanced sampling across angles/poses
- **Output:** Pose-specific buckets (front/three-quarter/profile/back)
- **Improves:** LoRA generalization

#### 7. Caption Generation
**Use VLM to generate 3D-aware captions**

```bash
python scripts/generic/training/prepare_training_data.py \
  --character-dirs /mnt/data/ai_data/datasets/3d-anime/{project}/clustered_refined/* \
  --output-dir /mnt/data/ai_data/training_data/3d_characters/{project} \
  --generate-captions \
  --caption-model qwen2_vl \
  --caption-prefix "a 3d animated character, pixar style, smooth shading"
```

- **Model:** Qwen2-VL or InternVL2
- **Focus:** Materials, lighting, camera angles
- **Output:** Image + caption pairs ready for training

#### 8. LoRA Training
**Train character-specific LoRA models**

```bash
conda run -n ai_env python sd-scripts/train_network.py \
  --config_file configs/3d_characters/{project}_{character}.toml
```

- **Dataset Size:** 200-500 images typically sufficient for 3D
- **Avoid:** Color jitter, horizontal flips (breaks PBR materials)
- **Framework:** Kohya_ss or preferred trainer

#### 9. Automated Evaluation
**Test and select best checkpoints**

```bash
python scripts/evaluation/test_lora_checkpoints.py \
  /path/to/lora/checkpoints \
  --base-model /path/to/base_model \
  --output-dir outputs/lora_testing/{project}_{character}
```

- **Metrics:** CLIP score, consistency, quality
- **Output:** Comparison grids, quality reports

---

## B. Audio Processing Path

#### Stage A1: Audio Extraction
**Extract audio track from video**

```bash
python scripts/generic/audio/audio_extractor.py \
  --input /path/to/movie.mp4 \
  --output /mnt/data/ai_data/datasets/3d-anime/{project}/audio/full_audio.wav \
  --format wav \
  --sample-rate 44100
```

- **Tool:** `audio_extractor.py`
- **Formats:** WAV (recommended), MP3, FLAC
- **Output:** High-quality audio for analysis

---

#### Stage A2: Voice Separation
**Separate vocals from background music and sound effects**

```bash
python scripts/generic/audio/voice_separator.py \
  /mnt/data/ai_data/datasets/3d-anime/{project}/audio/full_audio.wav \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/audio/separated \
  --model htdemucs \
  --two-stems vocals
```

- **Tool:** `voice_separator.py` (uses Demucs)
- **Separates:** Vocals, bass, drums, other (music/SFX)
- **Output:** Clean vocal track for character voice analysis

---

#### Stage A3: Speaker Diarization & Character Identification
**Identify who is speaking when, map to characters**

```bash
python scripts/generic/audio/speaker_diarization.py \
  --vocal-track /mnt/data/ai_data/datasets/3d-anime/{project}/audio/separated/vocals.wav \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/audio/diarization \
  --project {project} \
  --use-visual-sync
```

- **Tools:**
  - `speaker_diarization.py` - Temporal speaker segmentation
  - `character_voice_mapper.py` - Map speakers to characters
- **Features:**
  - Speaker embeddings (speaker recognition)
  - Temporal segmentation (who speaks when)
  - Visual-audio synchronization (match faces to voices)
  - Character voice profiles
- **Output:**
  - Speaker segments with timestamps
  - Character-to-voice mapping
  - Per-character voice samples

---

#### Stage A4: Voice Analysis & Feature Extraction
**Extract acoustic features for each character**

```bash
python scripts/generic/audio/voice_analyzer.py \
  --diarization-results /mnt/data/ai_data/datasets/3d-anime/{project}/audio/diarization \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/audio/analysis \
  --extract-features
```

- **Features Extracted:**
  - Pitch (F0) distribution
  - Timbre characteristics
  - Speaking rate
  - Emotional tone
  - Prosody patterns
- **Use Cases:**
  - Character voice cloning
  - Emotion recognition
  - Speaker verification
  - Multi-modal character recognition

---

## C. Temporal & Motion Analysis

#### Stage T1: Scene Structure Analysis
**Analyze narrative structure and shot boundaries**

```bash
python scripts/generic/analysis/scene_analyzer.py \
  --video /path/to/movie.mp4 \
  --frames-dir /mnt/data/ai_data/datasets/3d-anime/{project}/frames_processed \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/analysis/scenes \
  --detect-shots \
  --analyze-structure
```

- **Detects:**
  - Scene boundaries
  - Shot types (close-up, medium, wide, etc.)
  - Camera movements
  - Scene duration statistics
  - Narrative acts/sequences
- **Output:**
  - Scene segmentation JSON
  - Shot-level metadata
  - Narrative structure graph

---

#### Stage T2: Motion Analysis
**Analyze character and camera motion patterns**

```bash
python scripts/generic/analysis/motion_analyzer.py \
  --frames-dir /mnt/data/ai_data/datasets/3d-anime/{project}/frames_processed \
  --instances-dir /mnt/data/ai_data/datasets/3d-anime/{project}/instances_filtered \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/analysis/motion \
  --track-characters \
  --estimate-pose
```

- **Features:**
  - Optical flow analysis
  - Character tracking across frames
  - Pose estimation (RTM-Pose)
  - Action recognition
  - Camera motion classification
- **Use Cases:**
  - Action-conditioned generation
  - Motion-aware training
  - Pose diversity analysis
  - Dynamic scene understanding

---

#### Stage T3: Visual Effects Analysis
**Identify and catalog visual effects**

```bash
python scripts/generic/analysis/effect_analyzer.py \
  --frames-dir /mnt/data/ai_data/datasets/3d-anime/{project}/frames_processed \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/analysis/effects \
  --detect-effects
```

- **Detects:**
  - Lighting effects (magic glow, underwater, etc.)
  - Particle effects (water splashes, sparkles, etc.)
  - Weather effects (rain, snow, wind, etc.)
  - Camera effects (DoF, motion blur, lens flares)
  - Transformation sequences
- **Output:**
  - Effect catalog with timestamps
  - Effect-tagged frames
  - Effect intensity metadata

---

## D. Multi-Modal Integration

#### Stage M1: Audio-Visual Synchronization
**Align audio features with visual frames**

```bash
python scripts/generic/analysis/multimodal_sync.py \
  --video /path/to/movie.mp4 \
  --frames-dir /mnt/data/ai_data/datasets/3d-anime/{project}/frames_processed \
  --diarization-results /mnt/data/ai_data/datasets/3d-anime/{project}/audio/diarization \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/{project}/multimodal \
  --sync-precision 0.1
```

- **Synchronizes:**
  - Character faces with voices (lip sync detection)
  - Music beats with visual rhythm
  - Sound effects with visual events
- **Output:**
  - Synchronized audio-visual dataset
  - Character-voice-face triplets
  - Multimodal feature vectors

---

#### Stage M2: Integrated Dataset Construction
**Build multimodal datasets for various tasks**

```bash
python scripts/generic/analysis/dataset_builder.py \
  --project {project} \
  --task-type [character_multimodal|scene_understanding|action_recognition] \
  --output-dir /mnt/data/ai_data/training_data/multimodal/{project}
```

- **Dataset Types:**
  - Character multimodal (visual + voice + metadata)
  - Scene understanding (visual + audio + temporal)
  - Action recognition (visual + motion + audio)
  - Emotion analysis (visual + prosody + context)
- **Output Format:** HuggingFace datasets, PyTorch datasets, JSON metadata

---

## Film-Agnostic Architecture (New!)

### Core Principle

**One codebase, multiple films** - All tools are now generic and accept a `--project` parameter to load film-specific configurations.

### How It Works

```bash
# Same tool, different films
python script.py --project luca ...
python script.py --project toy_story ...
python script.py --project finding_nemo ...
```

Each film has its own configuration files:
- `configs/inpainting/{film}_prompts.json` - Character prompts for inpainting
- `configs/clustering/{film}_config.yaml` - Character names for clustering
- `configs/projects/{film}.yaml` - Overall project configuration

### Adding a New Film

**1. Create directory structure:**
```bash
mkdir -p /mnt/data/ai_data/datasets/3d-anime/{new_film}/{frames,instances,clustered}
```

**2. Create inpainting config (if using SD inpainting):**
```bash
cp configs/inpainting/template.json configs/inpainting/{new_film}_prompts.json
# Edit to add character descriptions
```

**3. Run the pipeline:**
```bash
# All steps use same --project parameter
python scripts/generic/segmentation/instance_segmentation.py \
  .../frames \
  --output-dir .../instances \
  --project {new_film}

python scripts/generic/enhancement/inpaint_occlusions.py \
  --project {new_film} \
  --method lama ...

python scripts/generic/clustering/face_identity_clustering.py \
  .../instances \
  --output-dir .../clustered \
  --project {new_film}
```

**That's it!** No code changes needed. The tools automatically:
- Load film-specific configurations
- Organize outputs correctly
- Fall back to generic mode if no config exists

### Benefits

- ✅ **No code duplication** - One implementation for all films
- ✅ **Easy to add films** - Just create configs
- ✅ **Maintainable** - Updates apply to all films
- ✅ **Flexible** - Can override with custom configs

---

## 3D-Specific Considerations

### Why 3D is Different

**Multi-character scenes:**
- 3D films have multiple characters per frame (2-3+ average)
- Traditional single-foreground extraction fails
- Solution: SAM2 instance-level segmentation

**Identity consistency:**
- 3D characters have consistent appearance across scenes
- Face recognition (ArcFace) works better than visual similarity (CLIP)
- Accuracy: 95%+ vs 60-70% with CLIP-only

**Rendering characteristics:**
- Soft anti-aliased edges (alpha threshold: 0.15 vs 0.25 for 2D)
- Intentional depth-of-field blur (blur threshold: 80 vs 100)
- Consistent lighting and materials

### Parameter Comparison

| Parameter | 2D Anime | 3D Animation |
|-----------|----------|--------------|
| Alpha Threshold | 0.25 | 0.15 |
| Blur Threshold | 100 | 80 |
| Min Cluster Size | 20-30 | 10-12 |
| Dataset Size | 1000+ | 200-500 |

See [`PARAMETERS_COMPARISON.md`](PARAMETERS_COMPARISON.md) for complete details.

---

## Prerequisites

### Required Software
- Python 3.10+
- CUDA-capable GPU (12GB+ VRAM)
- FFmpeg
- conda/mamba

### Environment Setup

```bash
# Create environment
conda create -n ai_env python=3.10 -y
conda activate ai_env

# Install PyTorch with CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements/all.txt
```

---

## Video Source Selection

### Ideal Sources
- Official movie releases (1080p or 4K)
- High-quality streaming rips
- Blu-ray sources

### Format Recommendations
- Container: MP4, MKV, TS
- Codec: H.264, H.265/HEVC
- Bitrate: 5-10 Mbps minimum

### Avoid
- Low-resolution sources (<720p)
- Heavily compressed videos
- Cam rips or poor quality encodes

---

## Expected Results

**From a 90-minute 3D film:**
- Frames: ~12,000
- Character instances (SAM2): ~3,000-4,000
- Identities found: 6-8 main characters
- Training images per character: 200-500

**Clustering Accuracy:**
- Identity Purity: 95%+
- Identity Coverage: 90%+
- Cross-scene consistency: Excellent

**Training Time:**
- Frame extraction: 1-2 hours
- SAM2 segmentation: 2-4 hours
- Identity clustering: 30 minutes
- Interactive review: 15-30 minutes
- Caption generation: 1-2 hours
- LoRA training: 4-8 hours

---

## Best Practices

### 1. Video Preparation
- Use highest quality source available
- Verify video integrity (no corruption)
- Check duration matches expected length

### 2. Frame Extraction
- Use scene-based mode for consistent quality
- Adjust `--scene-threshold` based on content (30.0 for 3D)
- Extract 10 frames per scene for good coverage

### 3. Interactive Review
- Spend 10-15 minutes reviewing clusters carefully
- Name clusters descriptively: `character_form_variant`
- Check face crops in `identity_*/faces/` for quality
- Merge split identities, split mixed clusters

### 4. Pose Subclustering
- Use for characters with >100 instances
- Enables balanced angle/pose sampling
- Improves LoRA generalization

### 5. Caption Generation
- Use Qwen2-VL-7B for best quality
- Include 3D-specific terms (materials, lighting, camera)
- Keep captions 40-77 tokens for SD compatibility

### 6. LoRA Training
- 200-500 images sufficient for 3D characters
- Avoid color jitter (breaks PBR materials)
- Avoid horizontal flips (breaks asymmetry)
- Use automated testing to select best checkpoint

---

## Troubleshooting

### Low Frame Count
- Lower `--scene-threshold`
- Increase `--frames-per-scene`
- Check video duration with ffprobe

### Instance Segmentation Issues
- Adjust `--min-size` (default: 16384 pixels)
- Use smaller SAM2 model if memory constrained
- Check visualization output

### Identity Clustering Problems
- Review face crops in `identity_*/faces/`
- Adjust `--min-cluster-size`
- Use interactive review to manually correct

### Memory Issues
- Reduce batch sizes
- Use smaller models (SAM2 small/base, RTM-Pose-S)
- Process fewer frames at a time

---

## Documentation Index

### Core Guides
- **Multi-Character Clustering:** [`docs/guides/MULTI_CHARACTER_CLUSTERING.md`](../guides/MULTI_CHARACTER_CLUSTERING.md)
- **Quick Start:** [`docs/getting-started/QUICK_START.md`](../getting-started/QUICK_START.md)
- **Interactive UI:** [`scripts/generic/clustering/interactive_ui/README.md`](../../scripts/generic/clustering/interactive_ui/README.md)

### Reference
- **3D Features:** [`3D_FEATURES.md`](3D_FEATURES.md)
- **Parameters:** [`PARAMETERS_COMPARISON.md`](PARAMETERS_COMPARISON.md)
- **Frame Extraction:** [`docs/guides/tools/UNIVERSAL_FRAME_EXTRACTION_GUIDE.md`](../guides/tools/UNIVERSAL_FRAME_EXTRACTION_GUIDE.md)
- **LoRA Testing:** [`docs/guides/tools/LORA_TESTING_GUIDE.md`](../guides/tools/LORA_TESTING_GUIDE.md)

### Character Info
- **Character Integration:** [`docs/guides/CHARACTER_INFO_INTEGRATION.md`](../guides/CHARACTER_INFO_INTEGRATION.md)
- **Character Files:** `docs/info/character_*.md`

---

**Ready to start?** Follow the [Quick Start Guide](../getting-started/QUICK_START.md) or run:

```bash
bash scripts/pipelines/run_multi_character_clustering.sh luca
```
