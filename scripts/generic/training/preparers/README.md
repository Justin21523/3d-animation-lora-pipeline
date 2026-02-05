# LoRA Data Preparers

Complete modular system for preparing training data for different types of LoRA adapters.

## Overview

Each preparer is a self-contained pipeline that:
1. Loads images from input directory
2. Applies quality filtering
3. Extracts features using SOTA models
4. Clusters by similarity
5. Generates captions
6. Assembles Kohya-format training dataset

All components (feature extractors, clusterers, caption engines, quality filters) are **pluggable** and configurable.

## Available Preparers

### 1. CharacterLoRAPreparer

**Purpose:** Character identity LoRA for learning specific character appearances.

**Use case:** Train LoRA to generate a specific character (e.g., "Elio", "Bryce", "Miguel").

**CLI:**
```bash
python scripts/generic/training/preparers/character_lora_preparer.py \
  --input-dir /path/to/character/images \
  --output-dir /path/to/output \
  --character-name "character_name" \
  --feature-extractor clip \
  --clusterer hdbscan \
  --caption-engine template \
  --min-cluster-size 12 \
  --repeats 10 \
  --device cuda
```

**Python API:**
```python
from preparers import CharacterLoRAPreparer

config = {
    'device': 'cuda',
    'repeats': 10,
    'feature_extractor': {'type': 'clip'},
    'clusterer': {'type': 'hdbscan', 'min_cluster_size': 12},
    'caption_engine': {'type': 'template'},
    'quality_filters': [
        {'type': 'blur', 'threshold': 100.0},
        {'type': 'size', 'min_width': 256, 'min_height': 256},
        {'type': 'dedup', 'threshold': 8}
    ]
}

preparer = CharacterLoRAPreparer(
    input_dir='/path/to/images',
    output_dir='/path/to/output',
    character_name='miguel',
    config=config
)

metadata = preparer.prepare()
```

**Output:** `{repeats}_{character_name}/` directory with images and captions.

---

### 2. PoseLoRAPreparer

**Purpose:** Pose-specific LoRA for learning character poses and body positions.

**Use case:** Train LoRA to control character poses (standing, sitting, running, specific angles).

**CLI:**
```bash
python scripts/generic/training/preparers/pose_lora_preparer.py \
  --input-dir /path/to/character/images \
  --output-dir /path/to/output \
  --character-name "character_name" \
  --feature-extractor clip \
  --clusterer hdbscan \
  --caption-engine template \
  --min-cluster-size 10 \
  --repeats 10 \
  --device cuda
```

**Key differences from character:**
- Focuses on pose similarity clustering
- Typically requires larger images (384x384+)
- Captions emphasize pose/angle/view

**Output:** `{repeats}_{character_name}_poses/` directory.

---

### 3. ExpressionLoRAPreparer

**Purpose:** Expression-specific LoRA for learning facial expressions.

**Use case:** Train LoRA to control character expressions (happy, sad, angry, surprised, etc.).

**CLI:**
```bash
python scripts/generic/training/preparers/expression_lora_preparer.py \
  --input-dir /path/to/character/images \
  --output-dir /path/to/output \
  --character-name "character_name" \
  --feature-extractor clip \
  --clusterer hdbscan \
  --caption-engine template \
  --min-cluster-size 8 \
  --repeats 10 \
  --device cuda
```

**Key differences:**
- Stricter blur threshold (120.0) for facial detail
- Clusters by expression similarity
- Captions emphasize emotion/expression

**Output:** `{repeats}_{character_name}_expressions/` directory.

---

### 4. BackgroundLoRAPreparer

**Purpose:** Background/scene LoRA for learning environments and locations.

**Use case:** Train LoRA to generate specific locations, lighting setups, or scene atmospheres.

**CLI:**
```bash
python scripts/generic/training/preparers/background_lora_preparer.py \
  --input-dir /path/to/background/images \
  --output-dir /path/to/output \
  --scene-name "scene_name" \
  --feature-extractor clip \
  --clusterer hdbscan \
  --caption-engine template \
  --min-cluster-size 15 \
  --repeats 5 \
  --device cuda
```

**Key differences:**
- Uses `--scene-name` instead of `--character-name`
- Aggressive deduplication (threshold: 5) for repetitive scenes
- Requires larger images (512x512+)
- Lower repeats (default: 5) as backgrounds need less reinforcement
- Captions emphasize setting, lighting, atmosphere

**Output:** `{repeats}_{scene_name}_background/` directory.

---

### 5. StyleLoRAPreparer

**Purpose:** Style LoRA for learning rendering styles, materials, and visual aesthetics.

**Use case:** Train LoRA to capture specific studio styles (Pixar, DreamWorks), lighting techniques, or material properties.

**CLI:**
```bash
python scripts/generic/training/preparers/style_lora_preparer.py \
  --input-dir /path/to/images \
  --output-dir /path/to/output \
  --style-name "style_name" \
  --feature-extractor clip \
  --clusterer kmeans \
  --n-clusters 5 \
  --caption-engine template \
  --repeats 8 \
  --device cuda
```

**Key differences:**
- Uses `--style-name` (e.g., "pixar_style", "dreamworks_style")
- Default clusterer is KMeans (fixed number of style buckets)
- Moderate deduplication (threshold: 8)
- Moderate repeats (default: 8)
- Captions emphasize rendering, materials, lighting, artistic style

**Output:** `{repeats}_{style_name}/` directory.

---

## Configuration System

All preparers support both CLI and Python API configuration.

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `device` | GPU device ('cuda' or 'cpu') | 'cuda' |
| `batch_size` | Feature extraction batch size | 32 |
| `caption_batch_size` | Caption generation batch size | 8 |
| `repeats` | Kohya repeats value | 10 (varies by type) |

### Feature Extractors

Available options:
- `clip` - OpenAI CLIP (default, fast and reliable)
- `eva_clip` - EVA-CLIP (improved CLIP variant)
- `dinov2` - DINOv2 (self-supervised, strong for poses)
- `siglip` - SigLIP (Google's improved CLIP)
- `internvl2` - InternVL2 vision tower (strongest for 3D content)

Example config:
```python
'feature_extractor': {
    'type': 'clip',
    'model_name': 'openai/clip-vit-large-patch14'  # Optional, model-specific
}
```

### Clusterers

Available options:
- `hdbscan` - Density-based, auto-k (default for most)
- `kmeans` - Partition-based, fixed-k (default for style)
- `spectral` - Graph-based clustering
- `agglomerative` - Hierarchical clustering
- `dbscan` - Density-based with fixed eps

Example config:
```python
'clusterer': {
    'type': 'hdbscan',
    'min_cluster_size': 12,
    'min_samples': 2,
    'metric': 'euclidean'
}
```

### Caption Engines

Available options:
- `template` - Fast template-based (default, no inference)
- `qwen2_vl` - Qwen2-VL (high-quality VLM)
- `internvl2` - InternVL2 (strongest for 3D animation)
- `llm_provider` - LLMProvider API (highest quality, requires API key)

Example config:
```python
'caption_engine': {
    'type': 'qwen2_vl',
    'max_length': 77,
    'temperature': 0.7,
    'prefix': 'a 3d animated character'
}
```

### Quality Filters

Available options:
- `blur` - Laplacian variance blur detection
- `size` - Minimum width/height filtering
- `dedup` - Perceptual hash deduplication

Example config:
```python
'quality_filters': [
    {'type': 'blur', 'threshold': 100.0},
    {'type': 'size', 'min_width': 512, 'min_height': 512},
    {'type': 'dedup', 'threshold': 8}
]
```

---

## Default Settings by Preparer Type

| Setting | Character | Pose | Expression | Background | Style |
|---------|-----------|------|------------|------------|-------|
| Min image size | 256x256 | 384x384 | 256x256 | 512x512 | 512x512 |
| Blur threshold | 100.0 | 100.0 | 120.0 | 80.0 | 100.0 |
| Dedup threshold | 8 | None | None | 5 | 8 |
| Min cluster size | 12 | 10 | 8 | 15 | N/A |
| Default clusterer | HDBSCAN | HDBSCAN | HDBSCAN | HDBSCAN | KMeans |
| Repeats | 10 | 10 | 10 | 5 | 8 |
| Caption prefix | "3d character" | "3d character" | "3d character" | "3d environment" | "{style_name}" |

---

## Output Format

All preparers produce Kohya-compatible training datasets:

```
output_dir/
├── {repeats}_{name}/
│   ├── image_001.png
│   ├── image_001.txt
│   ├── image_002.png
│   ├── image_002.txt
│   └── ...
└── preparation_metadata.json
```

**Metadata JSON:**
```json
{
  "preparer_type": "character_lora",
  "character_name": "miguel",
  "timestamp": "2025-11-22T10:30:00",
  "elapsed_seconds": 45.2,
  "config": { ... },
  "dataset_info": {
    "dataset_dir": "/path/to/10_miguel",
    "num_images": 450,
    "num_clusters": 3,
    "repeats": 10,
    "cluster_sizes": {0: 200, 1: 150, 2: 100}
  },
  "components": {
    "feature_extractor": "CLIPFeatureExtractor(...)",
    "clusterer": "HDBSCANClusterer(...)",
    "caption_engine": "TemplateCaptionEngine(...)"
  }
}
```

---

## Examples

### Example 1: Character LoRA with VLM Captions

```python
from preparers import CharacterLoRAPreparer

config = {
    'device': 'cuda',
    'repeats': 10,
    'feature_extractor': {'type': 'internvl2'},  # Strongest for 3D
    'clusterer': {'type': 'hdbscan', 'min_cluster_size': 15},
    'caption_engine': {'type': 'qwen2_vl'},  # High-quality captions
}

preparer = CharacterLoRAPreparer(
    input_dir='/data/miguel_images',
    output_dir='/data/miguel_lora',
    character_name='miguel',
    config=config
)

preparer.prepare()
```

### Example 2: Background LoRA with Aggressive Dedup

```python
from preparers import BackgroundLoRAPreparer

config = {
    'device': 'cuda',
    'repeats': 5,
    'feature_extractor': {'type': 'clip'},
    'clusterer': {'type': 'hdbscan', 'min_cluster_size': 20},
    'quality_filters': [
        {'type': 'size', 'min_width': 768, 'min_height': 768},
        {'type': 'dedup', 'threshold': 3}  # Very aggressive
    ]
}

preparer = BackgroundLoRAPreparer(
    input_dir='/data/beach_scenes',
    output_dir='/data/beach_lora',
    scene_name='tropical_beach',
    config=config
)

preparer.prepare()
```

### Example 3: Style LoRA with Fixed Clusters

```python
from preparers import StyleLoRAPreparer

config = {
    'device': 'cuda',
    'repeats': 8,
    'feature_extractor': {'type': 'clip'},
    'clusterer': {
        'type': 'kmeans',
        'n_clusters': 3  # 3 style variations
    },
    'caption_engine': {'type': 'llm_provider'}  # Highest quality
}

preparer = StyleLoRAPreparer(
    input_dir='/data/pixar_frames',
    output_dir='/data/pixar_style_lora',
    style_name='pixar_style',
    config=config
)

preparer.prepare()
```

---

## Best Practices

1. **Start with defaults:** Use CLI with default settings first to understand the pipeline
2. **Iterate on clustering:** Adjust `min_cluster_size` based on your dataset size
3. **Use VLM captions for quality:** Template is fast for testing, but VLM (Qwen2-VL, LLMProvider) produces better results
4. **Check metadata:** Always review `preparation_metadata.json` to understand cluster distribution
5. **Validate dataset:** Manually inspect a few images/captions before training
6. **Deduplication is critical:** Especially for backgrounds and styles (repetitive frames)
7. **Adjust repeats:** Character/pose need more (10), backgrounds need fewer (5)

---

## Troubleshooting

**Problem:** No clusters found (all labeled as noise)
- **Solution:** Reduce `min_cluster_size` or use KMeans with fixed k

**Problem:** Too many small clusters
- **Solution:** Increase `min_cluster_size` or use fewer feature dimensions

**Problem:** Poor caption quality
- **Solution:** Switch from `template` to `qwen2_vl` or `llm_provider`

**Problem:** Out of memory during feature extraction
- **Solution:** Reduce `batch_size` (try 16 or 8)

**Problem:** Dataset too large
- **Solution:** Increase deduplication aggressiveness (lower threshold)

---

## Architecture

All preparers follow the same modular architecture:

```
Preparer (High-level orchestration)
├── Feature Extractor (Vision encoder)
├── Clusterer (Grouping algorithm)
├── Caption Engine (Text generation)
└── Quality Filters (Image filtering)
```

Each component is **swappable** via configuration, enabling easy experimentation with different algorithms without code changes.
