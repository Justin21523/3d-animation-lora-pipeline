# Face-Based Identity Clustering Guide

## Overview

The **Face-Based Identity Clustering** pipeline automatically groups character instances by **WHO they are** (identity), not just visual similarity. This is critical for multi-character scenes where traditional CLIP-based clustering would group characters from the same scene together rather than separating different identities.

### Key Advantages

- **Correctly groups the SAME character** across:
  - Different poses and expressions
  - Different lighting conditions
  - Different backgrounds
  - Different camera angles

- **Separates DIFFERENT characters** even when they:
  - Appear in the same scene
  - Have similar clothing or colors
  - Share similar visual features

### Technical Approach

```
Instance Images
    ↓
Face Detection (RetinaFace/MTCNN/OpenCV)
    ↓
Face Recognition (InsightFace ArcFace R100)
    ↓
Identity Embeddings (512-dimensional)
    ↓
Dimensionality Reduction (Optional PCA)
    ↓
Clustering (HDBSCAN)
    ↓
Per-Character Folders
```

## Installation

### Dependencies

All dependencies are listed in `requirements/identity_clustering.txt`:

```bash
conda run -n ai_env pip install -r requirements/identity_clustering.txt
```

**Critical:** After installation, ensure numpy is downgraded to 1.26.4:

```bash
conda run -n ai_env pip install "numpy<2.0" --force-reinstall
```

### Verify Installation

```bash
conda run -n ai_env python -c "
import insightface
from sklearn.cluster import HDBSCAN
import umap
print('✅ All dependencies ready!')
"
```

## Basic Usage

### Input Requirements

- **Input directory:** Character instance images from SAM2 segmentation
- **Image format:** PNG or JPG
- **Face visibility:** At least 64×64 pixels of face must be visible
- **Minimum instances:** At least 10 instances per character recommended

### Command

```bash
conda run -n ai_env python scripts/generic/clustering/face_identity_clustering.py \
  /path/to/instances/ \
  --output-dir /path/to/output/ \
  --min-cluster-size 10 \
  --device cuda \
  --save-faces
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `instances_dir` | (required) | Directory with character instance images |
| `--output-dir` | (required) | Output directory for clustered identities |
| `--min-cluster-size` | 10 | Minimum instances per identity cluster |
| `--device` | cuda | Device to use (cuda or cpu) |
| `--save-faces` | False | Save detected face crops for inspection |

## Output Structure

```
output_dir/
├── identity_000/           # Character 0
│   ├── instance_0001.png   # Full character instance
│   ├── instance_0023.png
│   ├── faces/              # Face crops (if --save-faces)
│   │   ├── instance_0001.png
│   │   └── instance_0023.png
├── identity_001/           # Character 1
│   ├── instance_0005.png
│   └── ...
├── noise/                  # Instances that don't fit any cluster
│   └── instance_0999.png
└── identity_clustering.json  # Metadata and statistics
```

## Complete Workflow

### Step 1: Run SAM2 Instance Segmentation

First, extract character instances from frames:

```bash
conda run -n ai_env python scripts/generic/segmentation/instance_segmentation.py \
  /path/to/frames/ \
  --output-dir /path/to/instances/ \
  --device cuda
```

This produces: `/path/to/instances/instances/*.png`

### Step 2: Run Identity Clustering

Cluster instances by character identity:

```bash
conda run -n ai_env python scripts/generic/clustering/face_identity_clustering.py \
  /path/to/instances/instances/ \
  --output-dir /path/to/clustered/ \
  --min-cluster-size 10 \
  --device cuda \
  --save-faces
```

This produces: Per-character folders in `/path/to/clustered/`

### Step 3: Interactive Review (Optional)

Use the web-based UI to review, merge, split, or rename clusters:

```bash
conda run -n ai_env python scripts/generic/clustering/launch_interactive_review.py \
  /path/to/clustered/
```

This opens a browser interface for manual refinement.

### Step 4: Pose Subclustering (Advanced, Optional)

For balanced training data, subdivide each identity by pose and view:

```bash
conda run -n ai_env python scripts/generic/clustering/pose_subclustering.py \
  /path/to/clustered/identity_000/ \
  --output-dir /path/to/clustered/identity_000_poses/ \
  --device cuda
```

This creates pose-specific subfolders (front, three-quarter, profile, etc.)

## Advanced Configuration

### Face Detection Options

The pipeline automatically tries face detectors in this order:

1. **RetinaFace** (best for 3D characters)
2. **MTCNN** (fallback)
3. **OpenCV Haar Cascades** (last resort)

To install RetinaFace:

```bash
conda run -n ai_env pip install retinaface-pytorch
```

### Face Recognition Models

The pipeline uses **InsightFace** with ArcFace R100 model by default. The model will be automatically downloaded on first run (~100MB).

**Model location:** `~/.insightface/models/buffalo_l/`

### Clustering Parameters

**min_cluster_size:**
- **Higher values** (15-20): Require more instances per identity, reject small clusters as noise
- **Lower values** (5-10): Allow smaller identity clusters, useful for minor characters

**min_samples:**
- Controls how conservative the clustering is
- Default: 2 (based on the existing script)
- Higher values make clustering more conservative

**distance_threshold:**
- Maximum face distance to consider same identity
- Default: 0.5
- Lower values = stricter identity matching

## Quality Filtering

The pipeline automatically filters out:

1. **Instances with no detectable face**
2. **Faces smaller than min_face_size** (default: 64 pixels)
3. **Blurry instances** (based on Laplacian variance, if enabled)

Filtered instances are saved to the `noise/` folder.

## Performance Tips

### GPU Memory

- **Face detection:** ~2GB VRAM
- **Face recognition:** ~1GB VRAM
- **Clustering:** CPU-based (no GPU required)

### Speed Optimization

For large datasets (>10,000 instances):

1. **Batch processing:** Process in chunks of 5,000 instances
2. **Resume capability:** The pipeline will automatically skip already-processed images
3. **Disable face crops:** Remove `--save-faces` to save disk I/O

### Typical Processing Time

| Instances | GPU | Time |
|-----------|-----|------|
| 1,000 | RTX 3090 | ~5 min |
| 5,000 | RTX 3090 | ~20 min |
| 10,000 | RTX 3090 | ~40 min |

## Troubleshooting

### Error: "No module named 'insightface'"

```bash
conda run -n ai_env pip install insightface onnxruntime-gpu
# Then downgrade numpy
conda run -n ai_env pip install "numpy<2.0" --force-reinstall
```

### Error: "No faces detected in any images!"

**Possible causes:**
1. Instances are not character-focused (e.g., background segments)
2. Faces are too small (increase `min_face_size`)
3. Faces are at extreme angles (profile/back views)

**Solutions:**
- Check instance segmentation quality
- Lower `min_face_size` threshold
- Use pose-based clustering instead

### Too Many Clusters

If the pipeline creates too many identity clusters (same character split across multiple clusters):

1. **Increase `distance_threshold`** (make clustering less strict)
2. **Decrease `min_cluster_size`** (allow smaller clusters to merge)
3. Use the **interactive review tool** to manually merge clusters

### Too Few Clusters

If the pipeline merges different characters together:

1. **Decrease `distance_threshold`** (make clustering stricter)
2. **Increase `min_cluster_size`** (reject ambiguous small clusters)
3. Use the **interactive review tool** to manually split clusters

## Integration with Training Pipeline

After identity clustering, proceed to dataset preparation:

```bash
# For each character identity
for identity_dir in output_dir/identity_*/; do
  character_name=$(basename "$identity_dir")

  # Prepare training dataset
  conda run -n ai_env python scripts/generic/training/prepare_training_data.py \
    --character-dirs "$identity_dir" \
    --output-dir "/path/to/training_data/$character_name" \
    --character-name "$character_name" \
    --generate-captions \
    --caption-prefix "a 3d animated character, pixar style" \
    --target-size 400
done
```

## Comparison: Identity Clustering vs. CLIP Clustering

| Aspect | Face Identity Clustering | CLIP Visual Clustering |
|--------|--------------------------|------------------------|
| **Groups by** | Who (identity) | Visual similarity |
| **Best for** | Multi-character scenes | Single character, many poses |
| **Separates** | Different characters in same scene | Different scenes/backgrounds |
| **Accuracy** | High for frontal/3/4 views | Good for full-body similarity |
| **Limitations** | Requires visible face | May merge characters in same scene |
| **Speed** | Fast (face detection + clustering) | Slower (full-image embedding) |

**Recommendation:** Use face-based identity clustering when:
- You have multiple characters in the video
- Characters appear together in scenes
- You need to separate character identities reliably

## Examples

### Example 1: Multi-Character Animation

**Scenario:** Pixar movie with 3 main characters appearing in various scenes together

**Input:** 5,000 instances from SAM2 segmentation

**Command:**
```bash
conda run -n ai_env python scripts/generic/clustering/face_identity_clustering.py \
  /mnt/data/ai_data/datasets/3d-anime/movie/instances/instances/ \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/movie/clustered/ \
  --min-cluster-size 15 \
  --device cuda \
  --save-faces
```

**Expected Output:**
```
clustered/
├── identity_000/  # Main character 1 (1,800 instances)
├── identity_001/  # Main character 2 (1,500 instances)
├── identity_002/  # Main character 3 (1,200 instances)
├── identity_003/  # Supporting character (300 instances)
├── identity_004/  # Minor character (150 instances)
└── noise/         # Partial faces, side characters (50 instances)
```

### Example 2: Single Character Focus

**Scenario:** Extracting only the protagonist from a multi-character video

**Steps:**
1. Run identity clustering with default settings
2. Review clusters in interactive UI
3. Select the largest cluster (likely the protagonist)
4. Rename to character name

**Command:**
```bash
# Step 1: Cluster
conda run -n ai_env python scripts/generic/clustering/face_identity_clustering.py \
  instances/ --output-dir clustered/ --device cuda

# Step 2: Interactive review
conda run -n ai_env python scripts/generic/clustering/launch_interactive_review.py \
  clustered/

# Step 3: Rename largest cluster
mv clustered/identity_000/ clustered/protagonist/
```

## Related Tools

- **SAM2 Instance Segmentation:** `scripts/generic/segmentation/instance_segmentation.py`
- **Interactive Cluster Review:** `scripts/generic/clustering/launch_interactive_review.py`
- **Pose Subclustering:** `scripts/generic/clustering/pose_subclustering.py`
- **Training Data Preparation:** `scripts/generic/training/prepare_training_data.py`

## References

- [InsightFace Documentation](https://github.com/deepinsight/insightface)
- [HDBSCAN Algorithm](https://hdbscan.readthedocs.io/)
- [ArcFace Paper](https://arxiv.org/abs/1801.07698)
