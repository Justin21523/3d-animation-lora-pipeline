# Quality Filtering Guide

## Overview

The **Quality Filter** (`scripts/generic/training/quality_filter.py`) is an automated system that evaluates and selects high-quality, diverse character instances for LoRA training. It balances **quality metrics** (sharpness, completeness) with **diversity metrics** (pose, angle variations) to create optimal training datasets.

## Why Quality Filtering?

After clustering and inpainting, you typically have hundreds to thousands of character instances per cluster. Training on all of them would:
- Include low-quality images (blurry, incomplete)
- Have redundant near-duplicate frames
- Lack diversity in poses and angles
- Result in overfitting and poor generalization

Quality filtering solves these problems by:
1. **Rejecting low-quality images** based on objective metrics
2. **Ensuring diversity** across poses, angles, and visual appearance
3. **Stratified sampling** to maintain balanced representation
4. **Targeting optimal dataset size** (typically 200-500 per character for 3D)

---

## Quality Metrics

### 1. Sharpness (Laplacian Variance)

**What it measures:** Image sharpness/blur level using Laplacian edge detection variance.

**How it works:**
```python
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
sharpness = laplacian.var()
```

**Typical values:**
- **< 100:** Very blurry (reject)
- **100-200:** Acceptable sharpness
- **> 200:** Sharp image

**Default threshold:** `100`

**Why it matters:** Blurry images degrade training quality and teach the model incorrect details.

---

### 2. Completeness (Alpha Channel Coverage)

**What it measures:** Ratio of opaque pixels in the alpha channel.

**How it works:**
```python
alpha = image[:, :, 3]
completeness = (alpha > 240).sum() / alpha.size
```

**Typical values:**
- **< 0.70:** Significant transparency (possible segmentation issues)
- **0.85-0.95:** Good completeness
- **> 0.95:** Fully opaque

**Default threshold:** `0.85`

**Why it matters:** Images with large transparent regions may have failed segmentation or contain only partial characters.

---

### 3. Face Detection (Optional)

**What it measures:** Presence and confidence of face detection.

**How it works:**
Uses InsightFace's face detector to check if a face is present and its confidence score.

**Typical values:**
- **0.0:** No face detected
- **0.5-0.7:** Moderate confidence
- **> 0.8:** High confidence

**Why it matters:** For character LoRAs, face presence often indicates a good frontal/three-quarter view. Optional because profile/back views are also valuable.

---

### 4. Overall Quality Score

**Formula:**
```python
overall_score = (
    0.4 * normalized_sharpness +
    0.4 * completeness +
    0.2 * face_confidence
)
```

**Weighting rationale:**
- **40% sharpness:** Critical for training quality
- **40% completeness:** Ensures full character representation
- **20% face confidence:** Useful but not essential (allows back/profile views)

---

## Diversity Metrics

### CLIP-based Diversity Clustering

**Purpose:** Ensure selected images cover diverse poses, angles, and visual appearances.

**How it works:**

1. **Extract CLIP embeddings** for all images that pass quality checks
2. **Cluster embeddings** using K-Means (default: 5 clusters)
3. **Stratified sampling:** Select images proportionally from each cluster
4. **Within-cluster ranking:** Pick highest quality images from each cluster

**Example:**
- 500 images pass quality → 5 diversity clusters (100 each)
- Target: 200 images
- Sample 40 from each cluster (200 / 5)
- Within each cluster, pick top 40 by quality score

**Benefits:**
- Avoids bias toward a single pose/angle
- Ensures balanced representation across visual variations
- Combines quality and diversity in selection

---

## Usage

### Basic Command

```bash
python scripts/generic/training/quality_filter.py \
  --input-dir /path/to/clustered_inpainted \
  --output-dir /path/to/filtered \
  --target-per-cluster 200 \
  --min-sharpness 100 \
  --min-completeness 0.85 \
  --diversity-method clip \
  --device cuda
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | *required* | Directory with clustered character instances |
| `--output-dir` | *required* | Output directory for filtered instances |
| `--target-per-cluster` | 200 | Target number of images per cluster |
| `--min-sharpness` | 100.0 | Minimum Laplacian variance threshold |
| `--min-completeness` | 0.85 | Minimum alpha completeness ratio (0-1) |
| `--diversity-method` | clip | Diversity analysis method (`clip` or `none`) |
| `--diversity-clusters` | 5 | Number of diversity clusters for sampling |
| `--use-face-detection` | False | Enable face detection quality check |
| `--device` | cuda | Device for CLIP/face detection (`cuda` or `cpu`) |

---

## Example Workflow

### 1. Run Quality Filter

```bash
conda run -n ai_env python scripts/generic/training/quality_filter.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/clustered_v2_inpainted \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/clustered_filtered \
  --target-per-cluster 200 \
  --min-sharpness 100 \
  --min-completeness 0.85 \
  --diversity-method clip \
  --diversity-clusters 5 \
  --device cuda
```

**Output:**
```
======================================================================
AUTOMATED QUALITY FILTERING
======================================================================
Input: /mnt/data/.../clustered_v2_inpainted
Output: /mnt/data/.../clustered_filtered
Target per cluster: 200
Min sharpness: 100
Min completeness: 0.85
Diversity method: clip
Device: CUDA
======================================================================

Loading CLIP model...
✓ CLIP model loaded

📂 Found 57 character clusters

📂 Processing character_0 (89 instances)
  Quality check: 100%|████████████████| 89/89
  ✓ Selected 80 / 85 images

📂 Processing character_1 (234 instances)
  Quality check: 100%|████████████████| 234/234
  ✓ Selected 200 / 220 images

...

======================================================================
QUALITY FILTERING COMPLETE
======================================================================
Total input instances: 3,882
Passed quality check: 3,456 (89.0%)
Final selection: 2,100 (54.1%)

Rejection reasons:
  Low sharpness: 312
  Low completeness: 114
======================================================================

📁 Filtered instances saved to: /mnt/data/.../clustered_filtered
📊 Report saved to: /mnt/data/.../clustered_filtered/quality_filter_report.json
```

---

### 2. Analyze Results

```bash
python scripts/generic/training/analyze_quality_report.py \
  --report /mnt/data/ai_data/datasets/3d-anime/luca/clustered_filtered/quality_filter_report.json \
  --filtered-dir /mnt/data/ai_data/datasets/3d-anime/luca/clustered_filtered \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/clustered_filtered
```

**Output:**
- **quality_statistics.png:** Bar charts showing counts, pass rates, selection rates, rejection reasons per cluster
- **quality_distributions.png:** Histograms of sharpness, completeness, overall score distributions
- **Text summary** printed to console

---

### 3. Review Selected Images

Manually inspect a few clusters to verify quality and diversity:

```bash
# View thumbnails
ls /mnt/data/ai_data/datasets/3d-anime/luca/clustered_filtered/character_0/

# Check quality metrics for a cluster
cat /mnt/data/ai_data/datasets/3d-anime/luca/clustered_filtered/character_0/quality_metrics.json | head -50
```

---

## Output Structure

```
output_dir/
├── character_0/
│   ├── image_001.png          # Selected high-quality images
│   ├── image_042.png
│   ├── ...
│   └── quality_metrics.json   # Per-image quality scores
├── character_1/
│   ├── ...
├── ...
├── quality_filter_report.json # Overall statistics
├── quality_statistics.png     # Visualization (after analysis)
└── quality_distributions.png  # Distributions (after analysis)
```

### quality_filter_report.json

```json
{
  "parameters": {
    "target_per_cluster": 200,
    "min_sharpness": 100.0,
    "min_completeness": 0.85,
    "diversity_method": "clip"
  },
  "clusters": [
    {
      "cluster": "character_0",
      "total": 89,
      "passed_quality": 85,
      "selected": 80,
      "rejected_sharpness": 3,
      "rejected_completeness": 1
    },
    ...
  ]
}
```

### quality_metrics.json (per cluster)

```json
[
  {
    "path": "/path/to/image_001.png",
    "sharpness": 234.5,
    "completeness": 0.92,
    "face_detected": true,
    "face_confidence": 0.85,
    "alpha_mean": 242.3,
    "alpha_std": 15.2,
    "overall_score": 0.87
  },
  ...
]
```

---

## Tuning Guidelines

### For 3D Animation Characters

**Recommended settings:**
```bash
--target-per-cluster 200 \
--min-sharpness 100 \
--min-completeness 0.85 \
--diversity-method clip \
--diversity-clusters 5
```

**Rationale:**
- **200 images:** Sufficient for 3D character LoRA (consistent model)
- **Sharpness 100:** Rejects very blurry frames while allowing slight motion blur
- **Completeness 0.85:** Tolerates some transparency at edges (anti-aliasing)
- **5 diversity clusters:** Captures front/three-quarter/profile/back/action variations

---

### Adjusting for Different Scenarios

#### Too Many Rejections (< 60% pass rate)

**Lower thresholds:**
```bash
--min-sharpness 80 \
--min-completeness 0.80
```

**When to use:** Video has inherent blur, or segmentation left some transparency.

---

#### Need More Diversity

**Increase diversity clusters:**
```bash
--diversity-clusters 8
```

**When to use:** Large dataset with many pose variations; want to ensure fine-grained diversity.

---

#### Want More Images per Cluster

**Increase target:**
```bash
--target-per-cluster 400
```

**When to use:** Style LoRA (vs character LoRA), or very consistent character needing more examples.

---

#### Quality Over Diversity

**Disable diversity sampling:**
```bash
--diversity-method none
```

**Result:** Selects top-N images by quality score only (may bias toward similar poses).

---

## Technical Details

### Algorithm Flow

```
1. For each cluster directory:
   ├─ Load all PNG images
   ├─ For each image:
   │  ├─ Compute sharpness (Laplacian variance)
   │  ├─ Compute completeness (alpha ratio)
   │  ├─ (Optional) Detect face and confidence
   │  ├─ Compute overall quality score
   │  └─ Reject if below thresholds
   │
   ├─ For images that passed:
   │  └─ Extract CLIP embedding (if diversity enabled)
   │
   ├─ Cluster embeddings by K-Means (diversity groups)
   │
   ├─ Stratified sampling:
   │  ├─ Determine target per diversity cluster
   │  ├─ Within each diversity cluster:
   │  │  └─ Sort by quality score
   │  │  └─ Select top-N
   │  └─ If under target, add more from top quality
   │
   └─ Copy selected images to output directory
   └─ Save quality metrics JSON

2. Save overall report
```

---

### Memory and Performance

**GPU Memory Usage:**
- **CLIP (ViT-B/32):** ~400 MB
- **InsightFace (optional):** ~500 MB

**Processing Speed:**
- **Quality check:** ~10-20 images/sec (CPU-bound: read, Laplacian, alpha check)
- **CLIP embedding:** ~50-100 images/sec (GPU)
- **Clustering:** < 1 sec for typical cluster sizes

**Estimated Time:**
- 3,000 images, 50 clusters: ~5-10 minutes (with CLIP on GPU)

---

## Integration with Pipeline

Quality filtering fits between **inpainting** and **caption generation**:

```
Clustering → Inpainting → Quality Filter → Caption Generation → Training
```

**Why this order:**
1. **After inpainting:** Ensures completeness metrics are accurate (no missing alpha)
2. **Before captioning:** Reduces captioning cost by only processing selected images
3. **Before training:** Ensures only high-quality, diverse images enter the dataset

---

## Troubleshooting

### CLIP Not Available

**Error:**
```
⚠️  CLIP not installed, diversity analysis disabled
```

**Solution:**
```bash
conda run -n ai_env pip install git+https://github.com/openai/CLIP.git
```

Or use quality-only mode:
```bash
--diversity-method none
```

---

### Face Detection Fails

**Error:**
```
⚠️  Failed to load face detector: ...
```

**Solution:**
Face detection is optional. The script will continue without it. If you need it:
```bash
conda run -n ai_env pip install insightface onnxruntime-gpu
```

---

### All Images Rejected

**Symptom:** `Passed quality check: 0`

**Causes:**
1. Thresholds too strict
2. Input images actually low quality
3. Incorrect input directory (non-RGBA images)

**Solutions:**
1. Lower thresholds: `--min-sharpness 50 --min-completeness 0.70`
2. Check sample images manually
3. Verify input directory structure

---

## Best Practices

### 1. Inspect Before Running

Check a few sample images from input directory:
```bash
ls /path/to/clustered_inpainted/character_0/ | head -5
```

Verify they are RGBA PNGs with alpha channel.

---

### 2. Start with Defaults

Run with default parameters first, then adjust based on results:
```bash
python quality_filter.py --input-dir ... --output-dir ... --device cuda
```

Review pass rates and rejection reasons in the report.

---

### 3. Visualize Results

Always run the analysis script to generate plots:
```bash
python analyze_quality_report.py --report .../quality_filter_report.json --filtered-dir ...
```

Inspect distributions to understand your data.

---

### 4. Manual Spot Check

After filtering, randomly inspect a few clusters:
```bash
# Pick random images
ls /path/to/filtered/character_0/ | shuf | head -10
```

Open them in image viewer to confirm quality and diversity.

---

### 5. Iterate if Needed

If results aren't satisfactory:
1. Adjust thresholds
2. Change diversity cluster count
3. Re-run filter
4. Compare reports

---

## Advanced Usage

### Custom Quality Weights

Edit `scripts/generic/training/quality_filter.py` line ~280:

```python
overall_score = (
    0.5 * norm_sharpness +      # Prioritize sharpness
    0.3 * completeness +
    0.2 * face_confidence
)
```

---

### Additional Metrics

You can extend the system with:
- **Brightness/contrast checks**
- **Color histogram diversity**
- **Pose estimation confidence**
- **Perceptual hash deduplication**

Add new metrics to `evaluate_image_quality()` function.

---

## References

- **Laplacian Sharpness:** [OpenCV Docs](https://docs.opencv.org/4.x/d5/db5/tutorial_laplace_operator.html)
- **CLIP:** [OpenAI CLIP](https://github.com/openai/CLIP)
- **InsightFace:** [InsightFace Docs](https://github.com/deepinsight/insightface)
- **Stratified Sampling:** [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)

---

## Summary

The Quality Filter automates the critical task of selecting optimal training data by:
1. **Rejecting low-quality images** based on sharpness and completeness
2. **Ensuring diversity** through CLIP-based clustering and stratified sampling
3. **Balancing quality and diversity** via weighted scoring
4. **Providing transparency** through detailed metrics and reports

This results in cleaner, more diverse training datasets that produce better LoRA models with improved generalization and fewer artifacts.
