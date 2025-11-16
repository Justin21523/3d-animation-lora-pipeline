# Face Restoration Guide

## Overview

Face restoration enhances facial details in character instances, critical for accurate identity clustering and high-quality LoRA training.

**Model Used:** CodeFormer (primary) / GFPGAN (fallback)

**Purpose:**
- Enhance small/distant faces
- Fix compression artifacts
- Improve facial feature clarity
- Prepare instances for identity clustering

---

## Installation

### 1. Install Dependencies

```bash
conda run -n ai_env pip install -r requirements/face_restoration.txt
```

This installs:
- `basicsr` - Image restoration framework
- `facexlib` - Face utilities
- `gfpgan` - Face restoration model
- `retinaface-pytorch` - Face detection
- `insightface` - Alternative face detection

### 2. Model Download

Models will be automatically downloaded on first run to:
```
/mnt/c/AI_LLM_projects/ai_warehouse/models/face_restoration/
├── codeformer.pth          (~370 MB)
└── GFPGANv1.4.pth         (~350 MB, fallback)
```

---

## Quick Start

### Step 1: Test on Sample Images

Before processing all instances, test on a few samples to verify setup and tune parameters:

```bash
conda run -n ai_env python scripts/generic/enhancement/test_face_restoration.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances \
  --output-dir outputs/face_restoration_test \
  --num-samples 10 \
  --fidelity 0.7
```

**Check results:**
```bash
ls outputs/face_restoration_test/
# Look for *_comparison.jpg files showing before/after
```

### Step 2: Adjust Fidelity (if needed)

**Fidelity Parameter** (0.0 - 1.0):
- **0.5-0.6**: More restoration, less original style (may look "real photo" like)
- **0.7-0.8**: Balanced, **recommended for 3D anime**
- **0.9-1.0**: Preserve original style, minimal restoration

Test different values:
```bash
# More restoration
python scripts/generic/enhancement/test_face_restoration.py INPUT_DIR \
  --fidelity 0.5 --output-dir outputs/test_fidelity_0.5

# Preserve style (recommended)
python scripts/generic/enhancement/test_face_restoration.py INPUT_DIR \
  --fidelity 0.7 --output-dir outputs/test_fidelity_0.7

# Maximum preservation
python scripts/generic/enhancement/test_face_restoration.py INPUT_DIR \
  --fidelity 0.9 --output-dir outputs/test_fidelity_0.9
```

Compare results and choose the best fidelity value.

### Step 3: Process All Instances (in tmux)

Once satisfied with test results, process all instances:

```bash
# Create tmux session
tmux new-session -d -s luca_face_restore "conda run -n ai_env python scripts/generic/enhancement/face_restoration.py /mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_restored --fidelity 0.7 --upscale 2 --save-comparison 2>&1 | tee logs/face_restoration_\$(date +%Y%m%d_%H%M%S).log"

# Check status
tmux list-sessions

# Attach to view progress
tmux attach -t luca_face_restore
# Press Ctrl+B then D to detach
```

---

## Usage

### Basic Usage

```bash
python scripts/generic/enhancement/face_restoration.py INPUT_DIR \
  --output-dir OUTPUT_DIR \
  --fidelity 0.7 \
  --upscale 2
```

### Full Options

```bash
python scripts/generic/enhancement/face_restoration.py INPUT_DIR \
  --output-dir OUTPUT_DIR \
  --fidelity 0.7              # Restoration fidelity (0-1)
  --upscale 2                 # Upscale factor (1, 2, or 4)
  --device cuda               # cuda or cpu
  --save-comparison           # Save before/after comparisons
  --face-detector retinaface  # retinaface or yolov5
```

### Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fidelity` | 0.7 | Balance restoration vs. original style (higher = preserve style) |
| `--upscale` | 2 | Upscale factor (1=no upscale, 2=2x, 4=4x) |
| `--device` | cuda | Processing device (cuda or cpu) |
| `--save-comparison` | False | Save side-by-side comparisons (helpful for tuning) |
| `--face-detector` | retinaface | Face detection backend |

---

## Output Structure

```
output_dir/
├── restored/                      # Restored instances
│   ├── scene0000_pos1_*_inst0.png
│   └── ...
├── comparisons/                   # Before/after (if --save-comparison)
│   ├── scene0000_pos1_*_inst0_compare.jpg
│   └── ...
└── restoration_stats.json         # Processing statistics
```

---

## Resume Processing

The script supports **automatic resume**:
- Already processed files are detected and skipped
- Safe to interrupt (Ctrl+C) and restart
- Progress is preserved

```bash
# If interrupted, simply run the same command again
python scripts/generic/enhancement/face_restoration.py INPUT_DIR \
  --output-dir OUTPUT_DIR \
  --fidelity 0.7
# Will skip already processed files
```

---

## Expected Processing Time

**Speed:**
- With CUDA: ~5-10 images/second (depends on GPU)
- With CPU: ~0.5-1 images/second

**For Luca dataset:**
- ~4,300 instances (after SAM2 processing)
- Estimated time: 7-15 minutes (CUDA) or 1-2 hours (CPU)

---

## Quality Guidelines (3D Animation)

### ✅ Good Results

- Facial features are sharper
- Skin texture is preserved
- Eyes are clearer
- Maintains 3D animation style
- No artifacts or "real photo" look

### ⚠️ Adjust Fidelity If:

**Fidelity too low (< 0.6):**
- Faces look too "realistic"
- Lost 3D animation aesthetic
- Skin looks like real photo
- **Fix:** Increase fidelity to 0.7-0.8

**Fidelity too high (> 0.9):**
- Minimal improvement visible
- Faces still blurry
- No enhancement observed
- **Fix:** Decrease fidelity to 0.7

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'basicsr'"

**Solution:**
```bash
conda run -n ai_env pip install -r requirements/face_restoration.txt
```

### Issue: "CUDA out of memory"

**Solutions:**
1. Process in smaller batches
2. Reduce upscale factor to 1
3. Use CPU mode (slower but works)

```bash
python scripts/generic/enhancement/face_restoration.py INPUT_DIR \
  --output-dir OUTPUT_DIR \
  --upscale 1 \
  --device cpu
```

### Issue: "No faces detected" for all images

**Possible causes:**
1. Face detector not installed properly
2. Images are too small
3. Characters are not facing camera

**Solutions:**
```bash
# Try alternative face detector
python scripts/generic/enhancement/face_restoration.py INPUT_DIR \
  --output-dir OUTPUT_DIR \
  --face-detector yolov5

# Or skip face detection (process all images)
# Edit face_restoration.py: set self.face_detector = None
```

### Issue: Results look worse / artifacts

**Solution:** Increase fidelity to preserve original style
```bash
python scripts/generic/enhancement/face_restoration.py INPUT_DIR \
  --output-dir OUTPUT_DIR \
  --fidelity 0.85
```

---

## After Face Restoration

Once face restoration is complete, proceed to:

1. **✅ Identity Clustering** (ArcFace + HDBSCAN)
   - Restored faces → better identity embeddings
   - More accurate character grouping

2. **Quality Verification**
   - Review restored instances
   - Check for artifacts or over-restoration

3. **Dataset Assembly**
   - Combine with clustering results
   - Prepare for LoRA training

---

## Advanced: Batch Processing Script

Create a wrapper script for the full pipeline:

```bash
#!/bin/bash
# scripts/pipelines/enhance_and_cluster.sh

# Step 1: Face restoration
tmux new-session -d -s face_restore "conda run -n ai_env python scripts/generic/enhancement/face_restoration.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_restored \
  --fidelity 0.7 --upscale 2 2>&1 | tee logs/face_restore.log"

echo "⏳ Waiting for face restoration to complete..."
while tmux has-session -t face_restore 2>/dev/null; do
  sleep 30
done

echo "✅ Face restoration complete!"

# Step 2: Identity clustering (coming next)
# ...
```

---

## References

- [CodeFormer Paper](https://arxiv.org/abs/2206.11253)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [RetinaFace](https://github.com/serengil/retinaface)

