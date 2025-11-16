# Image Enhancement & Augmentation Strategy for 3D Character LoRA

## Overview

Post-processing pipeline for enhancing instance quality and augmenting training data while preserving 3D animation characteristics.

---

## Phase 1: Quality Enhancement (AI-Powered)

### 1.1 Super-Resolution (Optional)
**Model:** Real-ESRGAN or ESRGAN-Anime6B

**Use cases:**
- Upscale low-resolution instances (< 512px)
- Enhance fine details (hair, fabric texture, facial features)

**Parameters:**
```python
# For 3D characters, use conservative settings
scale_factor = 2  # Don't over-upscale
tile_size = 256   # Reduce VRAM usage
```

**⚠️ Caution:**
- May introduce artifacts in anti-aliased edges
- Only apply to instances < 512px on longest side
- Skip if original quality is already high

### 1.2 Deblurring (Conditional)
**Model:** NAFNet or DeblurGANv2

**Use cases:**
- Motion blur from fast character movement
- Depth-of-field blur (if unintentional)

**Quality filter:**
```python
# Only deblur if blur score > threshold
blur_score = cv2.Laplacian(image, cv2.CV_64F).var()
if blur_score < 80:  # 3D default threshold
    image = deblur_model(image)
```

**⚠️ Caution:**
- **DO NOT** deblur intentional cinematic DoF
- May reduce natural 3D softness
- Recommended: Manual review before batch processing

### 1.3 Face Restoration (High Priority)
**Model:** CodeFormer or GFPGAN

**Use cases:**
- Restore small/distant faces
- Fix compression artifacts
- Enhance facial feature clarity for identity clustering

**Parameters:**
```python
# Conservative fidelity to preserve 3D style
fidelity_weight = 0.7  # Balance between restoration and original style
upscale = 2            # Match super-resolution scale
```

**✅ Recommended:** Apply to all instances with detected faces

---

## Phase 2: Consistency Normalization

### 2.1 Color Correction
**Purpose:** Standardize lighting/color across different scenes

**Methods:**
1. **Histogram Equalization** (mild, per-channel)
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
3. **White Balance** (based on scene lighting)

**Parameters:**
```python
# 3D-specific: Preserve PBR material properties
clahe_clip_limit = 2.0  # Conservative to avoid flat lighting
adaptive_tile_size = 8
preserve_hue = True     # CRITICAL: Don't shift PBR colors
```

### 2.2 Background Cleanup
**Purpose:** Ensure clean alpha mattes

**Methods:**
1. **Edge refinement** (anti-aliasing preservation)
2. **Alpha matting** (Trimap + KNN matting)
3. **Background uniformity** (neutral gray or transparent)

**⚠️ DO NOT:**
- Use aggressive alpha thresholding (breaks soft edges)
- Apply heavy morphological operations (erodes 3D details)

---

## Phase 3: Data Augmentation (3D-Specific)

### ✅ ALLOWED Augmentations

#### 3.1 Cropping & Scaling
```python
# Random crop to focus on character
crop_scales = [0.8, 0.9, 1.0]  # Conservative range
min_character_ratio = 0.6      # Character must occupy 60% of crop

# Aspect ratio preservation
maintain_aspect_ratio = True   # CRITICAL for 3D proportions
```

#### 3.2 Rotation (Mild)
```python
# Only for in-plane rotation (simulate camera roll)
rotation_range = [-5, 5]  # degrees
fill_mode = 'nearest'     # Avoid color artifacts
```

#### 3.3 Brightness & Contrast (Mild)
```python
# Simulate lighting variations
brightness_range = [0.9, 1.1]  # ±10%
contrast_range = [0.95, 1.05]  # ±5%

# PRESERVE saturation (PBR materials)
adjust_saturation = False
```

#### 3.4 Gaussian Noise (Very Mild)
```python
# Simulate sensor noise
noise_std = 0.01  # Very subtle
apply_probability = 0.3  # Only 30% of augmented samples
```

---

### ❌ FORBIDDEN Augmentations (3D-Specific)

#### 3.5 Horizontal Flip
**Reason:** Breaks asymmetric accessories, character design

**Example:**
- Luca has asymmetric clothing details
- Flipping changes character identity subtly
- Confuses LoRA training

**Exception:** Only if character design is perfectly symmetric (rare)

#### 3.6 Color Jittering / Hue Shift
**Reason:** Destroys PBR material consistency

**Example:**
- Skin tone is a PBR property, not artistic choice
- Shifting hue breaks material learning
- Results in muddy colors during generation

#### 3.7 Elastic Deformation / Perspective Transform
**Reason:** 3D models have rigid geometry

**Example:**
- Elastic deformation violates skeletal constraints
- Perspective shift breaks camera model assumptions

---

## Phase 4: Quality Filtering (Post-Enhancement)

### 4.1 Automated Filters
```python
# Remove poor-quality augmented samples
filters = {
    'min_face_size': 64,           # Face clarity
    'blur_threshold': 80,          # Sharpness
    'brightness_range': [30, 220], # Exposure
    'saturation_range': [0.3, 2.0],# Color validity
    'edge_score': 0.3              # Detail preservation
}
```

### 4.2 Deduplication
```python
# Remove near-duplicates after augmentation
phash_threshold = 8   # Perceptual hash difference
ssim_threshold = 0.95 # Structural similarity
```

---

## Recommended Pipeline

```
Instance Extraction (SAM2)
    ↓
Quality Enhancement
    ├─ Face Restoration (CodeFormer) → All instances with faces
    ├─ Super-Resolution (Real-ESRGAN) → Only if < 512px
    └─ Deblurring (NAFNet) → Only if blur_score < 80
    ↓
Consistency Normalization
    ├─ Color Correction (CLAHE, mild)
    └─ Background Cleanup (Alpha refinement)
    ↓
Identity Clustering (ArcFace)
    ↓
Data Augmentation (Per-cluster)
    ├─ Cropping & Scaling (2-3 variants)
    ├─ Brightness/Contrast (2 variants)
    └─ Rotation (±5°, 2 variants)
    ↓
Quality Filtering
    ├─ Automated metrics
    └─ Deduplication
    ↓
Final Dataset (200-500 high-quality images per character)
```

---

## Target Dataset Size (3D Characters)

**Baseline (No Augmentation):**
- 200-300 high-quality instances per character
- Diverse poses, expressions, lighting
- Natural variation from 4,323 sampled frames

**With Conservative Augmentation:**
- 400-600 images per character
- 2-3x augmentation per original image
- Maintains 3D material consistency

**⚠️ Warning:**
- More data ≠ better quality for 3D
- 3D characters are inherently consistent (model identity)
- Focus on **pose/view diversity** over quantity
- Over-augmentation can introduce noise

---

## Implementation Priority

1. **Phase 1: Face Restoration** ← START HERE (critical for clustering)
2. **Phase 2: Background Cleanup** ← Ensures clean training data
3. **Phase 4: Quality Filtering** ← Remove poor instances
4. **Phase 3: Light Augmentation** ← Only if < 300 instances per character

**Optional:**
- Super-resolution (only if many low-res instances)
- Deblurring (only if significant motion blur detected)

---

## Tools & Models

| Task | Recommended Model | Alternative |
|------|------------------|-------------|
| Face Restoration | CodeFormer | GFPGAN, RestoreFormer |
| Super-Resolution | Real-ESRGAN | ESRGAN-Anime6B |
| Deblurring | NAFNet | DeblurGANv2 |
| Color Correction | OpenCV CLAHE | scikit-image |
| Deduplication | pHash + SSIM | ImageHash library |

---

## Next Steps

After SAM2 processing completes:

1. ✅ Verify instance quality
2. 🎯 **Apply face restoration** (prepare for clustering)
3. 🎯 **Identity clustering** (ArcFace embeddings)
4. Review clusters → Decide augmentation needs per character
5. Apply conservative augmentation only where needed
6. Final dataset assembly → LoRA training

