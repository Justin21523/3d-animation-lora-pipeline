# Background LoRA Training Guide
**Date**: 2025-11-15
**Status**: Phase 1 Complete - BrushNet Implementation

---

## Overview

This guide explains how to train **Background LoRA** adapters for 3D animation scenes. Background LoRAs complement Character LoRAs by providing contextually appropriate environments that can be composed together.

### Multi-LoRA System

```
Final Image = Base Model + Character LoRA (1.0) + Background LoRA (0.7) + [Optional: Pose/Expression LoRAs]
```

**Key Principle**: **Data Separation Purity**
- Character LoRA: Clean character instances (transparent backgrounds)
- Background LoRA: Clean backgrounds (characters completely removed)

---

## Prerequisites

### 1. SAM2 Segmentation Data

You need background layers from SAM2 segmentation (characters removed but NOT inpainted):

```
segmented/
├── character/     # Character instances (for Character LoRA)
├── background/    # Backgrounds with character holes (for Background LoRA)
└── masks/         # Alpha masks
```

If you don't have background layers, re-run segmentation:

```bash
python scripts/generic/segmentation/layered_segmentation.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/FILM/frames \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/FILM/segmented \
  --model sam2 \
  --extract-characters \
  --alpha-threshold 0.15 \
  --blur-threshold 80
```

### 2. Required Models

#### Option A: LaMa (Fast, Production-Ready)
```bash
pip install simple-lama-inpainting
```

#### Option B: BrushNet (SOTA, Text-Guided)
```bash
pip install diffusers transformers accelerate
# Models downloaded automatically on first run
```

---

## Workflow

### Step 1: Background Inpainting

Remove character remnants from background layers.

#### **Understanding SAM2 Background Output**

**⚠️ IMPORTANT**: SAM2 backgrounds have these characteristics:

1. **Black holes where characters were** - SAM2 只分割，不修補
2. **Geometric polygon shapes** - SAM2 遮罩邊緣
3. **Character silhouette remnants** - 部分角色輪廓可見

**Example from Luca dataset**:
```
背景圖：海底場景
問題：
- Luca 海怪形態的綠色輪廓仍可見
- 被角色遮擋的岩石/水草缺失（黑色區域）
- 需要 inpainting 填補這些區域
```

**This is NORMAL** - 這就是為什麼需要 inpainting！

#### **Method A: LaMa (Recommended for MVP)**

Fast batch processing:

```bash
python scripts/generic/inpainting/lama_batch_optimized.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/backgrounds \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_clean \
  --batch-size 16 \
  --device cuda
```

**What LaMa does**:
- Detects black regions (character holes)
- Fills with surrounding context (岩石紋理、水草延續)
- Fast but limited semantic understanding

**Speed**: ~2 images/second on GPU
**Quality**: Good for natural backgrounds (ocean, sky, buildings)

**Limitations**:
- May blur complex structures
- Cannot add semantic details ("windows" if surrounded by wall)

#### **Method B: BrushNet (SOTA Quality + Text Control)**

Text-guided inpainting with scene-specific prompts:

```bash
python scripts/generic/inpainting/brushnet_background_inpainting.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/backgrounds \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_brushnet \
  --prompt "3d animated underwater scene, coral reef, aquatic plants, blue gradient, pixar rendering, no characters" \
  --negative-prompt "characters, humans, sea monster, green figure, blurry, low quality" \
  --steps 50 \
  --guidance-scale 7.5 \
  --use-lama-first \
  --device cuda
```

**What BrushNet does**:
- **Understands scene semantics** (underwater → fills with coral/plants)
- **Text-guided control** - 可指定要補什麼內容
- **Removes character remnants** - negative prompt 防止幻覺
- **Structure-aware** - 雙分支保持岩石結構、水草方向

**How it fixes SAM2 issues**:
1. **Black holes** → Filled with contextually appropriate content
2. **Character silhouettes** → Completely removed via negative prompt
3. **Missing details** → Generated based on prompt + surrounding context

**Features**:
- Text-guided control ("Italian coastal town" vs "underwater scene")
- Quality validation (PSNR/SSIM)
- Auto-fallback to LaMa for simple cases
- Metadata export

**Speed**: ~0.2 images/second (slower but higher quality)

**Hybrid Strategy** (Recommended):
- Use `--use-lama-first` flag
- Simple backgrounds (< 15% mask) → LaMa (fast)
- Complex backgrounds → BrushNet (quality)

#### **How BrushNet Ensures Proper Context**

**Multi-layer mechanism**:

1. **Prompt Control** (Positive):
   ```python
   "3d animated underwater scene, coral reef, aquatic plants,
   blue gradient, diffuse lighting, pixar rendering"
   ```
   - Specifies WHAT to fill (coral, plants)
   - Defines style (Pixar 3D, blue gradient)

2. **Negative Prompt** (Critical for character removal):
   ```python
   "characters, people, humans, sea monster, green figure,
   luca, alberto, fish people, blurry, artifacts"
   ```
   - **Prevents character hallucination** - 關鍵！
   - Removes color remnants (green silhouette)

3. **Context from Surrounding Pixels**:
   - BrushNet sees岩石 texture → extends it
   - Sees水草 direction → continues the flow
   - Matches lighting (blue underwater gradient)

4. **Dual-Branch Architecture**:
   ```
   Structure Branch: Preserves edges (岩石輪廓、水草線條)
         +
   Texture Branch: Fills details (coral patterns, plant leaves)
         ↓
   Output: Structurally accurate + rich textures
   ```

5. **Quality Validation**:
   ```python
   if PSNR < 25.0 or SSIM < 0.85:
       retry_with_different_parameters()
   ```
   - Automatic quality check
   - Re-inpaint if not satisfactory

**Example: Luca Underwater Scene**

**Input (SAM2 output)**:
- Black hole where Luca was (30% of frame)
- Green silhouette remnants (sea monster form)
- Missing岩石 parts behind Luca

**BrushNet Processing**:
```python
Prompt: "underwater coral reef, rock formations, aquatic plants"
Negative: "sea monster, green character, luca"

Result:
1. Black hole → Filled with岩石 + coral
2. Green silhouette → Removed completely
3. Missing岩石 → Extended from visible parts
4. Textures match → 水草 flow consistent
```

**Output**:
- Clean underwater background
- No character traces
- Contextually appropriate (岩石/coral/水草)
- Ready for Background LoRA training

---

### Step 2: Scene Clustering

Group similar backgrounds together:

```bash
python scripts/generic/clustering/character_clustering.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_clean \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/scene_clusters \
  --min-cluster-size 15 \
  --min-samples 3 \
  --similarity-threshold 0.75 \
  --quality-filter
```

**Expected clusters** (Luca example):
- `cluster_0/` → Portorosso town (buildings, streets)
- `cluster_1/` → Ocean/beach scenes
- `cluster_2/` → Indoor houses
- `cluster_3/` → Underwater scenes
- `noise/` → Mixed/unclear scenes (discard)

---

### Step 3: Caption Generation

Generate VLM captions for each scene cluster:

```bash
python scripts/generic/training/prepare_training_data.py \
  --character-dirs /mnt/data/ai_data/datasets/3d-anime/luca/scene_clusters/cluster_0 \
  --output-dir /mnt/data/ai_data/training_data/portorosso_town \
  --character-name "portorosso" \
  --generate-captions \
  --caption-model qwen2_vl \
  --caption-prefix "3d animated background, italian coastal town, pixar style, detailed environment" \
  --target-size 300
```

**Caption template** for backgrounds:

```
3d animated background, pixar style,
{scene_type},           # "italian seaside town" / "ocean beach"
{architecture},         # "colorful buildings" / "coral reef"
{lighting},             # "warm sunlight" / "diffuse underwater light"
{atmosphere},           # "bright clear day" / "peaceful atmosphere"
no characters, empty scene
```

---

### Step 4: Train Background LoRA

#### Reuse Character LoRA's Best Hyperparameters

Assume your Character LoRA found optimal parameters:

```toml
# From Character LoRA Trial 35 (best result)
network_dim = 128
network_alpha = 96
learning_rate = 0.0001
train_batch_size = 1
gradient_accumulation_steps = 2
max_train_epochs = 20 (or 12 for faster)
```

#### Create Background LoRA Config

Copy and modify:

```bash
cp configs/training/sdxl_16gb_stable.toml configs/training/portorosso_town_bg.toml
```

**Key changes**:

```toml
# Dataset
[dataset]
train_data_dir = "/mnt/data/ai_data/training_data/portorosso_town"

# Output
output_dir = "/mnt/data/ai_data/models/lora/backgrounds/portorosso_town"
output_name = "portorosso_town_bg"

# Caption settings (different from character)
caption_prefix = "3d animated background, italian coastal town, pixar style"
keep_tokens = 3  # Keep "3d animated background"

# Augmentation (safer for backgrounds)
color_aug = false  # Still avoid color jitter (maintains Pixar style)
flip_aug = true    # OK for backgrounds (horizontal flip won't break asymmetry)
random_crop = false
```

#### Train

```bash
conda run -n kohya_ss accelerate launch --num_cpu_threads_per_process=2 \
  sd-scripts/sdxl_train_network.py \
  --config_file configs/training/portorosso_town_bg.toml
```

**Expected duration** (with 300 images, 5 repeats, 12 epochs):
- Total steps: ~18,000
- Time: ~5-6 hours (at 1.2s/step)

---

### Step 5: Test Background LoRA

#### Solo Test

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load Background LoRA
pipe.load_lora_weights(
    "/mnt/data/ai_data/models/lora/backgrounds/portorosso_town",
    weight_name="portorosso_town_bg.safetensors"
)

# Generate
image = pipe(
    "3d animated italian coastal town, colorful buildings, pixar style, sunny day",
    negative_prompt="people, characters, blurry, low quality",
    num_inference_steps=50,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 0.8}  # Background LoRA at 0.8
).images[0]
```

#### Composed with Character LoRA

```python
# Load both LoRAs
pipe.load_lora_weights([
    ("/path/to/luca_sdxl.safetensors", 1.0),           # Character (full strength)
    ("/path/to/portorosso_town_bg.safetensors", 0.7)   # Background (moderate)
])

# Generate composed scene
image = pipe(
    "luca paguro, young boy, standing in italian coastal town, colorful buildings, pixar style",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
```

**LoRA Scale Guidelines**:
- Character: **1.0** (core identity)
- Background: **0.6-0.8** (allows prompt to influence scene)
- Too high (> 0.9): Background overpowers prompt
- Too low (< 0.5): Background effect too weak

---

## Advanced: Multiple Background LoRAs

Train separate LoRAs for each major scene type:

```bash
# 1. Portorosso town
train_data: cluster_0/ (town scenes)
output: portorosso_town_bg.safetensors

# 2. Ocean/beach
train_data: cluster_1/ (water scenes)
output: ocean_beach_bg.safetensors

# 3. Indoor house
train_data: cluster_2/ (interior scenes)
output: italian_interior_bg.safetensors

# 4. Underwater
train_data: cluster_3/ (underwater scenes)
output: underwater_bg.safetensors
```

**Usage**:

```python
# Swap backgrounds easily
backgrounds = {
    "town": "portorosso_town_bg.safetensors",
    "beach": "ocean_beach_bg.safetensors",
    "underwater": "underwater_bg.safetensors"
}

# Load character + desired background
pipe.load_lora_weights([
    ("luca_sdxl.safetensors", 1.0),
    (backgrounds["underwater"], 0.7)
])
```

---

## Troubleshooting

### Issue 0: SAM2 Backgrounds Have "Geometric Patterns" or Black Holes

**Symptom**: Background images show:
- Black polygonal regions where characters were
- Character silhouettes or color remnants
- Missing environmental details (rocks, plants behind characters)

**Cause**: **This is NORMAL SAM2 behavior** - SAM2 only **segments** (separates character from background), it does NOT **inpaint** (fill the holes).

**Visual Example**:
```
Original Frame: Luca (green sea monster) in underwater scene with rocks
              ↓ SAM2 Segmentation
Background Output:
  ✓ Rocks/water visible around character
  ✗ Black hole where Luca was (~30% of frame)
  ✗ Green silhouette remnants from sea monster form
  ✗ Missing rock/coral parts that were behind Luca
```

**Solution** - Use inpainting to fix:

#### **Option A: LaMa (Fast, Good for Simple Scenes)**

```bash
# Process all 4589 backgrounds (~40 minutes)
python scripts/generic/inpainting/lama_batch_optimized.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/backgrounds \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_lama_clean \
  --batch-size 16 \
  --device cuda
```

**Pros**: Fast (~2 img/s), good for natural textures
**Cons**: May blur structures, can't understand scene semantics

#### **Option B: BrushNet (SOTA, Text-Guided)**

```bash
# Slower but higher quality (~6 hours for 4589 images)
python scripts/generic/inpainting/brushnet_background_inpainting.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/backgrounds \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_brushnet_clean \
  --prompt "3d animated scene, pixar style, detailed environment, no characters" \
  --negative-prompt "sea monster, green figure, luca, alberto, characters, people" \
  --use-lama-first \
  --device cuda
```

**Pros**: Understands semantics, removes character remnants completely
**Cons**: Slower (~0.2 img/s)

**Recommended Strategy**:
1. **Test on 50 images first** to verify quality
2. **Use `--use-lama-first`** - simple backgrounds → LaMa, complex → BrushNet
3. **Batch process overnight** for full dataset

**Expected Results After Inpainting**:
- ✅ Black holes filled with appropriate content
- ✅ Character silhouettes completely removed
- ✅ Environmental details extended naturally
- ✅ Ready for scene clustering & LoRA training

---

### Issue 1: Background LoRA Too Weak

**Symptom**: Prompt dominates, LoRA style barely visible

**Solutions**:
- Increase LoRA scale: 0.7 → 0.9
- Check training: May need more epochs or lower learning rate
- Ensure training data quality (clean inpainting, diverse scenes)

### Issue 2: Background LoRA Hallucinates Characters

**Symptom**: Characters appear even with "no people" in prompt

**Cause**: Training data had character remnants (poor inpainting)

**Solutions**:
- Re-inpaint with BrushNet + strong negative prompt
- Add "no people, empty scene" to all training captions
- Use higher quality inpainting (BrushNet > LaMa for character removal)

### Issue 3: Background Conflicts with Character

**Symptom**: Character edges blend into background unnaturally

**Cause**: Both LoRAs trained on different edge styles

**Solutions**:
- Lower background LoRA scale: 0.7 → 0.5
- Use regional prompting (separate character and background regions)
- Train character LoRA with more diverse backgrounds (or add background LoRA during training)

---

## Performance Comparison

| Inpainting Method | Speed (img/s) | Quality | Text Control | Best For |
|-------------------|---------------|---------|--------------|----------|
| **LaMa** | ~2.0 | Good | ❌ | Natural scenes, MVP |
| **BrushNet** | ~0.2 | Excellent | ✅ | Complex architecture, artistic control |
| **Hybrid** | ~1.0 | Excellent | ✅ | Production (best of both) |

**Recommendation for Luca**:
- Portorosso town (complex buildings) → **BrushNet**
- Ocean/beach (simple) → **LaMa** (faster)
- Underwater (artistic) → **BrushNet** with scene prompt

---

## Next Steps

1. ✅ Implement BrushNet inpainting
2. ⏳ Test on Luca backgrounds
3. ⏳ Train first Background LoRA (Portorosso town)
4. ⏳ Evaluate composition with Character LoRA
5. ⏳ Expand to other films (Onward, Orion)

---

## References

- **BrushNet Paper**: ECCV 2024 - "A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion"
- **LaMa Paper**: WACV 2022 - "Resolution-robust Large Mask Inpainting with Fourier Convolutions"
- **Multi-LoRA Composition**: See `docs/training/lora-composition.md`

---

**Status**: Implementation complete, ready for testing on Luca dataset.
