# 2D Anime vs 3D Animation Parameters

## Quick Reference Table

| Parameter | 2D Anime | 3D Animation | Reason |
|-----------|----------|--------------|--------|
| **Segmentation** |
| Alpha Threshold | 0.25 | 0.15 | 3D has soft anti-aliased edges |
| Blur Threshold | 100 | 80 | 3D uses intentional DoF blur |
| Min Character Size | 128 | 128 | Same minimum size |
| Model Preference | U²-Net, ISNet | U²-Net, SAM | SAM better for 3D objects |
| **Clustering** |
| Min Cluster Size | 20-25 | 10-15 | 3D characters more consistent |
| Min Samples | 3-5 | 2 | Tighter clusters in 3D |
| Similarity Threshold | 0.75 | 0.70 | Allow more variation capture |
| **Training Data** |
| Dataset Size (min) | 500-1000 | 200-500 | 3D needs fewer images |
| Dataset Size (ideal) | 1000-1500 | 400-600 | Optimal range |
| Caption Length | 40-77 tokens | 40-77 tokens | Same |
| **Augmentation** |
| Color Augmentation | ✅ Yes | ❌ No | 3D materials consistent |
| Horizontal Flip | ✅ Yes | ❌ No | Breaks 3D symmetry |
| Crop/Resize | ✅ Yes | ✅ Yes | Safe for both |
| Brightness | ⚠️ Careful | ⚠️ Careful | Minimal only |
| **Training** |
| Learning Rate | 1e-4 | 1e-4 | Same |
| Max Epochs | 20-25 | 15-20 | 3D trains faster |
| CLIP Skip | 2 | 2 | Same |
| Network Dim | 32 | 32 | Same |
| Network Alpha | 16 | 16 | Same |

## Detailed Explanations

### Segmentation Parameters

#### Alpha Threshold

**2D Anime: 0.25**
- 2D anime has hard cel-shading edges
- Clear separation between character and background
- Alpha is often binary (0 or 255)

**3D Animation: 0.15**
- 3D rendering uses anti-aliasing
- Soft alpha transitions at edges
- Gradual falloff from character to background
- Lower threshold captures soft edges without losing detail

**Example:**
```python
# 2D anime edge
[0, 0, 0, 255, 255, 255]  # Hard transition

# 3D animation edge
[0, 32, 128, 200, 240, 255]  # Soft gradient
```

#### Blur Threshold

**2D Anime: 100**
- 2D anime is usually sharp
- Blur indicates low quality or motion blur
- Higher threshold = stricter quality filter

**3D Animation: 80**
- 3D uses cinematic depth-of-field
- Background blur is intentional, not quality issue
- Lower threshold = accept artistic blur
- Still filter out truly blurry/low-quality frames

### Clustering Parameters

#### Min Cluster Size

**2D Anime: 20-25**
- 2D character art varies significantly
- Different animators, different episodes
- On-model vs off-model variations
- Need larger cluster for consensus

**3D Animation: 10-15**
- 3D character models are identical
- Same model in every frame
- Only lighting/angle varies
- Smaller cluster still reliable

**Why it matters:**
```
2D: 1000 images → 40 clusters (avg 25 images/cluster)
3D: 1000 images → 80 clusters (avg 12 images/cluster)
```

#### Min Samples

**2D Anime: 3-5**
- HDBSCAN needs more samples for reliable clusters
- Accounts for art variation
- Prevents false positives

**3D Animation: 2**
- Less variation means tighter clusters
- Can be more aggressive with clustering
- Faster convergence

### Training Data Requirements

#### Dataset Size

**Why 3D needs fewer images:**

1. **Consistency**: 3D model is identical in every frame
   - 2D: Artist variations, on/off-model
   - 3D: Perfect model consistency

2. **Angles**: 3D naturally covers all angles
   - 2D: Need explicit examples of all angles
   - 3D: Model rotates smoothly, interpolates well

3. **Lighting**: 3D lighting is consistent physics
   - 2D: Stylized lighting varies
   - 3D: Follows realistic light physics

**Practical implications:**
```
Character LoRA from 90-minute movie:

2D Anime:
- Extract 10,000 frames
- Segment 3,000 characters
- Cluster to 800 images per character
- Train with 500-1000 images

3D Animation:
- Extract 5,000 frames
- Segment 2,000 characters
- Cluster to 400 images per character
- Train with 200-400 images
```

### Augmentation Differences

#### Color Augmentation

**2D Anime: Enabled**
- Different episodes may have color variations
- Helps LoRA generalize across color palettes
- Prevents overfitting to specific colors

**3D Animation: Disabled**
- 3D materials have specific colors
- PBR (Physically Based Rendering) materials consistent
- Color augmentation breaks material properties
- Causes unrealistic color shifts in output

**Example:**
```python
# 2D: Can shift character's shirt from blue to cyan
# LoRA learns "shirt" concept, not specific blue

# 3D: Woody's vest is always same brown
# PBR material defines exact color
# Augmentation would create wrong material
```

#### Horizontal Flip

**2D Anime: Enabled**
- 2D characters often symmetric
- Flip increases dataset size
- Helps with pose variation

**3D Animation: Disabled**
- 3D characters have asymmetric details
- Accessories, patches, logos on specific side
- Flip creates "mirror universe" character
- Confuses LoRA about correct orientation

**Example:**
```
Woody (Toy Story):
- Sheriff badge on LEFT chest
- Holster on RIGHT hip
- If flipped: badge on right, holster on left
- Wrong! Breaks character consistency
```

### Caption Strategy

#### 2D Anime Captions
```
"1boy, anime style, cel-shaded, character name, blue hair,
school uniform, determined expression, outdoor background"
```

**Focus:**
- Art style (anime, cel-shaded)
- Character features (hair color, clothing)
- Booru-style tags

#### 3D Animation Captions
```
"a 3d animated character, character description, pixar style,
smooth shading, studio lighting, high quality render,
photorealistic materials"
```

**Focus:**
- 3D rendering terms (render, shading, materials)
- Production quality (studio lighting, high quality)
- Specific 3D style (pixar, dreamworks, disney)

### Training Epoch Requirements

**2D Anime: 20-25 epochs**
- More variation to learn
- Longer training for convergence
- Risk of underfitting <15 epochs

**3D Animation: 15-20 epochs**
- Less variation = faster learning
- Risk of overfitting >20 epochs
- Often converges by epoch 12-15

**Monitoring:**
```
2D: Loss plateaus around epoch 18-22
3D: Loss plateaus around epoch 12-16
```

## Common Misconceptions

### ❌ "Use same parameters for all animation"
Different animation styles need different processing. 2D and 3D are fundamentally different rendering techniques.

### ❌ "More data always better"
For 3D, 200 high-quality diverse images beats 1000 similar images. Focus on diversity over quantity.

### ❌ "3D is just easier 2D"
3D requires understanding of:
- PBR materials
- Lighting physics
- Anti-aliasing
- Depth of field
Different challenges, not easier.

### ❌ "Can train 3D on anime base model"
Can work, but better results with models trained on realistic/3D content. SD 1.5 or SDXL work well.

## Workflow Differences

### 2D Anime Workflow
```
Video → Dense frame extraction (1 frame/10 frames)
     → Heavy deduplication (many static frames)
     → Segmentation (cel-shading friendly)
     → Clustering (large clusters needed)
     → Manual curation (on-model selection)
     → Large dataset (500-1000 images)
     → Long training (20-25 epochs)
```

### 3D Animation Workflow
```
Video → Scene-based extraction (1 frame/scene)
     → Light deduplication (already diverse)
     → Segmentation (adjust for soft edges)
     → Clustering (smaller clusters OK)
     → Quick review (less curation needed)
     → Medium dataset (200-400 images)
     → Shorter training (15 epochs)
```

## Parameter Selection Flowchart

### Choosing Alpha Threshold

```
Does content have hard outlines? (cel-shading)
├─ Yes → Use 0.25 (2D anime)
└─ No → Does it have soft anti-aliased edges?
        ├─ Yes → Use 0.15 (3D animation)
        └─ Unclear → Test both, compare results
```

### Choosing Dataset Size

```
Is character model 100% consistent?
├─ Yes → 200-400 images (3D animation)
└─ No → Is there significant art variation?
        ├─ High variation → 1000+ images (2D TV anime)
        ├─ Medium variation → 500-800 images (2D movie)
        └─ Low variation → 400-600 images (consistent 2D)
```

## Testing Your Parameters

### Quick Parameter Test

1. **Extract 100 frames** from source
2. **Segment with default 2D params** (alpha=0.25, blur=100)
3. **Count successful segmentations**
4. **Segment with 3D params** (alpha=0.15, blur=80)
5. **Compare results:**
   - More characters found? → Use 3D params
   - Better edge quality? → Use 3D params
   - No difference? → Source might be 2.5D, test training

### A/B Training Test

1. **Prepare two datasets:**
   - Dataset A: 400 images, 2D params
   - Dataset B: 400 images, 3D params
2. **Train two LoRAs** (same config otherwise)
3. **Compare outputs:**
   - Which maintains 3D appearance better?
   - Which responds to lighting prompts better?
   - Which has fewer artifacts?

## Conclusion

Understanding parameter differences between 2D and 3D is crucial for optimal results:

**Key Takeaways:**
- 3D needs **softer thresholds** (0.15 vs 0.25)
- 3D allows **tighter clusters** (10 vs 20)
- 3D requires **fewer images** (200-400 vs 500-1000)
- 3D trains **faster** (15 vs 20-25 epochs)
- 3D **disables** color and flip augmentation

When in doubt, analyze your source content:
- Smooth shading? → 3D parameters
- Cel-shaded? → 2D parameters
- Hybrid (2.5D like Arcane)? → Test both approaches

For detailed workflow, see:
- `3D_PROCESSING_GUIDE.md`
- `3D_FEATURES.md`
- `.claude/claude.md`
