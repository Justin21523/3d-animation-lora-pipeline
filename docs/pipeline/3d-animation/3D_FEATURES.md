# 3D Animation Features

## Overview

This pipeline is specifically optimized for **3D animated content** (Pixar, DreamWorks, Disney 3D, etc.), which has fundamentally different characteristics from 2D anime.

## Key 3D Animation Characteristics

### 1. **Rendering Technology**

3D animation uses ray tracing and rasterization rendering:
- **Smooth shading** instead of cel-shading
- **Realistic lighting** with global illumination
- **Soft shadows** and ambient occlusion
- **Subsurface scattering** for skin/translucent materials
- **Specular highlights** and reflections
- **Depth of field** and motion blur

### 2. **Material Properties**

3D characters have consistent material properties:
- **PBR materials** (Physically Based Rendering)
- **Roughness and metallic maps**
- **Normal and bump mapping**
- **Translucency** for hair, clothing, skin
- **Consistent colors** across angles (no cel-shading variation)

### 3. **Character Consistency**

3D models maintain perfect consistency:
- **Same model** from all angles
- **Consistent proportions** across frames
- **Predictable lighting** response
- **No on-off art variation** (unlike 2D anime)

### 4. **Edge Characteristics**

3D rendering produces soft, anti-aliased edges:
- **Smooth alpha transitions** (not hard edges)
- **Sub-pixel anti-aliasing**
- **Transparency gradients** for hair/effects
- **No cel-shading outlines** (usually)

## Pipeline Optimizations for 3D

### Segmentation Adjustments

**Alpha Threshold: 0.15** (vs 0.25 for 2D)
- 3D has softer alpha gradients due to anti-aliasing
- Lower threshold captures smooth edge transitions
- Prevents losing detail at character boundaries

**Blur Threshold: 80** (vs 100 for 2D)
- 3D content may have intentional depth-of-field blur
- More lenient blur tolerance for background characters
- Preserves cinematically blurred frames

**Model Selection: U²-Net or SAM**
- U²-Net works well for general 3D segmentation
- SAM (Segment Anything) excels at 3D object detection
- ISNet can be used for high-precision needs

### Clustering Adjustments

**Min Cluster Size: 10-15** (vs 20-25 for 2D)
- 3D characters are more consistent across angles
- Tighter clusters form naturally
- Fewer samples needed for reliable identity

**Min Samples: 2** (vs 3-5 for 2D)
- HDBSCAN can use stricter parameters
- Less variation means tighter clusters
- Faster convergence

### Caption Templates

Use 3D-specific terminology:

```
Base: "a 3d animated character, [description], pixar style"
Quality: "high quality render, smooth shading, studio lighting"
Technical: "photorealistic materials, subsurface scattering, ambient occlusion"
```

Avoid 2D terms:
- ❌ "cel-shaded"
- ❌ "anime style"
- ❌ "hand-drawn"
- ✅ "rendered"
- ✅ "3d model"
- ✅ "smooth shading"

## Feature Comparison: 3D vs 2D

| Feature | 2D Anime | 3D Animation |
|---------|----------|--------------|
| Shading | Cel-shaded (flat colors) | Smooth gradients |
| Edges | Hard outlines | Anti-aliased soft edges |
| Lighting | Stylized, simple | Realistic, complex |
| Consistency | Variable across frames | Perfectly consistent |
| Materials | Flat colors | PBR with textures |
| Shadows | Hard shadows or none | Soft, realistic shadows |
| Alpha | Binary (on/off) | Gradual transitions |
| Blur | Usually sharp | Intentional DoF blur |

## Training Data Considerations

### Dataset Size

3D characters need **fewer training images** than 2D:
- **200-500 images**: Usually sufficient for character LoRA
- **500-1000 images**: For style LoRA or complex characters
- **1000+ images**: For multi-character or scene LoRA

Comparison:
- 2D anime: 500-1000 images minimum
- 3D animation: 200-500 images sufficient

Reason: 3D models are inherently consistent, so less variation to learn.

### Diversity Requirements

Focus on:
- **Expression variation**: Happy, sad, surprised, etc.
- **Pose variation**: Standing, sitting, running, jumping
- **Lighting variation**: Different times of day, environments
- **Camera angles**: Front, side, 3/4, back views

Less concern about:
- Style variation (3D model is consistent)
- Art quality variation (render quality is consistent)
- On/off-model issues (doesn't exist in 3D)

### Data Augmentation

**Disable certain augmentations** that break 3D consistency:
- ❌ **Color augmentation**: 3D materials have consistent colors
- ❌ **Horizontal flip**: Breaks model symmetry (characters aren't perfectly symmetric)
- ✅ **Crop/resize**: Safe for 3D
- ✅ **Brightness adjustment**: Safe in moderation

## Quality Metrics

### Segmentation Quality

For 3D content, check:
- **Alpha smoothness**: Gradual edge transitions preserved
- **Material separation**: Hair/skin/clothing correctly separated
- **Transparency handling**: Semi-transparent materials preserved
- **Lighting preservation**: Highlights/shadows not lost

### Clustering Quality

For 3D characters:
- **Tighter clusters**: Expect similarity >0.85 within clusters
- **Clear separation**: Different characters should cluster cleanly
- **Angle consistency**: Same character from different angles should cluster together

### Training Quality

Monitor for:
- **Material consistency**: Does LoRA preserve PBR properties?
- **Lighting response**: Does character respond to lighting prompts?
- **Shading quality**: Smooth gradients maintained?
- **3D appearance**: Still looks 3D, not flattened to 2D?

## Common 3D Animation Styles

### Pixar Style
- Smooth, cartoony proportions
- Warm, inviting lighting
- Soft shadows and ambient occlusion
- Subsurface scattering on skin
- Exaggerated expressions
- High production values

**Caption template:**
```
"pixar style 3d animation, [character], smooth shading, warm lighting, high quality render"
```

### DreamWorks Style
- More stylized than Pixar
- Vibrant, saturated colors
- Expressive, exaggerated features
- Dynamic poses and compositions

**Caption template:**
```
"dreamworks style 3d animation, [character], vibrant colors, expressive features"
```

### Disney 3D Style
- Photorealistic materials
- Magical lighting
- Detailed textures
- Cinematic quality

**Caption template:**
```
"disney 3d animation, [character], photorealistic materials, cinematic lighting"
```

### Illumination Style
- Simplified shapes
- Bright, playful colors
- Less detailed textures
- Comedy-focused

**Caption template:**
```
"illumination style 3d animation, [character], bright colors, simple shapes, stylized render"
```

## Technical Recommendations

### Hardware

3D content processing benefits from:
- **GPU VRAM**: 12GB+ recommended (3D renders are often high-res)
- **System RAM**: 32GB+ for large batch processing
- **Storage**: Fast SSD for latent caching

### Software

- **Segmentation**: U²-Net or SAM
- **Clustering**: CLIP ViT-L/14 (standard)
- **Captioning**: BLIP2 (works well for 3D)
- **Training**: Kohya_ss sd-scripts

### Performance

3D content processes **faster** in some ways:
- ✅ Clustering converges quicker (more consistent)
- ✅ Fewer training images needed
- ❌ Segmentation may be slower (higher resolution)
- ❌ More GPU memory needed for high-res renders

## Conclusion

3D animation requires different parameter tuning and processing strategies than 2D anime. The key is recognizing the **consistency** and **material properties** that make 3D unique, and adjusting your pipeline accordingly.

For detailed workflow, see `.claude/claude.md` and specific tool guides in `docs/guides/tools/`.
