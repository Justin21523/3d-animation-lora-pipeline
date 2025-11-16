# {Film Name} - Film-Specific Documentation

**Film:** {Film Name} ({Year})
**Studio:** {Studio Name} (e.g., Pixar, DreamWorks, Disney)
**Runtime:** {XX} minutes
**Setting:** {Brief setting description}

---

## Overview

{Brief 2-3 sentence description of the film}

## Key Visual Features

### Art Style
- **Visual Aesthetic:** {Describe the distinctive visual style}
- **Lighting:** {Typical lighting characteristics}
- **Color Palette:** {Dominant colors and tones}
- **Materials/Rendering:** {3D rendering characteristics}

### Scene Characteristics
- **Multi-Character Scenes:** {Typical number of characters per frame}
- **Camera Techniques:** {DoF, camera angles, etc.}
- **Special Visual Elements:** {Any unique visual features}

---

## Main Characters

### Primary Characters (Training Priority: High)

1. **{Character 1 Name}** (Role)
   - **Forms:** {If character has transformations/variants}
   - **Appearance:** {Brief physical description}
   - See: [`characters/character_{name}.md`](characters/character_{name}.md)

2. **{Character 2 Name}** (Role)
   - **Forms:** {Forms/variants}
   - **Appearance:** {Description}
   - See: [`characters/character_{name}.md`](characters/character_{name}.md)

### Supporting Characters (Training Priority: Medium)

3. **{Character 3 Name}** (Role)
   - {Brief description}
   - See: [`characters/character_{name}.md`](characters/character_{name}.md)

### Other Characters
- **{Character Name}** - {Brief role description}
- **{Character Name}** - {Brief role description}

---

## Configuration Files

### Inpainting Configuration
**Location:** `configs/inpainting/{film_name}_prompts.json`

Status: {✅ Created / ⏳ To be created}

Should contain:
- Character-specific prompts for all main characters
- Full body descriptions
- Body part-specific prompts
- Scene context prompts

### Clustering Configuration
**Location:** `configs/clustering/{film_name}_config.yaml`

Status: {✅ Created / ⏳ To be created}

Should define:
- Character names and forms
- Clustering parameters
- Quality filtering thresholds
- Character keywords

### Project Configuration
**Location:** `configs/projects/{film_name}.yaml`

Status: {✅ Created / ⏳ To be created}

Master configuration including:
- All pipeline parameters
- Path definitions
- Film-specific notes

---

## Processing Notes

### Frame Extraction
- **Status:** {Status}
- **Frames Extracted:** {Number}
- **Parameters:** {Scene threshold, etc.}

### SAM2 Instance Segmentation
- **Status:** {Status}
- **Instances Extracted:** {Number}
- **Parameters:** {points_per_side, min_size, etc.}
- **Notes:** {Any special considerations}

### Inpainting System
- **Status:** {Status}
- **Method:** {Recommended method}
- **Character Prompts:** {Status}

### Identity Clustering
- **Status:** {Status}
- **Expected Identities:** {Number} main characters
- **Forms to Track:** {List character forms}

---

## Special Considerations

### {Special Feature 1}
- {Description}
- {Impact on processing}
- {Recommendations}

### {Special Feature 2}
- {Description}
- {Impact on processing}
- {Recommendations}

---

## Training Recommendations

### Dataset Sizes (per character)
- **Main Characters:** {Recommended range} images each
- **Supporting Characters:** {Recommended range} images

### Caption Prefix
```
"a 3d animated character from {studio}'s {film}, {style description}"
```

### Style Tags
- `{film} style`
- `3d animation`
- `{distinctive style element}`
- `{another element}`

### Training Tips
- **Augmentations:** {Which to enable/disable and why}
- **Target images:** {Recommended count}
- **Epochs:** {Recommended range}
- **Special considerations:** {Any film-specific training notes}

---

## Timeline

| Date | Stage | Status |
|------|-------|--------|
| {Date} | Frame Extraction | {Status} |
| {Date} | SAM2 Segmentation | {Status} |
| {Date} | Inpainting | {Status} |
| {Date} | Identity Clustering | {Status} |
| {Date} | Interactive Review | {Status} |
| {Date} | Caption Generation | {Status} |
| {Date} | LoRA Training | {Status} |

---

## See Also

- **Generic Processing Guide:** [`docs/3d_anime_specific/3D_PROCESSING_GUIDE.md`](../../3d_anime_specific/3D_PROCESSING_GUIDE.md)
- **Inpainting Guide:** [`docs/guides/INPAINTING_GUIDE.md`](../../guides/INPAINTING_GUIDE.md)
- **Architecture Principles:** [`docs/ARCHITECTURE_DESIGN_PRINCIPLES.md`](../../ARCHITECTURE_DESIGN_PRINCIPLES.md)
- **Project Configuration System:** [`docs/PROJECT_CONFIGURATION_SYSTEM.md`](../../PROJECT_CONFIGURATION_SYSTEM.md)
