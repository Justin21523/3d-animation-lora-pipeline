# Luca - Film-Specific Documentation

**Film:** Luca (2021)
**Studio:** Pixar Animation Studios
**Director:** Enrico Casarosa
**Release:** June 18, 2021 (Disney+)
**Runtime:** 95 minutes
**Setting:** Italian Riviera (Portorosso, circa 1959)

---

## Overview

Luca is a coming-of-age story about sea monsters who transform into humans when dry. The story follows Luca Paguro and Alberto Scorfano as they experience an unforgettable summer in the Italian coastal town of Portorosso, filled with gelato, pasta, Vespa dreams, and the Portorosso Cup triathlon.

**Synopsis:** Set in a beautiful seaside town on the Italian Riviera, this film explores themes of friendship, identity, acceptance, and the courage to be different. The sea monsters serve as a metaphor for feeling like an outsider during childhood.

---

## Film Resources

### Structured Metadata
**Location:** [`film_metadata.json`](film_metadata.json)

Complete structured information including:
- Cast and characters
- Locations and settings
- Themes and cultural elements
- Color palette (RGB/hex values)
- Music and soundtrack details
- Easter eggs and trivia

### Visual Style Guide
**Location:** [`style_guide.md`](style_guide.md)

Technical reference for LoRA training and captioning:
- Lighting setups (Mediterranean, underwater, interior)
- PBR materials and shaders (skin, clothing, surfaces)
- Camera and cinematography techniques
- Color grading and mood
- Caption style guidelines
- Negative prompt recommendations

### Prompt Libraries
**Location:** `../../prompts/luca/`

Character-specific prompt collections:
- `luca_human_prompts.json` - Comprehensive prompts for Luca (human form)
- `alberto_human_prompts.json` - Alberto-specific prompts
- Organized by category (portrait, full_body, expressions, scenes, etc.)
- Includes positive and negative prompts with template variable expansion

**Usage:**
```python
from scripts.core.utils.prompt_loader import load_character_prompts

loader = load_character_prompts('luca_human', film='luca')
prompts = loader.get_balanced_sample(num_prompts=12)
```

## Key Visual Features

### Character Transformations
- **Dual Forms:** Main characters (Luca, Alberto) have both human and sea monster forms
- **Transformation Trigger:** Water contact causes instant transformation
- **Visual Distinction:** Sea monster forms have distinct scales, fins, and colors

### Art Style
- **Italian Coastal Aesthetic:** Warm, sun-drenched colors
- **Summer Lighting:** Bright, natural lighting with soft shadows
- **Underwater Scenes:** Different color grading, more depth-of-field blur
- **PBR Materials:** Pixar's signature smooth shading

### Scene Characteristics
- **Multi-Character Scenes:** Frequently 2-3+ characters per frame
- **Depth-of-Field:** Cinematic DoF blur in many shots
- **Color Palette:** Warm tones (yellows, oranges, teals, blues)

---

## Main Characters

### Primary Characters (Training Priority: High)

1. **Luca Paguro** (Protagonist)
   - **Human Form:** Teenage boy, teal striped shirt, brown wavy hair
   - **Sea Monster Form:** Blue-teal scales, fins, webbed hands
   - See: [`characters/character_luca.md`](characters/character_luca.md)

2. **Alberto Scorfano** (Deuteragonist)
   - **Human Form:** Teenage boy, green vest, curly dark hair
   - **Sea Monster Form:** Green scales, more robust build
   - See: [`characters/character_alberto.md`](characters/character_alberto.md)

3. **Giulia Marcovaldo** (Tritagonist)
   - **Form:** Human only (no transformation)
   - **Appearance:** Red hair, freckles, energetic personality
   - See: [`characters/character_guilia.md`](characters/character_guilia.md)

### Supporting Characters (Training Priority: Medium)

4. **Massimo Marcovaldo** (Giulia's Father)
   - Large build, beard, one arm, fisherman
   - See: [`characters/character_massimo.md`](characters/character_massimo.md)

5. **Ercole Visconti** (Antagonist)
   - Teenage bully, athletic build, arrogant
   - See: [`characters/character_ercole.md`](characters/character_ercole.md)

### Other Characters
- **Daniela Paguro** - Luca's mother (sea monster)
- **Lorenzo Paguro** - Luca's father (sea monster)
- **Grandma Paguro** - Adventurous grandmother

---

## Configuration Files

### Inpainting Configuration
**Location:** `configs/inpainting/luca_prompts.json`

Contains character-specific prompts for:
- Luca (human & sea monster forms)
- Alberto (human & sea monster forms)
- Giulia
- Massimo
- Ercole

Each character has:
- Full body descriptions
- Body part-specific prompts (face, torso, arms, hair, etc.)
- Scene context prompts

### Clustering Configuration
**Location:** `configs/clustering/luca_config.yaml`

Defines:
- Character names and forms
- Clustering parameters optimized for Luca
- Quality filtering thresholds
- Character keywords for identification

### Project Configuration
**Location:** `configs/projects/luca.yaml`

Master configuration including:
- All pipeline parameters
- Path definitions
- Processing history
- Film-specific notes

---

## Processing Notes

### SAM2 Instance Segmentation
- **Status:** In progress (81% complete as of 2025-11-09)
- **Instances Extracted:** 44,567
- **Parameters:** `points_per_side=20` (balanced mode)
- **Notes:** Stable, no hanging issues

### Inpainting System
- **Status:** Fully configured
- **Method:** LaMa recommended (balanced speed/quality)
- **Character Prompts:** All main characters defined
- **Auto-detection:** Enabled via filename parsing

### Identity Clustering
- **Status:** Ready (awaiting SAM2 completion)
- **Expected Identities:** 7-8 main characters
- **Forms to Track:**
  - Luca Human
  - Luca Sea Monster
  - Alberto Human
  - Alberto Sea Monster
  - Giulia
  - Massimo
  - Ercole

---

## Special Considerations

### Multi-Form Characters
- Luca and Alberto each need **TWO separate LoRA models** (human & sea monster)
- Keep forms separate during clustering and training
- Use form-specific prompts for captioning

### Underwater vs Land Scenes
- Different lighting and color grading
- May affect character appearance clustering
- Consider separate subclusters for underwater shots

### Character Interactions
- Many scenes with 2-3 characters together
- SAM2 instance segmentation handles overlapping characters
- Inpainting fills occluded regions

### Italian Cultural Elements
- Setting influences clothing, architecture, props
- Include "Italian coastal town" in captions
- Vespa scooters, cobblestone streets, etc.

---

## Training Recommendations

### Dataset Sizes (per character)
- **Main Characters (Luca, Alberto, Giulia):** 300-500 images each
- **Supporting (Massimo, Ercole):** 150-300 images
- **Each form separately:** Luca Human (300-400), Luca Sea Monster (200-300)

### Caption Prefix
```
"a 3d animated character from pixar's luca, smooth shading, italian coastal town setting"
```

### Style Tags
- `pixar luca style`
- `3d animation`
- `italian riviera`
- `summer lighting`
- `smooth shading`

### Training Tips
- **Disable color jitter** - Preserves distinctive color palette
- **Disable horizontal flips** - Character asymmetries
- **Target 400 images** per character form
- **Epochs: 10** typically sufficient for 3D characters

---

## Timeline

| Date | Stage | Status |
|------|-------|--------|
| 2025-11-09 | Frame Extraction | ✅ Complete |
| 2025-11-09 | SAM2 Segmentation | 🔄 In Progress (81%) |
| 2025-11-09 | Inpainting System | ✅ Ready |
| TBD | Identity Clustering | ⏳ Pending |
| TBD | Interactive Review | ⏳ Pending |
| TBD | Caption Generation | ⏳ Pending |
| TBD | LoRA Training | ⏳ Pending |

---

## Related Documentation

### Film-Specific
- **Film Metadata:** [`film_metadata.json`](film_metadata.json) - Structured film data
- **Visual Style Guide:** [`style_guide.md`](style_guide.md) - Technical art direction
- **Character Files:** [`characters/`](characters/) - Individual character documentation
- **Prompt Libraries:** `../../prompts/luca/` - Training/testing prompts

### Generic Guides
- **3D Processing Guide:** [`docs/3d_anime_specific/3D_PROCESSING_GUIDE.md`](../../3d_anime_specific/3D_PROCESSING_GUIDE.md)
- **Inpainting Guide:** [`docs/guides/INPAINTING_GUIDE.md`](../../guides/INPAINTING_GUIDE.md)
- **Architecture Principles:** [`docs/ARCHITECTURE_DESIGN_PRINCIPLES.md`](../../ARCHITECTURE_DESIGN_PRINCIPLES.md)
- **Multi-Character Clustering:** [`docs/guides/MULTI_CHARACTER_CLUSTERING.md`](../../guides/MULTI_CHARACTER_CLUSTERING.md)

---

**Last Updated:** 2025-11-11
**Documentation Version:** 2.0 (integrated film_info.md)
