# Luca - Visual Style Guide

**Film:** Luca (2021, Pixar)
**Director:** Enrico Casarosa
**Purpose:** Technical reference for LoRA training, captioning, and 3D animation pipeline

---

## Overview

Luca's visual style draws heavily from:
- **Italian cinema** (Fellini, neorealism)
- **Hayao Miyazaki** influence (hand-drawn texture feel)
- **Stop-motion aesthetics** (tactile, grounded)
- **Classic Italian Riviera** (1950s seaside nostalgia)

The film uses a **semi-stylized 3D approach** with softer shading, warmer colors, and less photorealistic textures compared to other Pixar films.

---

## 1. Lighting

### 1.1 Daytime / Outdoor Lighting

**Key characteristics:**
- **Warm Mediterranean sunlight** with soft yellow-orange tones
- **Soft shadows** (not harsh or high-contrast)
- **Bounce light** from terracotta walls and cobblestones
- **Atmospheric haze** over distant landscapes
- **Lens flare** used sparingly for nostalgia

**Typical setups:**
- **Key light:** Warm sun at 45-60° angle
- **Fill light:** Ambient sky blue (soft, diffused)
- **Rim light:** Golden hour backlight on hair/shoulders

**Caption tags:**
```
"warm italian sunlight", "soft shadows", "golden hour lighting",
"mediterranean glow", "coastal atmosphere"
```

---

### 1.2 Underwater Lighting

**Key characteristics:**
- **Blue-green volumetric lighting**
- **God rays** (crepuscular rays) filtering through water
- **Caustics** on surfaces (shimmering water patterns)
- **Gradient darkening** toward deeper areas
- **Bioluminescence** (subtle glows on sea monster features)

**Typical setups:**
- **Top-down directional** (surface light penetration)
- **Soft ambient** (scattered underwater light)
- **Accent lights** on character scales/fins

**Caption tags:**
```
"underwater volumetric lighting", "god rays through water",
"caustics patterns", "blue-green oceanic glow"
```

---

### 1.3 Interior / Night Lighting

**Key characteristics:**
- **Warm tungsten practical lights** (lamps, candles)
- **Soft ambient moonlight** (blue tones)
- **Cozy, intimate shadows**
- **Firelight** (flickering orange/red)

**Typical setups:**
- **Practical lights** from lamps, windows, candles
- **Moonlight fill** (cool blue)
- **Contrast between warm interiors and cool night**

**Caption tags:**
```
"warm tungsten lighting", "cozy interior glow", "moonlit scene",
"soft candlelight", "intimate atmosphere"
```

---

## 2. Materials (PBR Shaders)

### 2.1 Skin (Human Form)

**Characteristics:**
- **Semi-stylized subsurface scattering** (SSS)
- Soft, smooth gradients (not hyperrealistic)
- **Rosy cheeks** and **warm undertones**
- Minimal pores/blemishes (clean, cartoon-friendly)
- **Slight translucency** on ears, fingers

**Shader notes:**
- SSS depth: Medium (Pixar's "soft realism" range)
- Roughness: Low (slightly shiny, youthful skin)
- Freckles: Procedural or texture-based (Giulia)

**Caption tags:**
```
"smooth skin shading", "soft subsurface scattering", "rosy cheeks",
"warm skin tones", "pixar character skin"
```

---

### 2.2 Skin (Sea Monster Form)

**Characteristics:**
- **Iridescent scales** with color shift (teal → purple)
- **Wet-to-dry transition** (sparkles when transforming)
- **Translucent fins** (backlit glow effect)
- **Bioluminescent accents** (subtle)
- **Soft scale edges** (not harsh or reptilian)

**Shader notes:**
- Fresnel effect for iridescence
- Animated wetness map (drying process)
- Emission map for glowing patterns

**Caption tags:**
```
"iridescent scales", "sea monster transformation", "glowing fins",
"wet-to-dry transition", "bioluminescent accents"
```

---

### 2.3 Clothing & Fabrics

**Characteristics:**
- **Soft, deformable cloth** (natural wrinkles)
- **Matte cotton/linen** (minimal specularity)
- **Worn, lived-in textures** (not pristine)
- **Hand-drawn texture feel** (slightly stylized seams)

**Examples:**
- Luca's striped shirt: Soft cotton, warm tones
- Alberto's green vest: Rough linen, weathered
- Giulia's overalls: Denim with wrinkles

**Caption tags:**
```
"soft cotton fabric", "weathered clothing", "natural cloth wrinkles",
"hand-drawn texture details"
```

---

### 2.4 Hard Surfaces

**Vespa (metal):**
- **Glossy paint** (red or blue)
- **Chrome reflections** (handlebars, mirrors)
- **Scratches and wear** (nostalgic patina)

**Stone buildings:**
- **Rough plaster** (terracotta, cream, pastel colors)
- **Weathered paint** (peeling, aged)
- **Matte stone** with subtle color variation

**Water:**
- **Stylized reflections** (not photorealistic)
- **Gentle ripples** (hand-animated feel)
- **Color gradient** (turquoise shallows → deep blue)

**Caption tags:**
```
"glossy vespa paint", "chrome reflections", "weathered plaster walls",
"stylized water surface", "matte stone texture"
```

---

## 3. Camera / Cinematography

### 3.1 Shot Types

**Common compositions:**
- **Wide establishing shots** (coastal vistas, town overviews)
- **Three-quarter character shots** (standard dialogue)
- **Low-angle hero shots** (Luca/Alberto on Vespa)
- **Dutch angles** (dynamic action, hill descents)
- **Close-ups on eyes** (emotional beats)

**Focal lengths:**
- **24-35mm** (wide, environmental storytelling)
- **50mm** (intimate character moments)
- **85-100mm** (compressed perspective for landscapes)

---

### 3.2 Camera Movement

**Characteristics:**
- **Handheld-style shake** (minimal, adds energy)
- **Sweeping dolly shots** (coastal roads, hill descents)
- **Crane shots** (town reveals)
- **POV shots** (Luca's imagination sequences)

**Dream sequences:**
- **Float/drift camera** (weightless, surreal)
- **Rack focus** (shallow DOF transitions)

---

### 3.3 Depth of Field (DOF)

**Usage:**
- **Shallow DOF** for emotional closeups
- **Deep DOF** for action sequences (bike race)
- **Cinematic bokeh** (soft, circular highlights)
- **Foreground blur** (contextual storytelling)

**Caption tags:**
```
"shallow depth of field", "bokeh highlights", "cinematic focus",
"soft background blur"
```

---

## 4. Color Grading

### 4.1 Overall Palette

**Primary colors:**
- **Warm yellows/golds** (sunlight, sand)
- **Turquoise blues** (ocean, sky)
- **Terracotta/coral** (buildings, roofs)
- **Sea green** (foliage, water shadows)

**Mood:** Nostalgic, warm, inviting, slightly desaturated (vintage postcard feel)

---

### 4.2 Color by Scene Type

**Daytime exterior:**
- High saturation in blues/yellows
- Warm shadows (no pure black)
- Slight haze for atmosphere

**Underwater:**
- Blue-green dominant
- Low saturation reds (color absorption)
- Gradient from light to dark

**Sunset:**
- Orange/pink sky gradients
- Strong rim lights (golden/red)
- Deep blue shadows

**Night:**
- Cool blue moonlight
- Warm interior practicals
- Desaturated environment

---

### 4.3 LUTs / Post-Processing

**Pixar's approach for Luca:**
- Slight **film grain** (nostalgic texture)
- **Vignette** (subtle, not heavy-handed)
- **Bloom** on bright highlights (sunlight, water sparkles)
- **Color lift** in shadows (warm tones, not crushed blacks)

**Caption tags:**
```
"warm color grading", "nostalgic film aesthetic", "soft bloom",
"vintage summer tones", "mediterranean color palette"
```

---

## 5. Special Effects

### 5.1 Transformation Effect

**When sea monsters hit water or dry off:**
- **Sparkle particles** (twinkling stars)
- **Ripple distortion** (water surface interaction)
- **Scale emergence/retraction** (animated texture)
- **Color shift** (human skin ↔ blue-green scales)

**Caption tags:**
```
"sea monster transformation", "sparkle effect", "magical transition",
"water ripple distortion"
```

---

### 5.2 Water Simulation

**Characteristics:**
- **Stylized splashes** (not fully realistic)
- **Foam patterns** (hand-painted texture feel)
- **Caustics** (underwater light patterns)
- **Gentle wave motion** (not turbulent)

---

### 5.3 Weather & Atmosphere

**Rain:**
- **Heavy droplets** (visible, stylized)
- **Wet surfaces** (glossy roads, reflections)
- **Thunder lighting** (dramatic contrast)

**Fog/Haze:**
- **Atmospheric perspective** (distant hills fade)
- **God rays** (volumetric shafts)

---

## 6. Art Direction Notes

### 6.1 Influences

**Italian Cinema:**
- Federico Fellini (nostalgic, dreamlike)
- Vittorio De Sica (neorealism, human-centric)

**Hayao Miyazaki:**
- Hand-drawn texture aesthetic
- Soft edges and organic shapes
- Emphasis on food and environment

**Stop-Motion:**
- Tactile materials (wood, fabric)
- Slightly imperfect, handmade feel

---

### 6.2 Environment Design

**Portorosso town:**
- **Vertical composition** (cliffside layering)
- **Narrow streets** (cobblestone, arched doorways)
- **Pastel buildings** (peach, yellow, cream)
- **Hanging laundry** (adds life and color)
- **Fishing nets and boats** (dockside details)

**Alberto's tower:**
- **Cluttered, lived-in** (collected junk)
- **Warm wood tones**
- **Sunlight streaming through windows**

---

## 7. Technical Specifications

### 7.1 Aspect Ratio
- **1.85:1** (widescreen, not ultra-wide)

### 7.2 Rendering
- **Path-traced global illumination**
- **Multi-bounce lighting** (realistic but stylized)
- **Ambient occlusion** (soft contact shadows)

### 7.3 Anti-Aliasing
- **Soft edges** (3D animation priority)
- **Motion blur** (24fps cinematic feel)

---

## 8. Caption Style Guidelines

### 8.1 For LoRA Training

**Recommended format:**
```
"a 3d animated character, [character name] from pixar luca (2021),
[physical description], [pose/action], [expression],
[lighting description], smooth shading, studio lighting,
italian riviera background"
```

**Example:**
```
"a 3d animated character, luca paguro from pixar luca (2021),
young boy with wavy brown hair, riding bicycle down cobblestone hill,
excited expression, warm mediterranean sunlight, soft shadows,
smooth shading, depth of field, portorosso background"
```

---

### 8.2 Negative Prompts

**For 3D style consistency:**
```
2d, anime, cartoon, sketch, watercolor, painting, illustration,
flat colors, cel shading, line art, photorealistic, realistic,
live action, photography, bad anatomy, deformed, blurry,
low quality, jpeg artifacts
```

---

## 9. Quick Reference Table

| Element | Style | Tags |
|---------|-------|------|
| Lighting | Warm Mediterranean, soft shadows | `warm sunlight`, `soft lighting` |
| Skin | Semi-stylized SSS, rosy cheeks | `smooth shading`, `subsurface scattering` |
| Materials | PBR, matte surfaces, weathered | `pbr materials`, `stylized textures` |
| Camera | Wide vistas, 3/4 shots, shallow DOF | `cinematic angle`, `depth of field` |
| Color | Warm pastels, turquoise blues | `warm tones`, `mediterranean palette` |
| Water | Stylized, caustics, gentle waves | `stylized water`, `ocean caustics` |

---

## 10. Related Documentation

- **Film Metadata:** `docs/films/luca/film_metadata.json`
- **Character Files:** `docs/films/luca/characters/`
- **Prompt Libraries:** `prompts/luca/`
- **3D Processing Guide:** `docs/3d_anime_specific/3D_PROCESSING_GUIDE.md`

---

**Last Updated:** 2025-11-11
**Version:** 1.0
**For:** Luca LoRA training pipeline
