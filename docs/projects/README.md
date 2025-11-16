# Film-Specific Documentation

此目錄包含各電影的特定文檔與資訊。每部電影都有自己的子目錄，包含角色資訊、處理記錄和特殊考量。

This directory contains film-specific documentation and information. Each film has its own subdirectory with character info, processing notes, and special considerations.

---

## Directory Structure

```
docs/films/
├── README.md                    # 本文件 (This file)
├── template/                    # 新電影文檔模板 (Template for new films)
│   ├── README_template.md
│   ├── film_metadata_template.json
│   ├── style_guide_template.md
│   └── character_template.md
│
├── luca/                        # Luca 專用文檔
│   ├── README.md
│   ├── film_metadata.json      # Film metadata (structured)
│   ├── style_guide.md          # Visual style documentation
│   └── characters/
│       ├── character_luca.md
│       ├── character_alberto.md
│       ├── character_giulia.md
│       ├── character_massimo.md
│       └── character_ercole.md
│
└── {other_films}/               # 未來其他電影
    ├── README.md
    ├── film_metadata.json
    ├── style_guide.md
    └── characters/

prompts/                         # Character-specific prompt libraries
├── luca/
│   ├── luca_human_prompts.json
│   ├── alberto_human_prompts.json
│   └── ...
└── {other_films}/
```

---

## Currently Documented Films

### 🎬 Luca (2021)
**Path:** [`luca/`](luca/)
**Status:** ✅ Complete documentation (v2.0)
**Characters:** 6 main characters documented
**Resources:**
- ✅ Film metadata (JSON)
- ✅ Visual style guide
- ✅ Character files (6 characters)
- ✅ Prompt libraries (Luca, Alberto)
- ✅ Inpainting, Clustering, Project configs

---

## Adding a New Film

### Step 1: Create Directory Structure

```bash
mkdir -p docs/films/{new_film}/characters
```

### Step 2: Copy and Customize README

```bash
cp docs/films/template/README_template.md docs/films/{new_film}/README.md
# Edit the README to fill in film-specific information
```

### Step 3: Document Main Characters

```bash
cp docs/films/template/character_template.md docs/films/{new_film}/characters/character_{name}.md
# Create one file per main character
# Fill in physical descriptions, personality, roles, etc.
```

### Step 4: Create Configuration Files

```bash
# Inpainting configuration
cp configs/inpainting/template.json configs/inpainting/{new_film}_prompts.json
# Edit to add character descriptions

# Clustering configuration
cp configs/clustering/template.yaml configs/clustering/{new_film}_config.yaml
# Edit to define character names and forms

# Project configuration
cp configs/projects/template.yaml configs/projects/{new_film}.yaml
# Edit to define all pipeline parameters
```

### Step 5: Update Film Index

Add the new film to this README under "Currently Documented Films"

---

## Film Documentation Guidelines

### What to Include

**Film README (`{film}/README.md`):**
- Overview and key visual features
- Complete character list with priorities
- Configuration file locations and status
- Processing notes and timeline
- Special considerations for this film
- Training recommendations
- References to all film resources

**Film Metadata (`{film}/film_metadata.json`):**
- Structured metadata (title, studio, year, setting, director)
- Cast and voice actors
- Key locations and objects
- Themes and cultural elements
- Color palette (RGB/hex values)
- Music and soundtrack details
- Trivia and Easter eggs

**Style Guide (`{film}/style_guide.md`):**
- Lighting setups (daytime, underwater, interior/night)
- Materials and shaders (skin SSS, fabrics, hard surfaces)
- Camera and cinematography (shot types, movement, DOF)
- Color grading and mood
- Special effects (transformations, water simulation)
- Art direction influences
- Caption style guidelines (positive and negative prompts)
- Technical specifications

**Character Files (`{film}/characters/character_{name}.md`):**
- Physical appearance (detailed)
- Personality and role
- Forms/variants (if applicable)
- Relationships with other characters
- Scene contexts (where they typically appear)
- Visual tags for prompts

**Prompt Library (`prompts/{film}/{character}_prompts.json`):**
- Positive prompts by category (portrait, full_body, expressions, etc.)
- Negative prompts (style contamination, quality issues)
- Advanced negative categories
- Metadata and version info

### What NOT to Include

- ❌ Generic pipeline instructions (those go in `docs/guides/`)
- ❌ Code documentation (that belongs in `docs/reference/`)
- ❌ Processing logs (those are in output directories)

---

## Related Documentation

### Generic Guides (All Films)
- **Processing Guide:** [`docs/3d_anime_specific/3D_PROCESSING_GUIDE.md`](../3d_anime_specific/3D_PROCESSING_GUIDE.md)
- **Inpainting Guide:** [`docs/guides/INPAINTING_GUIDE.md`](../guides/INPAINTING_GUIDE.md)
- **Architecture:** [`docs/ARCHITECTURE_DESIGN_PRINCIPLES.md`](../ARCHITECTURE_DESIGN_PRINCIPLES.md)

### Configuration System
- **System Overview:** [`docs/PROJECT_CONFIGURATION_SYSTEM.md`](../PROJECT_CONFIGURATION_SYSTEM.md)
- **Config Templates:** `configs/{module}/template.*`

---

## Quick Reference

### Finding Film Information

```bash
# List all documented films
ls docs/films/

# View a film's overview
cat docs/films/{film}/README.md

# View a specific character
cat docs/films/{film}/characters/character_{name}.md
```

### Using Film Configurations

```bash
# Inpainting with film-specific prompts
python scripts/generic/enhancement/inpaint_occlusions.py \
  --project {film} \
  --method sd \
  --auto-detect-character

# Clustering with film-specific character names
python scripts/generic/clustering/face_identity_clustering.py \
  /path/to/instances \
  --project {film}

# Any generic tool can use --project parameter
python scripts/generic/{module}/{script}.py \
  --project {film} \
  {other args}
```

---

## Maintenance

### Updating Film Documentation

When processing a film:
1. Update the Timeline table in `{film}/README.md`
2. Add processing notes and statistics
3. Document any special issues encountered
4. Update configuration status (✅ Complete / 🔄 In Progress / ⏳ Pending)

### Version History

Each film's README should maintain a "Timeline" section tracking:
- Processing dates
- Stage completion status
- Any issues or solutions

---

## Template Usage

The `template/` directory contains:
- **README_template.md** - Copy this for new film overviews
- **character_template.md** - Copy this for each character

Always use templates to maintain consistency across films.

---

## Prompt Library System

**NEW:** Character-specific prompt libraries for training and evaluation.

**Location:** `prompts/{film_name}/{character}_prompts.json`

**Features:**
- ✅ Comprehensive prompts across 7+ categories
- ✅ Positive AND negative prompts
- ✅ Template variable expansion ({{base_negative}}, etc.)
- ✅ Balanced sampling across categories
- ✅ Film-accurate descriptions

**Usage:**
```python
from scripts.core.utils.prompt_loader import load_character_prompts

# Load prompts
loader = load_character_prompts('luca_human', film='luca')

# Get balanced sample
prompts = loader.get_balanced_sample(num_prompts=12)

# Get by category
portrait_prompts = loader.get_prompts_by_category('portrait')
```

**See:** `prompts/luca/` for complete implementation examples

---

**Last Updated:** 2025-11-11
**Films Documented:** 1 (Luca - complete v2.0: metadata, style guide, prompt libraries, character files)
**Films In Progress:** 0

**Luca v2.0 Updates:**
- ✅ Integrated film_info.md content into structured format
- ✅ Created film_metadata.json with comprehensive film data
- ✅ Created style_guide.md with technical art direction
- ✅ Updated README with resource references
- ✅ Established reusable structure for future films
