# Synthetic Data Generation System

A comprehensive system for generating diverse, high-quality synthetic training data for character LoRA models using vocabulary-based prompt generation.

## Overview

This system uses structured vocabulary YAMLs (expressions, poses, actions) to automatically generate thousands of varied prompts, helping improve LoRA generalization and reduce overfitting.

## Core Components

### 1. Prompt Generator (`prompt_generator.py`)

Main script for generating diverse training prompts from vocabulary components.

**Features:**
- Single-character prompt generation
- Multi-character interaction prompts
- Mixed mode (70% single, 30% interaction)
- Reproducible generation with seed support
- Vocabulary-driven randomization
- JSON/TXT output formats

**Usage Examples:**

```bash
# Generate 100 single-character prompts for Bryce
conda run -n ai_env python prompt_generator.py \
  --character bryce \
  --count 100 \
  --seed 42 \
  --output prompts/bryce_synthetic.txt

# Generate multi-character interaction prompts
conda run -n ai_env python prompt_generator.py \
  --characters bryce,caleb,elio \
  --mode interaction \
  --count 50 \
  --output prompts/interactions.txt

# Generate mixed prompts (single + interaction)
conda run -n ai_env python prompt_generator.py \
  --characters bryce,caleb \
  --mode mixed \
  --count 200 \
  --output prompts/mixed.json \
  --format json

# Disable specific features
conda run -n ai_env python prompt_generator.py \
  --character elio \
  --count 100 \
  --no-expressions \
  --no-camera \
  --output prompts/poses_only.txt
```

**Available Options:**
- `--vocab-dir`: Custom vocabulary directory (default: `prompts/generation/vocabulary`)
- `--character`: Single character token
- `--characters`: Comma-separated tokens for multi-character
- `--mode`: Generation mode (`single`/`interaction`/`mixed`)
- `--count`: Number of prompts to generate
- `--seed`: Random seed for reproducibility
- `--output`: Output file path
- `--format`: Output format (`txt`/`json`/`auto`)
- `--no-expressions`: Disable expression variations
- `--no-poses`: Disable pose variations
- `--no-actions`: Disable action variations
- `--no-camera`: Disable camera angle variations

### 2. Vocabulary System

Located in `prompts/generation/vocabulary/`:

#### expressions.yaml
- **18 total expressions** across 3 categories:
  - Basic emotions (6): happy, sad, angry, fearful, surprised, disgusted
  - Complex emotions (9): confused, excited, nervous, determined, embarrassed, proud, mischievous, thoughtful, curious
  - Neutral states (3): neutral, serious, sleepy
- Each expression includes synonyms and intensity modifiers
- Compatible with Pixar-style 3D characters

#### poses.yaml
- **19 total poses** across 4 categories:
  - Standing poses (4): neutral, confident, casual, hands on hips
  - Sitting poses (3): chair, cross-legged, relaxed
  - Action poses (8): walking, running, jumping, reaching, pointing, waving, arms crossed, hands clasped
  - Dynamic poses (4): dancing, falling, climbing, flying
- **11 camera angles**: front view, three-quarter view, side view, back view, close-up, medium shot, full body, etc.

#### actions.yaml
- **44 total actions** across multiple categories:
  - Movement: locomotion (6), rotation (2)
  - Hand actions: gestures (5), reaching (3), holding (2)
  - Interaction: communication (5), emotional (2), physical contact (3)
  - Object interaction: manipulation (4), tool use (3), creative (3)
  - Environmental: navigation (4), observation (2)
- Each action includes compatible poses and expressions

### 3. Prompt Quality

Generated prompts follow best practices:
- Character token always first
- Structured component ordering
- Pixar-style quality tags included
- 3D-specific terminology (smooth shading, clean render, etc.)
- Variety through synonym randomization
- Probability-based inclusion (expressions 70%, actions 50%, camera 50%)

**Example Generated Prompts:**
```
bryce, embarrassed expression, hopping, three-quarter view, clean render, detailed facial features
caleb, contemplative expression, carrying something, pixar style 3d animation, clean render
elio talking to bryce, pixar style 3d animation, detailed facial features
bryce and caleb standing together, clean render, professional lighting
```

## Pipeline Workflow

### Phase 1: Prompt Generation ✅ COMPLETE
```bash
# Generate prompts for all 14 characters
python prompt_generator.py --character bryce --count 500 --output prompts/bryce.txt
python prompt_generator.py --character caleb --count 500 --output prompts/caleb.txt
# ... (repeat for all 14 characters)
```

### Phase 2: Image Generation ✅ COMPLETE
```bash
# Generate synthetic images with automatic checkpointing
conda run -n ai_env python batch_image_generator.py \
  prompts/bryce_synthetic.json \
  --lora-path /mnt/c/ai_models/lora_sdxl/BEST_CHECKPOINTS_COLLECTION/BEST_bryce_lora_sdxl.safetensors \
  --character bryce \
  --base-model /mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors \
  --output-dir /tmp/synthetic_data/bryce \
  --checkpoint-dir /tmp/checkpoints/bryce \
  --steps 40 \
  --guidance 7.5 \
  --checkpoint-interval 50

# Features:
# - Checkpoint every 50 images (~3min intervals)
# - Deterministic resume (same seed → same image)
# - Real-time ETA tracking
# - GPU memory management
# - Validation image before batch
```

### Phase 3: Quality Filtering (PLANNED)
```bash
# Filter images based on CLIP score, aesthetic score, etc.
python quality_filter.py \
  --input-dir synthetic_data/ \
  --output-dir synthetic_data_filtered/ \
  --min-clip-score 0.25 \
  --min-aesthetic-score 5.0
```

### Phase 4: Dataset Organization (PLANNED)
```bash
# Organize filtered images into training datasets
python organize_dataset.py \
  --input-dir synthetic_data_filtered/ \
  --output-dir training_data_synthetic/ \
  --balance-by expression,pose
```

## System Status

✅ **Completed:**
- Prompt generator core implementation
- Vocabulary system (expressions, poses, actions)
- CLI interface with all features
- Single/interaction/mixed generation modes
- Reproducible generation with seeds
- JSON/TXT output formats

⏳ **In Progress:**
- Batch prompt generation scripts for all 14 characters

📋 **Planned:**
- Batch image generation pipeline
- Quality filtering system
- Dataset organization tools
- Training pipeline integration
- Evaluation metrics

## Character List

All 14 trained characters with best checkpoints:
1. elio
2. bryce
3. caleb
4. glordon
5. luca (human)
6. luca_seamonster
7. alberto (human)
8. alberto_seamonster
9. giulia
10. miguel
11. tyler
12. ian_lightfoot
13. barley_lightfoot
14. russell
15. orion

## Technical Details

**Dependencies:**
- Python 3.10+
- PyYAML
- Standard library (argparse, json, random, pathlib, dataclasses, enum)

**Performance:**
- ~0.5-1 second to generate 100 prompts
- Memory efficient (vocabularies loaded once)
- Deterministic with seed

**Extensibility:**
- Easy to add new vocabulary categories
- Template-based prompt construction
- Modular component selection
- Probability tuning per component type

## Future Enhancements

1. **Advanced Templates:**
   - Scene-based prompts
   - Emotion-action combos
   - Environmental context

2. **Quality Improvements:**
   - Grammar validation
   - Duplicate detection
   - Semantic coherence checking

3. **Integration:**
   - Direct integration with image generators
   - Auto-captioning validation
   - Training data pipeline hooks

4. **Analytics:**
   - Prompt diversity metrics
   - Component usage statistics
   - Generation quality reports

## Contributing

When adding new vocabulary:
1. Follow existing YAML structure
2. Include id, name, synonyms, description
3. Add to appropriate category
4. Update this README with counts
5. Test with prompt_generator.py

---

**Version:** 1.0.0
**Last Updated:** 2025-11-30
**Author:** AI Training Pipeline Team
