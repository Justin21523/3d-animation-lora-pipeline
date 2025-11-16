# Phase 3.3: Script Generalization Plan

**Date**: 2025-11-15
**Status**: In Progress
**Goal**: Make Luca-specific scripts configuration-driven and project-agnostic

---

## Overview

Convert hardcoded Luca-specific values to configuration-driven parameters, enabling the pipeline to work with any character/project by simply changing configuration files.

## Identified Hardcoded Values

### 1. Python Scripts

#### `luca_dataset_pipeline_simplified.py`
**Hardcoded items:**
- Line 73-76: Output directory names with "luca_" prefix
  ```python
  'face_matched': self.base_dir / 'luca_face_matched',
  'quality_filtered': self.base_dir / 'luca_quality_filtered',
  'augmented': self.base_dir / 'luca_augmented_comprehensive',
  'curated_400': self.base_dir / 'luca_curated_400',
  ```
- Line 53: Logger name "LucaPipeline"
- Line 570: Default config path `configs/projects/luca_dataset_prep_v2.yaml`

**Proposed changes:**
- Add `--project` parameter (default: project name from config)
- Read project name from config: `config.project.name`
- Generate directory names dynamically: `f"{project_name}_face_matched"`
- Logger name from config: `f"{project_name}Pipeline"`

#### `luca_dataset_preparation_pipeline.py`
**To be analyzed** - likely similar hardcoding patterns

### 2. Shell Scripts

#### `run_caption_generation.sh`, `run_instance_enhancement.sh`, etc.
**Typical hardcoding:**
- Project-specific paths: `/mnt/data/ai_data/datasets/3d-anime/luca/`
- Character profile paths: `configs/characters/luca.yaml`
- Output directory names

**Proposed changes:**
- Add PROJECT variable at top of script
- Read from environment or config file
- Use variable substitution for all paths

### 3. Workflow Scripts

#### `optimized_luca_pipeline.sh`, `run_complete_luca_pipeline.sh`
**Typical hardcoding:**
- Complete pipeline paths hardcoded to Luca
- TOML config file paths
- Model output names

**Proposed changes:**
- Accept `--project` flag
- Source project config from `configs/projects/${PROJECT}.yaml`
- Generate paths dynamically

---

## Implementation Strategy

### Approach: Backwards-Compatible Generalization

To maintain existing Luca workflows while enabling new projects:

#### Phase 3.3.1: Configuration Structure ✅ (Already exists)
- ✅ `configs/projects/luca.yaml` exists
- ✅ `configs/characters/luca.yaml` exists
- ✅ `configs/projects/template.yaml` exists

**Config structure:**
```yaml
project:
  name: "luca"           # Used for directory naming
  full_name: "Luca Character Dataset"

paths:
  base_dir: "/mnt/data/ai_data/datasets/3d-anime/luca"
  # All other paths derived from base_dir

characters:
  main: ["luca", "alberto", "giulia"]
```

#### Phase 3.3.2: Python Script Updates

**Step 1: Add project parameter to argparse**
```python
parser.add_argument(
    '--project-config',
    type=str,
    default='configs/projects/luca.yaml',
    help='Path to project configuration file'
)
```

**Step 2: Load project config early**
```python
from omegaconf import OmegaConf

def __init__(self, config_path: str):
    # Load project config
    self.project_config = OmegaConf.load(config_path)
    self.project_name = self.project_config.project.name

    # Load pipeline config (if separate)
    with open(pipeline_config, 'r') as f:
        self.config = yaml.safe_load(f)
```

**Step 3: Replace hardcoded values**
```python
# Old:
'face_matched': self.base_dir / 'luca_face_matched',

# New:
'face_matched': self.base_dir / f'{self.project_name}_face_matched',
```

**Step 4: Update logger**
```python
# Old:
return logging.getLogger("LucaPipeline")

# New:
return logging.getLogger(f"{self.project_name}Pipeline")
```

#### Phase 3.3.3: Shell Script Updates

**Step 1: Add project variable**
```bash
#!/bin/bash
# Project configuration
PROJECT="${1:-luca}"  # Default to luca if no argument
PROJECT_CONFIG="configs/projects/${PROJECT}.yaml"

# Verify project config exists
if [ ! -f "$PROJECT_CONFIG" ]; then
    echo "Error: Project config not found: $PROJECT_CONFIG"
    exit 1
fi
```

**Step 2: Load config values**
```bash
# Read project name from YAML (using yq or python)
PROJECT_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$PROJECT_CONFIG'))['project']['name'])")
BASE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$PROJECT_CONFIG'))['paths']['base_dir'])")
```

**Step 3: Use variables**
```bash
# Old:
INPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/instances"

# New:
INPUT_DIR="${BASE_DIR}/instances"
```

#### Phase 3.3.4: Workflow Script Updates

Similar pattern to shell scripts, but also:
- Update TOML config references
- Update model output paths
- Update character profile references

---

## Testing Strategy

### Test Cases

1. **Existing Luca workflow** - Must work unchanged
   ```bash
   python scripts/projects/luca/pipelines/luca_dataset_pipeline_simplified.py
   ```

2. **Luca with explicit config**
   ```bash
   python scripts/projects/luca/pipelines/luca_dataset_pipeline_simplified.py \
       --project-config configs/projects/luca.yaml
   ```

3. **New project (after creating alberto.yaml)**
   ```bash
   python scripts/projects/luca/pipelines/luca_dataset_pipeline_simplified.py \
       --project-config configs/projects/alberto.yaml
   ```

### Validation Checklist

- [ ] All hardcoded "luca" strings replaced with config lookups
- [ ] Default behavior unchanged (backwards compatible)
- [ ] New projects work with different config files
- [ ] Directory names generated correctly
- [ ] Logging reflects correct project name
- [ ] All paths resolve correctly
- [ ] Character profiles loaded from config
- [ ] No broken imports or references

---

## Implementation Order

### Priority 1: Python Pipelines (Most Critical)
1. ✅ Create this plan document
2. ⏳ `luca_dataset_pipeline_simplified.py` - Primary pipeline
3. ⏳ `luca_dataset_preparation_pipeline.py` - Secondary pipeline

### Priority 2: Individual Stage Scripts
4. ⏳ `run_caption_generation.sh`
5. ⏳ `run_instance_enhancement.sh`
6. ⏳ `run_pose_analysis.sh`
7. ⏳ `run_quality_filter.sh`

### Priority 3: Workflow Scripts
8. ⏳ `optimized_luca_pipeline.sh`
9. ⏳ `run_complete_luca_pipeline.sh`
10. ⏳ `run_luca_dataset_pipeline.sh`

### Priority 4: Training Script
11. ⏳ `auto_train_luca.sh`

---

## Breaking Changes

### Minimal Breaking Changes Expected

**Current behavior preserved:**
- Scripts run with default Luca config if no `--project-config` provided
- All Luca-specific paths remain functional
- Existing shell scripts work as-is (until updated)

**New capabilities:**
- Can specify different project configs
- Easy to add new characters/projects
- No code changes needed for new projects

### Migration Path for Users

**For existing Luca users:** No changes needed! Scripts work as before.

**For new projects:**
1. Copy `configs/projects/template.yaml` to `configs/projects/alberto.yaml`
2. Update paths and character info
3. Run scripts with `--project-config configs/projects/alberto.yaml`

---

## Documentation Updates Needed

After implementation:
- [ ] Update `scripts/projects/luca/README.md`
- [ ] Create `docs/reference/multi-project-guide.md`
- [ ] Update REORGANIZATION_CHANGELOG.md
- [ ] Add examples for other characters
- [ ] Update CLAUDE.md usage examples

---

## Success Criteria

Phase 3.3 is complete when:
1. ✅ Generalization plan documented
2. All 11 scripts updated with config-driven approach
3. Backward compatibility maintained (Luca workflows unchanged)
4. At least one new character (e.g., Alberto) successfully processed
5. Documentation updated with multi-project examples
6. No hardcoded "luca" strings in generic code sections
