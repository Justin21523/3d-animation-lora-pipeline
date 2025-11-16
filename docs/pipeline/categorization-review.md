# Categorization Review UI - Universal Guide

**Purpose:** Review and correct CLIP auto-categorization results for SAM2 instances
**Scope:** Film-agnostic, works with any 3D animation project
**Location:** `scripts/generic/review/categorization_review_ui.py`

---

## Overview

After running `instance_categorizer.py` to auto-classify SAM2 instances using CLIP, this UI allows you to:
- **Review all categorization results** (55K+ instances typical)
- **Correct misclassifications** (CLIP is not perfect)
- **Discard low-quality instances**
- **Re-classify to correct categories**
- **Batch process** thousands of instances efficiently

---

## Universal Usage Pattern

### Basic Command Structure

```bash
python scripts/generic/review/categorization_review_ui.py \
    --categorized-dir /path/to/<PROJECT>/instances_categorized \
    --output-dir /path/to/<PROJECT>/instances_reviewed \
    --project <project_name> \
    --port 5555
```

### Parameters (All Film-Agnostic)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--categorized-dir` | Directory with CLIP categorization results | `/mnt/data/.../luca/instances_categorized` |
| `--output-dir` | Directory to save reviewed results | `/mnt/data/.../luca/instances_reviewed` |
| `--project` | Project name (for display only) | `luca`, `toy_story`, `soul` |
| `--port` | Web server port | `5555` (default) |

---

## Example Usage Across Different Films

### Luca (Pixar, 2021)

```bash
python scripts/generic/review/categorization_review_ui.py \
    --categorized-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_categorized \
    --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_reviewed \
    --project luca \
    --port 5555
```

### Toy Story (Pixar, 1995)

```bash
python scripts/generic/review/categorization_review_ui.py \
    --categorized-dir /mnt/data/ai_data/datasets/3d-anime/toy_story/instances_categorized \
    --output-dir /mnt/data/ai_data/datasets/3d-anime/toy_story/instances_reviewed \
    --project toy_story \
    --port 5555
```

### Soul (Pixar, 2020)

```bash
python scripts/generic/review/categorization_review_ui.py \
    --categorized-dir /mnt/data/ai_data/datasets/3d-anime/soul/instances_categorized \
    --output-dir /mnt/data/ai_data/datasets/3d-anime/soul/instances_reviewed \
    --project soul \
    --port 5555
```

### Custom Project

```bash
python scripts/generic/review/categorization_review_ui.py \
    --categorized-dir /path/to/your_project/instances_categorized \
    --output-dir /path/to/your_project/instances_reviewed \
    --project your_project_name \
    --port 5555
```

---

## Input Requirements (Universal)

The tool expects a standard categorization directory structure created by `instance_categorizer.py`:

```
instances_categorized/
├── character/              # CLIP-classified character instances
├── human person/           # Human-like instances
├── object/                 # Generic objects
├── furniture/              # Furniture items
├── background/             # Background elements
├── vehicle/                # Vehicles
├── prop/                   # Props
├── accessory/              # Accessories
├── uncertain/              # Low-confidence predictions
└── categorization_results.json  # Metadata (required)
```

**Note:** Categories are fixed and universal across all films. CLIP uses these standard categories regardless of film content.

---

## Output Structure (Universal)

After review and export, results are organized in a standardized structure:

```
instances_reviewed/
├── character/              # Re-classified as character
├── human person/           # Re-classified as human
├── object/                 # Re-classified as object
├── furniture/              # Re-classified as furniture
├── background/             # Re-classified as background
├── vehicle/                # Re-classified as vehicle
├── prop/                   # Re-classified as prop
├── accessory/              # Re-classified as accessory
├── uncertain/              # Re-classified as uncertain
├── kept/                   # Confirmed correct (kept as-is)
├── discarded/              # Rejected instances (errors/low quality)
├── review_state.json       # Review progress (auto-saved)
└── review_report.json      # Final statistics and decisions
```

---

## UI Features (Film-Independent)

### 1. Multi-Level Filtering

**By Category:**
- Filter by any of the 9 standard categories
- View category-specific instances
- See instance counts per category

**By Status:**
- `All` - Show all instances
- `Pending` - Show unreviewed instances
- `Reviewed` - Show already-reviewed instances

**Combined Filtering:**
- Example: "Show only pending instances in the uncertain category"
- Example: "Show all reviewed character instances"

### 2. Three Review Actions

| Action | Keyboard | Description |
|--------|----------|-------------|
| **Keep** | `K` | Confirm categorization is correct |
| **Discard** | `D` | Reject instance (error/low quality) |
| **Re-classify** | `C` | Move to different category |

### 3. Batch Operations

- **Shift+Click** to select multiple instances
- **Select All** - Select all visible instances (respects filters)
- **Deselect All** - Clear selection
- **Batch Keep** - Keep all selected instances
- **Batch Discard** - Discard all selected instances
- **Batch Re-classify** - Move all selected to chosen category

**Use Case:** Quickly discard 1000+ background instances that are clearly wrong.

### 4. Keyboard Navigation

| Key | Action |
|-----|--------|
| `K` | Keep current instance |
| `D` | Discard current instance |
| `C` | Toggle re-classify menu |
| `→` | Next image |
| `←` | Previous image |
| `Esc` | Close modal |
| `Shift+Click` | Multi-select |

---

## Workflow (Universal Steps)

### Step 1: Launch UI

```bash
# Replace paths with your project
bash /tmp/launch_categorization_review.sh
# OR
python scripts/generic/review/categorization_review_ui.py --categorized-dir ... --output-dir ...
```

### Step 2: Open Browser

Navigate to: **http://localhost:5555**

### Step 3: Review Strategy

**Recommended approach:**

1. **Start with "uncertain" category** (70% of instances typically)
   - These are low-confidence CLIP predictions
   - Quickly discard obvious errors
   - Re-classify salvageable instances

2. **Review "character" category** (25% typically)
   - This is your target category for LoRA training
   - Ensure quality and correct classification
   - Discard partial views, occlusions, blurry instances

3. **Spot-check other categories** (5% typically)
   - Verify furniture, background, vehicle classifications
   - Re-classify any characters that were missed

### Step 4: Batch Process

For large volumes:
1. Filter by category
2. Select All (or Shift+Click multiple)
3. Batch Discard or Batch Re-classify
4. Repeat for each category

### Step 5: Export Results

Click **"💾 Export Results"** button to:
- Copy files to output directory
- Generate `review_report.json`
- Save final statistics

---

## Resume Interrupted Sessions (Universal)

The UI automatically saves progress to `review_state.json`:

- **Interrupt at any time:** Press `Ctrl+C`
- **Resume later:** Re-run the same command
- **Progress restored:** All Keep/Discard/Re-classify decisions are loaded

**Note:** Progress is project-specific (stored in `--output-dir`), so you can work on multiple films in parallel.

---

## Integration with Pipeline (Universal)

### Before This Tool

```
SAM2 Segmentation → Instance Categorizer (CLIP)
                          ↓
                   Auto-categorization
                   (55K+ instances, ~30% accuracy)
```

### After This Tool

```
Manual Review UI → Corrected Categorization
                        ↓
                   Clean dataset
                   (10K+ character instances, 95%+ accuracy)
```

### Next Steps

```
Reviewed Instances → Context-Aware Inpainting → Face Identity Clustering → LoRA Training
```

---

## Performance Tips (Universal)

### Efficient Review

1. **Use filters aggressively**
   - Don't review all 55K at once
   - Filter by category first

2. **Leverage batch operations**
   - Select 100+ instances at once
   - One-click batch discard

3. **Keyboard shortcuts**
   - Navigate with arrow keys
   - Use K/D/C for quick decisions
   - Avoid mouse clicks when possible

4. **Multiple sessions**
   - Review 5K-10K instances per session
   - Take breaks to avoid fatigue
   - Progress is auto-saved

### Expected Time

- **55K instances:** ~3-5 hours total
- **10K instances/hour** with efficient workflow
- **Faster with batch operations** on obvious errors

---

## Quality Guidelines (Universal)

### Keep Criteria (Character Category)

✅ **Keep if:**
- Full or majority of character visible
- Clear face (not blurry)
- Good lighting
- Minimal occlusion
- Clean segmentation edges

❌ **Discard if:**
- Partial character (<30% visible)
- Blurry face or motion blur
- Heavy occlusion
- Poor segmentation (artifacts)
- Extreme angles (top/bottom view)

### Re-classify Criteria

🔄 **Re-classify if:**
- CLIP chose wrong category (e.g., character → background)
- Instance fits better in different category
- Uncertain → specific category after inspection

---

## Troubleshooting (Universal)

### Issue: UI won't start

**Solution:**
```bash
# Check if port is in use
lsof -i :5555
# Kill existing process
kill -9 <PID>
# Restart UI
```

### Issue: Images not loading

**Check:**
- `categorization_results.json` exists in `--categorized-dir`
- Image files exist in category subfolders
- File permissions are correct

### Issue: Progress not saved

**Check:**
- `review_state.json` in `--output-dir`
- Write permissions on output directory
- Disk space available

### Issue: Export fails

**Check:**
- Sufficient disk space for copied files
- Output directory write permissions
- No filename conflicts

---

## Advanced Usage

### Custom Port

```bash
# If port 5555 is in use
python scripts/generic/review/categorization_review_ui.py \
    --categorized-dir ... \
    --output-dir ... \
    --port 8080  # Use different port
```

### Multiple Projects in Parallel

```bash
# Terminal 1: Review Luca on port 5555
python ... --project luca --port 5555

# Terminal 2: Review Toy Story on port 5556
python ... --project toy_story --port 5556
```

---

## API Endpoints (For Automation)

The UI exposes REST APIs for programmatic access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/instances` | GET | Get all instances with stats |
| `/api/image/<category>/<filename>` | GET | Serve instance image |
| `/api/review` | POST | Review single instance |
| `/api/batch` | POST | Batch review multiple instances |
| `/api/export` | POST | Export final results |

**Example:** Automated batch processing script:

```python
import requests

# Mark all instances in "furniture" as discarded
response = requests.post('http://localhost:5555/api/batch', json={
    'filenames': furniture_filenames,
    'action': 'discard'
})
```

---

## File Formats (Universal)

### review_state.json

```json
{
  "reviews": {
    "frame_00001_instance_0001.png": {
      "action": "keep",
      "original_category": "character",
      "new_category": "character",
      "timestamp": "2025-11-10T12:34:56"
    }
  },
  "kept": ["frame_00001_instance_0001.png"],
  "discarded": ["frame_00002_instance_0005.png"],
  "reclassified": ["frame_00003_instance_0010.png"],
  "last_updated": "2025-11-10T12:34:56"
}
```

### review_report.json

```json
{
  "project": "luca",
  "total_instances": 55249,
  "reviewed_instances": 55249,
  "export_stats": {
    "kept": 12000,
    "discarded": 40000,
    "reclassified": 3249,
    "unchanged": 0
  },
  "timestamp": "2025-11-10T14:00:00"
}
```

---

## Summary

**Key Points:**
1. ✅ **100% Film-Agnostic** - Works with any 3D animation project
2. ✅ **No Code Changes** - Just change paths and project name
3. ✅ **Standardized Output** - Same structure for all films
4. ✅ **Resume-Friendly** - Auto-saves progress
5. ✅ **Efficient Workflow** - Keyboard shortcuts + batch operations

**Universal Command:**
```bash
python scripts/generic/review/categorization_review_ui.py \
    --categorized-dir /path/to/<ANY_FILM>/instances_categorized \
    --output-dir /path/to/<ANY_FILM>/instances_reviewed \
    --project <any_name>
```

**Next Tool in Pipeline:**
```bash
# After review completes
python scripts/generic/enhancement/inpaint_context_aware.py \
    --instances-dir /path/to/<ANY_FILM>/instances_reviewed/kept \
    --frames-dir /path/to/<ANY_FILM>/frames \
    --output-dir /path/to/<ANY_FILM>/instances_inpainted
```

---

**Version:** 1.0.0
**Last Updated:** 2025-11-10
**Compatibility:** All 3D animation projects
