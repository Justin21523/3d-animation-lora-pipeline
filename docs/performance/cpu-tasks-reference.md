# CPU Tasks Quick Reference - Ready to Execute

**Last Updated:** 2025-11-12 22:45
**Purpose:** Pure CPU tasks that can run in parallel with GPU optimization

---

## Currently Running CPU Tasks

### 1. ✅ Phase 1a: Scene Deduplication - **COMPLETE**
- **Status:** Complete
- **Result:** 4323 → 3074 frames (28.8% dedup)
- **Output:** `/mnt/data/ai_data/datasets/3d-anime/luca/frames_deduplicated/`

### 2. 🔄 Phase 1b: Quality Filtering - **RUNNING**
- **Status:** Running (PID: 105758)
- **Started:** 2025-11-12 22:45
- **Estimated Time:** 15-20 minutes
- **Log:** `/tmp/phase1b_quality.log`
- **Output:** `/mnt/data/ai_data/datasets/3d-anime/luca/frames_quality_filtered/`
- **Expected Result:** 3074 → ~2500 quality frames

---

## Ready to Start (Pure CPU Tasks)

### 3. ⏸️ Pose Data Preparation
**Input:** 542 Luca character instances
**Process:** RTM-Pose keypoint detection + clustering
**Time:** 40-55 minutes (CPU)
**Priority:** HIGH - Enables Pose LoRA dataset

**To Start:**
```bash
# Create and run pose preparation script
nohup bash /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/workflows/pose_preparation.sh > /tmp/pose_prep.log 2>&1 &
```

**Expected Output:**
- Keypoints JSON files
- Pose clusters (standing/walking/running/sitting)
- 300-500 curated pose examples
- Location: `/mnt/data/ai_data/datasets/3d-anime/luca/pose_data/`

---

### 4. ⏸️ Expression Data Preparation
**Input:** 542 Luca character instances
**Process:** RetinaFace detection + face cropping + quality filtering
**Time:** 20-30 minutes (CPU)
**Priority:** HIGH - Enables Expression LoRA dataset

**To Start:**
```bash
# Create and run expression preparation script
nohup bash /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/workflows/expression_preparation.sh > /tmp/expr_prep.log 2>&1 &
```

**Expected Output:**
- Face-cropped images (200-400)
- Emotion clustering (happy/sad/surprised/angry/neutral)
- Quality-filtered close-ups
- Location: `/mnt/data/ai_data/datasets/3d-anime/luca/expression_data/`

---

### 5. ⏸️ Lighting Classification
**Input:** 3074 deduplicated frames (or ~2500 quality-filtered)
**Process:** Histogram-based lighting condition classification
**Time:** 30 minutes (CPU)
**Priority:** MEDIUM - Enables Lighting LoRA dataset

**To Start:**
```bash
# Create and run lighting classification script
nohup bash /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/workflows/lighting_classification.sh > /tmp/lighting.log 2>&1 &
```

**Expected Output:**
- Frames categorized by:
  - Time of day (morning/noon/afternoon/evening/night)
  - Weather (sunny/cloudy/overcast)
  - Indoor vs outdoor
- 400-600 diverse lighting examples
- Location: `/mnt/data/ai_data/datasets/3d-anime/luca/lighting_data/`

---

## Scripts Status

| Script File | Status | Notes |
|-------------|--------|-------|
| `phase1a_scene_deduplication.sh` | ✅ Complete | Standalone dedup version created |
| `phase1b_quality_filtering.sh` | ✅ Created & Running | Inline Python code |
| `pose_preparation.sh` | ❌ Need to create | RTM-Pose CPU mode |
| `expression_preparation.sh` | ❌ Need to create | RetinaFace + crop |
| `lighting_classification.sh` | ❌ Need to create | Histogram analysis |

---

## Resource Usage Estimates

### Current System Load (During Optimization)
- **GPU:** 95-100% (hyperparameter optimization)
- **CPU:** 10-20% (idle capacity available)
- **Memory:** ~30GB used / 64GB total
- **Disk I/O:** Low

### If All CPU Tasks Run in Parallel
- **CPU:** 50-70% (safe, won't bottleneck GPU)
- **Memory:** +5-8GB (still plenty of headroom)
- **Disk I/O:** Moderate read/write (acceptable)
- **Priority:** All run with `nice -n 19` (low priority)

**Verdict:** ✅ Safe to run all 4 CPU tasks in parallel

---

## Execution Timeline

### Immediate (Now)
- ✅ Phase 1a Complete (2025-11-12 22:40)
- 🔄 Phase 1b Running (Est. complete: 23:00-23:05)

### Next 1 Hour (Can start immediately)
- Launch Pose Preparation (~40-55 min)
- Launch Expression Preparation (~20-30 min)
- Launch Lighting Classification (~30 min)
- **All complete by:** ~23:45-00:00

### Result by Midnight
- **4 CPU-only phases complete**
- **Ready for GPU-intensive phases** when optimization finishes
- **~2-3 hours of processing time saved** by parallel execution

---

## Monitoring Commands

### Check All Running CPU Tasks
```bash
ps aux | grep -E "phase1b|pose_prep|expr_prep|lighting" | grep -v grep
```

### View Logs in Real-Time
```bash
# Phase 1b Quality Filtering
tail -f /tmp/phase1b_quality.log

# Pose Preparation (when started)
tail -f /tmp/pose_prep.log

# Expression Preparation (when started)
tail -f /tmp/expr_prep.log

# Lighting Classification (when started)
tail -f /tmp/lighting.log
```

### Quick Status Check
```bash
echo "=== CPU Task Status ===" && \
echo "Phase 1a: $(ls /mnt/data/ai_data/datasets/3d-anime/luca/frames_deduplicated/*.jpg 2>/dev/null | wc -l) frames" && \
echo "Phase 1b: $(ls /mnt/data/ai_data/datasets/3d-anime/luca/frames_quality_filtered/*.jpg 2>/dev/null | wc -l) frames" && \
echo "Pose Data: $(ls /mnt/data/ai_data/datasets/3d-anime/luca/pose_data/ 2>/dev/null | wc -l) files" && \
echo "Expression Data: $(ls /mnt/data/ai_data/datasets/3d-anime/luca/expression_data/ 2>/dev/null | wc -l) files" && \
echo "Lighting Data: $(ls /mnt/data/ai_data/datasets/3d-anime/luca/lighting_data/ 2>/dev/null | wc -l) files"
```

---

## Data Flow Summary

```
Raw Video (Luca Film)
↓
[DONE] Scene Sampling → 4323 frames
↓
[DONE] Phase 1a Dedup → 3074 unique scenes
↓
[RUNNING] Phase 1b Quality → ~2500 quality frames
↓
├─→ [READY] Background Processing (needs GPU later)
├─→ [READY] Style LoRA (reuse quality frames)
└─→ [READY] Lighting Classification (CPU, can start now)

Character Instances (542 images)
↓
├─→ [READY] Pose Preparation (CPU, can start now)
├─→ [READY] Expression Preparation (CPU, can start now)
└─→ [DONE] Character LoRA (already in training)
```

---

## Next Actions (Prioritized)

1. ✅ **Phase 1b running** - Wait for completion (~15 min)
2. 🔄 **Create remaining CPU scripts** - Pose, Expression, Lighting
3. 🔄 **Launch all CPU tasks** - Parallel execution
4. ⏳ **Monitor progress** - Check logs every 30 min
5. ⏳ **Await GPU availability** - For segmentation/captioning phases

---

## Important Notes

- **All CPU tasks use `nice -n 19`** - Won't interfere with GPU optimization
- **Progress tracked in** `docs/workflows/MULTI_LORA_PREPARATION_MASTER_PLAN.md`
- **Logs saved to** `/tmp/` for debugging
- **Final data locations** under `/mnt/data/ai_data/datasets/3d-anime/luca/`
- **Review checkpoints** after Expression clustering (manual labeling)

---

## Total Time Savings

**By running CPU tasks in parallel during GPU optimization:**
- Pose: 40-55 min
- Expression: 20-30 min
- Lighting: 30 min
- Quality Filtering: 15-20 min

**Net Savings: ~2-3 hours of wallclock time**

**Without parallel processing:** Would need to wait for GPU → ~13 hours total
**With parallel processing:** GPU tasks only → ~10 hours total

---

## References

- **Master Plan:** `docs/workflows/MULTI_LORA_PREPARATION_MASTER_PLAN.md`
- **SDXL Migration:** `docs/guides/SD15_TO_SDXL_MIGRATION.md`
- **Multi-LoRA System:** `docs/guides/MULTI_TYPE_LORA_SYSTEM.md`
- **LoRA Composition:** `docs/guides/LORA_COMPOSITION_QUICKSTART.md`
