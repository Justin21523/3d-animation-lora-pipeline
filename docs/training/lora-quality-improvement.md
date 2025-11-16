# Improving LoRA Quality - Comprehensive Guide

## Problem Diagnosis

Based on Trial 3.5 evaluation, the following patterns were observed:

### ✅ What Works Well
- **Close-up portraits**: Excellent consistency (41.4% of training data)
- **Simple backgrounds**: Stable character appearance
- **Progressive improvement**: Later epochs consistently better

### ❌ What Needs Improvement
- **Far/wide shots**: Character appears smaller, age inconsistency
- **Full-body scenes**: Body proportions unstable
- **Multi-character scenes**: Character deforms when others present
- **Complex backgrounds**: Character appearance changes
- **Occlusion**: Body distortion when partially hidden

---

## Root Cause Analysis

### Dataset Imbalances (From `analyze_training_dataset.py`)

| Category | Current | Recommended | Impact |
|----------|---------|-------------|--------|
| **Close-up** | 41.4% ✅ | 30-40% | Over-optimized for faces |
| **Medium shot** | 0.0% ❌ | 25-30% | No mid-range reference |
| **Full-body** | 10.2% ❌ | 25-30% | Poor body consistency |
| **Far/wide** | 0.8% ❌ | 5-10% | Fails at distance |
| **Multi-character** | 0.0% ❌ | 15-20% | No context handling |
| **Occlusion** | 0.0% ❌ | 8-12% | Distorts when hidden |
| **Complex BG** | 0.0% ❌ | 20-30% | Background-dependent |

**Conclusion**: The model excels at what it's trained on (close-ups with simple backgrounds) but fails on underrepresented scenarios.

---

## Improvement Strategies

### Strategy A: Quick Fix - Data Augmentation (1-2 days)

**Use existing images with synthetic variations**

#### Step 1: Augment current dataset
```bash
python scripts/data_curation/augment_dataset_for_balance.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/training_ready/1_luca \
  /tmp/luca_dataset_analysis.json \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/training_balanced_v1 \
  --strategy balanced
```

**What this does:**
- Crops full-body shots → creates medium shots
- Adds context descriptions → simulates complex backgrounds
- Generates caption variations → improves prompt robustness
- **Target**: 450-500 images with better balance

#### Step 2: Verify balance
```bash
python scripts/analysis/analyze_training_dataset.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/training_balanced_v1 \
  --output-json /tmp/balanced_v1_analysis.json
```

#### Step 3: Retrain with balanced dataset
```bash
# Update training config to point to new dataset
# Train for 12-14 epochs (similar to trial 3.5)
```

**Expected improvements:**
- ✅ Better medium-shot consistency
- ✅ More stable body proportions
- ⚠️ Multi-character still weak (needs real data)
- ⚠️ Occlusion still problematic

**Effort**: Low | **Impact**: Medium

---

### Strategy B: Targeted Re-extraction (3-5 days)

**Extract missing scene types from source video**

#### Step 1: Identify gaps

Create a targeted extraction profile:

```json
{
  "priority_scenes": {
    "full_body": {
      "target": 120,
      "keywords": ["standing", "walking", "full body visible"],
      "min_character_height_ratio": 0.6
    },
    "medium_shot": {
      "target": 100,
      "keywords": ["waist up", "half body"],
      "min_character_height_ratio": 0.4
    },
    "multi_character": {
      "target": 80,
      "required_characters": ["luca", "alberto"],
      "required_characters_alt": ["luca", "giulia"]
    },
    "complex_backgrounds": {
      "target": 80,
      "scene_types": ["town", "plaza", "crowded street"]
    },
    "occlusion": {
      "target": 40,
      "keywords": ["behind", "partial", "foreground objects"]
    }
  }
}
```

#### Step 2: Re-run clustering with filters

```bash
# Extract instances with specific criteria
python scripts/generic/clustering/character_clustering.py \
  --input-dir .../frames \
  --output-dir .../clustered_targeted \
  --min-size 200 \          # Larger characters only
  --aspect-ratio 0.5-2.0 \  # Allow more variety
  --use-face-detection \
  --pose-estimation         # NEW: Track poses
```

#### Step 3: Manual curation with interactive tool

```bash
python scripts/generic/clustering/interactive_character_selector.py \
  --cluster-dir .../clustered_targeted \
  --output-dir .../curated_v2 \
  --show-metadata \         # Show pose, size, context info
  --filter-mode balanced    # Prioritize underrepresented types
```

#### Step 4: Combine with existing dataset

```bash
# Merge curated_v2 with training_ready/1_luca
# Target final composition:
#   - Close-up: 140 (35%)
#   - Medium: 100 (25%)
#   - Full-body: 100 (25%)
#   - Wide: 30 (7.5%)
#   - Multi-char: 60 (15%)
#   Total: ~400 images
```

**Expected improvements:**
- ✅ All shot types well-represented
- ✅ Multi-character handling improved
- ✅ Occlusion robustness
- ✅ Background variety

**Effort**: Medium | **Impact**: High

---

### Strategy C: Training Parameter Tuning (Immediate)

**Optimize training without changing data**

#### Current issues sugggest:
1. **Overfitting to close-ups** → Reduce learning rate or add regularization
2. **Poor generalization** → Increase network rank or use dropout

#### Recommended changes:

```toml
[model]
network_dim = 64        # Increase from 32 (more capacity)
network_alpha = 32      # Half of network_dim

[training]
learning_rate = 8e-5    # Reduce from 1e-4 (less overfitting)
lr_scheduler = "cosine_with_restarts"  # Better than constant
lr_warmup_steps = 100

# Regularization
min_snr_gamma = 5.0     # NEW: Stabilize training
noise_offset = 0.05     # NEW: Better dark/light handling

# Data augmentation (careful with 3D!)
color_aug = false       # Keep false for 3D
flip_aug = false        # Keep false for asymmetric features

[advanced]
gradient_checkpointing = true   # Allow larger batches
mixed_precision = "fp16"
gradient_accumulation_steps = 2  # Effective batch size = 2x
```

#### Run ablation study

Train 3 variants simultaneously:

1. **trial_3.6_higher_rank**: network_dim=64
2. **trial_3.6_lower_lr**: learning_rate=8e-5
3. **trial_3.6_regularized**: +min_snr_gamma, +noise_offset

Compare after 10 epochs.

**Expected improvements:**
- ✅ Better generalization to unseen scenarios
- ✅ Less overfitting to dominant patterns
- ⚠️ Won't fix data imbalance fundamentally

**Effort**: Low | **Impact**: Low-Medium

---

### Strategy D: Multi-Stage Training (Advanced)

**Train in phases with different data mixes**

#### Phase 1: Foundation (epochs 1-6)
- **Data**: Balanced dataset (all shot types equal)
- **Goal**: Learn character features across contexts
- **LR**: 1e-4

#### Phase 2: Refinement (epochs 7-12)
- **Data**: Weighted sampling (favor weak areas)
  - Full-body: 35%
  - Multi-character: 25%
  - Close-up: 20%
  - Medium: 15%
  - Wide: 5%
- **Goal**: Strengthen weak scenarios
- **LR**: 5e-5

#### Phase 3: Polish (epochs 13-16)
- **Data**: Original distribution (close-up heavy)
- **Goal**: Final quality on primary use-case
- **LR**: 2e-5

**Implementation**:
```bash
# Phase 1
train_network.py --config luca_phase1.toml --epochs 6

# Phase 2 (resume from phase 1)
train_network.py --config luca_phase2.toml --epochs 6 \
  --resume checkpoint_phase1_epoch6.safetensors

# Phase 3 (resume from phase 2)
train_network.py --config luca_phase3.toml --epochs 4 \
  --resume checkpoint_phase2_epoch12.safetensors
```

**Expected improvements:**
- ✅ Best of both worlds: strong everywhere
- ✅ Prioritizes important scenarios
- ⚠️ Complex to set up

**Effort**: High | **Impact**: Very High

---

## Recommended Action Plan

### Immediate (Today)
1. ✅ Run `analyze_training_dataset.py` (DONE)
2. Review trial 3.5 outputs by category:
   ```bash
   # View close-up outputs (should be good)
   ls trial_3.5/evaluation/luca_trial3.5-000014/img_*_p00*_v00.png

   # View full-body prompts (should show problems)
   ls trial_3.5/evaluation/luca_trial3.5-000014/img_*_p01*_v00.png
   ```

### Week 1: Quick Win
1. Run `augment_dataset_for_balance.py` with "balanced" strategy
2. Verify new balance with `analyze_training_dataset.py`
3. Train trial 3.6 with augmented dataset
4. Compare trial 3.5 vs 3.6 outputs

**Expected**: 30-40% improvement in full-body consistency

### Week 2: Comprehensive Fix
1. Re-extract targeted scenes from source video
2. Manual curation focusing on gaps
3. Build fully balanced dataset (~400 images)
4. Train trial 4.0 with multi-stage approach

**Expected**: 70-80% improvement across all scenarios

### Ongoing: Testing & Iteration
1. Create standardized test suite:
   ```bash
   prompts/luca/luca_test_suite.json
     - 10 close-up prompts
     - 10 medium prompts
     - 10 full-body prompts
     - 10 multi-character prompts
     - 10 complex background prompts
   ```

2. Automated comparison:
   ```bash
   python scripts/evaluation/compare_lora_models.py \
     --models trial_3.5 trial_3.6 trial_4.0 \
     --test-suite prompts/luca/luca_test_suite.json \
     --output trial_comparison.html
   ```

---

## Success Metrics

Track these metrics per checkpoint:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Close-up consistency** | >90% | CLIP similarity between variations |
| **Full-body consistency** | >75% | CLIP similarity between variations |
| **Multi-char robustness** | >70% | Character remains recognizable |
| **Background independence** | >70% | Character consistent across BGs |
| **Occlusion handling** | >65% | No distortion when partially hidden |

**Overall target**: >80% average across all metrics

---

## Troubleshooting Common Issues

### Issue: Character age changes in different scenarios
**Cause**: Training data has different maturity levels mixed
**Fix**: Filter by consistent age representation, use "teenage boy" in all captions

### Issue: Character proportions unstable
**Cause**: Insufficient full-body training examples
**Fix**: Increase full-body to 25-30% of dataset

### Issue: LoRA strength too high causes artifacts
**Cause**: Overfitting to training distribution
**Fix**: Reduce network_dim, increase regularization, use lower LoRA scale (0.7-0.8)

### Issue: Character disappears in complex scenes
**Cause**: No complex background training data
**Fix**: Add 20-30% complex background samples

---

## Tools Reference

### Analysis
- `scripts/analysis/analyze_training_dataset.py` - Diagnose imbalances
- `scripts/evaluation/test_lora_checkpoints.py` - Automated testing
- `scripts/evaluation/compare_lora_models.py` - Cross-model comparison

### Data Curation
- `scripts/data_curation/augment_dataset_for_balance.py` - Synthetic augmentation
- `scripts/generic/clustering/interactive_character_selector.py` - Manual review
- `scripts/generic/clustering/character_clustering.py` - Auto clustering

### Training
- `train_network.py` - Main training script
- `configs/training/*.toml` - Training configurations

---

## Conclusion

Your trial 3.5 shows **excellent close-up quality** but suffers from **severe data imbalance**:
- 0% multi-character scenes
- 0% occlusion examples
- 10% full-body (need 25-30%)
- 0% complex backgrounds

**Recommended priority**:
1. **This week**: Strategy A (augmentation) for quick 30% improvement
2. **Next week**: Strategy B (re-extraction) for comprehensive fix
3. **Ongoing**: Strategy D (multi-stage) for optimal results

The foundation is solid — you just need better data distribution!
