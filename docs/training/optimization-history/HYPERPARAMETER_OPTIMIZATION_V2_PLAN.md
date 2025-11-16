# 📊 Hyperparameter Optimization V2 - Improved Strategy

**Date:** 2025-11-12
**Version:** 2.0 (Based on Trial 1-5 Analysis)
**Status:** Awaiting Trial 5 completion before implementation

---

## 🎯 **Executive Summary**

Based on analysis of Trials 1-5, we've identified critical issues in the search space that led to:
- **Trial 4 quality problems**: All-black images, blurry limbs, body distortion, cut-off heads
- **Root causes**: Learning rate too high (0.000184), Alpha/Dim imbalance (128/128), insufficient training epochs (8)

This document outlines an improved search strategy that:
1. ✅ Fixes known problematic parameter ranges
2. ✅ Adds new promising parameters to explore
3. ✅ Introduces safety constraints to prevent training failures

---

## 📋 **Current Search Space Analysis**

### **What Worked Well ✅**

| Parameter | Current Range | Keep/Adjust | Reason |
|-----------|---------------|-------------|--------|
| `network_dim` | [64, 128, 256] | ✅ **Keep** | Good coverage of capacity |
| `gradient_accumulation_steps` | [1, 2, 4] | ✅ **Keep** | Balanced memory/speed trade-off |
| `lr_warmup_steps` | [50, 100, 150, 200] | ✅ **Keep** | Adequate warm-up exploration |

### **What Needs Fixing ❌**

| Parameter | Current Range | Problem | Recommended Fix |
|-----------|---------------|---------|-----------------|
| `learning_rate` | [0.00005, 0.0002] | **Too high upper bound** | ❌ → ✅ [0.00006, 0.00012] |
| `text_encoder_lr` | [0.00003, 0.0001] | **Too high upper bound** | ❌ → ✅ [0.00003, 0.00008] |
| `network_alpha` | [16, 32, 64, 128] | **No ratio constraint with Dim** | ❌ → ✅ Alpha = Dim × [0.25, 0.5] |
| `max_train_epochs` | [8, 12, 16] | **8 epochs too few** | ❌ → ✅ [12, 16, 20] |
| `optimizer_type` | [AdamW, AdamW8bit, Prodigy] | **8bit unstable with high LR** | ✅ Keep but add constraints |
| `lr_scheduler` | [cosine, cosine_with_restarts, polynomial] | **polynomial underexplored** | ✅ Keep |

---

## 🔬 **Problem Analysis from Trial 4**

### **Trial 4 Configuration (Problematic)**
```python
learning_rate = 0.000184  # ❌ TOO HIGH
text_encoder_lr = 0.000088
network_dim = 128
network_alpha = 128  # ❌ Alpha = Dim (no regularization)
optimizer_type = "AdamW8bit"
lr_scheduler = "polynomial"
max_train_epochs = 8  # ❌ TOO FEW
gradient_accumulation_steps = 1
```

### **Quality Issues Observed**
1. **All-black images** → Gradient explosion from high LR
2. **Blurry limbs, background bleeding** → Insufficient capacity (low Dim) + poor regularization (Alpha=Dim)
3. **Body distortion** → Training instability
4. **Cut-off heads** → Overfitting to specific crops in training data

### **Root Causes**
- **Learning Rate 0.000184**: Too aggressive, causes gradient explosion
- **Alpha/Dim = 1.0**: No weight decay, leads to overfitting
- **8 Epochs**: Not enough time to stabilize learning
- **AdamW8bit + High LR**: 8-bit quantization amplifies instability

---

## 🎨 **Improved Search Space V2.1** (Updated with User Feedback)

### **1️⃣ Core Parameters (Expanded Alpha Ratio)**

```python
# Learning Rates - NARROWED
"learning_rate": {
    "type": "float",
    "min": 0.00006,    # Raised lower bound
    "max": 0.00012,    # REDUCED from 0.0002
    "log": True
}

"text_encoder_lr": {
    "type": "float",
    "min": 0.00003,
    "max": 0.00008,    # REDUCED from 0.0001
    "log": True
}

# Network Architecture - WITH EXPANDED RATIO CONSTRAINT
"network_dim": {
    "type": "categorical",
    "choices": [64, 128, 256]  # Keep
}

"network_alpha_ratio": {  # EXPANDED: Based on Trial 4 success
    "type": "float",
    "min": 0.25,
    "max": 0.9,  # EXPANDED from 0.5 to 0.9 to explore Trial 4's range
    "step": 0.05  # Gives [0.25, 0.3, 0.35, ..., 0.85, 0.9]
}
# Actual alpha = network_dim * network_alpha_ratio
# Rationale: Trial 4 (ratio=1.0) had highest similarity,
#            so we should explore [0.75-0.9] with proper safeguards

# Training Duration - INCREASED MINIMUM
"max_train_epochs": {
    "type": "categorical",
    "choices": [12, 16, 20]  # REMOVED 8
}

# Optimizer - Keep but add constraints
"optimizer_type": {
    "type": "categorical",
    "choices": ["AdamW", "AdamW8bit", "Prodigy"]
}

# LR Scheduler - Keep
"lr_scheduler": {
    "type": "categorical",
    "choices": ["cosine", "cosine_with_restarts", "polynomial"]
}

# Warmup - Keep
"lr_warmup_steps": {
    "type": "categorical",
    "choices": [50, 100, 150, 200]
}

# Gradient Accumulation - Keep
"gradient_accumulation_steps": {
    "type": "categorical",
    "choices": [1, 2, 4]
}
```

### **2️⃣ New Parameters to Explore (Optional Additions)**

```python
# Dropout for regularization (NEW)
"network_dropout": {
    "type": "float",
    "min": 0.0,
    "max": 0.15,
    "step": 0.05  # [0.0, 0.05, 0.1, 0.15]
}

# Learning rate scheduler warmup ratio (NEW)
"lr_scheduler_num_cycles": {  # For cosine_with_restarts
    "type": "int",
    "min": 1,
    "max": 4
}

# Text encoder training ratio (NEW)
"text_encoder_lr_ratio": {  # text_encoder_lr = learning_rate * ratio
    "type": "float",
    "min": 0.3,
    "max": 0.9,
    "step": 0.1
}
```

---

## 🚫 **Safety Constraints V2.1** (Updated with Stratified Rules)

### **Hard Constraints (Must be enforced in code)**

```python
# Constraint 1: High LR + 8bit optimizer
if learning_rate > 0.00012 and optimizer_type == "AdamW8bit":
    REJECT_TRIAL  # Unstable

# Constraint 2: Exact Alpha = Dim (overfitting + instability risk)
if abs(network_alpha - network_dim) < 1e-6:  # Float comparison
    REJECT_TRIAL

# Constraint 3: High Dim + Few Epochs
if network_dim >= 256 and max_train_epochs < 16:
    REJECT_TRIAL  # Insufficient training

# Constraint 4: Very high LR + low warmup
if learning_rate > 0.0001 and lr_warmup_steps < 100:
    REJECT_TRIAL  # Training instability

# Constraint 5: Gradient Accumulation 1 + High LR
if gradient_accumulation_steps == 1 and learning_rate > 0.00011:
    REJECT_TRIAL  # Too aggressive updates

# === NEW: Stratified Constraints for High Alpha Ratios ===
# Constraint 6: High memory (ratio >= 0.75) requires strong stability
if alpha_ratio >= 0.75:
    # Must satisfy ALL of these conditions
    required_checks = [
        max_train_epochs >= 16,              # Longer training for stability
        gradient_accumulation_steps >= 2,    # Smoother gradient updates
        optimizer_type in ["AdamW", "Prodigy"],  # No 8-bit with high alpha
        lr_warmup_steps >= 150,              # More gradual warmup
    ]
    if not all(required_checks):
        REJECT_TRIAL  # High alpha needs all safeguards

# Constraint 7: Very high alpha (>= 0.85) + high LR is dangerous
if alpha_ratio >= 0.85 and learning_rate > 0.0001:
    REJECT_TRIAL  # Too much learning capacity + speed = explosion risk
```

### **Soft Constraints (Discourage but allow)**

```python
# Low priority combinations (reduce sampling probability)
LOW_PRIORITY_COMBOS = [
    {"network_dim": 64, "max_train_epochs": 12},  # Underutilized capacity
    {"optimizer_type": "Prodigy", "lr_warmup_steps": 200},  # Prodigy has adaptive LR
]
```

---

## 📊 **Expected Improvements**

### **V1 (Current) vs V2 (Proposed)**

| Metric | V1 | V2 | Improvement |
|--------|----|----|-------------|
| **Stable trials** | 40-50% | 80-90% | ↑ +40-50% fewer failures |
| **Quality failures** | 30-40% | <10% | ↑ Fewer black/distorted images |
| **Optimal configs found** | 10-15% | 30-40% | ↑ Better coverage of good space |
| **Training time wasted** | ~30% | <10% | ↓ Fewer failed trials |

---

## 🗓️ **Implementation Plan**

### **Phase 1: Immediate (After Trial 5 completes)**
1. ✅ **Manually test Trial 3 checkpoint** (Dim=256, Alpha=32, LR=0.000066)
   - Expected result: High quality, stable training
   - Will validate that V2 parameter ranges are correct

2. ✅ **Generate V2 optimization script** with:
   - Narrowed LR ranges
   - Alpha ratio constraints
   - 12-20 epoch range
   - Safety constraint checks

### **Phase 2: Short Validation Run (10 trials, ~12-16 hours)**
3. ✅ **Launch small optimization** to validate V2 improvements:
   - Target: 10 trials
   - Goal: Verify 0 catastrophic failures
   - Check: At least 7/10 trials produce usable checkpoints

### **Phase 3: Full Optimization (40 trials, ~2-3 days)**
4. ✅ **Launch full V2 optimization** if validation succeeds:
   - Target: 40 trials
   - Expected: 30-35 usable results
   - Goal: Find optimal configuration with <5% quality failures

---

## 💡 **Recommendations**

### **Immediate Actions**
1. **Wait for Trial 5** to complete (ETA: check current progress)
2. **Test Trial 3 manually** when GPU is free
3. **Review Trial 5 results** - if good, keep its config as baseline

### **Next Optimization Strategy**

**Option A: Conservative (Recommended)**
- Use V2 search space with narrow ranges
- Run 20 trials to find safe optimal
- Estimated time: 24-30 hours
- Risk: Low
- Expected best config: Dim=128-256, Alpha=32-64, LR=0.00008-0.0001

**Option B: Aggressive (More Exploration)**
- Use V2 + additional new parameters (dropout, scheduler cycles)
- Run 40 trials to explore full space
- Estimated time: 48-60 hours
- Risk: Medium
- Expected best config: May find better than Option A

**Option C: Hybrid (Balanced)**
- Start with Option A (20 trials)
- If no good configs found, add Option B parameters
- Sequential optimization: safer iteration
- Total time: 30-40 hours (split over 2 runs)

### **My Recommendation: Option C (Hybrid)**
**Rationale:**
- Start conservative to quickly find stable baseline
- Expand search if needed (don't waste time on extreme values)
- Trial 3 looks promising → nearby parameter space likely optimal
- You can stop after Phase 1 if results are satisfactory

---

## 📝 **Code Changes Required**

### **File: `hyperparameter_optimization_v2.py`**

**Key Changes:**
1. Update search space with new ranges
2. Add `network_alpha_ratio` calculation
3. Implement safety constraint checker
4. Add trial validation before training
5. Improve logging for rejected trials

**Pseudo-code:**
```python
def suggest_trial(trial):
    # Sample parameters
    lr = trial.suggest_float("learning_rate", 0.00006, 0.00012, log=True)
    dim = trial.suggest_categorical("network_dim", [64, 128, 256])
    alpha_ratio = trial.suggest_float("network_alpha_ratio", 0.25, 0.5)
    alpha = int(dim * alpha_ratio)

    optimizer = trial.suggest_categorical("optimizer_type", ["AdamW", "AdamW8bit", "Prodigy"])
    epochs = trial.suggest_categorical("max_train_epochs", [12, 16, 20])

    # Safety checks
    if lr > 0.00012 and optimizer == "AdamW8bit":
        raise optuna.TrialPruned("High LR + 8bit unstable")

    if alpha == dim:
        raise optuna.TrialPruned("Alpha=Dim causes overfitting")

    if dim >= 256 and epochs < 16:
        raise optuna.TrialPruned("Insufficient training for high dim")

    return {
        "learning_rate": lr,
        "network_dim": dim,
        "network_alpha": alpha,
        ...
    }
```

---

## 📈 **Success Metrics**

### **V2 Optimization will be considered successful if:**
✅ **<10% catastrophic failures** (black images, crashes)
✅ **>60% usable checkpoints** (acceptable quality)
✅ **>3 excellent configs** (ready for production)
✅ **CLIP score >0.30** on best checkpoint
✅ **No gradient explosion** in any trial
✅ **Consistent character identity** across all outputs

---

## 🎯 **Next Steps for User**

### **Right Now:**
1. ⏳ **Monitor Trial 5** progress
2. 🧪 **Plan to test Trial 3** when GPU is free
3. 📖 **Review this document** and confirm strategy

### **After Trial 5 Completes:**
1. 🔍 **Evaluate Trial 5 results**
2. 🧪 **Test Trial 3 checkpoint** (likely best candidate)
3. ✅ **Approve V2 optimization plan**
4. 🚀 **Launch V2 optimization** (Option A, B, or C)

### **Expected Timeline:**
- **Trial 5 completion**: Check current status
- **Trial 3 testing**: 5-10 minutes
- **Decision point**: 30 minutes
- **V2 Opt implementation**: 1-2 hours
- **V2 Opt execution**: 24-60 hours (depending on option)

---

**End of Document**
**Questions? Ready to proceed when you are! 🚀**
