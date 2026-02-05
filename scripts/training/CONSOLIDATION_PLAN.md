# Training Scripts Consolidation Plan

**Date:** 2025-11-22
**Status:** Proposed

## Current State

- **Total files:** 24 scripts (12 .sh, 12 .py)
- **Total lines:** ~2,500 lines
- **Problem:** Multiple overlapping scripts for similar functions

## Consolidation Strategy

### Phase 1: Merge Training Orchestration

**Create:** `unified_training_orchestrator.py`

**Merges (5 files → 1):**
- `auto_train_all_sequential.py` (172 lines)
- `launch_iterative_training.py` (91 lines)
- `pure_iterative_training.py` (203 lines)
- `launch_iteration_v6.py` (162 lines)
- `start_training_tmux.sh` (40 lines)

**Features:**
- Sequential training of multiple characters
- Iterative training with evaluation feedback
- Tmux session management
- Progress tracking and logging
- Configurable training strategies

**Usage:**
```bash
python unified_training_orchestrator.py \
  --mode sequential \
  --characters luca alberto giulia \
  --base-config configs/training/base.toml \
  --session-name lora_training
```

### Phase 2: Merge Hyperparameter Optimization

**Create:** `hyperparameter_optimizer.py`

**Merges (5 files → 1):**
- `lora_hyperparameter_search.py` (253 lines)
- `lora_hyperparameter_search_optuna.py` (432 lines)
- `iterative_lora_optimizer.py` (1,361 lines)
- `run_hyperparameter_search.sh` (152 lines)
- `run_hyperparameter_search_optuna.sh` (170 lines)

**Features:**
- Random/Grid/Bayesian search (Optuna)
- Iterative optimization with feedback
- Multi-objective optimization
- Early stopping
- Result tracking and visualization

**Usage:**
```bash
python hyperparameter_optimizer.py \
  --method optuna \
  --n-trials 20 \
  --base-config configs/training/luca.toml \
  --output-dir hyperparameter_search/luca
```

### Phase 3: Merge Training Management

**Create:** `training_manager.py`

**Merges (4 files → 1):**
- `auto_train_memory_safe.sh` (127 lines)
- `restart_training.sh` (102 lines)
- `safe_restart_training.sh` (271 lines)
- `stop_training.sh` (85 lines)

**Features:**
- Start/stop/restart training
- Memory-safe training (monitor VRAM)
- Safe checkpoint recovery
- Process management
- Graceful shutdown

**Usage:**
```bash
python training_manager.py start --config luca.toml --memory-limit 14GB
python training_manager.py restart --checkpoint last
python training_manager.py stop --graceful
```

### Phase 4: Merge Monitoring

**Create:** `training_monitor.py`

**Merges (3 files → 1):**
- `monitor_training.sh` (99 lines)
- `check_progress.sh` (152 lines)
- `STATUS_SUMMARY.sh` (101 lines)

**Features:**
- Real-time training progress
- Loss/metric tracking
- ETA calculation
- Status summary dashboard
- Alert on anomalies

**Usage:**
```bash
python training_monitor.py --watch --interval 30
python training_monitor.py --summary --all-sessions
```

### Phase 5: Keep Standalone Tools

**Keep as-is:**
- ✅ `checkpoint_evaluator.py` - Comprehensive evaluation tool
- ✅ `fix_lighting_captions.py` - Special-purpose caption fixer
- ✅ `test_single_training.py` - Quick testing tool

**SDXL-specific (keep or merge later):**
- `start_sdxl_16gb_training.sh` - SDXL configuration
- `launch_iterative_optimization.sh` - SDXL workflow
- `sdxl_train_safe_checkpointing.py` - SDXL checkpointing
- `patch_safe_checkpointing.py` - Checkpointing patches

## Implementation Plan

### Step 1: Create Unified Training Orchestrator

```python
# unified_training_orchestrator.py structure

class TrainingOrchestrator:
    """Unified training orchestration system"""

    def __init__(self, mode='sequential', config=None):
        self.mode = mode  # sequential, parallel, iterative
        self.config = config

    def run_sequential(self, characters):
        """Train characters one by one"""

    def run_iterative(self, character, n_iterations):
        """Iterative training with evaluation feedback"""

    def run_parallel(self, characters, n_gpus):
        """Train multiple characters in parallel"""
```

### Step 2: Create Hyperparameter Optimizer

```python
# hyperparameter_optimizer.py structure

class HyperparameterOptimizer:
    """Unified hyperparameter optimization"""

    def __init__(self, method='optuna', n_trials=20):
        self.method = method  # random, grid, optuna, iterative

    def optimize(self, base_config, search_space):
        """Run optimization"""

    def suggest_improvements(self, evaluation_result):
        """Iterative improvement based on eval"""
```

### Step 3: Implement and Test

1. Create unified scripts
2. Test with existing configs
3. Verify feature parity
4. Update documentation
5. Delete old scripts

## Benefits

### Before (Current)
- 24 scripts
- ~2,500 lines
- Overlapping functionality
- Hard to maintain
- Inconsistent interfaces

### After (Proposed)
- 8 core files:
  - `unified_training_orchestrator.py` (~400 lines)
  - `hyperparameter_optimizer.py` (~500 lines)
  - `training_manager.py` (~300 lines)
  - `training_monitor.py` (~200 lines)
  - `checkpoint_evaluator.py` (keep, ~300 lines)
  - `fix_lighting_captions.py` (keep, ~150 lines)
  - `test_single_training.py` (keep, ~100 lines)
  - SDXL utilities (~500 lines)
- Total: ~2,450 lines (similar total, much better organized)
- **67% fewer files** (24 → 8)
- Single unified interface per function
- Easier to maintain and extend

## Files to Delete After Consolidation

```bash
# Training orchestration (→ unified_training_orchestrator.py)
rm scripts/training/auto_train_all_sequential.py
rm scripts/training/launch_iterative_training.py
rm scripts/training/pure_iterative_training.py
rm scripts/training/launch_iteration_v6.py
rm scripts/training/start_training_tmux.sh

# Hyperparameter optimization (→ hyperparameter_optimizer.py)
rm scripts/training/lora_hyperparameter_search.py
rm scripts/training/lora_hyperparameter_search_optuna.py
rm scripts/training/iterative_lora_optimizer.py
rm scripts/training/run_hyperparameter_search.sh
rm scripts/training/run_hyperparameter_search_optuna.sh

# Training management (→ training_manager.py)
rm scripts/training/auto_train_memory_safe.sh
rm scripts/training/restart_training.sh
rm scripts/training/safe_restart_training.sh
rm scripts/training/stop_training.sh

# Monitoring (→ training_monitor.py)
rm scripts/training/monitor_training.sh
rm scripts/training/check_progress.sh
rm scripts/training/STATUS_SUMMARY.sh

# Total: 17 files to delete
```

## Risks and Mitigation

### Risk 1: Feature Loss
**Mitigation:** Comprehensive testing before deletion, keep backups

### Risk 2: Workflow Disruption
**Mitigation:** Maintain backward compatibility aliases

### Risk 3: SDXL Specifics
**Mitigation:** Keep SDXL scripts separate until migration

## Timeline

- **Week 1:** Design unified interfaces
- **Week 2:** Implement core orchestrator
- **Week 3:** Implement optimizer and manager
- **Week 4:** Testing and documentation
- **Week 5:** Migration and cleanup

## Next Steps

1. Review and approve this plan
2. Create feature parity checklist
3. Implement unified orchestrator
4. Test with real training runs
5. Migrate users to new scripts
6. Delete old scripts

## Questions for User

1. Should we merge SDXL scripts now or later?
2. Priority: orchestrator or optimizer first?
3. Keep iterative training as separate mode or integrate?
4. Any specific workflows that must be preserved?
