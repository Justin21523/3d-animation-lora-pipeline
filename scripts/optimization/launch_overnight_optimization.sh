#!/usr/bin/bash
# Extended Hyperparameter Optimization Launch Script
# Specialized for facial quality issues: consistency, cropping, missing features
#
# Target Issues:
#   - 臉型不一致 (Face shape inconsistency)
#   - 截斷問題 (Cropping issues)
#   - 五官缺失 (Missing facial features)

set -e

echo "=========================================="
echo "🚀 LAUNCHING EXTENDED HYPERPARAMETER OPTIMIZATION"
echo "=========================================="
echo "Target issues:"
echo "  - 臉型不一致 (Face shape inconsistency)"
echo "  - 截斷問題 (Cropping issues)"
echo "  - 五官缺失 (Missing facial features)"
echo "=========================================="
echo ""

# Configuration
DATASET_CONFIG="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/luca_human_dataset.toml"
BASE_MODEL="/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors"
OUTPUT_DIR="/mnt/data/ai_data/models/lora/luca/optimization_overnight"
STUDY_NAME="luca_facial_quality_optimization"
N_TRIALS=50  # Extended for thorough search (no time limit)
DEVICE="cuda"

# Python executable
PYTHON="/home/b0979/.conda/envs/kohya_ss/bin/python"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/progress_checks"

echo "📋 Configuration:"
echo "  Dataset: $DATASET_CONFIG"
echo "  Base Model: $BASE_MODEL"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Study Name: $STUDY_NAME"
echo "  Trials: $N_TRIALS (extendable)"
echo "  Device: $DEVICE"
echo ""

# Check if optimization script exists
OPTIM_SCRIPT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/optimization/optuna_hyperparameter_search.py"
if [ ! -f "$OPTIM_SCRIPT" ]; then
    echo "❌ ERROR: Optimization script not found at $OPTIM_SCRIPT"
    exit 1
fi

# Check if dataset config exists
if [ ! -f "$DATASET_CONFIG" ]; then
    echo "❌ ERROR: Dataset config not found at $DATASET_CONFIG"
    exit 1
fi

# Check if base model exists
if [ ! -f "$BASE_MODEL" ]; then
    echo "❌ ERROR: Base model not found at $BASE_MODEL"
    exit 1
fi

echo "✅ All checks passed"
echo ""
echo "=========================================="
echo "🏁 STARTING OPTIMIZATION"
echo "=========================================="
echo "Estimated time: 18-24+ hours for 50 trials"
echo "Progress will be logged to: $OUTPUT_DIR/optimization.log"
echo "Intermediate results: $OUTPUT_DIR/progress_checks/"
echo ""
echo "You can monitor progress with:"
echo "  tail -f $OUTPUT_DIR/optimization.log"
echo ""
echo "To check current best trial:"
echo "  sqlite3 $OUTPUT_DIR/optuna_study.db 'SELECT number, value FROM trials ORDER BY value LIMIT 5;'"
echo ""
echo "=========================================="
echo ""

# Launch optimization in background
nohup $PYTHON "$OPTIM_SCRIPT" \
  --dataset-config "$DATASET_CONFIG" \
  --base-model "$BASE_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --study-name "$STUDY_NAME" \
  --n-trials $N_TRIALS \
  --device "$DEVICE" \
  > "$OUTPUT_DIR/optimization.log" 2>&1 &

PID=$!

echo "✅ Optimization launched successfully!"
echo ""
echo "Process ID (PID): $PID"
echo "Log file: $OUTPUT_DIR/optimization.log"
echo ""
echo "=========================================="
echo "📊 MONITORING COMMANDS"
echo "=========================================="
echo ""
echo "1. Watch live progress:"
echo "   tail -f $OUTPUT_DIR/optimization.log"
echo ""
echo "2. Check current status:"
echo "   ps aux | grep $PID"
echo ""
echo "3. View best 10 trials so far:"
echo "   sqlite3 $OUTPUT_DIR/optuna_study.db \\"
echo "     'SELECT number, value FROM trials WHERE state=\"COMPLETE\" ORDER BY value LIMIT 10;'"
echo ""
echo "4. Count completed trials:"
echo "   sqlite3 $OUTPUT_DIR/optuna_study.db 'SELECT COUNT(*) FROM trials WHERE state=\"COMPLETE\";'"
echo ""
echo "5. Check intermediate checkpoint images:"
echo "   ls -lh $OUTPUT_DIR/progress_checks/"
echo ""
echo "6. Kill optimization (if needed):"
echo "   kill $PID"
echo ""
echo "=========================================="
echo ""
echo "Optimization is now running in the background."
echo "Will continue until optimal parameters are found or $N_TRIALS trials complete."
echo ""
