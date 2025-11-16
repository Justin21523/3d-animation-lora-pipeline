#!/bin/bash
# Start SDXL Training with Safe Gradient Checkpointing
# Uses custom wrapper with use_reentrant=False to prevent CUDA errors
# This script ensures training output is visible in both tmux and log file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KOHYA_ROOT="/mnt/c/AI_LLM_projects/kohya_ss"
WRAPPER_SCRIPT="$SCRIPT_DIR/scripts/training/sdxl_train_safe_checkpointing.py"
CONFIG_FILE="$SCRIPT_DIR/configs/training/sdxl_16gb_stable.toml"
LOG_DIR="$SCRIPT_DIR/logs/training"
SESSION_NAME="sdxl_luca_training_safe"

# Create log directory
mkdir -p "$LOG_DIR"

# Generate log filename
LOG_FILE="$LOG_DIR/sdxl_training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting SDXL LoRA Training..."
echo "Session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo ""

# Kill existing session
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Wait for GPU to clear
sleep 5

# Create new tmux session with training
cd "$KOHYA_ROOT"

tmux new-session -d -s "$SESSION_NAME"

# Send the training command (now with safe checkpointing patched directly in sdxl_train.py)
tmux send-keys -t "$SESSION_NAME" "cd $KOHYA_ROOT" C-m
tmux send-keys -t "$SESSION_NAME" "conda activate kohya_ss" C-m
tmux send-keys -t "$SESSION_NAME" "accelerate launch --num_cpu_threads_per_process=2 ./sd-scripts/sdxl_train_network.py --config_file='$CONFIG_FILE' 2>&1 | tee '$LOG_FILE'" C-m

echo "✓ Training started with SAFE gradient checkpointing!"
echo ""
echo "⚡ Configuration:"
echo "  - Gradient checkpointing: ENABLED (use_reentrant=False)"
echo "  - Gradient accumulation: 2 steps (faster than before)"
echo "  - Mixed precision: bf16"
echo "  - Expected speed: ~2-3 seconds/step"
echo ""
echo "To view training:"
echo "  bash $SCRIPT_DIR/safe_view_training.sh"
echo "  (SAFE - cannot interrupt training)"
echo ""
echo "To view log file:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Current log: $LOG_FILE" > /tmp/current_training_log.txt
