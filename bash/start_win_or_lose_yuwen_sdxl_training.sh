#!/bin/bash
set -euo pipefail

# Start SDXL identity LoRA training for Win or Lose / yuwen.
# Uses safe gradient checkpointing wrapper (use_reentrant=False).
#
# Session: yuwen_sdxl_training
# Logs: /mnt/data/training/lora/win-or-lose/yuwen_identity/logs_master/

REPO_ROOT="/mnt/c/ai_projects/3d-animation-lora-pipeline"
SESSION="yuwen_sdxl_training"
CONFIG_FILE="$REPO_ROOT/configs/training/character_loras_sdxl/win_or_lose_yuwen_identity_sdxl.toml"
WRAPPER="$REPO_ROOT/scripts/training/sdxl_train_safe_checkpointing.py"

OUT_DIR="/mnt/data/training/lora/win-or-lose/yuwen_identity"
MASTER_LOG_DIR="$OUT_DIR/logs_master"
mkdir -p "$MASTER_LOG_DIR"

LOG_FILE="$MASTER_LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "Starting yuwen SDXL identity LoRA training..."
echo "Session: $SESSION"
echo "Config: $CONFIG_FILE"
echo "Log: $LOG_FILE"

# Kill any existing session with same name
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 2

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "cd \"$REPO_ROOT\"" C-m
tmux send-keys -t "$SESSION" "mkdir -p \"$OUT_DIR/logs\"" C-m
tmux send-keys -t "$SESSION" "conda run -n kohya_ss python -u \"$WRAPPER\" --config_file \"$CONFIG_FILE\" 2>&1 | tee \"$LOG_FILE\"" C-m

echo "✓ Training launched"
echo "Attach: tmux attach -t $SESSION"
echo "Log: tail -f $LOG_FILE"

