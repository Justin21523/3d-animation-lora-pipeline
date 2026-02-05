#!/bin/bash
#
# Start Universal LoRA Training in tmux
#
# Creates a tmux session with:
# - Window 0: Safety monitor
# - Window 1: Training orchestrator (universal-only)
#
# Author: LLMProvider Tooling
# Date: 2025-12-04
#

set -euo pipefail

SESSION_NAME="lora_training"
SCRIPT_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/batch"
CONFIG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/synthetic_loras_filtered"
SD_SCRIPTS="/mnt/c/ai_projects/kohya_ss/sd-scripts"

echo "=" | tr '=' '=' | head -c 80
echo
echo "Starting Universal LoRA Training in tmux"
echo "=" | tr '=' '=' | head -c 80
echo
echo "Session name: $SESSION_NAME"
echo "Training: 3 universal LoRAs only"
echo "Estimated time: ~188 hours (7.8 days)"
echo

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Session '$SESSION_NAME' already exists!"
    echo
    read -p "Kill existing session and restart? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo "✅ Killed existing session"
    else
        echo "Attaching to existing session..."
        tmux attach -t "$SESSION_NAME"
        exit 0
    fi
fi

# Create new tmux session (detached)
echo "Creating tmux session..."
tmux new-session -d -s "$SESSION_NAME" -n "safety-monitor"

# Window 0: Safety monitor
echo "Setting up safety monitor..."
tmux send-keys -t "$SESSION_NAME:0" "cd /mnt/c/ai_projects/3d-animation-lora-pipeline" C-m
tmux send-keys -t "$SESSION_NAME:0" "bash scripts/batch/training_safety_monitor.sh" C-m

# Window 1: Training orchestrator
echo "Setting up training orchestrator..."
tmux new-window -t "$SESSION_NAME:1" -n "training"
tmux send-keys -t "$SESSION_NAME:1" "cd /mnt/c/ai_projects/3d-animation-lora-pipeline" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo '⏳ Waiting 5 seconds before starting training...'" C-m
tmux send-keys -t "$SESSION_NAME:1" "sleep 5" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo '🚀 Starting Universal LoRA Training'" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo '   - universal_pose (4,577 images, ~57 hours)'" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo '   - universal_action (4,982 images, ~73 hours)'" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo '   - universal_expression (3,997 images, ~58 hours)'" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo 'Press Enter to start training...'" C-m
tmux send-keys -t "$SESSION_NAME:1" "python3 scripts/batch/train_all_synthetic_loras_sequential.py \\" C-m
tmux send-keys -t "$SESSION_NAME:1" "  --config-dir $CONFIG_DIR \\" C-m
tmux send-keys -t "$SESSION_NAME:1" "  --sd-scripts $SD_SCRIPTS \\" C-m
tmux send-keys -t "$SESSION_NAME:1" "  --universal-only" C-m

echo
echo "=" | tr '=' '=' | head -c 80
echo
echo "✅ tmux session created: $SESSION_NAME"
echo
echo "Available windows:"
echo "  0: safety-monitor  - System resource monitoring"
echo "  1: training        - Training orchestrator (waiting for Enter)"
echo
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To detach: Ctrl+B then D"
echo "To switch windows: Ctrl+B then 0/1"
echo
echo "Training will START when you:"
echo "  1. Attach to session: tmux attach -t $SESSION_NAME"
echo "  2. Switch to training window: Ctrl+B then 1"
echo "  3. Press Enter to confirm"
echo
echo "=" | tr '=' '=' | head -c 80
echo
