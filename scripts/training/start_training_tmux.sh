#!/bin/bash
# Start iterative LoRA training in tmux session for long-running stability

SESSION_NAME="lora_training"
LOG_DIR="/mnt/data/ai_data/models/lora/luca/iterative_overnight_v5"

# Kill existing session if any
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Set up logging
tmux send-keys -t $SESSION_NAME "cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline" C-m
tmux send-keys -t $SESSION_NAME "mkdir -p $LOG_DIR" C-m

# Start training with output redirection
tmux send-keys -t $SESSION_NAME "python scripts/training/launch_iterative_training.py 2>&1 | tee $LOG_DIR/training.log" C-m

echo "=================================================================="
echo "✓ Training started in tmux session: $SESSION_NAME"
echo "=================================================================="
echo ""
echo "Useful commands:"
echo "  tmux attach -t $SESSION_NAME     # Attach to session"
echo "  tmux detach                       # Detach (Ctrl+B, D)"
echo "  tail -f $LOG_DIR/training.log     # Watch log"
echo "  nvidia-smi                        # Check GPU"
echo "  tmux kill-session -t $SESSION_NAME  # Stop training"
echo ""
echo "=================================================================="
