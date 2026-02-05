#!/bin/bash
# Launch 6-character training pipeline in TMUX session "sd"

set -e

TMUX_SESSION="sd"
PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/batch/train_and_evaluate_6_characters.sh"

echo "==========================================
"
echo "Launching Training in TMUX Session: $TMUX_SESSION"
echo "=========================================="
echo ""

# Check if tmux session exists
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "✅ TMUX session '$TMUX_SESSION' exists"
    echo "Sending training command to session..."

    # Send command to existing session
    tmux send-keys -t "$TMUX_SESSION" "cd $PROJECT_ROOT" C-m
    tmux send-keys -t "$TMUX_SESSION" "bash $TRAINING_SCRIPT" C-m

    echo ""
    echo "✅ Training launched in existing TMUX session '$TMUX_SESSION'"
else
    echo "⚠️  TMUX session '$TMUX_SESSION' does not exist"
    echo "Creating new session '$TMUX_SESSION'..."

    # Create new session and run training
    tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT_ROOT"
    tmux send-keys -t "$TMUX_SESSION" "bash $TRAINING_SCRIPT" C-m

    echo ""
    echo "✅ Created TMUX session '$TMUX_SESSION' and launched training"
fi

echo ""
echo "==========================================
"
echo "Training Pipeline Status"
echo "=========================================="
echo ""
echo "TMUX session: $TMUX_SESSION"
echo "Training script: $TRAINING_SCRIPT"
echo ""
echo "To attach to session:"
echo "  tmux attach -t $TMUX_SESSION"
echo ""
echo "To detach from session (inside tmux):"
echo "  Press Ctrl+B, then D"
echo ""
echo "To check training logs:"
echo "  tail -f logs/training/train_evaluate_6chars_*.log"
echo "=========================================="
