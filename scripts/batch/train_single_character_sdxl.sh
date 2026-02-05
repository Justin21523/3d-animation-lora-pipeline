#!/usr/bin/bash
#
# Train Single Super Wings Character SDXL LoRA
# Usage: bash train_single_character_sdxl.sh <character_name>
#

set -e

CHARACTER=$1

if [ -z "$CHARACTER" ]; then
    echo "Usage: $0 <character_name>"
    echo "Available characters: jett, jerome, donnie, chase, flip, todd, paul, bello, beard"
    exit 1
fi

CONFIGS_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/super_wings_loras_sdxl"
KOHYA_DIR="/mnt/c/ai_projects/kohya_ss/sd-scripts"
CONFIG_FILE="$CONFIGS_DIR/super-wings-${CHARACTER}-identity-sdxl.toml"
SESSION_NAME="sw_${CHARACTER}_sdxl"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config not found: $CONFIG_FILE"
    exit 1
fi

echo "=================================================================="
echo "Super Wings SDXL LoRA Training"
echo "=================================================================="
echo "Character: $CHARACTER"
echo "Config: $CONFIG_FILE"
echo "Session: $SESSION_NAME"
echo ""
echo "Training parameters:"
echo "  - Epochs: 5"
echo "  - Checkpoint every: 1 epoch"
echo "  - Will produce 5 checkpoints for testing"
echo ""
echo "=================================================================="
echo ""

# Kill existing session if exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
sleep 1

# Create new session
tmux new-session -d -s "$SESSION_NAME"

# Setup and run training
tmux send-keys -t "$SESSION_NAME" "cd $KOHYA_DIR" C-m
sleep 1
tmux send-keys -t "$SESSION_NAME" "conda activate kohya_ss" C-m
sleep 2

# Launch training
tmux send-keys -t "$SESSION_NAME" "accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 4 sdxl_train_network.py --config_file=$CONFIG_FILE" C-m

echo "✓ Training started in tmux session: $SESSION_NAME"
echo ""
echo "Monitor with:"
echo "  tmux attach -t $SESSION_NAME"
echo "  (Detach with: Ctrl+B then D)"
echo ""
echo "Outputs will be saved to:"
echo "  /mnt/data/training/lora/super-wings/${CHARACTER}_identity/"
echo ""
echo "After training completes, test the 5 checkpoints before training next character!"
echo ""
