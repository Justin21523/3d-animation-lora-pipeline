#!/usr/bin/bash
"""
Train All Super Wings SDXL LoRAs Sequentially

Trains all 9 Super Wings character LoRAs using successful SDXL configuration.
Each character runs in its own tmux session with automatic monitoring.

Author: LLMProvider Tooling
Date: 2025-12-13
"""

set -e

CONFIGS_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/super_wings_loras_sdxl"
KOHYA_DIR="/mnt/c/ai_projects/kohya_ss/sd-scripts"
LOG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/logs/super_wings_training"

# Characters in processing order
CHARACTERS=(
    "jett"
    "jerome"
    "donnie"
    "chase"
    "flip"
    "todd"
    "paul"
    "bello"
    "beard"
)

echo "=================================================================="
echo "Super Wings SDXL LoRA Training - Sequential Batch"
echo "=================================================================="
echo "Total characters: ${#CHARACTERS[@]}"
echo "Configs dir: $CONFIGS_DIR"
echo "Kohya dir: $KOHYA_DIR"
echo ""

# Check if configs exist
echo "Checking configs..."
for char in "${CHARACTERS[@]}"; do
    config_file="$CONFIGS_DIR/super-wings-${char}-identity-sdxl.toml"
    if [ ! -f "$config_file" ]; then
        echo "❌ Config not found: $config_file"
        echo "   Run prepare_super_wings_complete_with_aug.py first!"
        exit 1
    fi
    echo "  ✓ $char: $(basename $config_file)"
done

echo ""
echo "All configs found!"
echo ""

# Function to train a single character
train_character() {
    local char=$1
    local session_name="sw_${char}_sdxl"
    local config_file="$CONFIGS_DIR/super-wings-${char}-identity-sdxl.toml"

    echo "=================================================================="
    echo "Training: $char"
    echo "=================================================================="
    echo "Config: $config_file"
    echo "Session: $session_name"
    echo ""

    # Kill existing session if exists
    tmux kill-session -t "$session_name" 2>/dev/null || true
    sleep 1

    # Create new session
    tmux new-session -d -s "$session_name"

    # Setup and run training
    tmux send-keys -t "$session_name" "cd $KOHYA_DIR" C-m
    sleep 1
    tmux send-keys -t "$session_name" "conda activate kohya_ss" C-m
    sleep 2

    # Launch training
    tmux send-keys -t "$session_name" "accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 4 sdxl_train_network.py --config_file=$config_file" C-m

    echo "✓ Training started in tmux session: $session_name"
    echo "  View with: tmux attach -t $session_name"
    echo "  Detach with: Ctrl+B then D"
    echo ""
}

# Function to wait for training to complete
wait_for_completion() {
    local session_name=$1
    local char=$2

    echo "Waiting for $char to complete..."

    while tmux has-session -t "$session_name" 2>/dev/null; do
        sleep 30
        echo "  [$char] Still training... ($(date '+%H:%M:%S'))"
    done

    echo "✓ $char training complete!"
    echo ""
}

# Main training loop
echo "Starting sequential training..."
echo ""

for i in "${!CHARACTERS[@]}"; do
    char="${CHARACTERS[$i]}"
    current=$((i + 1))
    total="${#CHARACTERS[@]}"

    echo "[$current/$total] Processing: $char"
    echo ""

    # Train character
    train_character "$char"

    # Wait for completion before starting next
    wait_for_completion "sw_${char}_sdxl" "$char"
done

echo "=================================================================="
echo "All Training Complete!"
echo "=================================================================="
echo "Processed: ${#CHARACTERS[@]} characters"
echo ""
echo "Next steps:"
echo "  1. Evaluate all checkpoints"
echo "  2. Select best epoch for each character"
echo "  3. Generate test images"
echo ""
echo "Training outputs in:"
echo "  /mnt/data/training/lora/super-wings/"
