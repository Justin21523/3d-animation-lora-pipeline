#!/bin/bash
#
# Automatic LoRA Training Script with tmux - Project-Agnostic
# Waits for caption completion, then starts training in tmux session
#
# Usage:
#   ./auto_train_luca.sh [project]     # project defaults to "luca" if not specified
#
# Examples:
#   ./auto_train_luca.sh              # Use luca project
#   ./auto_train_luca.sh alberto      # Use alberto project
#

set -e

# ============================================================
# Project Configuration
# ============================================================

PROJECT="${1:-luca}"  # Default to luca if no argument provided
PROJECT_CONFIG="configs/projects/${PROJECT}.yaml"

# Verify project config exists
if [ ! -f "$PROJECT_CONFIG" ]; then
    echo "❌ Error: Project config not found: $PROJECT_CONFIG"
    echo "Available projects in configs/projects/:"
    ls -1 configs/projects/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//' || echo "  (none found)"
    exit 1
fi

# Read project configuration from YAML
PROJECT_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$PROJECT_CONFIG'))['project']['name'])")
BASE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$PROJECT_CONFIG'))['paths']['base_dir'])")

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
DATA_DIR="${BASE_DIR}/${PROJECT_NAME}_final_data"
CONFIG_FILE="$PROJECT_ROOT/configs/training/${PROJECT_NAME}_final_v1_pure.toml"
LOG_DIR="$PROJECT_ROOT/logs/training"

echo "=========================================="
echo "${PROJECT_NAME^^} LoRA Auto-Training Script"
echo "=========================================="
echo "Project: ${PROJECT_NAME}"
echo "Data directory: $DATA_DIR"
echo "Config file: $CONFIG_FILE"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Function to check caption completion
check_captions_complete() {
    local total_images=$(find "$DATA_DIR" -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l)
    local total_captions=$(find "$DATA_DIR" -type f -name "*.txt" | wc -l)

    echo "Images: $total_images"
    echo "Captions: $total_captions"

    if [ "$total_images" -eq "$total_captions" ] && [ "$total_images" -gt 0 ]; then
        return 0  # Complete
    else
        return 1  # Not complete
    fi
}

# Wait for caption generation to complete
echo "Step 1: Waiting for caption generation to complete..."
echo ""

while ! check_captions_complete; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Captions not ready yet, waiting 60 seconds..."
    sleep 60
done

echo ""
echo "✓ All captions generated!"
echo ""

# Verify dataset format
echo "Step 2: Verifying dataset format..."
total_images=$(find "$DATA_DIR" -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l)
total_captions=$(find "$DATA_DIR" -type f -name "*.txt" | wc -l)

echo "  Total images: $total_images"
echo "  Total captions: $total_captions"
echo ""

if [ "$total_images" -ne "$total_captions" ]; then
    echo "ERROR: Image count ($total_images) does not match caption count ($total_captions)"
    exit 1
fi

echo "✓ Dataset format verified!"
echo ""

# Create tmux session for training
echo "Step 3: Starting LoRA training in tmux session..."
echo ""

SESSION_NAME="${PROJECT_NAME}_lora_training"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "WARNING: tmux session '$SESSION_NAME' already exists!"
    echo "Attaching to existing session..."
    tmux attach -t "$SESSION_NAME"
    exit 0
fi

# Create new tmux session
tmux new-session -d -s "$SESSION_NAME" -n "training"

# Send training command to tmux session
tmux send-keys -t "$SESSION_NAME" "cd /mnt/c/AI_LLM_projects/kohya_ss" C-m
tmux send-keys -t "$SESSION_NAME" "conda activate kohya_ss" C-m
tmux send-keys -t "$SESSION_NAME" "echo '========================================'" C-m
tmux send-keys -t "$SESSION_NAME" "echo '${PROJECT_NAME^^} LoRA Training - Session Started'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Session: $SESSION_NAME'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Config: $CONFIG_FILE'" C-m
tmux send-keys -t "$SESSION_NAME" "echo '========================================'" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m

# Start training
tmux send-keys -t "$SESSION_NAME" "python sd-scripts/train_network.py --config_file $CONFIG_FILE 2>&1 | tee $LOG_DIR/${PROJECT_NAME}_final_v1_pure_\$(date +%Y%m%d_%H%M%S).log" C-m

echo "✓ Training started in tmux session: $SESSION_NAME"
echo ""
echo "To monitor training progress:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach from session: Ctrl+B, then D"
echo "To list sessions: tmux ls"
echo ""
echo "Training log will be saved to: $LOG_DIR/"
echo ""

# Create a monitoring script
cat > "$PROJECT_ROOT/monitor_${PROJECT_NAME}_training.sh" << MONITOR_EOF
#!/bin/bash
# Monitor ${PROJECT_NAME} training progress

SESSION_NAME="${PROJECT_NAME}_lora_training"

if tmux has-session -t "\$SESSION_NAME" 2>/dev/null; then
    echo "Training session is running!"
    echo "Attaching to session..."
    tmux attach -t "\$SESSION_NAME"
else
    echo "Training session not found!"
    echo "Available sessions:"
    tmux ls
fi
MONITOR_EOF

chmod +x "$PROJECT_ROOT/monitor_${PROJECT_NAME}_training.sh"

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Training will continue in background."
echo "Use './monitor_${PROJECT_NAME}_training.sh' to check progress."
echo ""
