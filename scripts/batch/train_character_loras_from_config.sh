#!/bin/bash
#
# Universal Automated LoRA Training Pipeline (with tmux)
#
# This script provides a generic, reusable workflow for training multiple LoRAs:
# 1. Reads character list from YAML config file
# 2. Waits for caption generation to complete
# 3. Organizes training data in Kohya SS format
# 4. Trains LoRAs sequentially in tmux session
# 5. Automatically evaluates each LoRA after training
# 6. Generates comparison reports
#
# Usage:
#   bash scripts/batch/train_character_loras_from_config.sh <config_file> [session_name]
#
# Example:
#   bash scripts/batch/train_character_loras_from_config.sh configs/characters/new_batch.yaml lora_batch1
#
# Monitor tmux session:
#   tmux attach -t <session_name>
#

set -e

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
cd "$PROJECT_ROOT"

# ANSI colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
CONFIG_FILE="${1:-configs/characters/current_batch.yaml}"
SESSION_NAME="${2:-lora_training}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file not found: $CONFIG_FILE${NC}"
    echo ""
    echo "Usage: $0 <config_file> [session_name]"
    echo ""
    echo "Example:"
    echo "  $0 configs/characters/new_batch.yaml lora_batch1"
    exit 1
fi

echo "==========================================="
echo "Universal LoRA Training Pipeline"
echo "==========================================="
echo "Config: $CONFIG_FILE"
echo "Session: $SESSION_NAME"
echo ""

# Parse YAML config file using Python
read -r -d '' PYTHON_PARSE_CONFIG << 'EOF' || true
import sys
import yaml
from pathlib import Path

config_file = sys.argv[1]

with open(config_file) as f:
    config = yaml.safe_load(f)

# Print characters in shell-parseable format
for char in config.get('characters', []):
    movie = char['movie']
    char_dir = char.get('char_dir', char['char_name'])
    char_name = char['char_name']
    display_name = char['display_name']
    source_type = char.get('source_type', 'inpainted')
    img_count = char.get('image_count', 0)
    repeats = char.get('repeats', 10)
    epochs = char.get('epochs', 14)
    batch_size = char.get('batch_size', 6)

    print(f"{movie}:{char_dir}:{char_name}:{display_name}:{source_type}:{img_count}:{repeats}:{epochs}:{batch_size}")
EOF

# Load characters from config
mapfile -t CHARACTERS < <(python3 -c "$PYTHON_PARSE_CONFIG" "$CONFIG_FILE")

if [ ${#CHARACTERS[@]} -eq 0 ]; then
    echo -e "${RED}❌ No characters found in config file${NC}"
    exit 1
fi

echo "Found ${#CHARACTERS[@]} characters to train:"
for char_def in "${CHARACTERS[@]}"; do
    IFS=':' read -r movie char_dir char_name display_name source_type img_count repeats epochs batch_size <<< "$char_def"
    echo "  - $display_name ($movie): $img_count images"
done
echo ""

# Step 1: Wait for caption generation to complete
echo -e "${BLUE}Step 1: Waiting for caption generation...${NC}"
echo ""

for char_def in "${CHARACTERS[@]}"; do
    IFS=':' read -r movie char_dir char_name display_name source_type img_count repeats epochs batch_size <<< "$char_def"

    if [ "$source_type" = "augmented" ]; then
        IMAGE_DIR="/mnt/data/ai_data/datasets/3d-anime/$movie/lora_data/characters_augmented/$char_dir"
    else
        IMAGE_DIR="/mnt/data/ai_data/datasets/3d-anime/$movie/lora_data/characters_inpainted/$char_dir"
    fi

    echo -n "  Checking $display_name captions... "

    # Wait for captions with timeout
    TIMEOUT=7200  # 2 hours max wait
    ELAPSED=0

    while true; do
        caption_count=$(find "$IMAGE_DIR" -maxdepth 1 -name "*.txt" 2>/dev/null | wc -l)

        if [ "$caption_count" -ge "$img_count" ]; then
            echo -e "${GREEN}✅ Complete ($caption_count captions)${NC}"
            break
        fi

        if [ $ELAPSED -ge $TIMEOUT ]; then
            echo -e "${YELLOW}⚠️  Timeout waiting for captions${NC}"
            echo -e "${YELLOW}   Current: $caption_count / $img_count${NC}"
            break
        fi

        echo -ne "\r  $display_name: $caption_count / $img_count captions (${ELAPSED}s)... "
        sleep 30
        ELAPSED=$((ELAPSED + 30))
    done
done

echo ""

# Step 2: Organize training data in Kohya SS format
echo -e "${BLUE}Step 2: Organizing training data...${NC}"
echo ""

for char_def in "${CHARACTERS[@]}"; do
    IFS=':' read -r movie char_dir char_name display_name source_type img_count repeats epochs batch_size <<< "$char_def"

    if [ "$source_type" = "augmented" ]; then
        SOURCE_DIR="/mnt/data/ai_data/datasets/3d-anime/$movie/lora_data/characters_augmented/$char_dir"
    else
        SOURCE_DIR="/mnt/data/ai_data/datasets/3d-anime/$movie/lora_data/characters_inpainted/$char_dir"
    fi

    TARGET_DIR="/mnt/data/ai_data/datasets/3d-anime/$movie/lora_data/training_data/${char_name}_identity"
    KOHYA_DIR="$TARGET_DIR/${repeats}_${char_name}"

    echo -e "  ${display_name}:"
    echo -e "    Source: $SOURCE_DIR"
    echo -e "    Target: $KOHYA_DIR"

    # Create Kohya SS directory structure
    mkdir -p "$KOHYA_DIR"

    # Copy images and captions
    rsync -av --include="*.png" --include="*.jpg" --include="*.txt" --exclude="*" "$SOURCE_DIR/" "$KOHYA_DIR/" 2>&1 | grep -v "sending incremental file list" || true

    # Count copied files
    img_copied=$(find "$KOHYA_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l)
    txt_copied=$(find "$KOHYA_DIR" -maxdepth 1 -type f -name "*.txt" | wc -l)

    echo -e "    ${GREEN}✅ Organized: $img_copied images, $txt_copied captions${NC}"
done

echo ""

# Step 3: Create tmux session and train sequentially
echo -e "${BLUE}Step 3: Starting sequential LoRA training in tmux...${NC}"
echo ""

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create new tmux session
tmux new-session -d -s "$SESSION_NAME"

# Setup training loop in tmux
for idx in "${!CHARACTERS[@]}"; do
    char_def="${CHARACTERS[$idx]}"
    IFS=':' read -r movie char_dir char_name display_name source_type img_count repeats epochs batch_size <<< "$char_def"

    num=$((idx + 1))
    total=${#CHARACTERS[@]}

    echo -e "${YELLOW}[$num/$total] Queueing: $display_name${NC}"

    TRAIN_DIR="/mnt/data/ai_data/datasets/3d-anime/$movie/lora_data/training_data/${char_name}_identity"
    LORA_OUTPUT="/mnt/data/ai_data/models/lora/$movie/${char_name}_identity"
    CONFIG_FILE_TRAIN="$PROJECT_ROOT/configs/training/character_loras/${movie}_${char_name}_identity.toml"

    # Check if training config exists
    if [ ! -f "$CONFIG_FILE_TRAIN" ]; then
        echo -e "  ${RED}⚠️  Training config not found: $CONFIG_FILE_TRAIN${NC}"
        echo -e "  ${YELLOW}Skipping $display_name${NC}"
        continue
    fi

    # Build training command with evaluation
    TRAIN_CMD="
echo '=========================================='
echo '[$num/$total] Training: $display_name ($movie)'
echo '=========================================='
echo ''
echo 'Config: $CONFIG_FILE_TRAIN'
echo 'Output: $LORA_OUTPUT'
echo ''

cd /mnt/c/AI_LLM_projects/kohya_ss/sd-scripts

conda run -n kohya_ss accelerate launch --num_cpu_threads_per_process=4 train_network.py \\
  --config_file=$CONFIG_FILE_TRAIN \\
  2>&1 | tee $PROJECT_ROOT/logs/train_${char_name}_$(date +%Y%m%d_%H%M%S).log

TRAIN_EXIT_CODE=\${PIPESTATUS[0]}

if [ \$TRAIN_EXIT_CODE -ne 0 ]; then
    echo ''
    echo '=========================================='
    echo '❌ Training FAILED: $display_name'
    echo '=========================================='
    echo ''
else
    echo ''
    echo '=========================================='
    echo '✅ Training complete: $display_name'
    echo 'Starting automated evaluation...'
    echo '=========================================='
    echo ''

    cd $PROJECT_ROOT

    # Check if test prompts exist
    if [ -f prompts/lora_testing/${char_name}_identity_prompts.json ]; then
        /home/b0979/.conda/envs/ai_env/bin/python scripts/evaluation/test_lora_checkpoints.py \\
          $LORA_OUTPUT \\
          --base-model /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors \\
          --output-dir $LORA_OUTPUT/evaluations \\
          --prompts-file prompts/lora_testing/${char_name}_identity_prompts.json \\
          --num-variations 4 \\
          --steps 30 \\
          --cfg-scale 7.5 \\
          --seed 42 \\
          --device cuda \\
          2>&1 | tee $PROJECT_ROOT/logs/eval_${char_name}_$(date +%Y%m%d_%H%M%S).log

        echo ''
        echo '✅ Evaluation complete: $display_name'
    else
        echo '⚠️  Test prompts not found, skipping evaluation'
    fi

    echo ''
    echo '✅ $display_name completed!'
fi

echo ''
"

    # Send command to tmux
    tmux send-keys -t "$SESSION_NAME" "$TRAIN_CMD" C-m

    # Wait briefly between characters
    if [ $idx -lt $((${#CHARACTERS[@]} - 1)) ]; then
        tmux send-keys -t "$SESSION_NAME" "sleep 5" C-m
    fi
done

# Final completion message
tmux send-keys -t "$SESSION_NAME" "
echo '=========================================='
echo '✅ ALL CHARACTER LORAS COMPLETED!'
echo '=========================================='
echo ''
echo 'Training session: $SESSION_NAME'
echo 'Config file: $CONFIG_FILE'
echo ''
echo 'Results:'
" C-m

for char_def in "${CHARACTERS[@]}"; do
    IFS=':' read -r movie char_dir char_name display_name source_type img_count repeats epochs batch_size <<< "$char_def"
    tmux send-keys -t "$SESSION_NAME" "echo '  - $display_name: /mnt/data/ai_data/models/lora/$movie/${char_name}_identity/'" C-m
done

tmux send-keys -t "$SESSION_NAME" "
echo ''
echo 'Evaluation reports in evaluations/ subdirectories'
echo '=========================================='
" C-m

echo ""
echo "==========================================="
echo -e "${GREEN}✅ Training Pipeline Started!${NC}"
echo "==========================================="
echo ""
echo "Training sequence:"
for idx in "${!CHARACTERS[@]}"; do
    char_def="${CHARACTERS[$idx]}"
    IFS=':' read -r movie char_dir char_name display_name source_type img_count repeats epochs batch_size <<< "$char_def"
    num=$((idx + 1))
    echo -e "  $num. ${BLUE}$display_name${NC} ($movie) - $img_count images, $epochs epochs"
done
echo ""
echo "Tmux session: $SESSION_NAME"
echo -e "${YELLOW}Monitor with:${NC}"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo -e "${YELLOW}Detach from tmux:${NC}"
echo "  Ctrl+B, then D"
echo ""
echo -e "${YELLOW}Check session status:${NC}"
echo "  tmux list-sessions"
echo ""
echo "==========================================="
