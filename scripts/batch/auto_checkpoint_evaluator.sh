#!/bin/bash
#
# Automatic SDXL Checkpoint Evaluator
# Monitors checkpoint directory and auto-evaluates new checkpoints
#
# Usage: bash scripts/batch/auto_checkpoint_evaluator.sh <lora_dir> <character_id>
# Example: bash scripts/batch/auto_checkpoint_evaluator.sh /mnt/data/ai_data/models/lora_sdxl/coco/miguel_identity miguel
#

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <lora_dir> <character_id>"
    echo "Example: $0 /mnt/data/ai_data/models/lora_sdxl/coco/miguel_identity miguel"
    exit 1
fi

LORA_DIR="$1"
CHAR_ID="$2"

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
BASE_MODEL="/home/b0979/models/sdxl/sd_xl_base_1.0.safetensors"
PROMPTS_FILE="${PROJECT_ROOT}/prompts/lora_testing/${CHAR_ID}_identity_test.txt"
MONITOR_LOG="${PROJECT_ROOT}/logs/checkpoint_monitor_${CHAR_ID}_$(date +%Y%m%d_%H%M%S).log"

# Create evaluated checkpoints tracking file
EVALUATED_FILE="${LORA_DIR}/.evaluated_checkpoints.txt"
touch "$EVALUATED_FILE"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================================" | tee -a "$MONITOR_LOG"
echo "🤖 AUTOMATIC CHECKPOINT EVALUATOR STARTED" | tee -a "$MONITOR_LOG"
echo "================================================================================" | tee -a "$MONITOR_LOG"
echo "LoRA Directory: $LORA_DIR" | tee -a "$MONITOR_LOG"
echo "Character: $CHAR_ID" | tee -a "$MONITOR_LOG"
echo "Base Model: $BASE_MODEL" | tee -a "$MONITOR_LOG"
echo "Prompts: $PROMPTS_FILE" | tee -a "$MONITOR_LOG"
echo "Monitor Log: $MONITOR_LOG" | tee -a "$MONITOR_LOG"
echo "================================================================================" | tee -a "$MONITOR_LOG"
echo "" | tee -a "$MONITOR_LOG"

# Function to evaluate a checkpoint
evaluate_checkpoint() {
    local checkpoint_path="$1"
    local checkpoint_name=$(basename "$checkpoint_path" .safetensors)
    local eval_dir="${LORA_DIR}/eval_${checkpoint_name}"

    # Create temporary directory with only this checkpoint
    local temp_dir="/tmp/eval_${checkpoint_name}_$$"
    mkdir -p "$temp_dir"
    cp "$checkpoint_path" "$temp_dir/"

    echo "" | tee -a "$MONITOR_LOG"
    echo "================================================================================" | tee -a "$MONITOR_LOG"
    echo -e "${BLUE}🎨 EVALUATING: $checkpoint_name${NC}" | tee -a "$MONITOR_LOG"
    echo "================================================================================" | tee -a "$MONITOR_LOG"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MONITOR_LOG"
    echo "Checkpoint: $checkpoint_path" | tee -a "$MONITOR_LOG"
    echo "Output: $eval_dir" | tee -a "$MONITOR_LOG"
    echo "" | tee -a "$MONITOR_LOG"

    # Run SDXL evaluation
    conda run -n ai_env python "${PROJECT_ROOT}/scripts/evaluation/sdxl_lora_evaluator.py" \
        "$temp_dir" \
        --base-model "$BASE_MODEL" \
        --output-dir "$eval_dir" \
        --prompts-file "$PROMPTS_FILE" \
        --num-images-per-prompt 4 \
        --num-inference-steps 30 \
        --guidance-scale 7.5 \
        --seed 42 \
        --device cuda \
        2>&1 | tee -a "$MONITOR_LOG"

    local exit_code=${PIPESTATUS[0]}

    # Cleanup temp directory
    rm -rf "$temp_dir"

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ Evaluation complete: $checkpoint_name${NC}" | tee -a "$MONITOR_LOG"
        echo "$checkpoint_name" >> "$EVALUATED_FILE"

        # Display results
        if [ -f "${eval_dir}/checkpoint_comparison.json" ]; then
            echo "" | tee -a "$MONITOR_LOG"
            echo "📊 Results:" | tee -a "$MONITOR_LOG"
            cat "${eval_dir}/checkpoint_comparison.json" | python -m json.tool 2>/dev/null | grep -E "(clip_score|consistency)" | tee -a "$MONITOR_LOG" || true
        fi
    else
        echo -e "${RED}❌ Evaluation failed for $checkpoint_name${NC}" | tee -a "$MONITOR_LOG"
    fi

    echo "" | tee -a "$MONITOR_LOG"
}

# Monitor loop
echo -e "${YELLOW}👀 Monitoring for new checkpoints (checking every 60 seconds)...${NC}" | tee -a "$MONITOR_LOG"
echo "Press Ctrl+C to stop monitoring" | tee -a "$MONITOR_LOG"
echo "" | tee -a "$MONITOR_LOG"

while true; do
    # Find all checkpoint files
    while IFS= read -r checkpoint; do
        checkpoint_name=$(basename "$checkpoint" .safetensors)

        # Check if already evaluated
        if ! grep -q "^${checkpoint_name}$" "$EVALUATED_FILE" 2>/dev/null; then
            echo -e "${BLUE}🔔 New checkpoint detected: $checkpoint_name${NC}" | tee -a "$MONITOR_LOG"
            evaluate_checkpoint "$checkpoint"
        fi
    done < <(find "$LORA_DIR" -maxdepth 1 -name "*.safetensors" -type f 2>/dev/null | sort)

    # Check if training is complete
    if [ -f "${LORA_DIR}/.training_complete" ]; then
        echo "" | tee -a "$MONITOR_LOG"
        echo "================================================================================" | tee -a "$MONITOR_LOG"
        echo -e "${GREEN}✅ TRAINING COMPLETE - STOPPING MONITOR${NC}" | tee -a "$MONITOR_LOG"
        echo "================================================================================" | tee -a "$MONITOR_LOG"
        break
    fi

    # Status update
    num_checkpoints=$(find "$LORA_DIR" -maxdepth 1 -name "*.safetensors" -type f 2>/dev/null | wc -l)
    num_evaluated=$(wc -l < "$EVALUATED_FILE" 2>/dev/null || echo 0)
    echo "[$(date '+%H:%M:%S')] Status: $num_evaluated/$num_checkpoints evaluated" | tee -a "$MONITOR_LOG"

    sleep 60
done

echo "" | tee -a "$MONITOR_LOG"
echo "================================================================================" | tee -a "$MONITOR_LOG"
echo "Monitor session ended at: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MONITOR_LOG"
echo "================================================================================" | tee -a "$MONITOR_LOG"
