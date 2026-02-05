#!/bin/bash
#
# Universal SDXL Training with Automatic Checkpoint Evaluation
# Usage: bash scripts/batch/train_sdxl_with_auto_eval.sh <config_file>
# Example: bash scripts/batch/train_sdxl_with_auto_eval.sh configs/training/character_loras_sdxl/coco_miguel_identity_sdxl.toml
#

set -e

# Check if config file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 configs/training/character_loras_sdxl/coco_miguel_identity_sdxl.toml"
    exit 1
fi

CONFIG_FILE="$1"

# Convert to absolute path if relative
if [[ ! "$CONFIG_FILE" = /* ]]; then
    CONFIG_FILE="$(pwd)/$CONFIG_FILE"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract character info from config file path
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .toml)
# Extract film and character from filename (e.g., "coco_miguel_identity_sdxl" -> film="coco", char="miguel")
FILM=$(echo "$CONFIG_BASENAME" | cut -d'_' -f1)
CHAR_ID=$(echo "$CONFIG_BASENAME" | cut -d'_' -f2)

PROJECT_ROOT="/mnt/c/ai_projects/3d-animation-lora-pipeline"
KOHYA_SS_DIR="/mnt/c/ai_projects/kohya_ss/sd-scripts"
BASE_MODEL="/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"

# Extract output_dir from config file
OUTPUT_DIR=$(grep "^output_dir" "$CONFIG_FILE" | sed 's/.*"\(.*\)".*/\1/')

# Construct prompts file path
PROMPTS_FILE="${PROJECT_ROOT}/prompts/lora_testing/${CHAR_ID}_identity_test.txt"

# Validate prompts file exists (FAIL FAST - don't auto-generate generic prompts)
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "❌ ERROR: Prompts file not found: $PROMPTS_FILE"
    echo "   Please create character-specific prompts first."
    echo "   Generic auto-generated prompts hurt training quality."
    exit 1
fi

LOG_FILE="${PROJECT_ROOT}/logs/${CONFIG_BASENAME}_training_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "SDXL LoRA TRAINING WITH AUTO-EVALUATION"
echo "================================================================================"
echo "Character: ${FILM}/${CHAR_ID}"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Base Model: $BASE_MODEL"
echo "Prompts: $PROMPTS_FILE"
echo "Log: $LOG_FILE"
echo "================================================================================"
echo ""

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Check if in tmux
if [ -z "$TMUX" ]; then
    echo -e "${YELLOW}⚠️  WARNING: Not running in tmux!${NC}"
    echo -e "${YELLOW}Training may be interrupted if terminal disconnects.${NC}"
    echo ""
else
    echo -e "${GREEN}✅ Running in tmux: $(tmux display-message -p '#S')${NC}"
    echo ""
fi

# Function to evaluate a single checkpoint
evaluate_checkpoint() {
    local checkpoint_path="$1"
    local checkpoint_name=$(basename "$checkpoint_path" .safetensors)
    local eval_dir="${OUTPUT_DIR}/eval_${checkpoint_name}"

    echo ""
    echo "================================================================================"
    echo "🎨 EVALUATING CHECKPOINT: $checkpoint_name"
    echo "================================================================================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Create temporary directory with only this checkpoint (SDXL evaluator requires a directory)
    local temp_dir="/tmp/eval_${checkpoint_name}_$$"
    mkdir -p "$temp_dir"
    cp "$checkpoint_path" "$temp_dir/"

    # Run simple image generation (evaluation)
    conda run -n ai_env python "${PROJECT_ROOT}/scripts/evaluation/simple_lora_image_generator.py" \
        "$temp_dir" \
        --base-model "$BASE_MODEL" \
        --output-dir "$eval_dir" \
        --prompts-file "$PROMPTS_FILE" \
        --num-images-per-prompt 4 \
        --steps 30 \
        --guidance 7.5 \
        --seed 42 \
        --device cuda \
        2>&1 | tee -a "$LOG_FILE"

    local exit_code=${PIPESTATUS[0]}

    # Cleanup temp directory
    rm -rf "$temp_dir"

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ Image generation complete: $checkpoint_name${NC}"

        # Display image count if metadata exists
        if [ -f "${eval_dir}/${checkpoint_name}/metadata.json" ]; then
            echo ""
            echo "📊 Generated images for $checkpoint_name:"
            num_images=$(cat "${eval_dir}/${checkpoint_name}/metadata.json" | python -m json.tool 2>/dev/null | grep '"num_images"' | sed 's/.*: \(.*\),*/\1/')
            echo "  Total: $num_images images"
            echo "  Location: ${eval_dir}/${checkpoint_name}/"
        fi
    else
        echo -e "${RED}❌ Image generation failed for $checkpoint_name${NC}"
    fi

    echo ""
}

# Function to monitor and evaluate checkpoints
monitor_and_evaluate() {
    local last_evaluated=""

    echo ""
    echo "================================================================================"
    echo "👀 CHECKPOINT MONITOR STARTED"
    echo "================================================================================"
    echo "Output directory: $OUTPUT_DIR"
    echo "Checking for new checkpoints every 60 seconds..."
    echo "Will auto-evaluate each checkpoint as it's saved"
    echo ""

    while true; do
        # Find all checkpoint files
        checkpoints=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.safetensors" -type f 2>/dev/null | sort)

        for checkpoint in $checkpoints; do
            checkpoint_name=$(basename "$checkpoint")

            # Check if this checkpoint has been evaluated
            if [[ "$last_evaluated" != *"$checkpoint_name"* ]]; then
                echo -e "${BLUE}🔔 New checkpoint detected: $checkpoint_name${NC}"
                evaluate_checkpoint "$checkpoint"
                last_evaluated="${last_evaluated} ${checkpoint_name}"
            fi
        done

        # Check if training is complete
        if [ -f "${OUTPUT_DIR}/.training_complete" ]; then
            echo ""
            echo "================================================================================"
            echo "✅ TRAINING COMPLETE - STOPPING MONITOR"
            echo "================================================================================"
            break
        fi

        sleep 60
    done
}

# Start monitoring in background
monitor_and_evaluate & 
MONITOR_PID=$!

echo ""
echo "================================================================================"
echo "🚀 STARTING SDXL TRAINING"
echo "================================================================================"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Run training
cd "$KOHYA_SS_DIR"

# Use Python unbuffered mode (-u) for real-time output
export PYTHONUNBUFFERED=1
conda run -n kohya_ss accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 4 \
    sdxl_train_network.py \
    --config_file="$CONFIG_FILE" \
    2>&1 | tee -a "$LOG_FILE"

TRAINING_EXIT_CODE=$?

# Mark training as complete
touch "${OUTPUT_DIR}/.training_complete"

# Wait for monitor to finish
wait $MONITOR_PID

echo ""
echo "================================================================================"
echo "TRAINING SESSION COMPLETE"
echo "================================================================================"
echo "Ended at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully${NC}"
else
    echo -e "${RED}❌ Training failed with exit code $TRAINING_EXIT_CODE${NC}"
fi

echo ""
echo "Character: ${FILM}/${CHAR_ID}"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Generate comprehensive comparison report if training succeeded
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "================================================================================"
    echo "📊 GENERATING COMPREHENSIVE COMPARISON REPORT"
    echo "================================================================================"
    echo ""

    # Collect all evaluation results
    eval_results=()
    for eval_dir in $(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "eval_*" | sort); do
        checkpoint_name=$(basename "$eval_dir" | sed 's/^eval_//')
        eval_json="${eval_dir}/${checkpoint_name}/evaluation.json"

        if [ -f "$eval_json" ]; then
            eval_results+=("$eval_json")
        fi
    done

    if [ ${#eval_results[@]} -gt 0 ]; then
        echo "Found ${#eval_results[@]} evaluated checkpoint(s)"
        echo ""

        # Create comparison summary
        comparison_file="${OUTPUT_DIR}/final_comparison_summary.txt"

        echo "=== SDXL LoRA CHECKPOINT COMPARISON ===" > "$comparison_file"
        echo "Character: ${FILM}/${CHAR_ID}" >> "$comparison_file"
        echo "Training completed: $(date '+%Y-%m-%d %H:%M:%S')" >> "$comparison_file"
        echo "" >> "$comparison_file"
        echo "Checkpoint Rankings (by Aggregate Score):" >> "$comparison_file"
        echo "" >> "$comparison_file"

        # Extract scores and rank checkpoints
        declare -A scores
        for eval_json in "${eval_results[@]}"; do
            checkpoint=$(basename "$(dirname "$eval_json")")
            aggregate=$(cat "$eval_json" | python -m json.tool 2>/dev/null | grep '"aggregate_score"' | sed 's/.*: \(.*\),*/\1/')
            clip=$(cat "$eval_json" | python -m json.tool 2>/dev/null | grep '"clip_score_mean"' | sed 's/.*: \(.*\),*/\1/')
            face=$(cat "$eval_json" | python -m json.tool 2>/dev/null | grep '"face_consistency_mean"' | sed 's/.*: \(.*\),*/\1/')
            diversity=$(cat "$eval_json" | python -m json.tool 2>/dev/null | grep '"diversity_mean"' | sed 's/.*: \(.*\),*/\1/')
            quality=$(cat "$eval_json" | python -m json.tool 2>/dev/null | grep '"quality_mean"' | sed 's/.*: \(.*\),*/\1/')

            scores["$checkpoint"]="$aggregate|$clip|$face|$diversity|$quality"
        done

        # Sort by aggregate score and display
        rank=1
        best_checkpoint=""
        for checkpoint in $(for k in "${!scores[@]}"; do echo "$k ${scores[$k]}"; done | sort -t'|' -k2 -rn | awk '{print $1}'); do
            IFS='|' read -r agg clip face div qual <<< "${scores[$checkpoint]}"

            if [ $rank -eq 1 ]; then
                best_checkpoint="$checkpoint"
                echo "🏆 RANK $rank: $checkpoint (BEST)" | tee -a "$comparison_file"
            else
                echo "   RANK $rank: $checkpoint" | tee -a "$comparison_file"
            fi

            printf "   Aggregate: %.4f | CLIP: %.4f | Face: %.4f | Diversity: %.4f | Quality: %.4f\n" \
                "$agg" "$clip" "$face" "$div" "$qual" | tee -a "$comparison_file"
            echo "" | tee -a "$comparison_file"

            ((rank++))
        done

        echo "================================================================================" | tee -a "$comparison_file"
        echo "🏆 RECOMMENDED CHECKPOINT: $best_checkpoint" | tee -a "$comparison_file"
        echo "================================================================================" | tee -a "$comparison_file"
        echo ""

        echo -e "${GREEN}✅ Comparison report saved to: $comparison_file${NC}"
        echo ""

        # Copy best checkpoint to a convenient location
        best_checkpoint_path="${OUTPUT_DIR}/${best_checkpoint}.safetensors"
        if [ -f "$best_checkpoint_path" ]; then
            cp "$best_checkpoint_path" "${OUTPUT_DIR}/BEST_${best_checkpoint}.safetensors"
            echo -e "${GREEN}✅ Best checkpoint copied to: ${OUTPUT_DIR}/BEST_${best_checkpoint}.safetensors${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  No evaluation results found${NC}"
    fi

    echo ""
fi

echo "All evaluations:"
echo "  ls -d ${OUTPUT_DIR}/eval_*"
echo ""
echo "Comparison summary:"
echo "  cat ${OUTPUT_DIR}/final_comparison_summary.txt"
echo ""
echo "================================================================================"
