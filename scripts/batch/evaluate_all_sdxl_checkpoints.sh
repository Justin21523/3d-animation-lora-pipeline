#!/bin/bash
#
# Batch SDXL LoRA Checkpoint Evaluator
# Evaluates all checkpoints for: orion, ian, caleb, tyler, bryce
#
# Usage: bash scripts/batch/evaluate_all_sdxl_checkpoints.sh
#

set -e

PROJECT_ROOT="/mnt/c/ai_projects/3d-animation-lora-pipeline"
BASE_MODEL="/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/evaluation/sdxl_lora_evaluator.py"

# Character configurations (name, lora_dir, prompt)
declare -A LORA_DIRS=(
    ["orion"]="/mnt/c/ai_models/lora_sdxl/orion/orion_identity"
    ["ian_lightfoot"]="/mnt/c/ai_models/lora_sdxl/onward/ian_lightfoot_identity"
    ["caleb"]="/mnt/c/ai_models/lora_sdxl/elio/caleb_identity"
    ["tyler"]="/mnt/c/ai_models/lora_sdxl/turning-red/tyler_identity"
    ["bryce"]="/mnt/c/ai_models/lora_sdxl/elio/bryce_identity"
)

declare -A PROMPT_FILES=(
    ["orion"]="${PROJECT_ROOT}/prompts/single_character/orion_prompts.txt"
    ["ian_lightfoot"]="${PROJECT_ROOT}/prompts/single_character/ian_lightfoot_prompts.txt"
    ["caleb"]="${PROJECT_ROOT}/prompts/single_character/caleb_prompts.txt"
    ["tyler"]="${PROJECT_ROOT}/prompts/single_character/tyler_prompts.txt"
    ["bryce"]="${PROJECT_ROOT}/prompts/single_character/bryce_prompts.txt"
)

NEGATIVE_PROMPT="multiple people, duplicate, clone, two characters, extra limbs, extra arms, extra legs, extra hands, deformed, distorted, disfigured, bad anatomy, wrong anatomy, mutation, mutated, ugly, blurry, low quality, jpeg artifacts, watermark, text, bad proportions, gross proportions"

echo "================================================================================"
echo "🎨 BATCH SDXL LoRA CHECKPOINT EVALUATION - ALL CHARACTERS"
echo "================================================================================"
echo "Characters: orion, ian_lightfoot, caleb, tyler, bryce"
echo "Base Model: $BASE_MODEL"
echo "================================================================================"
echo ""

# Log file
LOG_FILE="${PROJECT_ROOT}/logs/checkpoint_evaluation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Log file: $LOG_FILE"
echo ""

# Evaluate each character
for char_name in orion ian_lightfoot caleb tyler bryce; do
    lora_dir="${LORA_DIRS[$char_name]}"
    prompt_file="${PROMPT_FILES[$char_name]}"
    output_dir="${lora_dir}/evaluation_results"

    echo "================================================================================"
    echo "Evaluating: $char_name"
    echo "================================================================================"
    echo "LoRA directory: $lora_dir"
    echo "Output directory: $output_dir"
    echo "Prompts file: $prompt_file"
    echo ""

    # Check if directory exists
    if [ ! -d "$lora_dir" ]; then
        echo "❌ Directory not found: $lora_dir"
        echo "Skipping $char_name..."
        echo "" | tee -a "$LOG_FILE"
        continue
    fi

    # Check if prompts file exists
    if [ ! -f "$prompt_file" ]; then
        echo "❌ Prompts file not found: $prompt_file"
        echo "Skipping $char_name..."
        echo "" | tee -a "$LOG_FILE"
        continue
    fi

    # Count checkpoints
    num_checkpoints=$(find "$lora_dir" -maxdepth 1 -name "*.safetensors" -type f | wc -l)
    echo "Found $num_checkpoints checkpoint(s)"

    if [ $num_checkpoints -eq 0 ]; then
        echo "❌ No checkpoints found for $char_name"
        echo "Skipping..." | tee -a "$LOG_FILE"
        echo ""
        continue
    fi

    # List checkpoints
    echo "Checkpoints to evaluate:"
    find "$lora_dir" -maxdepth 1 -name "*.safetensors" -type f -exec basename {} \; | sort
    echo ""

    # Create output directory
    mkdir -p "$output_dir"

    # Run evaluation
    echo "Starting evaluation at: $(date '+%Y-%m-%d %H:%M:%S')"

    conda run -n ai_env python "$EVAL_SCRIPT" \
        "$lora_dir" \
        --base-model "$BASE_MODEL" \
        --output-dir "$output_dir" \
        --prompts-file "$prompt_file" \
        --negative-prompt "$NEGATIVE_PROMPT" \
        --num-images-per-prompt 4 \
        --num-inference-steps 40 \
        --guidance-scale 7.5 \
        --seed 42 \
        --device cuda 2>&1 | tee -a "$LOG_FILE"

    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✅ $char_name evaluation completed successfully"
        echo "Results: $output_dir"

        # Show best checkpoint if available
        if [ -f "$output_dir/checkpoint_comparison.json" ]; then
            echo ""
            echo "🏆 Best checkpoint for $char_name:"
            python -m json.tool "$output_dir/checkpoint_comparison.json" 2>/dev/null | grep -A 3 '"best_checkpoint"' || echo "  (see checkpoint_comparison.json)"
        fi
    else
        echo ""
        echo "❌ $char_name evaluation failed with exit code: $exit_code"
    fi

    echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "Waiting 10 seconds before next character..."
    sleep 10
    echo ""
done

echo "================================================================================"
echo "ALL CHARACTER EVALUATIONS COMPLETED"
echo "================================================================================"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log file: $LOG_FILE"
echo "================================================================================"
echo ""
echo "📊 Results Summary:"
for char_name in orion ian_lightfoot caleb tyler bryce; do
    lora_dir="${LORA_DIRS[$char_name]}"
    output_dir="${lora_dir}/evaluation_results"

    if [ -d "$output_dir" ]; then
        echo "  ✅ $char_name: $output_dir"
    else
        echo "  ❌ $char_name: NOT EVALUATED"
    fi
done
echo ""
echo "================================================================================"
