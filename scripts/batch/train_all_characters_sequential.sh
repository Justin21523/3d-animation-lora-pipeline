#!/bin/bash
# Sequential training script for all 5 characters
# Order: orion → ian → caleb → tyler → bryce

set -e

PROJECT_ROOT="/mnt/c/ai_projects/3d-animation-lora-pipeline"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/batch/train_sdxl_with_auto_eval.sh"

# Character configs in order
CONFIGS=(
    "$PROJECT_ROOT/configs/training/character_loras_sdxl/orion_orion_sdxl.toml"
    "$PROJECT_ROOT/configs/training/character_loras_sdxl/onward_ian_lightfoot_sdxl.toml"
    "$PROJECT_ROOT/configs/training/character_loras_sdxl/elio_caleb_sdxl.toml"
    "$PROJECT_ROOT/configs/training/character_loras_sdxl/turning-red_tyler_sdxl.toml"
    "$PROJECT_ROOT/configs/training/character_loras_sdxl/elio_bryce_sdxl.toml"
)

CHARACTER_NAMES=(
    "orion"
    "ian_lightfoot"
    "caleb"
    "tyler"
    "bryce"
)

echo "================================================================================"
echo "SEQUENTIAL CHARACTER LORA TRAINING"
echo "================================================================================"
echo "Total characters: ${#CONFIGS[@]}"
echo "Training order:"
for i in "${!CHARACTER_NAMES[@]}"; do
    echo "  $((i+1)). ${CHARACTER_NAMES[$i]}"
done
echo ""
echo "Each character will train for 5 epochs with checkpoint testing after each epoch"
echo "================================================================================"
echo ""

# Log file
LOG_FILE="$PROJECT_ROOT/logs/sequential_training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Master log: $LOG_FILE"
echo ""

# Train each character
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    char_name="${CHARACTER_NAMES[$i]}"
    
    echo "================================================================================"
    echo "[$((i+1))/${#CONFIGS[@]}] TRAINING: $char_name"
    echo "================================================================================"
    echo "Config: $config"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Run training
    bash "$TRAIN_SCRIPT" "$config" 2>&1 | tee -a "$LOG_FILE"
    
    exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✅ $char_name training completed successfully"
        echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    else
        echo ""
        echo "❌ $char_name training failed with exit code: $exit_code"
        echo "Failed at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        echo "Do you want to continue with the next character? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Training sequence aborted."
            exit 1
        fi
    fi
    
    echo ""
done

echo "================================================================================"
echo "ALL TRAINING COMPLETED"
echo "================================================================================"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Master log: $LOG_FILE"
echo "================================================================================"
