#!/bin/bash
#
# Sequential SDXL Training with Auto-Evaluation
# Supports flexible character configuration
#
# Usage:
#   bash train_all_sdxl_sequential.sh                              # Train all 12 default characters
#   bash train_all_sdxl_sequential.sh coco elio luca              # Train specific movies
#   bash train_all_sdxl_sequential.sh miguel bryce alberto        # Train specific characters
#   bash train_all_sdxl_sequential.sh --config custom_list.txt    # Train from config file
#

set -e

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
cd "$PROJECT_ROOT"

# Default training order (all 12 characters)
DEFAULT_CONFIGS=(
    "configs/training/character_loras_sdxl/coco_miguel_identity_sdxl.toml"
    "configs/training/character_loras_sdxl/elio_bryce_identity_sdxl.toml"
    "configs/training/character_loras_sdxl/elio_caleb_identity_sdxl.toml"
    "configs/training/character_loras_sdxl/elio_elio_identity_sdxl.toml"
    "configs/training/character_loras_sdxl/elio_glordon_identity_sdxl.toml"
    "configs/training/character_loras_sdxl/luca_alberto_identity_sdxl.toml"
    "configs/training/character_loras_sdxl/luca_giulia_identity_sdxl.toml"
    "configs/training/character_loras_sdxl/onward_barley_lightfoot_identity_sdxl.toml"
    "configs/training/character_loras_sdxl/onward_ian_lightfoot_identity_sdxl.toml"
    "configs/training/character_loras_sdxl/orion_orion_identity_sdxl.toml"
    "configs/training/character_loras_sdxl/turning-red_tyler_identity_sdxl.toml"
    "configs/training/character_loras_sdxl/up_russell_identity_sdxl.toml"
)

# Function to find config by character/movie name
find_config() {
    local name="$1"
    local found=""

    # Search for exact match
    for config in "${DEFAULT_CONFIGS[@]}"; do
        if [[ "$config" == *"${name}_identity_sdxl.toml" ]] || [[ "$config" == *"${name}_"*"_identity_sdxl.toml" ]]; then
            echo "$config"
            return 0
        fi
    done

    # Search for partial match
    for config in "${DEFAULT_CONFIGS[@]}"; do
        if [[ "$config" == *"$name"* ]]; then
            echo "$config"
            return 0
        fi
    done

    return 1
}

# Parse arguments
CONFIGS=()

if [ $# -eq 0 ]; then
    # No arguments: use all default configs
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
    echo "Using all 12 default characters"
elif [ "$1" == "--config" ] && [ -f "$2" ]; then
    # Load from config file
    echo "Loading character list from: $2"
    while IFS= read -r line; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^# ]] && continue

        # If line is a full path, use it directly
        if [ -f "$line" ]; then
            CONFIGS+=("$line")
        else
            # Try to find matching config
            config=$(find_config "$line")
            if [ -n "$config" ]; then
                CONFIGS+=("$config")
            else
                echo "⚠️  Warning: No config found for '$line', skipping"
            fi
        fi
    done < "$2"
elif [ "$1" == "--list" ]; then
    # List all available characters
    echo "Available characters:"
    for config in "${DEFAULT_CONFIGS[@]}"; do
        char_name=$(basename "$config" .toml | sed 's/_identity_sdxl//')
        echo "  - $char_name"
    done
    exit 0
else
    # Parse character/movie names from arguments
    echo "Looking for specified characters: $@"
    for arg in "$@"; do
        # Check if it's a full path
        if [ -f "$arg" ]; then
            CONFIGS+=("$arg")
            echo "  ✓ Added: $arg"
            continue
        fi

        # Try to find matching config
        config=$(find_config "$arg")
        if [ -n "$config" ]; then
            CONFIGS+=("$config")
            echo "  ✓ Found: $(basename "$config" .toml)"
        else
            echo "  ✗ Not found: $arg"
        fi
    done
fi

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "❌ Error: No valid configurations found"
    echo ""
    echo "Usage:"
    echo "  $0                              # Train all 12 characters"
    echo "  $0 miguel bryce alberto        # Train specific characters"
    echo "  $0 coco elio luca              # Train all characters from movies"
    echo "  $0 --config custom_list.txt    # Train from file"
    echo "  $0 --list                      # List available characters"
    exit 1
fi

TOTAL_CHARS=${#CONFIGS[@]}
CURRENT=0
BATCH_LOG="${PROJECT_ROOT}/logs/batch_training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$BATCH_LOG")"

echo "================================================================================" | tee -a "$BATCH_LOG"
echo "🚀 BATCH SDXL TRAINING WITH AUTO-EVALUATION" | tee -a "$BATCH_LOG"
echo "================================================================================" | tee -a "$BATCH_LOG"
echo "Total characters: $TOTAL_CHARS" | tee -a "$BATCH_LOG"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"
echo "Training queue:" | tee -a "$BATCH_LOG"
for i in "${!CONFIGS[@]}"; do
    char_name=$(basename "${CONFIGS[$i]}" .toml | sed 's/_identity_sdxl//')
    echo "  $((i+1)). $char_name" | tee -a "$BATCH_LOG"
done
echo "================================================================================" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"
echo "⚙️  Each character will:" | tee -a "$BATCH_LOG"
echo "   1. Train for 10 epochs (~6-8 hours per character)" | tee -a "$BATCH_LOG"
echo "   2. Save checkpoints every 2 epochs (Epoch 2, 4, 6, 8, 10)" | tee -a "$BATCH_LOG"
echo "   3. Auto-evaluate EACH checkpoint immediately after saving" | tee -a "$BATCH_LOG"
echo "   4. Generate comprehensive comparison report at the end" | tee -a "$BATCH_LOG"
echo "   5. Copy the BEST checkpoint to BEST_*.safetensors" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"
echo "📊 Estimated total time: $((TOTAL_CHARS * 7))h - $((TOTAL_CHARS * 9))h" | tee -a "$BATCH_LOG"
echo "================================================================================" | tee -a "$BATCH_LOG"

BATCH_START=$(date +%s)

# Track success/failure
SUCCEEDED=()
FAILED=()

for config in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    CHAR_NAME=$(basename "$config" .toml | sed 's/_identity_sdxl//')

    echo "" | tee -a "$BATCH_LOG"
    echo "================================================================================" | tee -a "$BATCH_LOG"
    echo "[$CURRENT/$TOTAL_CHARS] Training: $CHAR_NAME" | tee -a "$BATCH_LOG"
    echo "Config: $config" | tee -a "$BATCH_LOG"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$BATCH_LOG"
    echo "================================================================================" | tee -a "$BATCH_LOG"

    CHAR_START=$(date +%s)

    # Run training with auto-evaluation
    if bash "${PROJECT_ROOT}/scripts/batch/train_sdxl_with_auto_eval.sh" "$config"; then
        CHAR_END=$(date +%s)
        CHAR_DURATION=$((CHAR_END - CHAR_START))

        echo "" | tee -a "$BATCH_LOG"
        echo "✅ $CHAR_NAME completed successfully in $((CHAR_DURATION / 3600))h $((CHAR_DURATION % 3600 / 60))m" | tee -a "$BATCH_LOG"

        SUCCEEDED+=("$CHAR_NAME")
    else
        CHAR_END=$(date +%s)
        CHAR_DURATION=$((CHAR_END - CHAR_START))

        echo "" | tee -a "$BATCH_LOG"
        echo "❌ $CHAR_NAME FAILED after $((CHAR_DURATION / 3600))h $((CHAR_DURATION % 3600 / 60))m" | tee -a "$BATCH_LOG"
        echo "⚠️  Continuing with next character..." | tee -a "$BATCH_LOG"

        FAILED+=("$CHAR_NAME")
    fi

    # Show progress
    ELAPSED=$((CHAR_END - BATCH_START))
    REMAINING=$((TOTAL_CHARS - CURRENT))
    if [ $CURRENT -gt 0 ]; then
        EST=$((ELAPSED / CURRENT * REMAINING))
        echo "⏱️  Estimated remaining time: $((EST / 3600))h $((EST % 3600 / 60))m" | tee -a "$BATCH_LOG"
    fi

    echo "📊 Progress: ${#SUCCEEDED[@]} succeeded, ${#FAILED[@]} failed, $REMAINING remaining" | tee -a "$BATCH_LOG"
done

BATCH_END=$(date +%s)
TOTAL=$((BATCH_END - BATCH_START))

echo "" | tee -a "$BATCH_LOG"
echo "================================================================================" | tee -a "$BATCH_LOG"
echo "🎉 BATCH TRAINING COMPLETE" | tee -a "$BATCH_LOG"
echo "================================================================================" | tee -a "$BATCH_LOG"
echo "Total time: $((TOTAL / 3600))h $((TOTAL % 3600 / 60))m" | tee -a "$BATCH_LOG"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"
echo "📊 Summary:" | tee -a "$BATCH_LOG"
echo "  ✅ Succeeded: ${#SUCCEEDED[@]}/${TOTAL_CHARS}" | tee -a "$BATCH_LOG"
echo "  ❌ Failed:    ${#FAILED[@]}/${TOTAL_CHARS}" | tee -a "$BATCH_LOG"

if [ ${#SUCCEEDED[@]} -gt 0 ]; then
    echo "" | tee -a "$BATCH_LOG"
    echo "✅ Successful characters:" | tee -a "$BATCH_LOG"
    for char in "${SUCCEEDED[@]}"; do
        echo "   - $char" | tee -a "$BATCH_LOG"
    done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "" | tee -a "$BATCH_LOG"
    echo "❌ Failed characters:" | tee -a "$BATCH_LOG"
    for char in "${FAILED[@]}"; do
        echo "   - $char" | tee -a "$BATCH_LOG"
    done
    echo "" | tee -a "$BATCH_LOG"
    echo "⚠️  Please check individual logs for failure details" | tee -a "$BATCH_LOG"
fi

echo "================================================================================" | tee -a "$BATCH_LOG"

# Exit with error if any failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
