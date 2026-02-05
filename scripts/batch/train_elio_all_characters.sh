#!/bin/bash
# Train All Elio Character Identity LoRAs
# Runs all 5 characters in sequence
# Date: 2025-11-21

KOHYA_SCRIPTS="/mnt/c/AI_LLM_projects/kohya_ss/sd-scripts"
CONFIG_DIR="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/character_loras"
LOG_DIR="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/logs"

echo "=========================================="
echo "Elio Character LoRA Training - Batch Mode"
echo "=========================================="
echo ""
echo "Training 5 characters in sequence:"
echo "  1. Elio (679 images, 12 epochs, ~2-3 hours)"
echo "  2. Glordon (201 images, 15 epochs, ~2-3 hours)"
echo "  3. Bryce (201 images, 15 epochs, ~2-3 hours)"
echo "  4. Olga (204 images, 15 epochs, ~2-3 hours)"
echo "  5. Caleb (195 images, 18 epochs, ~2-3 hours)"
echo ""
echo "Total estimated time: 10-15 hours"
echo ""
read -p "Press Enter to start training..."

# Array of characters and their configs
declare -a CHARACTERS=(
    "elio:elio_elio_identity.toml"
    "glordon:elio_glordon_identity.toml"
    "bryce:elio_bryce_identity.toml"
    "olga:elio_olga_identity.toml"
    "caleb:elio_caleb_identity.toml"
)

# Process each character
for item in "${CHARACTERS[@]}"; do
    char="${item%%:*}"
    config="${item##*:}"

    echo ""
    echo "=========================================="
    echo "Training: $char"
    echo "Config: $config"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    # Train with Kohya
    cd "$KOHYA_SCRIPTS"
    conda run -n kohya_ss accelerate launch \
        --num_cpu_threads_per_process=4 \
        train_network.py \
        --config_file="$CONFIG_DIR/$config" \
        2>&1 | tee "$LOG_DIR/train_${char}_identity.log"

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✅ $char training completed successfully"
        echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo ""
        echo "❌ $char training failed with exit code $exit_code"
        echo "Check log: $LOG_DIR/train_${char}_identity.log"

        read -p "Continue to next character? (y/n): " continue_choice
        if [[ ! "$continue_choice" =~ ^[Yy]$ ]]; then
            echo "Training stopped by user"
            exit 1
        fi
    fi

    echo ""
done

echo ""
echo "=========================================="
echo "All Elio Character Training Complete!"
echo "=========================================="
echo ""
echo "Summary:"
for item in "${CHARACTERS[@]}"; do
    char="${item%%:*}"
    checkpoint_dir="/mnt/data/ai_data/models/lora/elio/${char}_identity"
    if [ -d "$checkpoint_dir" ]; then
        checkpoint_count=$(ls -1 "$checkpoint_dir"/*.safetensors 2>/dev/null | wc -l)
        echo "  $char: $checkpoint_count checkpoints"
    else
        echo "  $char: No checkpoints found"
    fi
done
echo ""
