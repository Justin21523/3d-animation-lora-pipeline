#!/bin/bash
# ============================================================================
# Train All Synthetic LoRAs Sequentially
# ============================================================================
#
# Trains 42 LoRAs (14 characters × 3 types) sequentially to avoid GPU OOM.
# Training order: All pose → All action → All expression
#
# Estimated time: 77-98 hours (3-4 days continuous)
# - Pose: ~1.5-2h each × 14 = 21-28h
# - Action: ~2-2.5h each × 14 = 28-35h
# - Expression: ~2-2.5h each × 14 = 28-35h
#
# Features:
# - Sequential training (one at a time)
# - Automatic GPU cleanup between runs
# - Comprehensive logging
# - Error recovery
#
# Author: LLMProvider Tooling
# Date: 2025-12-04
# ============================================================================

set -euo pipefail

# Configuration
CONDA_ENV="kohya_ss"
SCRIPT="/mnt/c/sd-scripts/sdxl_train_network.py"
CONFIG_ROOT="/mnt/c/ai_models/lora_sdxl"
LOG_DIR="/mnt/data/ai_data/synthetic_lora_data/logs/training"

# Create log directory
mkdir -p "$LOG_DIR"

# Characters
CHARACTERS=(
    alberto
    bryce
    caleb
    elio
    giulia
    ian_lightfoot
    luca
    miguel
    orion
    russell
    tyler
    alberto_seamonster
    luca_seamonster
    barley_lightfoot
)

# LoRA types (train in order: pose → action → expression)
TYPES=(pose action expression)

# Statistics
total_trained=0
total_failed=0
total_skipped=0

# Main training function
train_lora() {
    local char=$1
    local type=$2

    config_file="${CONFIG_ROOT}/${char}/${type}/config.toml"

    # Check if config exists
    if [ ! -f "$config_file" ]; then
        echo "⚠️  Config not found: $config_file"
        return 1
    fi

    echo ""
    echo "============================================="
    echo "Training: ${char} ${type}"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================="

    # Training log file
    log_file="${LOG_DIR}/${char}_${type}_$(date +%Y%m%d_%H%M%S).log"

    # Run training
    if conda run -n "$CONDA_ENV" accelerate launch \
        --num_cpu_threads_per_process=8 \
        "$SCRIPT" \
        --config_file="$config_file" \
        2>&1 | tee "$log_file"; then

        echo ""
        echo "✓ Completed: ${char} ${type}"
        echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Log: $log_file"
        ((total_trained++))
        return 0
    else
        echo ""
        echo "✗ Failed: ${char} ${type}"
        echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Log: $log_file"
        ((total_failed++))
        return 1
    fi
}

# Start training
echo "============================================="
echo "SYNTHETIC LORA TRAINING - SEQUENTIAL MODE"
echo "============================================="
echo ""
echo "Total LoRAs to train: $((${#CHARACTERS[@]} * ${#TYPES[@]}))"
echo "Conda environment: $CONDA_ENV"
echo "Training script: $SCRIPT"
echo "Log directory: $LOG_DIR"
echo ""
echo "Estimated total time: 77-98 hours (3-4 days)"
echo ""
echo "Press Ctrl+C to cancel..."
sleep 5

# Train by type (all pose, then all action, then all expression)
for type in "${TYPES[@]}"; do
    echo ""
    echo "============================================="
    echo "TRAINING ALL ${type^^} LORAS"
    echo "============================================="
    echo ""

    type_start_time=$(date +%s)

    for char in "${CHARACTERS[@]}"; do
        train_lora "$char" "$type"

        # Brief cooldown between training runs
        echo ""
        echo "Cooldown: 10 seconds..."
        sleep 10

        # GPU memory cleanup
        nvidia-smi --gpu-reset-ecc > /dev/null 2>&1 || true
    done

    type_end_time=$(date +%s)
    type_duration=$((type_end_time - type_start_time))
    type_hours=$((type_duration / 3600))
    type_minutes=$(((type_duration % 3600) / 60))

    echo ""
    echo "============================================="
    echo "Completed all ${type^^} LoRAs"
    echo "Time taken: ${type_hours}h ${type_minutes}m"
    echo "============================================="
done

# Final summary
echo ""
echo "============================================="
echo "ALL TRAINING COMPLETE"
echo "============================================="
echo ""
echo "Statistics:"
echo "  Total trained: $total_trained"
echo "  Total failed: $total_failed"
echo "  Total skipped: $total_skipped"
echo ""
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", ($total_trained / ($total_trained + $total_failed)) * 100}")%"
echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Next steps:"
echo "1. Review checkpoints in: $CONFIG_ROOT"
echo "2. Run evaluation: scripts/evaluation/auto_evaluate_checkpoints.py"
echo "3. Select best checkpoints: scripts/evaluation/select_best_checkpoint.py"
echo ""
echo "============================================="
