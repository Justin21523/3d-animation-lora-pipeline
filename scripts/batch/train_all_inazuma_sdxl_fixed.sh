#!/bin/bash
# Train all 7 Inazuma Eleven characters sequentially with fixed config
# Prevents NaN loss with proper gradient clipping and stable learning rates

set -e

# Configuration
KOHYA_ROOT="/mnt/c/ai_projects/kohya_ss/sd-scripts"
CONDA_ENV="kohya_ss"
CONFIG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl"
LOG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_LOG="$LOG_DIR/inazuma_batch_training_FIXED_${TIMESTAMP}.log"

# Character list
CHARACTERS=(
    "endou_mamoru"
    "gouenji_shuuya"
    "fudou_akio"
    "matsukaze_tenma"
    "inamori_asuto"
    "nosaka_yuuma"
    "utsunomiya_toramaru"
)

echo "======================================================================" | tee -a "$BATCH_LOG"
echo "🚀 Inazuma Eleven SDXL LoRA Batch Training (FIXED)" | tee -a "$BATCH_LOG"
echo "======================================================================" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"
echo "Fixes applied:" | tee -a "$BATCH_LOG"
echo "  - Learning rate: 8e-5 → 5e-5" | tee -a "$BATCH_LOG"
echo "  - Added max_grad_norm = 1.0 (gradient clipping)" | tee -a "$BATCH_LOG"
echo "  - Disabled min_snr_gamma (was causing NaN)" | tee -a "$BATCH_LOG"
echo "  - Disabled xformers (not available)" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"
echo "Characters: ${#CHARACTERS[@]}" | tee -a "$BATCH_LOG"
echo "Estimated time: ~9 hours per character = ~63 hours total" | tee -a "$BATCH_LOG"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"

# Change to Kohya directory
cd "$KOHYA_ROOT"

# Train each character
for idx in "${!CHARACTERS[@]}"; do
    char_id="${CHARACTERS[$idx]}"
    char_num=$((idx + 1))

    echo "======================================================================" | tee -a "$BATCH_LOG"
    echo "[$char_num/${#CHARACTERS[@]}] Training: $char_id" | tee -a "$BATCH_LOG"
    echo "======================================================================" | tee -a "$BATCH_LOG"

    config_file="$CONFIG_DIR/inazuma_${char_id}_sdxl.toml"

    if [ ! -f "$config_file" ]; then
        echo "❌ Config not found: $config_file" | tee -a "$BATCH_LOG"
        continue
    fi

    echo "Config: $config_file" | tee -a "$BATCH_LOG"
    char_log="$LOG_DIR/inazuma_${char_id}_sdxl_training_FIXED_${TIMESTAMP}.log"
    echo "Log: $char_log" | tee -a "$BATCH_LOG"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$BATCH_LOG"
    echo "" | tee -a "$BATCH_LOG"

    # Run training with unbuffered output
    if PYTHONUNBUFFERED=1 conda run -n "$CONDA_ENV" python -u sdxl_train_network.py \
        --config_file "$config_file" \
        2>&1 | tee "$char_log"; then

        echo "" | tee -a "$BATCH_LOG"
        echo "✅ $char_id completed at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$BATCH_LOG"
        echo "" | tee -a "$BATCH_LOG"
    else
        echo "" | tee -a "$BATCH_LOG"
        echo "❌ $char_id FAILED at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$BATCH_LOG"
        echo "Check log: $char_log" | tee -a "$BATCH_LOG"
        echo "" | tee -a "$BATCH_LOG"
    fi

    # Show progress
    completed=$((idx + 1))
    remaining=$((${#CHARACTERS[@]} - completed))
    echo "Progress: $completed/${#CHARACTERS[@]} completed, $remaining remaining" | tee -a "$BATCH_LOG"
    echo "" | tee -a "$BATCH_LOG"

    # Brief pause between characters
    if [ $completed -lt ${#CHARACTERS[@]} ]; then
        echo "⏸ Cooling down for 10 seconds..." | tee -a "$BATCH_LOG"
        sleep 10
    fi
done

echo "======================================================================" | tee -a "$BATCH_LOG"
echo "✅ ALL TRAINING COMPLETED!" | tee -a "$BATCH_LOG"
echo "======================================================================" | tee -a "$BATCH_LOG"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"
echo "Summary:" | tee -a "$BATCH_LOG"
echo "  Batch log: $BATCH_LOG" | tee -a "$BATCH_LOG"
echo "  Individual logs: $LOG_DIR/inazuma_*_FIXED_${TIMESTAMP}.log" | tee -a "$BATCH_LOG"
echo "" | tee -a "$BATCH_LOG"
echo "Next steps:" | tee -a "$BATCH_LOG"
echo "  1. Review logs for any NaN loss (should be gone)" | tee -a "$BATCH_LOG"
echo "  2. Test checkpoints with evaluation script" | tee -a "$BATCH_LOG"
echo "  3. Select best epoch per character" | tee -a "$BATCH_LOG"
