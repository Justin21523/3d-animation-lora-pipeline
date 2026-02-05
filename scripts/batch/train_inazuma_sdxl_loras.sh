#!/bin/bash
#
# Inazuma Eleven SDXL LoRA Batch Training Script
# Sequential training of all 7 characters with automatic checkpoint evaluation
# Adapted from 3D character pipeline to 2D anime optimization
#
# Usage:
#   tmux new-session -s inazuma_training
#   bash scripts/batch/train_inazuma_sdxl_loras.sh
#
# Features:
#   - Sequential training (one character at a time)
#   - Automatic checkpoint saving every 2 epochs
#   - Timeline-aware evaluation prompts
#   - Progress logging and time tracking
#   - GPU memory safety checks
#   - tmux-friendly (survives disconnections)

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Paths
PROJECT_ROOT="/mnt/c/ai_projects/3d-animation-lora-pipeline"
CONFIG_DIR="$PROJECT_ROOT/configs/training/character_loras_sdxl"
LOG_DIR="$PROJECT_ROOT/logs"
KOHYA_ROOT="/mnt/c/ai_projects/kohya_ss/sd-scripts"

# Training environment
CONDA_ENV="kohya_ss"

# Character list (training order)
CHARACTERS=(
    "endou_mamoru"
    "gouenji_shuuya"
    "fudou_akio"
    "matsukaze_tenma"
    "inamori_asuto"
    "nosaka_yuuma"
    "utsunomiya_toramaru"
)

# Create log directory
mkdir -p "$LOG_DIR"

# Batch log file
BATCH_LOG="$LOG_DIR/inazuma_batch_training_$(date +%Y%m%d_%H%M%S).log"

# ============================================================================
# Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BATCH_LOG"
}

check_gpu() {
    if ! nvidia-smi &> /dev/null; then
        log "ERROR: nvidia-smi not available. GPU not detected."
        exit 1
    fi
    log "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | tee -a "$BATCH_LOG"
}

estimate_time_remaining() {
    local completed=$1
    local total=$2
    local elapsed=$3

    if [ $completed -eq 0 ]; then
        echo "N/A"
        return
    fi

    local avg_time_per_char=$((elapsed / completed))
    local remaining_chars=$((total - completed))
    local remaining_seconds=$((avg_time_per_char * remaining_chars))

    local hours=$((remaining_seconds / 3600))
    local minutes=$(((remaining_seconds % 3600) / 60))

    echo "${hours}h ${minutes}m"
}

train_character() {
    local char_id=$1
    local index=$2
    local total=$3

    local config_file="$CONFIG_DIR/inazuma_${char_id}_sdxl.toml"
    local char_log="$LOG_DIR/inazuma_${char_id}_sdxl_training_$(date +%Y%m%d_%H%M%S).log"

    log "=========================================="
    log "Training Character [$index/$total]: $char_id"
    log "Config: $config_file"
    log "Log: $char_log"
    log "=========================================="

    # Verify config exists
    if [ ! -f "$config_file" ]; then
        log "ERROR: Config file not found: $config_file"
        return 1
    fi

    # Check GPU before training
    check_gpu

    # Start training
    local start_time=$(date +%s)
    log "Starting training at $(date '+%Y-%m-%d %H:%M:%S')"

    # Run Kohya SDXL training script
    cd "$KOHYA_ROOT"
    PYTHONUNBUFFERED=1 conda run -n "$CONDA_ENV" python -u sdxl_train_network.py \
        --config_file "$config_file" \
        2>&1 | tee "$char_log"

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))

    if [ $exit_code -eq 0 ]; then
        log "✓ Training completed successfully for $char_id"
        log "  Duration: ${hours}h ${minutes}m"
    else
        log "✗ Training failed for $char_id (exit code: $exit_code)"
        log "  Check log: $char_log"
        return $exit_code
    fi

    return 0
}

# ============================================================================
# Main Training Loop
# ============================================================================

log "=========================================="
log "Inazuma Eleven SDXL LoRA Batch Training"
log "=========================================="
log "Total characters: ${#CHARACTERS[@]}"
log "Configuration directory: $CONFIG_DIR"
log "Log directory: $LOG_DIR"
log "Kohya root: $KOHYA_ROOT"
log "Conda environment: $CONDA_ENV"
log ""

# Initial GPU check
check_gpu
log ""

# Training loop
total_chars=${#CHARACTERS[@]}
completed_chars=0
failed_chars=()
batch_start_time=$(date +%s)

for i in "${!CHARACTERS[@]}"; do
    char_id="${CHARACTERS[$i]}"
    index=$((i + 1))

    # Train character
    if train_character "$char_id" "$index" "$total_chars"; then
        completed_chars=$((completed_chars + 1))
        log ""
        log "Progress: $completed_chars/$total_chars characters completed"

        # Estimate remaining time
        batch_elapsed=$(($(date +%s) - batch_start_time))
        remaining_time=$(estimate_time_remaining $completed_chars $total_chars $batch_elapsed)
        log "Estimated time remaining: $remaining_time"
        log ""
    else
        failed_chars+=("$char_id")
        log "WARNING: Training failed for $char_id. Continuing with next character..."
        log ""
    fi

    # Small delay between characters
    if [ $index -lt $total_chars ]; then
        log "Waiting 30 seconds before next character..."
        sleep 30
    fi
done

# ============================================================================
# Final Report
# ============================================================================

batch_end_time=$(date +%s)
total_duration=$((batch_end_time - batch_start_time))
total_hours=$((total_duration / 3600))
total_minutes=$(((total_duration % 3600) / 60))

log "=========================================="
log "Batch Training Completed"
log "=========================================="
log "Total characters: $total_chars"
log "Successfully trained: $completed_chars"
log "Failed: ${#failed_chars[@]}"

if [ ${#failed_chars[@]} -gt 0 ]; then
    log "Failed characters:"
    for char in "${failed_chars[@]}"; do
        log "  - $char"
    done
fi

log ""
log "Total training time: ${total_hours}h ${total_minutes}m"
log "Average time per character: $((total_duration / total_chars / 60)) minutes"
log ""
log "Batch log saved to: $BATCH_LOG"
log ""

# Check output directories
log "Checking output directories..."
for char_id in "${CHARACTERS[@]}"; do
    output_dir="/mnt/data/ai_data/models/lora_sdxl/inazuma-eleven/${char_id}_identity"
    if [ -d "$output_dir" ]; then
        checkpoint_count=$(find "$output_dir" -name "*.safetensors" -type f 2>/dev/null | wc -l)
        log "  $char_id: $checkpoint_count checkpoints found"
    else
        log "  $char_id: ⚠ Output directory not found"
    fi
done

log ""
log "=========================================="
log "Next Steps:"
log "=========================================="
log "1. Review training logs in: $LOG_DIR"
log "2. Evaluate checkpoints:"
log "   python scripts/evaluation/sdxl_lora_evaluator.py \\"
log "     --lora-dir /mnt/data/ai_data/models/lora_sdxl/inazuma-eleven"
log "3. Test with timeline-specific prompts:"
log "   - 'inazuma_endou_mamoru, timeline_original, soccer_uniform'"
log "   - 'inazuma_endou_mamoru, timeline_go, adult, coach_outfit'"
log "=========================================="

if [ ${#failed_chars[@]} -eq 0 ]; then
    log "✓ All characters trained successfully!"
    exit 0
else
    log "⚠ Some characters failed. Check logs for details."
    exit 1
fi
