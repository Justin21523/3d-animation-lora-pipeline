#!/bin/bash
# ============================================================================
# Monitor Synthetic LoRA Training Progress
# ============================================================================
#
# Real-time monitoring dashboard for training progress.
# Updates every 5 seconds with:
# - GPU status (utilization, memory, temperature)
# - Active training processes
# - Recent checkpoints
# - Training log tail
# - Progress estimation
#
# Usage: bash scripts/batch/monitor_synthetic_training.sh
#
# Author: LLMProvider Tooling
# Date: 2025-12-04
# ============================================================================

# Configuration
CHECKPOINT_DIR="/mnt/c/ai_models/lora_sdxl"
LOG_DIR="/mnt/data/ai_data/synthetic_lora_data/logs/training"
REFRESH_INTERVAL=5  # seconds

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Clear screen function
clear_screen() {
    clear
}

# Get GPU info
get_gpu_info() {
    echo -e "${BLUE}=== GPU Status ===${NC}"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    awk -F, '{
        printf "GPU %d: %s\n", $1, $2
        printf "  Utilization: %d%% | Memory: %d/%d MB | Temp: %d°C\n", $3, $4, $5, $6
    }'
    echo ""
}

# Get active training
get_active_training() {
    echo -e "${BLUE}=== Active Training ===${NC}"

    active=$(ps aux | grep "sdxl_train_network.py" | grep -v grep | tail -1)

    if [ -n "$active" ]; then
        # Extract character/type from process
        echo "$active" | awk '{
            for(i=1; i<=NF; i++) {
                if ($i ~ /config_file/) {
                    split($(i+1), path, "/")
                    char = path[length(path)-1]
                    type = path[length(path)]
                    gsub(/config\.toml/, "", type)
                    printf "  "
                    printf "Training: %s/%s\n", char, type
                }
            }
        }'

        # Get PID and runtime
        pid=$(echo "$active" | awk '{print $2}')
        runtime=$(ps -p $pid -o etime= | xargs)
        echo "  PID: $pid | Runtime: $runtime"
    else
        echo "  No active training"
    fi
    echo ""
}

# Get recent checkpoints
get_recent_checkpoints() {
    echo -e "${BLUE}=== Recent Checkpoints (last 10, past 2h) ===${NC}"

    find "$CHECKPOINT_DIR" -name "*.safetensors" -type f -mmin -120 2>/dev/null | \
    sort -r | \
    head -10 | \
    while read -r ckpt; do
        size=$(du -h "$ckpt" | cut -f1)
        time=$(stat -c %y "$ckpt" | cut -d'.' -f1)
        name=$(basename "$ckpt")
        echo "  [$time] $name ($size)"
    done

    if [ ! -s <(find "$CHECKPOINT_DIR" -name "*.safetensors" -type f -mmin -120 2>/dev/null) ]; then
        echo "  No recent checkpoints"
    fi
    echo ""
}

# Get training log tail
get_log_tail() {
    echo -e "${BLUE}=== Training Logs (last 15 lines) ===${NC}"

    # Find most recent log file
    recent_log=$(find "$LOG_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | \
                 sort -rn | \
                 head -1 | \
                 cut -d' ' -f2-)

    if [ -n "$recent_log" ]; then
        echo "  Log: $(basename "$recent_log")"
        echo "  ---"
        tail -15 "$recent_log" 2>/dev/null | sed 's/^/  /'
    else
        echo "  No log files found"
    fi
    echo ""
}

# Get progress estimation
get_progress() {
    echo -e "${BLUE}=== Progress Estimation ===${NC}"

    # Count total checkpoints
    total_checkpoints=$(find "$CHECKPOINT_DIR" -name "*.safetensors" -type f 2>/dev/null | wc -l)

    # Estimate total expected (42 LoRAs × ~13 checkpoints each)
    expected_total=546
    progress_pct=$(awk "BEGIN {printf \"%.1f\", ($total_checkpoints / $expected_total) * 100}")

    echo "  Total checkpoints: $total_checkpoints / ~$expected_total"
    echo "  Estimated progress: $progress_pct%"

    # Estimate remaining time (rough)
    if [ $total_checkpoints -gt 0 ]; then
        # Get oldest checkpoint time
        oldest=$(find "$CHECKPOINT_DIR" -name "*.safetensors" -type f -printf '%T@\n' 2>/dev/null | \
                 sort -n | head -1)

        if [ -n "$oldest" ]; then
            now=$(date +%s)
            elapsed=$((now - ${oldest%.*}))
            elapsed_hours=$((elapsed / 3600))

            if [ $elapsed_hours -gt 0 ]; then
                avg_time_per_ckpt=$((elapsed / total_checkpoints))
                remaining_ckpts=$((expected_total - total_checkpoints))
                remaining_seconds=$((remaining_ckpts * avg_time_per_ckpt))
                remaining_hours=$((remaining_seconds / 3600))

                echo "  Elapsed: ${elapsed_hours}h"
                echo "  Estimated remaining: ~${remaining_hours}h"
            fi
        fi
    fi
    echo ""
}

# Main monitoring loop
main() {
    echo -e "${GREEN}"
    echo "============================================="
    echo "SYNTHETIC LORA TRAINING MONITOR"
    echo "============================================="
    echo -e "${NC}"
    echo "Refresh interval: ${REFRESH_INTERVAL}s"
    echo "Press Ctrl+C to exit"
    echo ""
    sleep 2

    while true; do
        clear_screen

        echo -e "${GREEN}=== Synthetic LoRA Training Monitor ===${NC}"
        echo "Last update: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""

        get_gpu_info
        get_active_training
        get_recent_checkpoints
        get_progress
        get_log_tail

        echo -e "${YELLOW}Press Ctrl+C to exit | Refreshing in ${REFRESH_INTERVAL}s...${NC}"

        sleep $REFRESH_INTERVAL
    done
}

# Run monitor
main
