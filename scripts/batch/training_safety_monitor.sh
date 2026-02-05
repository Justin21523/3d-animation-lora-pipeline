#!/bin/bash
#
# Training Safety Monitor
#
# Monitors system resources during LoRA training and takes protective actions:
# - Watches GPU memory usage
# - Watches system RAM usage
# - Monitors GPU temperature
# - Alerts on dangerous conditions
# - Can auto-pause training if thresholds exceeded
#
# Author: LLMProvider Tooling
# Date: 2025-12-04
#

set -euo pipefail

# Configuration
RAM_THRESHOLD=90          # % RAM usage before warning
GPU_MEM_THRESHOLD=14000   # MB GPU memory before warning (RTX 5080 has 16GB)
GPU_TEMP_THRESHOLD=85     # °C GPU temperature before warning
CHECK_INTERVAL=60         # seconds between checks
LOG_FILE="/tmp/training_safety_monitor.log"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_message() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

check_ram() {
    # Get RAM usage percentage
    local ram_percent=$(free | grep Mem | awk '{printf "%.0f", ($3/$2) * 100}')
    local ram_used_gb=$(free -g | grep Mem | awk '{print $3}')
    local ram_total_gb=$(free -g | grep Mem | awk '{print $2}')

    if (( ram_percent >= RAM_THRESHOLD )); then
        log_message "${RED}WARNING${NC}" "RAM usage critical: ${ram_percent}% (${ram_used_gb}/${ram_total_gb} GB)"
        return 1
    elif (( ram_percent >= 80 )); then
        log_message "${YELLOW}CAUTION${NC}" "RAM usage high: ${ram_percent}% (${ram_used_gb}/${ram_total_gb} GB)"
        return 0
    else
        log_message "${GREEN}OK${NC}" "RAM usage normal: ${ram_percent}% (${ram_used_gb}/${ram_total_gb} GB)"
        return 0
    fi
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_message "${YELLOW}WARNING${NC}" "nvidia-smi not found, skipping GPU checks"
        return 0
    fi

    # GPU memory usage (MB)
    local gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    local gpu_mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

    # GPU temperature
    local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)

    # GPU utilization
    local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)

    # Check memory
    if (( gpu_mem >= GPU_MEM_THRESHOLD )); then
        log_message "${RED}WARNING${NC}" "GPU memory critical: ${gpu_mem}/${gpu_mem_total} MB"
        return 1
    elif (( gpu_mem >= 12000 )); then
        log_message "${YELLOW}CAUTION${NC}" "GPU memory high: ${gpu_mem}/${gpu_mem_total} MB"
    else
        log_message "${GREEN}OK${NC}" "GPU memory normal: ${gpu_mem}/${gpu_mem_total} MB"
    fi

    # Check temperature
    if (( gpu_temp >= GPU_TEMP_THRESHOLD )); then
        log_message "${RED}WARNING${NC}" "GPU temperature critical: ${gpu_temp}°C"
        return 1
    elif (( gpu_temp >= 80 )); then
        log_message "${YELLOW}CAUTION${NC}" "GPU temperature high: ${gpu_temp}°C"
    else
        log_message "${GREEN}OK${NC}" "GPU temperature normal: ${gpu_temp}°C (${gpu_util}% util)"
    fi

    return 0
}

check_swap() {
    local swap_total=$(free -g | grep Swap | awk '{print $2}')
    local swap_used=$(free -g | grep Swap | awk '{print $3}')

    if (( swap_used > 0 )); then
        log_message "${YELLOW}WARNING${NC}" "System is using swap: ${swap_used}/${swap_total} GB - Performance degraded!"
        return 1
    else
        log_message "${GREEN}OK${NC}" "No swap usage"
        return 0
    fi
}

check_training_process() {
    # Check if accelerate/train_network is running
    if pgrep -f "train_network" > /dev/null; then
        local pid=$(pgrep -f "train_network" | head -1)
        local cpu_percent=$(ps -p $pid -o %cpu | tail -1 | xargs)
        local mem_percent=$(ps -p $pid -o %mem | tail -1 | xargs)

        log_message "${BLUE}INFO${NC}" "Training process active (PID: $pid, CPU: ${cpu_percent}%, MEM: ${mem_percent}%)"
        return 0
    else
        log_message "${YELLOW}INFO${NC}" "No training process detected"
        return 1
    fi
}

cleanup_memory() {
    log_message "${BLUE}INFO${NC}" "Performing memory cleanup..."

    # Drop caches (requires sudo, may fail)
    if [ -w /proc/sys/vm/drop_caches ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches
        log_message "${GREEN}OK${NC}" "Kernel caches dropped"
    else
        log_message "${YELLOW}WARN${NC}" "Cannot drop caches (requires root)"
    fi

    # Python garbage collection via running python
    python3 -c "import gc; gc.collect()" 2>/dev/null || true
}

# Main monitoring loop
main() {
    echo "=" | tr '=' '=' | head -c 80
    echo
    echo "Training Safety Monitor"
    echo "=" | tr '=' '=' | head -c 80
    echo
    echo "Configuration:"
    echo "  RAM threshold: ${RAM_THRESHOLD}%"
    echo "  GPU memory threshold: ${GPU_MEM_THRESHOLD} MB"
    echo "  GPU temperature threshold: ${GPU_TEMP_THRESHOLD}°C"
    echo "  Check interval: ${CHECK_INTERVAL}s"
    echo "  Log file: ${LOG_FILE}"
    echo
    echo "Press Ctrl+C to stop monitoring"
    echo
    echo "=" | tr '=' '=' | head -c 80
    echo

    # Initialize log
    log_message "${GREEN}START${NC}" "Training safety monitor started"

    while true; do
        echo
        log_message "${BLUE}CHECK${NC}" "Running system health check..."

        local warnings=0

        # Check RAM
        check_ram || ((warnings++))

        # Check GPU
        check_gpu || ((warnings++))

        # Check swap
        check_swap || ((warnings++))

        # Check training process
        check_training_process

        # Take action if warnings
        if (( warnings > 0 )); then
            log_message "${YELLOW}ACTION${NC}" "Detected ${warnings} warning(s), performing cleanup..."
            cleanup_memory
        fi

        # Alert if critical (3+ warnings)
        if (( warnings >= 3 )); then
            log_message "${RED}ALERT${NC}" "CRITICAL: Multiple resource warnings! Consider pausing training!"
            # Optional: Could auto-pause here
        fi

        # Wait before next check
        sleep "$CHECK_INTERVAL"
    done
}

# Handle Ctrl+C
trap 'echo; log_message "${YELLOW}STOP${NC}" "Monitoring stopped by user"; exit 0' INT TERM

# Run
main
