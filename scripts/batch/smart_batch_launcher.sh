#!/bin/bash
# Smart Batch Launcher - Waits for GPU to be available before starting batch processing
# This ensures no GPU competition with existing processes (e.g., Luca SAM2)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-configs/batch/sam2_lama.yaml}"

# GPU memory threshold (MB) - start batch only if GPU memory < this value
GPU_MEM_THRESHOLD=5000  # 5GB

# Process name patterns to check
PROCESS_PATTERNS=(
    "instance_segmentation.py"
    "sam2_background_inpainting.py"
)

echo "======================================================================="
echo "🧠 Smart Batch Launcher"
echo "======================================================================="
echo ""
echo "Config: $CONFIG"
echo "GPU Memory Threshold: ${GPU_MEM_THRESHOLD} MB"
echo ""

cd "$PROJECT_ROOT"

# Function to check GPU memory usage
check_gpu_memory() {
    local gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "$gpu_mem"
}

# Function to check if competing processes are running
check_competing_processes() {
    local count=0
    for pattern in "${PROCESS_PATTERNS[@]}"; do
        local procs=$(ps aux | grep -E "$pattern" | grep -v grep | grep -v "$0" || true)
        if [ -n "$procs" ]; then
            echo "  ⚠️  Found competing process: $pattern"
            echo "$procs" | head -1 | awk '{print "      PID " $2 ": " $11 " " $12 " " $13 " " $14}'
            ((count++))
        fi
    done
    echo "$count"
}

echo "🔍 Checking system status..."
echo ""

# Initial check
gpu_mem=$(check_gpu_memory)
competing_procs=$(check_competing_processes)

echo ""
echo "Current GPU Memory: ${gpu_mem} MB"
echo "Competing Processes: $competing_procs"
echo ""

if [ "$gpu_mem" -lt "$GPU_MEM_THRESHOLD" ] && [ "$competing_procs" -eq 0 ]; then
    echo "✅ GPU is available! Starting batch processing immediately..."
    echo ""
    exec bash "$SCRIPT_DIR/run_batch_processing.sh" "$CONFIG" "${@:2}"
fi

echo "⏳ GPU is currently busy. Waiting for availability..."
echo ""
echo "Will start batch processing when:"
echo "  1. GPU memory usage < ${GPU_MEM_THRESHOLD} MB"
echo "  2. No SAM2/LaMa processes running"
echo ""
echo "You can safely disconnect. This script will continue in background."
echo ""
echo "Press Ctrl+C to cancel (will NOT affect running processes)"
echo ""

# Polling loop
CHECK_INTERVAL=300  # 5 minutes
WAIT_COUNT=0

while true; do
    gpu_mem=$(check_gpu_memory)
    competing_procs=$(check_competing_processes)

    WAIT_COUNT=$((WAIT_COUNT + 1))
    WAIT_TIME=$((WAIT_COUNT * CHECK_INTERVAL / 60))

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Check #${WAIT_COUNT} (waited ${WAIT_TIME} min)"
    echo "  GPU Memory: ${gpu_mem} MB (threshold: ${GPU_MEM_THRESHOLD})"
    echo "  Competing Processes: $competing_procs"

    if [ "$gpu_mem" -lt "$GPU_MEM_THRESHOLD" ] && [ "$competing_procs" -eq 0 ]; then
        echo ""
        echo "🎉 GPU is now available!"
        echo ""
        echo "======================================================================="
        echo "🚀 Starting Batch Processing"
        echo "======================================================================="
        echo ""

        # Start batch processing
        exec bash "$SCRIPT_DIR/run_batch_processing.sh" "$CONFIG" "${@:2}"
    fi

    echo "  ⏳ Still waiting... (next check in $((CHECK_INTERVAL / 60)) min)"
    echo ""

    sleep "$CHECK_INTERVAL"
done
