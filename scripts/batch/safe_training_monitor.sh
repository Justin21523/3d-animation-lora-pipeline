#!/bin/bash
# Safe Training Monitor - Prevents OOM and CUDA crashes
# Monitors GPU memory, system memory, and training process health

CONFIG_FILE="$1"
TRAINING_PID=""
MAX_GPU_MEMORY_PERCENT=95
MAX_SYSTEM_MEMORY_PERCENT=90
CHECK_INTERVAL=30

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file.toml>"
    exit 1
fi

# Function to check GPU memory
check_gpu_memory() {
    local gpu_mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    local gpu_mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    local gpu_percent=$((gpu_mem_used * 100 / gpu_mem_total))
    echo $gpu_percent
}

# Function to check system memory
check_system_memory() {
    local mem_percent=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    echo $mem_percent
}

# Function to check if process is running
is_running() {
    if [ -n "$TRAINING_PID" ] && kill -0 $TRAINING_PID 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Cleanup function
cleanup() {
    echo "[$(date)] Training monitor stopped"
    if is_running; then
        echo "[$(date)] Gracefully stopping training process..."
        kill -SIGTERM $TRAINING_PID 2>/dev/null
        sleep 5
        if is_running; then
            kill -SIGKILL $TRAINING_PID 2>/dev/null
        fi
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "[$(date)] Starting safe training monitor..."
echo "[$(date)] Config: $CONFIG_FILE"
echo "[$(date)] Max GPU memory: ${MAX_GPU_MEMORY_PERCENT}%"
echo "[$(date)] Max system memory: ${MAX_SYSTEM_MEMORY_PERCENT}%"

# Start training in background
cd /mnt/c/ai_projects/kohya_ss/sd-scripts
conda run -n kohya_ss accelerate launch --num_cpu_threads_per_process=2 sdxl_train_network.py \
    --config_file "$CONFIG_FILE" &
TRAINING_PID=$!

echo "[$(date)] Training started with PID: $TRAINING_PID"

# Monitor loop
while is_running; do
    sleep $CHECK_INTERVAL

    # Check GPU memory
    gpu_percent=$(check_gpu_memory)
    if [ $gpu_percent -gt $MAX_GPU_MEMORY_PERCENT ]; then
        echo "[$(date)] WARNING: GPU memory at ${gpu_percent}% (threshold: ${MAX_GPU_MEMORY_PERCENT}%)"
        echo "[$(date)] Emergency stop to prevent OOM crash"
        kill -SIGTERM $TRAINING_PID
        sleep 3
        echo "[$(date)] Training stopped safely"
        exit 1
    fi

    # Check system memory
    sys_percent=$(check_system_memory)
    if [ $sys_percent -gt $MAX_SYSTEM_MEMORY_PERCENT ]; then
        echo "[$(date)] WARNING: System memory at ${sys_percent}% (threshold: ${MAX_SYSTEM_MEMORY_PERCENT}%)"
        echo "[$(date)] Emergency stop to prevent system freeze"
        kill -SIGTERM $TRAINING_PID
        sleep 3
        echo "[$(date)] Training stopped safely"
        exit 1
    fi

    # Check CUDA errors in recent logs
    if dmesg | tail -20 | grep -qi "cuda\|gpu hang\|out of memory"; then
        echo "[$(date)] WARNING: CUDA error detected in system logs"
        echo "[$(date)] Emergency stop to prevent crash"
        kill -SIGTERM $TRAINING_PID
        sleep 3
        echo "[$(date)] Training stopped safely"
        exit 1
    fi

    echo "[$(date)] Status OK - GPU: ${gpu_percent}%, System: ${sys_percent}%, PID: $TRAINING_PID"
done

# Training completed normally
wait $TRAINING_PID
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Training completed successfully"
else
    echo "[$(date)] Training exited with code: $EXIT_CODE"
fi

exit $EXIT_CODE
