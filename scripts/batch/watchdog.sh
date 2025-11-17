#!/bin/bash
# Batch Processing Watchdog - Auto-restart on failure with safety checks
# Monitors the batch processing and automatically restarts if it crashes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-configs/batch/sam2_lama.yaml}"

LOG_DIR="$PROJECT_ROOT/logs/batch_processing"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"
RESTART_COUNT_FILE="$LOG_DIR/restart_count.txt"
MAX_RESTARTS=10  # Maximum consecutive restarts before giving up

# GPU memory threshold for detecting stuck processes (MB)
GPU_MEM_STUCK_THRESHOLD=15000  # If GPU memory stays > 15GB for 30+ min, likely stuck
STUCK_CHECK_INTERVAL=1800  # 30 minutes

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

# Initialize restart counter
if [ ! -f "$RESTART_COUNT_FILE" ]; then
    echo "0" > "$RESTART_COUNT_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$WATCHDOG_LOG"
}

get_restart_count() {
    cat "$RESTART_COUNT_FILE" 2>/dev/null || echo "0"
}

increment_restart_count() {
    local count=$(get_restart_count)
    echo $((count + 1)) > "$RESTART_COUNT_FILE"
}

reset_restart_count() {
    echo "0" > "$RESTART_COUNT_FILE"
}

is_batch_processor_running() {
    ps aux | grep -E "batch_processor.py.*$CONFIG" | grep -v grep | grep -v watchdog > /dev/null
    return $?
}

get_batch_processor_pid() {
    ps aux | grep -E "batch_processor.py.*$CONFIG" | grep -v grep | grep -v watchdog | awk '{print $2}' | head -1
}

get_gpu_memory() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0"
}

check_stuck_process() {
    # Check if GPU memory is high but no progress for extended period
    local gpu_mem=$(get_gpu_memory)

    if [ "$gpu_mem" -gt "$GPU_MEM_STUCK_THRESHOLD" ]; then
        # Check if any log files have been modified in last 30 minutes
        local recent_logs=$(find "$LOG_DIR" -name "sam2_*.log" -mmin -30 2>/dev/null | wc -l)

        if [ "$recent_logs" -eq 0 ]; then
            log "⚠️  Possible stuck process detected (GPU: ${gpu_mem}MB, no log updates)"
            return 0  # Stuck
        fi
    fi

    return 1  # Not stuck
}

kill_batch_processor() {
    local pid=$(get_batch_processor_pid)

    if [ -n "$pid" ]; then
        log "🛑 Killing batch processor (PID: $pid)"
        kill -9 "$pid" 2>/dev/null || true

        # Also kill any child processes (SAM2, LaMa)
        pkill -9 -f "instance_segmentation.py" 2>/dev/null || true
        pkill -9 -f "sam2_background_inpainting.py" 2>/dev/null || true

        sleep 3
    fi
}

start_batch_processor() {
    log "🚀 Starting batch processor..."

    nohup bash "$SCRIPT_DIR/run_batch_processing.sh" "$CONFIG" \
        > "$LOG_DIR/batch_processor_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

    local new_pid=$!
    log "✅ Batch processor started (PID: $new_pid)"

    # Reset restart counter after successful start
    sleep 10
    if is_batch_processor_running; then
        reset_restart_count
        log "✅ Batch processor confirmed running"
    fi
}

# Main watchdog loop
log "========================================================================"
log "🐕 Batch Processing Watchdog Started"
log "========================================================================"
log "Config: $CONFIG"
log "Max restarts: $MAX_RESTARTS"
log "Stuck check interval: ${STUCK_CHECK_INTERVAL}s"
log ""

# Start initial process if not running
if ! is_batch_processor_running; then
    log "📋 Batch processor not running, starting initial instance..."
    start_batch_processor
else
    log "✅ Batch processor already running"
fi

LAST_STUCK_CHECK=$(date +%s)

while true; do
    sleep 60  # Check every minute

    RESTART_COUNT=$(get_restart_count)

    # Check if we've exceeded max restarts
    if [ "$RESTART_COUNT" -ge "$MAX_RESTARTS" ]; then
        log "❌ Maximum restart limit reached ($MAX_RESTARTS). Stopping watchdog."
        log "   Please investigate the issue manually."
        exit 1
    fi

    # Check if batch processor is running
    if ! is_batch_processor_running; then
        log "⚠️  Batch processor stopped unexpectedly!"
        increment_restart_count
        RESTART_COUNT=$(get_restart_count)

        log "🔄 Attempting restart #$RESTART_COUNT..."

        # Clean up any zombie processes
        kill_batch_processor

        # Wait before restart
        sleep 5

        # Restart
        start_batch_processor

        continue
    fi

    # Periodic stuck process check (every 30 minutes)
    CURRENT_TIME=$(date +%s)
    TIME_SINCE_LAST_CHECK=$((CURRENT_TIME - LAST_STUCK_CHECK))

    if [ "$TIME_SINCE_LAST_CHECK" -ge "$STUCK_CHECK_INTERVAL" ]; then
        if check_stuck_process; then
            log "⚠️  Stuck process detected, forcing restart..."
            increment_restart_count

            kill_batch_processor
            sleep 5
            start_batch_processor
        fi

        LAST_STUCK_CHECK=$CURRENT_TIME
    fi

    # Log periodic status
    if [ $((SECONDS % 600)) -eq 0 ]; then  # Every 10 minutes
        local pid=$(get_batch_processor_pid)
        local gpu_mem=$(get_gpu_memory)
        log "💚 Watchdog active (Processor PID: $pid, GPU: ${gpu_mem}MB, Restarts: $RESTART_COUNT)"
    fi
done
