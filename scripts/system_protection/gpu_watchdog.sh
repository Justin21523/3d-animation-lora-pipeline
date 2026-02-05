#!/bin/bash
# GPU Memory Watchdog - Monitors CUDA OOM situations

LOG_FILE="/tmp/gpu_watchdog.log"
CHECK_INTERVAL=15
GPU_CRITICAL=95  # % memory usage

echo "═══════════════════════════════════════════════════════════════" >> "$LOG_FILE"
echo "GPU Watchdog started at $(date)" >> "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════" >> "$LOG_FILE"

send_notification() {
    local urgency=$1
    local title=$2
    local message=$3

    if command -v notify-send &> /dev/null; then
        DISPLAY=:0 notify-send -u "$urgency" "$title" "$message"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$urgency] $title: $message" >> "$LOG_FILE"
}

while true; do
    if command -v nvidia-smi &> /dev/null; then
        # Get GPU memory usage
        GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits)

        while IFS=',' read -r gpu_id mem_used mem_total temp; do
            mem_used=$(echo $mem_used | xargs)
            mem_total=$(echo $mem_total | xargs)
            temp=$(echo $temp | xargs)

            mem_pct=$(awk "BEGIN {printf \"%.0f\", ($mem_used/$mem_total)*100}")

            # Critical GPU memory
            if [ "$mem_pct" -ge "$GPU_CRITICAL" ]; then
                send_notification "critical" "🔥 GPU Memory Critical!" \
                    "GPU $gpu_id: ${mem_pct}% (${mem_used}MB/${mem_total}MB)"
            fi

            # High temperature
            if [ "$temp" -ge 85 ]; then
                send_notification "normal" "🌡️  GPU Temperature High" \
                    "GPU $gpu_id: ${temp}°C"
            fi

        done <<< "$GPU_INFO"
    fi

    sleep "$CHECK_INTERVAL"
done
