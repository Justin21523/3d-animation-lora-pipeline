#!/bin/bash
# Memory Watchdog - Monitors and prevents OOM situations
# Run in background: nohup /tmp/memory_watchdog.sh &

LOG_FILE="/tmp/memory_watchdog.log"
CHECK_INTERVAL=10  # seconds
RAM_CRITICAL=90    # % - critical threshold
RAM_WARNING=80     # % - warning threshold
SWAP_CRITICAL=80   # % - swap usage critical

echo "═══════════════════════════════════════════════════════════════" >> "$LOG_FILE"
echo "Memory Watchdog started at $(date)" >> "$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════" >> "$LOG_FILE"

# Send desktop notification
send_notification() {
    local urgency=$1
    local title=$2
    local message=$3

    # Try notify-send (most Linux desktops)
    if command -v notify-send &> /dev/null; then
        DISPLAY=:0 notify-send -u "$urgency" "$title" "$message"
    fi

    # Log
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$urgency] $title: $message" >> "$LOG_FILE"
}

# Get memory usage percentage
get_mem_usage() {
    free | awk '/^Mem:/ {printf "%.0f", ($3/$2) * 100}'
}

get_swap_usage() {
    free | awk '/^Swap:/ {if ($2 > 0) printf "%.0f", ($3/$2) * 100; else print "0"}'
}

# Get top memory consumers
get_top_memory_processes() {
    ps aux --sort=-%mem | head -6 | tail -5 | awk '{printf "  • %s: %.1f%% (PID: %s)\n", $11, $4, $2}'
}

while true; do
    MEM_USAGE=$(get_mem_usage)
    SWAP_USAGE=$(get_swap_usage)

    # Critical: Memory > 90%
    if [ "$MEM_USAGE" -ge "$RAM_CRITICAL" ]; then
        send_notification "critical" "🚨 Memory Critical!" \
            "RAM usage at ${MEM_USAGE}%! System may crash soon."

        echo "Top memory consumers:" >> "$LOG_FILE"
        get_top_memory_processes >> "$LOG_FILE"

        # Optional: Kill low-priority processes
        # Uncomment to enable automatic killing
        # pkill -f "chrome|chromium|firefox" 2>/dev/null

    # Warning: Memory > 80%
    elif [ "$MEM_USAGE" -ge "$RAM_WARNING" ]; then
        send_notification "normal" "⚠️  Memory Warning" \
            "RAM usage at ${MEM_USAGE}%. Consider closing unused applications."
    fi

    # Critical: Swap usage high
    if [ "$SWAP_USAGE" -ge "$SWAP_CRITICAL" ]; then
        send_notification "normal" "⚠️  High Swap Usage" \
            "Swap at ${SWAP_USAGE}%. Performance may degrade."
    fi

    sleep "$CHECK_INTERVAL"
done
