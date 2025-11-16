#!/bin/bash
# SAM2 Status Monitor
# Usage:
#   bash check_sam2.sh          # Single check
#   bash check_sam2.sh --watch  # Auto-refresh every 30s

INSTANCES_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances"
TOTAL_SCENES=1441
WATCH_MODE=false
REFRESH_INTERVAL=30

# Parse arguments
if [ "$1" == "--watch" ] || [ "$1" == "-w" ]; then
    WATCH_MODE=true
fi

show_status() {
clear
echo "═══════════════════════════════════════════════════════════"
echo "           🎬 SAM2 Instance Segmentation Status"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if running
if tmux has-session -t luca_sam2 2>/dev/null; then
    echo "✅ Status: RUNNING"
else
    echo "❌ Status: NOT RUNNING"
    exit 1
fi

# Count instances
INSTANCE_COUNT=$(ls "$INSTANCES_DIR" 2>/dev/null | wc -l)
echo "📊 Total instances: $(printf "%'d" $INSTANCE_COUNT)"

# Get latest scene
LATEST_SCENE=$(ls "$INSTANCES_DIR" 2>/dev/null | grep -o 'scene[0-9]*' | sort -u | tail -1)
LATEST_SCENE_NUM=$(echo "$LATEST_SCENE" | grep -o '[0-9]*' | sed 's/^0*//')  # Remove leading zeros
echo "🎞️  Latest scene: $LATEST_SCENE"

# Calculate progress
PROGRESS=$((LATEST_SCENE_NUM * 100 / TOTAL_SCENES))
echo "📈 Progress: ${LATEST_SCENE_NUM}/${TOTAL_SCENES} scenes ($PROGRESS%)"

# Progress bar
BAR_LENGTH=50
FILLED=$((PROGRESS * BAR_LENGTH / 100))
printf "["
for i in $(seq 1 $FILLED); do printf "█"; done
for i in $(seq $((FILLED + 1)) $BAR_LENGTH); do printf "░"; done
printf "] %d%%\n" $PROGRESS

# GPU status
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
echo "🎮 GPU: ${GPU_UTIL}% utilization, ${GPU_MEM} memory"

# Check latest file time
LATEST_FILE=$(ls -t "$INSTANCES_DIR" 2>/dev/null | head -1)
if [ -n "$LATEST_FILE" ]; then
    LATEST_TIME=$(stat -c %Y "$INSTANCES_DIR/$LATEST_FILE" 2>/dev/null)
    CURRENT_TIME=$(date +%s)
    IDLE_TIME=$((CURRENT_TIME - LATEST_TIME))

    echo "⏱️  Last update: $(stat -c %y "$INSTANCES_DIR/$LATEST_FILE" 2>/dev/null | cut -d'.' -f1)"

    if [ $IDLE_TIME -lt 60 ]; then
        echo "✅ Activity: Processing normally (${IDLE_TIME}s ago)"
    elif [ $IDLE_TIME -lt 180 ]; then
        echo "⏳ Activity: Processing complex frame (${IDLE_TIME}s idle)"
    else
        echo "⚠️  Activity: POSSIBLY STUCK (${IDLE_TIME}s idle)"
    fi
fi

# Estimate remaining time
REMAINING_SCENES=$((TOTAL_SCENES - LATEST_SCENE_NUM))
# Assume average 15 scenes/min
EST_MINUTES=$((REMAINING_SCENES / 15))
EST_HOURS=$((EST_MINUTES / 60))
EST_MIN_REMAINDER=$((EST_MINUTES % 60))

echo ""
echo "⏰ Estimated remaining: ~${EST_HOURS}h ${EST_MIN_REMAINDER}m"

# Check for failed frames
FAILED_LOG="/mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/failed_frames.json"
if [ -f "$FAILED_LOG" ]; then
    FAILED_COUNT=$(grep -o "\"total_failed\":" "$FAILED_LOG" | wc -l)
    if [ $FAILED_COUNT -gt 0 ]; then
        echo "⚠️  Failed frames: Check $FAILED_LOG"
    fi
fi

echo "═══════════════════════════════════════════════════════════"
echo ""
if [ "$WATCH_MODE" = true ]; then
    echo "🔄 Auto-refresh mode: Updates every ${REFRESH_INTERVAL}s"
    echo "   Press Ctrl+C to stop"
else
    echo "💡 Tips:"
    echo "  - Auto-refresh: bash scripts/utils/check_sam2.sh --watch"
    echo "  - View tmux session: tmux attach -t luca_sam2"
    echo "  - Alternative monitor: bash scripts/utils/watch_sam2.sh"
fi
echo ""
}

# Main execution
if [ "$WATCH_MODE" = true ]; then
    echo "🔍 Starting SAM2 auto-refresh monitor..."
    echo "   Refresh interval: ${REFRESH_INTERVAL} seconds"
    echo "   Press Ctrl+C to stop"
    echo ""
    sleep 2

    while true; do
        show_status
        sleep $REFRESH_INTERVAL
    done
else
    show_status
fi
