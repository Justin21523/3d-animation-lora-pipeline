#!/bin/bash
# Real-time SAM2 monitoring
# Alerts if no new files in 3 minutes

INSTANCES_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances"
ALERT_THRESHOLD=180  # 3 minutes

echo "🔍 Starting SAM2 real-time monitor..."
echo "Will alert if no new files for ${ALERT_THRESHOLD}s"
echo "Press Ctrl+C to stop"
echo ""

LAST_COUNT=$(ls "$INSTANCES_DIR" 2>/dev/null | wc -l)
LAST_CHANGE_TIME=$(date +%s)

while true; do
    sleep 30

    CURRENT_COUNT=$(ls "$INSTANCES_DIR" 2>/dev/null | wc -l)
    CURRENT_TIME=$(date +%s)

    if [ "$CURRENT_COUNT" -gt "$LAST_COUNT" ]; then
        # New files detected
        NEW_FILES=$((CURRENT_COUNT - LAST_COUNT))
        LATEST_SCENE=$(ls "$INSTANCES_DIR" | grep -o 'scene[0-9]*' | sort -u | tail -1)

        echo "[$(date '+%H:%M:%S')] ✓ +${NEW_FILES} instances | Latest: ${LATEST_SCENE} | Total: ${CURRENT_COUNT}"

        LAST_COUNT=$CURRENT_COUNT
        LAST_CHANGE_TIME=$CURRENT_TIME
    else
        # No new files
        IDLE_TIME=$((CURRENT_TIME - LAST_CHANGE_TIME))

        if [ $IDLE_TIME -gt $ALERT_THRESHOLD ]; then
            echo "[$(date '+%H:%M:%S')] ⚠️  WARNING: No new files for ${IDLE_TIME}s! Possibly stuck on complex frame."
        else
            echo "[$(date '+%H:%M:%S')] ⏳ Processing complex frame... (idle ${IDLE_TIME}s)"
        fi
    fi
done
