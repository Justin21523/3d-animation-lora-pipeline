#!/bin/bash
# Monitor training loss values in real-time

LOG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/logs"

echo "🔍 Monitoring SDXL training loss values..."
echo "Press Ctrl+C to stop"
echo ""

# Find the latest training log
LOG_FILE=$(ls -t $LOG_DIR/inazuma_endou_mamoru_sdxl_training_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No training log found"
    exit 1
fi

echo "📝 Watching: $(basename $LOG_FILE)"
echo ""
echo "Expected: Loss values should be 0.05 - 0.15 (NOT nan)"
echo "========================================================"
echo ""

# Monitor loss values
tail -f "$LOG_FILE" | grep --line-buffered "avr_loss" | while read line; do
    echo "$line"

    # Check for NaN
    if echo "$line" | grep -q "avr_loss=nan"; then
        echo ""
        echo "❌ ERROR: NaN loss detected! Training will fail."
        echo "   Please stop training and check configuration."
        echo ""
    fi
done
