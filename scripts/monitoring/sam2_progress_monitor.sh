#!/bin/bash
# SAM2 Progress Monitor
# Monitors SAM2 processing by checking output files (works even without logs)

set -e

OUTPUT_DIR="${1:-/mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2}"
TOTAL_FRAMES="${2:-14411}"
CHECK_INTERVAL="${3:-60}"  # seconds

echo "======================================================================="
echo "📊 SAM2 Progress Monitor"
echo "======================================================================="
echo ""
echo "Output Directory: $OUTPUT_DIR"
echo "Total Frames:     $TOTAL_FRAMES"
echo "Check Interval:   ${CHECK_INTERVAL}s"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo "======================================================================="
echo ""

# Store previous values for speed calculation
PREV_FRAMES=0
PREV_INSTANCES=0
PREV_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED_TOTAL=$((CURRENT_TIME - PREV_TIME))

    # Count files (efficient methods)
    if [ -d "$OUTPUT_DIR/instances" ]; then
        # Get latest frame number from filename
        LATEST_FRAME=$(ls "$OUTPUT_DIR/instances/" 2>/dev/null | \
            grep -oP 'frame\d+' | \
            sed 's/frame//' | \
            sort -n | \
            tail -1 || echo "0")

        # Count total instances (approximation to avoid slow counting)
        INSTANCE_COUNT=$(find "$OUTPUT_DIR/instances" -name "*.png" -type f 2>/dev/null | wc -l)

    else
        LATEST_FRAME=0
        INSTANCE_COUNT=0
    fi

    # Calculate progress
    PROGRESS_PCT=$(echo "scale=2; ($LATEST_FRAME / $TOTAL_FRAMES) * 100" | bc)
    REMAINING_FRAMES=$((TOTAL_FRAMES - LATEST_FRAME))

    # Calculate speed (frames/min and instances/min)
    if [ $ELAPSED_TOTAL -gt 0 ]; then
        FRAMES_DELTA=$((LATEST_FRAME - PREV_FRAMES))
        INSTANCES_DELTA=$((INSTANCE_COUNT - PREV_INSTANCES))

        FRAMES_PER_MIN=$(echo "scale=2; ($FRAMES_DELTA / $ELAPSED_TOTAL) * 60" | bc)
        INSTANCES_PER_MIN=$(echo "scale=2; ($INSTANCES_DELTA / ELAPSED_TOTAL) * 60" | bc)

        # Average instances per frame
        if [ $LATEST_FRAME -gt 0 ]; then
            AVG_INST_PER_FRAME=$(echo "scale=2; $INSTANCE_COUNT / $LATEST_FRAME" | bc)
        else
            AVG_INST_PER_FRAME="0"
        fi

        # Estimate remaining time
        if [ $(echo "$FRAMES_PER_MIN > 0" | bc) -eq 1 ]; then
            REMAINING_MINUTES=$(echo "scale=0; $REMAINING_FRAMES / $FRAMES_PER_MIN" | bc)
            REMAINING_HOURS=$(echo "scale=1; $REMAINING_MINUTES / 60" | bc)
        else
            REMAINING_HOURS="N/A"
        fi
    else
        FRAMES_PER_MIN="0"
        INSTANCES_PER_MIN="0"
        AVG_INST_PER_FRAME="0"
        REMAINING_HOURS="N/A"
    fi

    # Display progress
    echo "[$(date +'%Y-%m-%d %H:%M:%S')]"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 Progress:         $LATEST_FRAME / $TOTAL_FRAMES frames ($PROGRESS_PCT%)"
    echo "📊 Instances:        $INSTANCE_COUNT total ($AVG_INST_PER_FRAME per frame)"
    echo "⚡ Speed:            $FRAMES_PER_MIN frames/min"
    echo "⏳ Remaining:        $REMAINING_FRAMES frames (~$REMAINING_HOURS hours)"
    echo ""

    # Update previous values
    PREV_FRAMES=$LATEST_FRAME
    PREV_INSTANCES=$INSTANCE_COUNT
    PREV_TIME=$CURRENT_TIME

    sleep "$CHECK_INTERVAL"
done
