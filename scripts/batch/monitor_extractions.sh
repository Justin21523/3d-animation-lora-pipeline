#!/bin/bash
# Simple monitoring script for batch frame extraction
# Shows frame counts and process status for all projects

PROJECTS=("coco" "elio" "turning-red" "up")
BASE_DIR="/mnt/data/ai_data/datasets/3d-anime"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Batch Frame Extraction Monitor                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

while true; do
    clear
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================="

    for proj in "${PROJECTS[@]}"; do
        frames_dir="$BASE_DIR/$proj/frames"
        count=0

        if [ -d "$frames_dir" ]; then
            count=$(ls -1 "$frames_dir" 2>/dev/null | grep -c ".jpg$" || echo "0")
        fi

        # Check if process is running
        if ps aux | grep -q "[u]niversal_frame_extractor.py /mnt/c/raw_videos/$proj"; then
            status="🟢 RUNNING"
            cpu=$(ps aux | grep "[u]niversal_frame_extractor.py /mnt/c/raw_videos/$proj" | awk '{print $3}' | head -1 | tr -d '\n')
        else
            status="⚪ IDLE"
            cpu="0.0"
        fi

        printf "%-15s : %5d frames | %s (CPU: %s%%)\n" "$proj" "$count" "$status" "$cpu"
    done

    echo ""
    echo "=================================="
    echo "Press Ctrl+C to exit monitoring"
    echo ""

    sleep 10
done
