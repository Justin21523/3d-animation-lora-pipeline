#!/bin/bash
# Real-time Batch Processing Monitor
# Shows current status of all processing tasks

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
DATA_ROOT="/mnt/data/ai_data/datasets/3d-anime"

cd "$PROJECT_ROOT"

echo "========================================================================"
echo "🔍 Batch Processing Monitor"
echo "========================================================================"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check GPU status
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | \
    awk -F, '{printf "   Memory: %s / %s | Utilization: %s | Temp: %s°C\n", $1, $2, $3, $4}'
echo ""

# Check running processes
echo "⚙️  Active Processes:"
SAM2_PID=$(ps aux | grep "instance_segmentation.py" | grep -v grep | awk '{print $2}' | head -1)
LAMA_PID=$(ps aux | grep "sam2_background_inpainting.py" | grep -v grep | awk '{print $2}' | head -1)
BATCH_PID=$(ps aux | grep "batch_processor.py.*sam2_lama" | grep -v grep | awk '{print $2}' | head -1)
WATCHDOG_PID=$(ps aux | grep "watchdog.sh" | grep -v grep | awk '{print $2}' | head -1)

[ -n "$BATCH_PID" ] && echo "   ✅ Batch Processor: PID $BATCH_PID" || echo "   ❌ Batch Processor: NOT RUNNING"
[ -n "$WATCHDOG_PID" ] && echo "   ✅ Watchdog: PID $WATCHDOG_PID" || echo "   ❌ Watchdog: NOT RUNNING"
[ -n "$SAM2_PID" ] && echo "   🔄 SAM2: PID $SAM2_PID" || echo "   ⏸️  SAM2: Idle"
[ -n "$LAMA_PID" ] && echo "   🔄 LaMa: PID $LAMA_PID" || echo "   ⏸️  LaMa: Idle"
echo ""

# Check current film being processed
if [ -n "$SAM2_PID" ]; then
    CURRENT_FILM=$(ps -p $SAM2_PID -o cmd --no-headers | grep -oP '3d-anime/\K[^/]+' | head -1)
    echo "🎬 Current Film: $CURRENT_FILM"
    echo ""
fi

# Check progress for each film
echo "📊 Films Progress:"
for film in coco elio onward orion turning-red up; do
    TOTAL_FRAMES=$(ls -1 "$DATA_ROOT/$film/frames_final/"*.jpg 2>/dev/null | wc -l || echo "0")

    if [ "$TOTAL_FRAMES" -eq 0 ]; then
        TOTAL_FRAMES=$(ls -1 "$DATA_ROOT/$film/frames/"*.jpg 2>/dev/null | wc -l || echo "0")
    fi

    SAM2_INSTANCES=$(find "$DATA_ROOT/$film/${film}_instances_sam2_v2/instances" -name "*.png" 2>/dev/null | wc -l || echo "0")
    SAM2_BACKGROUNDS=$(find "$DATA_ROOT/$film/${film}_instances_sam2_v2/backgrounds" -name "*.jpg" 2>/dev/null | wc -l || echo "0")
    LAMA_BACKGROUNDS=$(find "$DATA_ROOT/$film/backgrounds_lama_v2" -name "*.jpg" 2>/dev/null | wc -l || echo "0")

    if [ "$TOTAL_FRAMES" -gt 0 ]; then
        SAM2_PERCENT=$(echo "scale=1; $SAM2_BACKGROUNDS * 100 / $TOTAL_FRAMES" | bc)
        LAMA_PERCENT=$(echo "scale=1; $LAMA_BACKGROUNDS * 100 / $TOTAL_FRAMES" | bc)

        printf "   %-15s Total: %4d | SAM2: %4d (%5.1f%%) | LaMa: %4d (%5.1f%%)\n" \
            "$film" "$TOTAL_FRAMES" "$SAM2_BACKGROUNDS" "$SAM2_PERCENT" "$LAMA_BACKGROUNDS" "$LAMA_PERCENT"
    fi
done

echo ""
echo "📝 Recent Logs:"
echo "   Watchdog: logs/batch_processing/watchdog.log"
echo "   Batch:    logs/batch_processing_restart.log"
echo "   Latest:   $(ls -t logs/batch_processing/sam2_*.log 2>/dev/null | head -1)"
echo ""
echo "========================================================================"
echo "💡 Commands:"
echo "   Watch progress: watch -n 10 'bash scripts/batch/monitor.sh'"
echo "   Check logs:     tail -f logs/batch_processing/watchdog.log"
echo "   GPU monitor:    watch -n 5 nvidia-smi"
echo "========================================================================"
