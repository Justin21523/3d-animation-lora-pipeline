#!/bin/bash
#
# Real-time Training Monitor
# Shows GPU stats, checkpoint progress, and training output
#

OUTPUT_DIR="${1:-/mnt/data/ai_data/models/lora_sdxl/coco/miguel_identity}"
LOG_FILE="${2:-logs/coco_miguel_identity_sdxl_training_*.log}"

echo "=================================="
echo "  Real-time Training Monitor"
echo "=================================="
echo "Output Dir: $OUTPUT_DIR"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                          SDXL Training Monitor                                ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    # GPU Status
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 GPU Status"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
        awk -F', ' '{printf "GPU Util: %3s%%  |  VRAM: %5sMB / %5sMB  |  Temp: %2s°C  |  Power: %3sW\n", $1, $2, $3, $4, $5}'
    echo ""

    # Checkpoint Status
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💾 Checkpoints"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ -d "$OUTPUT_DIR" ]; then
        CHECKPOINT_COUNT=$(ls "$OUTPUT_DIR"/*.safetensors 2>/dev/null | wc -l)
        echo "Total checkpoints: $CHECKPOINT_COUNT"
        if [ $CHECKPOINT_COUNT -gt 0 ]; then
            echo ""
            ls -lht "$OUTPUT_DIR"/*.safetensors 2>/dev/null | head -5 | \
                awk '{printf "  %s  %6s  %s\n", $9, $5, $6" "$7" "$8}' | \
                sed 's|.*/||'
        fi
    else
        echo "Waiting for checkpoints..."
    fi
    echo ""

    # Training Log (last 15 lines)
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📝 Training Log (last 15 lines)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Try multiple log sources
    LOG_FOUND=0

    # Try the specified log file pattern
    LATEST_LOG=$(ls -t $LOG_FILE 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ] && [ -s "$LATEST_LOG" ]; then
        tail -15 "$LATEST_LOG" 2>/dev/null
        LOG_FOUND=1
    fi

    # If no log found, try to find training process output
    if [ $LOG_FOUND -eq 0 ]; then
        # Find the training process
        TRAIN_PID=$(ps aux | grep "sdxl_train_network.py" | grep -v grep | awk '{print $2}' | head -1)
        if [ -n "$TRAIN_PID" ]; then
            echo "Training process PID: $TRAIN_PID (waiting for output...)"
            echo "Hint: Output may be buffered. Check back in a few minutes."
        else
            echo "No training process found."
        fi
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Refreshing every 5 seconds... (Press Ctrl+C to exit)"

    sleep 5
done
