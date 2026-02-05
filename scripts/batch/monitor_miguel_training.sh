#!/bin/bash
# Monitor Miguel LoRA Training Progress
# Quick status check script

echo "========================================"
echo "Miguel Identity LoRA Training Monitor"
echo "========================================"
echo ""

# Check if training process is running
if ps aux | grep "train_network.py.*miguel" | grep -v grep | grep -q .; then
    echo "✅ Training Status: RUNNING"

    # Get process info
    echo ""
    echo "Process Info:"
    ps aux | grep "train_network.py.*miguel" | grep -v grep | head -1 | awk '{print "  PID: " $2 ", CPU: " $3 "%, MEM: " $4 "%, Runtime: " $10}'

    # Check GPU status
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | awk -F',' '{printf "  GPU: %s%%, VRAM: %s%% (%s/%s MB), Temp: %s°C, Power: %sW\n", $1, $2, $3, $4, $5, $6}'

    # Check for checkpoints
    echo ""
    echo "Checkpoints:"
    CHECKPOINT_DIR="/mnt/data/ai_data/models/lora/coco/miguel_identity"
    CHECKPOINT_COUNT=$(ls -1 "$CHECKPOINT_DIR"/*.safetensors 2>/dev/null | wc -l)
    if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
        echo "  Found $CHECKPOINT_COUNT checkpoint(s):"
        ls -lh "$CHECKPOINT_DIR"/*.safetensors | awk '{print "    " $9 " (" $5 ")"}'
    else
        echo "  No checkpoints yet (saved every 2 epochs)"
    fi

    # Check log directory
    echo ""
    echo "Latest Log:"
    LATEST_LOG=$(ls -t "$CHECKPOINT_DIR"/logs/miguel_identity*/network_train/events.out.* 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        LOG_SIZE=$(du -h "$LATEST_LOG" | cut -f1)
        LOG_TIME=$(stat -c %y "$LATEST_LOG" | cut -d'.' -f1)
        echo "  $LATEST_LOG"
        echo "  Size: $LOG_SIZE, Last modified: $LOG_TIME"
    else
        echo "  No log file found"
    fi

else
    echo "❌ Training Status: NOT RUNNING"
    echo ""
    echo "Check if training completed or encountered errors."
    echo "Last checkpoint:"
    ls -lth /mnt/data/ai_data/models/lora/coco/miguel_identity/*.safetensors 2>/dev/null | head -1
fi

echo ""
echo "========================================"
echo "To view TensorBoard:"
echo "  tensorboard --logdir=/mnt/data/ai_data/models/lora/coco/miguel_identity/logs"
echo ""
echo "Expected completion:"
echo "  Optimistic: 2.2 hours (~$(date -d '+2 hours 12 minutes' '+%H:%M'))"
echo "  Typical: 3-4 hours (~$(date -d '+3 hours 30 minutes' '+%H:%M'))"
echo "  Conservative: 5-6 hours (~$(date -d '+5 hours 30 minutes' '+%H:%M'))"
echo "========================================"
