#!/bin/bash
#
# Watch for Epoch 6 completion and stop training
#

LORA_DIR="/mnt/data/ai_data/models/lora_sdxl/coco/miguel_identity"
CHECKPOINT_FILE="${LORA_DIR}/miguel_identity_lora_sdxl-000006.safetensors"

echo "================================================================================"
echo "⏰ Monitoring for Epoch 6 completion"
echo "================================================================================"
echo "Target: $CHECKPOINT_FILE"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

while true; do
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo ""
        echo "================================================================================"
        echo "✅ EPOCH 6 CHECKPOINT DETECTED!"
        echo "================================================================================"
        echo "File: $CHECKPOINT_FILE"
        echo "Size: $(du -h "$CHECKPOINT_FILE" | cut -f1)"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""

        # Wait a few seconds to ensure file is fully written
        echo "⏳ Waiting 30 seconds to ensure checkpoint is fully saved..."
        sleep 30

        # Stop training
        echo ""
        echo "🛑 Stopping training..."

        # Find and kill training processes
        training_pids=$(ps aux | grep "sdxl_train_network.py" | grep -v grep | awk '{print $2}')

        if [ -n "$training_pids" ]; then
            echo "Found training processes: $training_pids"
            for pid in $training_pids; do
                echo "  Killing PID: $pid"
                kill $pid
            done

            # Wait for processes to terminate
            sleep 5

            # Force kill if still running
            for pid in $training_pids; do
                if ps -p $pid > /dev/null 2>&1; then
                    echo "  Force killing PID: $pid"
                    kill -9 $pid
                fi
            done

            echo ""
            echo "✅ Training stopped successfully"
        else
            echo "⚠️  No training processes found (may have already stopped)"
        fi

        echo ""
        echo "================================================================================"
        echo "📦 Available checkpoints:"
        ls -lht "$LORA_DIR"/*.safetensors
        echo ""
        echo "================================================================================"
        echo "📝 Next steps:"
        echo "  1. Test the 3 checkpoints (Epoch 2, 4, 6)"
        echo "  2. Decide whether to continue or start batch training"
        echo ""
        echo "To test checkpoints, run:"
        echo "  bash scripts/batch/evaluate_all_sdxl_checkpoints.sh miguel"
        echo ""
        echo "To start batch training, run:"
        echo "  bash scripts/batch/train_all_sdxl_sequential.sh bryce alberto ..."
        echo "================================================================================"

        break
    fi

    # Status update every minute
    echo "[$(date '+%H:%M:%S')] Still waiting for Epoch 6... (checking every 60s)"
    sleep 60
done
