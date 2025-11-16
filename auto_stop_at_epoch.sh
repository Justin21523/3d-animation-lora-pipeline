#!/bin/bash
# Auto-stop training at specified epoch
# Usage: bash auto_stop_at_epoch.sh <epoch_number>

TARGET_EPOCH=${1:-12}
SESSION_NAME="sdxl_luca_training_safe"
LOG_FILE=$(cat /tmp/current_training_log.txt 2>/dev/null | head -1 | cut -d' ' -f3)

if [ -z "$LOG_FILE" ]; then
    echo "❌ Cannot find training log file"
    exit 1
fi

echo "🔍 Monitoring training progress..."
echo "   Target epoch: $TARGET_EPOCH"
echo "   Log file: $LOG_FILE"
echo ""
echo "⚠️  Training will be stopped when epoch $TARGET_EPOCH completes"
echo "   Press Ctrl+C to cancel monitoring (training will continue)"
echo ""

while true; do
    # Check if training session exists
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "✓ Training session ended"
        exit 0
    fi

    # Check current epoch from log
    CURRENT_EPOCH=$(grep -oP "epoch \K\d+(?=/)" "$LOG_FILE" 2>/dev/null | tail -1)

    if [ -n "$CURRENT_EPOCH" ]; then
        echo -ne "\r⏳ Current epoch: $CURRENT_EPOCH / Target: $TARGET_EPOCH   "

        if [ "$CURRENT_EPOCH" -ge "$TARGET_EPOCH" ]; then
            echo ""
            echo ""
            echo "🎯 Target epoch $TARGET_EPOCH reached!"
            echo "   Stopping training gracefully..."

            # Send Ctrl+C to training session
            tmux send-keys -t "$SESSION_NAME" C-c

            sleep 5

            # Verify it stopped
            if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
                echo "✅ Training stopped successfully at epoch $TARGET_EPOCH"
            else
                echo "⚠️  Training session still running, trying again..."
                tmux send-keys -t "$SESSION_NAME" C-c
                sleep 3
            fi

            echo ""
            echo "📊 Final checkpoint should be at epoch $TARGET_EPOCH"
            echo "   Location: /mnt/data/ai_data/models/lora/luca/sdxl_trial1/"

            exit 0
        fi
    fi

    sleep 30  # Check every 30 seconds
done
