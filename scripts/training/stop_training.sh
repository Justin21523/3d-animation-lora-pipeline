#!/bin/bash
# Stop LoRA training gracefully
# 優雅停止 LoRA 訓練

echo "========================================================================"
echo "STOPPING LORA TRAINING"
echo "========================================================================"
echo

# Find training process
TRAINING_PID=$(ps aux | grep "[l]aunch_iterative_training.py" | awk '{print $2}')

if [ -z "$TRAINING_PID" ]; then
    echo "❌ No training process found."
    echo "   Training is not running."
    exit 1
fi

echo "Found training process: PID $TRAINING_PID"
echo

# Check current progress
CHECKPOINT_FILE="/mnt/data/ai_data/models/lora/luca/iterative_overnight_v5/checkpoint.json"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Current progress:"
    cat "$CHECKPOINT_FILE" | python3 -m json.tool 2>/dev/null || cat "$CHECKPOINT_FILE"
    echo
fi

# Ask for confirmation
read -p "Stop training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Send SIGTERM for graceful shutdown
echo "Sending graceful stop signal (SIGTERM)..."
kill $TRAINING_PID

# Wait for process to exit
echo "Waiting for process to exit..."
for i in {1..30}; do
    if ! ps -p $TRAINING_PID > /dev/null 2>&1; then
        echo "✓ Training stopped successfully."
        break
    fi
    sleep 1
    echo -n "."
done
echo

# Check if still running
if ps -p $TRAINING_PID > /dev/null 2>&1; then
    echo "⚠️  Process still running. Force stopping..."
    kill -9 $TRAINING_PID
    sleep 2
    if ! ps -p $TRAINING_PID > /dev/null 2>&1; then
        echo "✓ Training force stopped."
    else
        echo "❌ Failed to stop training. Please check manually."
        exit 1
    fi
fi

# Verify GPU is free
echo
echo "Checking GPU status..."
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

echo
echo "========================================================================"
echo "✓ TRAINING STOPPED"
echo "========================================================================"
echo "You can now:"
echo "  1. Optimize captions: python scripts/utils/optimize_character_captions.py"
echo "  2. Modify training parameters: nano scripts/training/iterative_lora_optimizer.py"
echo "  3. Restart training: bash scripts/training/restart_training.sh"
echo "========================================================================"
