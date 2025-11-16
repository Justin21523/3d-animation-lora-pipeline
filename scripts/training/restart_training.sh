#!/bin/bash
# Restart LoRA training (resumes from checkpoint)
# 重啟 LoRA 訓練（從 checkpoint 恢復）

echo "========================================================================"
echo "RESTART LORA TRAINING"
echo "========================================================================"
echo

# Check if training is already running
RUNNING_PID=$(ps aux | grep "[l]aunch_iterative_training.py" | awk '{print $2}')
if [ ! -z "$RUNNING_PID" ]; then
    echo "❌ Training is already running (PID: $RUNNING_PID)"
    echo "   Stop it first: bash scripts/training/stop_training.sh"
    exit 1
fi

# Check checkpoint
CHECKPOINT_FILE="/mnt/data/ai_data/models/lora/luca/iterative_overnight_v5/checkpoint.json"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "✓ Checkpoint found. Training will resume from:"
    cat "$CHECKPOINT_FILE" | python3 -m json.tool 2>/dev/null || cat "$CHECKPOINT_FILE"
    echo
else
    echo "⚠️  No checkpoint found. Training will start from iteration 1."
    echo
fi

# Check GPU availability
echo "Checking GPU status..."
GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ "$GPU_USAGE" -gt 50 ]; then
    echo "⚠️  GPU utilization is high: ${GPU_USAGE}%"
    nvidia-smi
    echo
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Change to project directory
cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline

# Start training
LOG_FILE="/mnt/data/ai_data/models/lora/luca/iterative_overnight_v5/training.log"

echo "Starting training..."
echo "Log file: $LOG_FILE"
echo

nohup python scripts/training/launch_iterative_training.py \
  > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

# Wait a moment to check if it started successfully
sleep 3

if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "========================================================================"
    echo "✓ TRAINING STARTED SUCCESSFULLY"
    echo "========================================================================"
    echo "PID: $TRAIN_PID"
    echo "Log: $LOG_FILE"
    echo
    echo "Monitor with:"
    echo "  tail -f $LOG_FILE"
    echo "  bash scripts/monitoring/monitor_lora_training.sh"
    echo
    echo "Stop with:"
    echo "  bash scripts/training/stop_training.sh"
    echo "========================================================================"

    # Show initial output
    echo
    echo "Initial output:"
    echo "----------------------------------------"
    sleep 2
    tail -20 "$LOG_FILE"
else
    echo "========================================================================"
    echo "❌ TRAINING FAILED TO START"
    echo "========================================================================"
    echo "Check log file for errors: $LOG_FILE"
    tail -50 "$LOG_FILE"
    exit 1
fi
