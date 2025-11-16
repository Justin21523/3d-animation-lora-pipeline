#!/bin/bash
# SAM2 Progress Monitor
# Quick status check for SAM2 instance segmentation

INSTANCES_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances"
TOTAL_SCENES=1441
TOTAL_FRAMES=4323

echo "========================================="
echo "🎬 SAM2 Instance Segmentation Monitor"
echo "========================================="
echo ""

# Check if process is running
if tmux has-session -t luca_sam2 2>/dev/null; then
    echo "✅ Status: Running"
else
    echo "❌ Status: Not running"
    exit 1
fi

# Count instances
INSTANCE_COUNT=$(ls "$INSTANCES_DIR" 2>/dev/null | wc -l)
echo "📊 Total instances: $INSTANCE_COUNT"

# Get latest scene
LATEST_SCENE=$(ls "$INSTANCES_DIR" 2>/dev/null | grep -o 'scene[0-9]*' | sort -u | tail -1)
LATEST_SCENE_NUM=$(echo "$LATEST_SCENE" | grep -o '[0-9]*')
echo "🎞️  Latest scene: $LATEST_SCENE"

# Calculate progress
PROGRESS=$((LATEST_SCENE_NUM * 100 / TOTAL_SCENES))
echo "📈 Progress: ${LATEST_SCENE_NUM}/${TOTAL_SCENES} scenes ($PROGRESS%)"

# Check GPU
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
echo "🎮 GPU: ${GPU_UTIL}% utilization, ${GPU_MEM} memory"

# Check latest file time
LATEST_FILE=$(ls -t "$INSTANCES_DIR" 2>/dev/null | head -1)
if [ -n "$LATEST_FILE" ]; then
    LATEST_TIME=$(stat -c %y "$INSTANCES_DIR/$LATEST_FILE" 2>/dev/null | cut -d'.' -f1)
    echo "⏱️  Last update: $LATEST_TIME"
fi

echo "========================================="
