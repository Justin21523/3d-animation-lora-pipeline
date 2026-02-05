#!/bin/bash
#
# SDXL Training Monitor - Real-time Progress Display
# Usage: bash scripts/batch/monitor_sdxl_training.sh
#

clear
echo "========================================="
echo "SDXL Training Monitor"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Find latest training log
LATEST_LOG=$(ls -t /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/logs/*miguel*training*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No training log found"
    exit 1
fi

echo "📄 Monitoring log: $(basename $LATEST_LOG)"
echo ""

# Function to display training stats
show_stats() {
    echo -e "${BLUE}=== GPU Status ===${NC}"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader | \
        awk -F, '{printf "  GPU %s: %s%%, VRAM: %sMB/%sMB, Temp: %s°C, Power: %sW\n", $1, $3, $4, $5, $6, $7}'

    echo ""
    echo -e "${BLUE}=== Training Process ===${NC}"
    ps aux | grep "sdxl_train_network" | grep -v grep | awk '{printf "  PID: %s, CPU: %s%%, MEM: %s%%, Runtime: %s\n", $2, $3, $4, $10}'

    echo ""
    echo -e "${BLUE}=== Latest Training Output ===${NC}"

    # Try to get output from actual training process (not the wrapper)
    # Look for lines with epoch, step, loss info
    tail -100 "$LATEST_LOG" 2>/dev/null | grep -E "(epoch|step|loss|saving|saved)" | tail -20 || \
        echo "  ⏳ Initializing... (載入模型、緩存latents中)"

    echo ""
    echo -e "${BLUE}=== Checkpoints ===${NC}"
    ls -lth /mnt/data/ai_data/models/lora_sdxl/coco/miguel_identity/*.safetensors 2>/dev/null | \
        head -3 | awk '{printf "  %s  %s  %s\n", $6, $7, $9}' || \
        echo "  No checkpoints yet"
}

# Main monitoring loop
echo "========================================="
echo "Press Ctrl+C to exit"
echo "Refreshing every 5 seconds..."
echo "========================================="
echo ""

while true; do
    show_stats
    echo ""
    echo "----------------------------------------"
    echo "Last update: $(date '+%H:%M:%S')"
    echo "========================================="
    sleep 5
    clear
done
