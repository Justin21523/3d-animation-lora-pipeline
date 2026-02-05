#!/bin/bash
#
# Monitor Elio SDXL Training Progress
# Usage: bash scripts/batch/monitor_elio_training.sh
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================"
echo "🎯 ELIO SDXL TRAINING MONITOR"
echo "========================================"
echo ""

# Check if tmux session exists
if ! tmux has-session -t elio_sdxl_training 2>/dev/null; then
    echo -e "${RED}❌ Tmux session 'elio_sdxl_training' not found${NC}"
    echo "Please start training first"
    exit 1
fi

echo -e "${GREEN}✅ Training session active${NC}"
echo ""

# GPU Status
echo "========================================"
echo "📊 GPU STATUS"
echo "========================================"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r idx name temp gpu_util mem_util mem_used mem_total; do
    echo -e "${BLUE}GPU $idx:${NC} $name"
    echo "  Temperature: ${temp}°C"
    echo "  GPU Util: ${gpu_util}%"
    echo "  VRAM: ${mem_used}MB / ${mem_total}MB (${mem_util}%)"
done
echo ""

# Training Process
echo "========================================"
echo "⚙️  TRAINING PROCESSES"
echo "========================================"
ps aux | grep -E "(sdxl_train|accelerate)" | grep -v grep | while read -r line; do
    pid=$(echo $line | awk '{print $2}')
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    time=$(echo $line | awk '{print $10}')
    cmd=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i}')
    echo -e "${YELLOW}PID${NC} $pid | CPU: $cpu% | MEM: $mem% | Time: $time"
    echo "  Command: ${cmd:0:80}..."
done
echo ""

# Latest Training Output
echo "========================================"
echo "📝 LATEST TRAINING OUTPUT"
echo "========================================"
tmux capture-pane -t elio_sdxl_training -p | tail -30
echo ""

# Checkpoints
OUTPUT_DIR="/mnt/data/training/lora/elio/elio_identity"
if [ -d "$OUTPUT_DIR" ]; then
    echo "========================================"
    echo "💾 CHECKPOINTS"
    echo "========================================"
    ls -lht "$OUTPUT_DIR"/*.safetensors 2>/dev/null | head -5 | while read -r line; do
        echo "$line"
    done
    checkpoint_count=$(ls "$OUTPUT_DIR"/*.safetensors 2>/dev/null | wc -l)
    echo ""
    echo -e "${GREEN}Total checkpoints: $checkpoint_count${NC}"
fi
echo ""

# Training Log
LOG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/logs"
LATEST_LOG=$(ls -t "$LOG_DIR"/elio_elio_identity_sdxl_training_*.log 2>/dev/null | head -1)
if [ -f "$LATEST_LOG" ]; then
    echo "========================================"
    echo "📋 TRAINING LOG (Last 20 lines)"
    echo "========================================"
    echo "Log file: $LATEST_LOG"
    tail -20 "$LATEST_LOG"
fi

echo ""
echo "========================================"
echo "Press Ctrl+C to exit monitoring"
echo "Refresh in 10 seconds..."
echo "========================================"
