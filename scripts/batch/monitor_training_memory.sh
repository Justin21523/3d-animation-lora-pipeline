#!/bin/bash
################################################################################
# Training Memory Monitor with Automatic Safety Shutdown
#
# Monitors GPU VRAM, System RAM, and training processes
# Automatically stops training if memory usage exceeds safe thresholds
#
# Usage:
#   bash scripts/batch/monitor_training_memory.sh
#
# Author: LLMProvider Tooling
# Date: 2025-12-04
################################################################################

# Color codes
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thresholds (in percentage)
GPU_VRAM_WARN_THRESHOLD=85
GPU_VRAM_CRITICAL_THRESHOLD=95
SYSTEM_RAM_WARN_THRESHOLD=85
SYSTEM_RAM_CRITICAL_THRESHOLD=95

# Monitoring interval (seconds)
INTERVAL=5

# Log file
LOG_FILE="/mnt/c/ai_projects/3d-animation-lora-pipeline/logs/memory_monitor.log"

# Emergency stop flag
EMERGENCY_STOP=false

# Initialize log
mkdir -p "$(dirname "$LOG_FILE")"
echo "=== Memory Monitor Started $(date) ===" >> "$LOG_FILE"

# Function to get GPU memory usage
get_gpu_usage() {
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "%d %.1f %d", $1, ($1/$2)*100, $3}'
}

# Function to get system memory usage
get_ram_usage() {
    free | grep Mem | awk '{printf "%d %.1f", $3/1024/1024, ($3/$2)*100}'
}

# Function to stop all training
emergency_stop_training() {
    echo -e "${RED}🚨 EMERGENCY STOP TRIGGERED!${NC}"
    echo "$(date): EMERGENCY STOP - Memory threshold exceeded" >> "$LOG_FILE"

    # Kill all training sessions
    tmux kill-session -t train_universal_pose 2>/dev/null
    tmux kill-session -t train_universal_action 2>/dev/null
    tmux kill-session -t train_universal_expression 2>/dev/null

    echo -e "${RED}✅ All training sessions stopped${NC}"
    echo "$(date): All training sessions terminated" >> "$LOG_FILE"

    EMERGENCY_STOP=true
}

# Function to display status
display_status() {
    local gpu_used=$1
    local gpu_percent=$2
    local gpu_util=$3
    local ram_gb=$4
    local ram_percent=$5
    local iteration=$6

    clear
    echo "================================================================================"
    echo "訓練記憶體監控器 (按 Ctrl+C 停止監控)"
    echo "================================================================================"
    echo ""
    echo "📊 GPU VRAM (RTX 5080):"

    # GPU status with color coding
    if (( $(echo "$gpu_percent >= $GPU_VRAM_CRITICAL_THRESHOLD" | bc -l) )); then
        echo -e "   ${RED}⚠️  CRITICAL: ${gpu_used} MB (${gpu_percent}% used)${NC}"
        echo -e "   ${RED}🚨 超過臨界值！即將自動停止訓練...${NC}"
        emergency_stop_training
    elif (( $(echo "$gpu_percent >= $GPU_VRAM_WARN_THRESHOLD" | bc -l) )); then
        echo -e "   ${YELLOW}⚠️  WARNING: ${gpu_used} MB (${gpu_percent}% used)${NC}"
    else
        echo -e "   ${GREEN}✅ OK: ${gpu_used} MB (${gpu_percent}% used)${NC}"
    fi

    echo -e "   GPU 使用率: ${gpu_util}%"
    echo ""

    echo "💾 System RAM:"
    if (( $(echo "$ram_percent >= $SYSTEM_RAM_CRITICAL_THRESHOLD" | bc -l) )); then
        echo -e "   ${RED}⚠️  CRITICAL: ${ram_gb} GB (${ram_percent}% used)${NC}"
        echo -e "   ${RED}🚨 超過臨界值！即將自動停止訓練...${NC}"
        emergency_stop_training
    elif (( $(echo "$ram_percent >= $SYSTEM_RAM_WARN_THRESHOLD" | bc -l) )); then
        echo -e "   ${YELLOW}⚠️  WARNING: ${ram_gb} GB (${ram_percent}% used)${NC}"
    else
        echo -e "   ${GREEN}✅ OK: ${ram_gb} GB (${ram_percent}% used)${NC}"
    fi
    echo ""

    echo "🔍 Training Sessions:"
    if tmux ls 2>/dev/null | grep -q "train_universal"; then
        tmux ls 2>/dev/null | grep "train_universal" | while read -r line; do
            echo -e "   ${GREEN}✅${NC} $line"
        done
    else
        echo -e "   ${YELLOW}⚠️  No training sessions running${NC}"
    fi
    echo ""

    echo "📈 安全閾值設定:"
    echo "   GPU VRAM: 警告 ${GPU_VRAM_WARN_THRESHOLD}%, 臨界 ${GPU_VRAM_CRITICAL_THRESHOLD}%"
    echo "   系統 RAM: 警告 ${SYSTEM_RAM_WARN_THRESHOLD}%, 臨界 ${SYSTEM_RAM_CRITICAL_THRESHOLD}%"
    echo ""

    echo "⏰ 監控週期: 每 ${INTERVAL} 秒"
    echo "📝 日誌: ${LOG_FILE}"
    echo ""
    echo "================================================================================"
}

# Main monitoring loop
iteration=0
while true; do
    if [ "$EMERGENCY_STOP" = true ]; then
        echo ""
        echo -e "${RED}訓練已因記憶體超限被自動停止${NC}"
        echo "請檢查日誌: $LOG_FILE"
        exit 1
    fi

    # Get current memory stats
    read gpu_used gpu_percent gpu_util < <(get_gpu_usage)
    read ram_gb ram_percent < <(get_ram_usage)

    # Display status
    display_status "$gpu_used" "$gpu_percent" "$gpu_util" "$ram_gb" "$ram_percent" "$iteration"

    # Log to file every 12 iterations (1 minute)
    if (( iteration % 12 == 0 )); then
        echo "$(date): GPU=${gpu_percent}% (${gpu_used}MB), RAM=${ram_percent}% (${ram_gb}GB)" >> "$LOG_FILE"
    fi

    sleep "$INTERVAL"
    ((iteration++))
done
