#!/bin/bash
################################################################################
# Train Universal LoRAs Sequentially
#
# This script trains the 3 universal LoRAs one at a time to avoid RAM overflow.
# Training order: pose → action → expression
#
# Usage:
#   bash scripts/batch/train_universal_loras_sequential.sh
#
# Author: LLMProvider Tooling
# Date: 2025-12-04
################################################################################

set -e

# Configuration
KOHYA_DIR="/mnt/c/ai_projects/kohya_ss/sd-scripts"
CONFIG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/synthetic_loras_filtered"
LOG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/logs/synthetic_training"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Training sequence
LORA_TYPES=("pose" "action" "expression")

echo "================================================================================"
echo "Universal LoRA 順序訓練啟動器"
echo "================================================================================"
echo ""
echo "訓練順序: pose → action → expression"
echo "策略: 一次只訓練一個，避免 RAM 超限"
echo ""

# Check if Kohya directory exists
if [ ! -d "$KOHYA_DIR" ]; then
    echo -e "${RED}❌ Error: Kohya sd-scripts directory not found at $KOHYA_DIR${NC}"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Function to train a single LoRA
train_lora() {
    local lora_type=$1
    local config_file="${CONFIG_DIR}/universal_${lora_type}_sdxl.toml"
    local session_name="train_universal_${lora_type}"
    local log_file="${LOG_DIR}/universal_${lora_type}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "================================================================================"
    echo -e "${BLUE}訓練 #$((current_idx + 1))/3: universal_${lora_type}${NC}"
    echo "================================================================================"
    echo ""
    echo "配置文件: $config_file"
    echo "日誌文件: $log_file"
    echo ""

    # Kill existing session if it exists
    tmux kill-session -t "$session_name" 2>/dev/null || true

    # Create new tmux session
    echo -e "${BLUE}🚀 啟動 tmux session: ${session_name}${NC}"
    tmux new-session -d -s "$session_name"

    # Send training command
    tmux send-keys -t "$session_name" "cd $KOHYA_DIR" Enter
    sleep 1
    tmux send-keys -t "$session_name" "conda activate kohya_ss" Enter
    sleep 2
    tmux send-keys -t "$session_name" "echo '=== Training universal_${lora_type} LoRA ===' | tee $log_file" Enter
    tmux send-keys -t "$session_name" "echo 'Config: $config_file' | tee -a $log_file" Enter
    tmux send-keys -t "$session_name" "echo 'Started at: \$(date)' | tee -a $log_file" Enter
    tmux send-keys -t "$session_name" "echo '' | tee -a $log_file" Enter
    sleep 1
    tmux send-keys -t "$session_name" "python sdxl_train_network.py --config_file=\"$config_file\" 2>&1 | tee -a $log_file" Enter

    echo -e "${GREEN}✅ Tmux session 已啟動${NC}"
    echo ""
    echo "監控指令:"
    echo -e "  ${YELLOW}tmux attach -t ${session_name}${NC}"
    echo -e "  ${YELLOW}tail -f ${log_file}${NC}"
    echo ""

    # Wait for training to start
    echo "等待訓練啟動..."
    sleep 10

    # Monitor training progress
    echo -e "${BLUE}📊 開始監控訓練進度...${NC}"
    echo "（每 30 秒檢查一次，按 Ctrl+C 停止監控但不停止訓練）"
    echo ""

    local check_count=0
    while true; do
        # Check if tmux session still exists
        if ! tmux has-session -t "$session_name" 2>/dev/null; then
            echo ""
            echo -e "${GREEN}✅ Training session completed or stopped${NC}"
            break
        fi

        # Check if training process is still running
        if ! pgrep -f "sdxl_train_network.py.*universal_${lora_type}" > /dev/null; then
            # Wait a bit more to ensure it's really finished
            sleep 5
            if ! pgrep -f "sdxl_train_network.py.*universal_${lora_type}" > /dev/null; then
                echo ""
                echo -e "${GREEN}✅ Training process completed${NC}"
                break
            fi
        fi

        # Every 10 checks (5 minutes), show latest progress
        if (( check_count % 10 == 0 )); then
            echo "--- $(date +%H:%M:%S) --- 訓練進行中..."
            if [ -f "$log_file" ]; then
                # Show last few lines with epoch/step info
                tail -n 3 "$log_file" | grep -E "epoch|steps|loss" || echo "  (等待訓練數據...)"
            fi
        fi

        sleep 30
        ((check_count++))
    done

    echo ""
    echo -e "${GREEN}✅ universal_${lora_type} 訓練完成！${NC}"
    echo "完成時間: $(date)"
    echo ""

    # Kill tmux session
    tmux kill-session -t "$session_name" 2>/dev/null || true

    # Show training summary if available
    if [ -f "$log_file" ]; then
        echo "訓練摘要:"
        grep -E "epoch.*saved" "$log_file" | tail -n 5 || echo "  (無摘要資訊)"
        echo ""
    fi
}

# Main training loop
echo "開始順序訓練..."
echo ""

for current_idx in "${!LORA_TYPES[@]}"; do
    lora_type="${LORA_TYPES[$current_idx]}"

    # Show progress
    echo "================================================================================"
    echo "進度: $((current_idx + 1))/3"
    echo "================================================================================"

    # Train this LoRA
    train_lora "$lora_type"

    # Small gap between trainings
    if [ $current_idx -lt $((${#LORA_TYPES[@]} - 1)) ]; then
        echo ""
        echo -e "${BLUE}⏸️  等待 10 秒後開始下一個訓練...${NC}"
        echo ""
        sleep 10
    fi
done

# Final summary
echo ""
echo "================================================================================"
echo -e "${GREEN}🎉 所有 Universal LoRAs 訓練完成！${NC}"
echo "================================================================================"
echo ""
echo "訓練完成時間: $(date)"
echo ""
echo "📁 模型輸出位置:"
echo "   /mnt/c/ai_models/lora_sdxl/synthetic/universal_pose/"
echo "   /mnt/c/ai_models/lora_sdxl/synthetic/universal_action/"
echo "   /mnt/c/ai_models/lora_sdxl/synthetic/universal_expression/"
echo ""
echo "📝 訓練日誌:"
echo "   ${LOG_DIR}/"
echo ""
echo "下一步: Phase 6 - Checkpoint Evaluation & Selection"
echo "   python scripts/evaluation/test_lora_checkpoints.py"
echo ""
echo "================================================================================"
