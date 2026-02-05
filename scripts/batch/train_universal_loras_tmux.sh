#!/bin/bash
################################################################################
# Train Universal LoRAs in Tmux Sessions
#
# This script launches training for all 3 universal LoRAs (pose, action, expression)
# in separate tmux sessions for easy monitoring.
#
# Usage:
#   bash scripts/batch/train_universal_loras_tmux.sh
#
# Author: LLMProvider Tooling
# Date: 2025-12-04
################################################################################

set -e  # Exit on error

# Configuration
KOHYA_DIR="/mnt/c/ai_projects/kohya_ss/sd-scripts"
CONFIG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/synthetic_loras_filtered"
LOG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/logs/synthetic_training"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "Universal LoRA 訓練啟動器"
echo "================================================================================"
echo ""

# Check if Kohya directory exists
if [ ! -d "$KOHYA_DIR" ]; then
    echo "❌ Error: Kohya sd-scripts directory not found at $KOHYA_DIR"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Function to launch training in tmux
launch_training() {
    local lora_type=$1
    local config_file=$2
    local session_name="train_universal_${lora_type}"
    local log_file="${LOG_DIR}/universal_${lora_type}_$(date +%Y%m%d_%H%M%S).log"

    echo -e "${BLUE}📦 啟動 universal_${lora_type} 訓練...${NC}"

    # Kill existing session if it exists
    tmux kill-session -t "$session_name" 2>/dev/null || true

    # Create new tmux session
    tmux new-session -d -s "$session_name"

    # Send training command
    tmux send-keys -t "$session_name" "cd $KOHYA_DIR" Enter
    tmux send-keys -t "$session_name" "conda activate kohya_ss" Enter
    tmux send-keys -t "$session_name" "echo '=== Training universal_${lora_type} LoRA ===' | tee $log_file" Enter
    tmux send-keys -t "$session_name" "echo 'Config: $config_file' | tee -a $log_file" Enter
    tmux send-keys -t "$session_name" "echo 'Started at: \$(date)' | tee -a $log_file" Enter
    tmux send-keys -t "$session_name" "echo '' | tee -a $log_file" Enter
    tmux send-keys -t "$session_name" "python sdxl_train_network.py --config_file=\"$config_file\" 2>&1 | tee -a $log_file" Enter

    echo -e "   Session: ${GREEN}$session_name${NC}"
    echo -e "   Log: ${log_file}"
    echo -e "   監控指令: ${YELLOW}tmux attach -t $session_name${NC}"
    echo ""

    sleep 2  # Give tmux time to start
}

# Dataset statistics
echo "📊 資料集統計:"
for lora_type in pose action expression; do
    dataset_dir="/mnt/data/ai_data/synthetic_lora_data/datasets/universal_${lora_type}/1_universal_${lora_type}"
    if [ -d "$dataset_dir" ]; then
        img_count=$(ls "$dataset_dir"/*.png 2>/dev/null | wc -l)
        echo "   universal_${lora_type}: ${img_count} 張"
    fi
done
echo ""

# Launch training sessions
echo "🚀 啟動訓練 sessions..."
echo ""

launch_training "pose" "${CONFIG_DIR}/universal_pose_sdxl.toml"
launch_training "action" "${CONFIG_DIR}/universal_action_sdxl.toml"
launch_training "expression" "${CONFIG_DIR}/universal_expression_sdxl.toml"

# Summary
echo "================================================================================"
echo "✅ 所有訓練已啟動！"
echo "================================================================================"
echo ""
echo "📋 監控指令:"
echo "   列出所有 sessions:  tmux ls"
echo "   連接 pose:         tmux attach -t train_universal_pose"
echo "   連接 action:       tmux attach -t train_universal_action"
echo "   連接 expression:   tmux attach -t train_universal_expression"
echo ""
echo "   離開 session (不停止訓練): Ctrl+B, 然後按 D"
echo ""
echo "📊 TensorBoard 監控:"
echo "   tensorboard --logdir=${LOG_DIR}"
echo ""
echo "⏸️  停止訓練:"
echo "   tmux kill-session -t train_universal_pose"
echo "   tmux kill-session -t train_universal_action"
echo "   tmux kill-session -t train_universal_expression"
echo ""
echo "📁 模型輸出位置:"
echo "   /mnt/c/ai_models/lora_sdxl/synthetic/universal_pose/"
echo "   /mnt/c/ai_models/lora_sdxl/synthetic/universal_action/"
echo "   /mnt/c/ai_models/lora_sdxl/synthetic/universal_expression/"
echo ""
echo "🔍 訓練預估時間 (RTX 5080, batch_size=4, gradient_accum=2):"
echo "   universal_pose:       ~10,000 steps, 25 epochs  (~8-10 小時)"
echo "   universal_action:     ~10,500 steps, 25 epochs  (~9-11 小時)"
echo "   universal_expression: ~10,000 steps, 25 epochs  (~8-10 小時)"
echo ""
echo "================================================================================"
