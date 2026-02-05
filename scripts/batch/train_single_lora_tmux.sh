#!/bin/bash
################################################################################
# Train Single Universal LoRA in Tmux
#
# Usage:
#   bash scripts/batch/train_single_lora_tmux.sh pose
#   bash scripts/batch/train_single_lora_tmux.sh action
#   bash scripts/batch/train_single_lora_tmux.sh expression
#
# Author: LLMProvider Tooling
# Date: 2025-12-04
################################################################################

LORA_TYPE=$1

if [ -z "$LORA_TYPE" ]; then
    echo "❌ Error: Please specify LoRA type (pose, action, or expression)"
    echo ""
    echo "Usage:"
    echo "  bash $0 pose"
    echo "  bash $0 action"
    echo "  bash $0 expression"
    exit 1
fi

# Configuration
KOHYA_DIR="/mnt/c/ai_projects/kohya_ss/sd-scripts"
CONFIG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/synthetic_loras_filtered"
LOG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/logs/synthetic_training"

CONFIG_FILE="${CONFIG_DIR}/universal_${LORA_TYPE}_sdxl.toml"
SESSION_NAME="train_universal_${LORA_TYPE}"
LOG_FILE="${LOG_DIR}/universal_${LORA_TYPE}_$(date +%Y%m%d_%H%M%S).log"

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "Universal ${LORA_TYPE^^} LoRA 訓練啟動"
echo "================================================================================"
echo ""
echo "配置文件: $CONFIG_FILE"
echo "日誌文件: $LOG_FILE"
echo "Tmux session: $SESSION_NAME"
echo ""

# Kill existing session if it exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  停止現有的 $SESSION_NAME session..."
    tmux kill-session -t "$SESSION_NAME"
    sleep 2
fi

# Create new tmux session and run training
echo "🚀 啟動訓練..."
tmux new-session -d -s "$SESSION_NAME" bash -c "
    cd $KOHYA_DIR && \
    conda activate kohya_ss && \
    echo '================================================================================' | tee $LOG_FILE && \
    echo 'Training universal_${LORA_TYPE} LoRA' | tee -a $LOG_FILE && \
    echo 'Config: $CONFIG_FILE' | tee -a $LOG_FILE && \
    echo 'Started at: \$(date)' | tee -a $LOG_FILE && \
    echo '================================================================================' | tee -a $LOG_FILE && \
    echo '' | tee -a $LOG_FILE && \
    python sdxl_train_network.py --config_file=\"$CONFIG_FILE\" 2>&1 | tee -a $LOG_FILE && \
    echo '' | tee -a $LOG_FILE && \
    echo '================================================================================' | tee -a $LOG_FILE && \
    echo '✅ Training completed at: \$(date)' | tee -a $LOG_FILE && \
    echo '================================================================================' | tee -a $LOG_FILE && \
    echo '' && \
    echo 'Press Enter to close this session...' && \
    read
"

sleep 3

# Verify training started
if pgrep -f "sdxl_train_network.*universal_${LORA_TYPE}" > /dev/null; then
    echo ""
    echo "✅ 訓練已成功啟動！"
    echo ""
    echo "📊 監控指令:"
    echo "   連接 tmux:    tmux attach -t $SESSION_NAME"
    echo "   查看日誌:      tail -f $LOG_FILE"
    echo "   離開 tmux:    按 Ctrl+B 然後按 D"
    echo ""
    echo "⏸️  停止訓練:"
    echo "   tmux kill-session -t $SESSION_NAME"
    echo ""
else
    echo ""
    echo "⚠️  訓練可能未成功啟動，請檢查日誌:"
    echo "   tail -f $LOG_FILE"
    echo ""
fi
