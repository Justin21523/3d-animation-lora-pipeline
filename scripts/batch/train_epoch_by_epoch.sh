#!/usr/bin/bash
#
# Epoch-by-Epoch SDXL LoRA Training with Testing Workflow
# Usage: bash train_epoch_by_epoch.sh <character> <current_epoch>
#
# This script trains ONE epoch at a time, allowing checkpoint testing between epochs
#

set -e

CHARACTER=$1
CURRENT_EPOCH=${2:-1}  # Default to epoch 1 if not specified
MAX_EPOCHS=5

if [ -z "$CHARACTER" ]; then
    echo "Usage: $0 <character_name> [current_epoch]"
    echo "Available characters: jett, jerome, donnie, chase, flip, todd, paul, bello, beard"
    exit 1
fi

CONFIGS_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/super_wings_loras_sdxl"
KOHYA_DIR="/mnt/c/ai_projects/kohya_ss/sd-scripts"
BASE_CONFIG="$CONFIGS_DIR/super-wings-${CHARACTER}-identity-sdxl.toml"
EPOCH_CONFIG="/tmp/super-wings-${CHARACTER}-epoch${CURRENT_EPOCH}.toml"
OUTPUT_DIR="/mnt/data/training/lora/super-wings/${CHARACTER}_identity"
SESSION_NAME="sw_${CHARACTER}_ep${CURRENT_EPOCH}"

# Check if base config exists
if [ ! -f "$BASE_CONFIG" ]; then
    echo "❌ Config not found: $BASE_CONFIG"
    exit 1
fi

# Find the latest checkpoint if resuming
RESUME_FROM=""
if [ $CURRENT_EPOCH -gt 1 ]; then
    PREV_EPOCH=$((CURRENT_EPOCH - 1))
    # Look for checkpoint from previous epoch
    LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "*-$(printf "%06d" $((PREV_EPOCH * 700))).safetensors" 2>/dev/null | head -1)

    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "⚠️  找不到 epoch $PREV_EPOCH 的 checkpoint"
        echo "可用的 checkpoints:"
        ls -lh "$OUTPUT_DIR"/*.safetensors 2>/dev/null || echo "無"
        read -p "是否從頭開始訓練? (y/n): " confirm
        if [ "$confirm" != "y" ]; then
            exit 1
        fi
    else
        RESUME_FROM="$LATEST_CHECKPOINT"
        echo "📁 將從 checkpoint 繼續訓練: $(basename $RESUME_FROM)"
    fi
fi

# Create epoch-specific config
cp "$BASE_CONFIG" "$EPOCH_CONFIG"

# Modify config to train only up to current epoch
sed -i "s/max_train_epochs = 5/max_train_epochs = $CURRENT_EPOCH/g" "$EPOCH_CONFIG"

echo "=================================================================="
echo "Super Wings SDXL LoRA - Epoch-by-Epoch Training"
echo "=================================================================="
echo "角色: $CHARACTER"
echo "當前 Epoch: $CURRENT_EPOCH/$MAX_EPOCHS"
echo "Config: $EPOCH_CONFIG"
echo "Session: $SESSION_NAME"
if [ -n "$RESUME_FROM" ]; then
    echo "恢復自: $(basename $RESUME_FROM)"
fi
echo ""
echo "此次訓練將完成 epoch $CURRENT_EPOCH 並產生 checkpoint"
echo "=================================================================="
echo ""

# Kill existing session if exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
sleep 1

# Create new session
tmux new-session -d -s "$SESSION_NAME"

# Setup and run training
tmux send-keys -t "$SESSION_NAME" "cd $KOHYA_DIR" C-m
sleep 1
tmux send-keys -t "$SESSION_NAME" "conda activate kohya_ss" C-m
sleep 2

# Build training command
TRAIN_CMD="accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 4 sdxl_train_network.py --config_file=$EPOCH_CONFIG"

if [ -n "$RESUME_FROM" ]; then
    TRAIN_CMD="$TRAIN_CMD --network_weights=\"$RESUME_FROM\""
fi

# Launch training
tmux send-keys -t "$SESSION_NAME" "$TRAIN_CMD" C-m

echo "✓ Epoch $CURRENT_EPOCH 訓練已啟動: $SESSION_NAME"
echo ""
echo "監控訓練:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "訓練完成後:"
echo "  1. 測試 checkpoint: $OUTPUT_DIR/super-wings-${CHARACTER}-identity-sdxl-$(printf "%06d" $((CURRENT_EPOCH * 700))).safetensors"
echo "  2. 如果滿意，繼續下一個 epoch:"
echo "     bash $0 $CHARACTER $((CURRENT_EPOCH + 1))"
echo ""
if [ $CURRENT_EPOCH -eq $MAX_EPOCHS ]; then
    echo "🎉 這是最後一個 epoch！完成後即可訓練下一個角色。"
fi
echo ""
