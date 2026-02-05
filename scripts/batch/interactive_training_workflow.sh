#!/usr/bin/bash
#
# Interactive SDXL LoRA Training Workflow
# Trains all 9 Super Wings characters, one epoch at a time with testing checkpoints
#

set -e

CHARACTERS=("jett" "jerome" "donnie" "chase" "flip" "todd" "paul" "bello" "beard")
MAX_EPOCHS=5
SCRIPT_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/batch"
TRAINING_SCRIPT="$SCRIPT_DIR/train_epoch_by_epoch.sh"

echo "=================================================================="
echo "🎬 Super Wings SDXL LoRA - Interactive Training Workflow"
echo "=================================================================="
echo ""
echo "訓練計畫:"
echo "  • 9 個角色: ${CHARACTERS[@]}"
echo "  • 每個角色 5 個 epochs"
echo "  • 每個 epoch 後停下來測試 checkpoint"
echo "  • 總共需要測試 45 個 checkpoints"
echo ""
echo "=================================================================="
echo ""

read -p "是否開始訓練? (y/n): " start_confirm
if [ "$start_confirm" != "y" ]; then
    echo "已取消"
    exit 0
fi

# Loop through each character
for CHAR_IDX in "${!CHARACTERS[@]}"; do
    CHARACTER="${CHARACTERS[$CHAR_IDX]}"
    CHAR_NUM=$((CHAR_IDX + 1))

    echo ""
    echo "=================================================================="
    echo "🎯 角色 $CHAR_NUM/9: $CHARACTER"
    echo "=================================================================="
    echo ""

    # Loop through epochs for this character
    for EPOCH in $(seq 1 $MAX_EPOCHS); do
        echo ""
        echo "──────────────────────────────────────────────────────────────"
        echo "📊 $CHARACTER - Epoch $EPOCH/$MAX_EPOCHS"
        echo "──────────────────────────────────────────────────────────────"
        echo ""

        # Run training for this epoch
        bash "$TRAINING_SCRIPT" "$CHARACTER" "$EPOCH"

        echo ""
        echo "⏳ 等待 epoch $EPOCH 訓練完成..."
        echo "   請在另一個終端監控: tmux attach -t sw_${CHARACTER}_ep${EPOCH}"
        echo ""
        read -p "Epoch $EPOCH 是否已完成訓練? (y/n): " epoch_done

        while [ "$epoch_done" != "y" ]; do
            echo "繼續等待..."
            sleep 10
            read -p "Epoch $EPOCH 是否已完成訓練? (y/n): " epoch_done
        done

        echo ""
        echo "🧪 請測試 checkpoint..."
        OUTPUT_DIR="/mnt/data/training/lora/super-wings/${CHARACTER}_identity"
        echo "   Checkpoint 位置: $OUTPUT_DIR"
        ls -lh "$OUTPUT_DIR"/*.safetensors 2>/dev/null || echo "⚠️  找不到 checkpoint"
        echo ""

        read -p "Checkpoint 測試是否滿意? (y/n/retry): " test_result

        if [ "$test_result" == "retry" ]; then
            echo "重新訓練 epoch $EPOCH..."
            # Clean up this epoch's checkpoint
            find "$OUTPUT_DIR" -name "*-$(printf "%06d" $EPOCH).safetensors" -delete 2>/dev/null || true
            EPOCH=$((EPOCH - 1))  # Retry this epoch
            continue
        elif [ "$test_result" != "y" ]; then
            echo "❌ 訓練中止"
            exit 1
        fi

        echo "✅ Epoch $EPOCH checkpoint 通過測試"

        if [ $EPOCH -eq $MAX_EPOCHS ]; then
            echo ""
            echo "🎉 $CHARACTER 的 5 個 epochs 全部完成！"
            echo ""
        fi
    done

    echo ""
    echo "✅ $CHARACTER 訓練完成！($CHAR_NUM/9)"
    echo ""

    if [ $CHAR_NUM -lt 9 ]; then
        read -p "是否繼續訓練下一個角色 (${CHARACTERS[$((CHAR_IDX + 1))]})? (y/n): " next_char
        if [ "$next_char" != "y" ]; then
            echo "暫停訓練流程"
            echo "稍後可繼續從 ${CHARACTERS[$((CHAR_IDX + 1))]} 開始"
            exit 0
        fi
    fi
done

echo ""
echo "=================================================================="
echo "🎊 全部 9 個角色訓練完成！"
echo "=================================================================="
echo ""
echo "總結:"
echo "  ✅ 完成角色數: 9/9"
echo "  ✅ 總 checkpoints: 45"
echo ""
echo "下一步: 選出每個角色的最佳 checkpoint 並進行最終評估"
echo ""
