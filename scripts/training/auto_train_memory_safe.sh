#!/bin/bash
# 記憶體安全的自動化訓練腳本

set -e

CHARACTERS=(2 3 4 5 6 7)
BASE_MODEL="/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/anything-v5-PrtRE.safetensors"
EVAL_DIR="/mnt/data/ai_data/lora_evaluation"

# 清理記憶體函數
cleanup_memory() {
    echo "🧹 清理記憶體..."
    pkill -f train_network.py 2>/dev/null || true
    pkill -f evaluate_lora.py 2>/dev/null || true
    sleep 5
    sync
    echo "✓ 記憶體清理完成"
}

# 檢查記憶體函數
check_memory() {
    AVAILABLE=$(free -g | awk '/^Mem:/{print $7}')
    echo "📊 可用記憶體: ${AVAILABLE}GB"
    if [ "$AVAILABLE" -lt 10 ]; then
        echo "⚠️  記憶體不足，等待釋放..."
        cleanup_memory
        sleep 10
    fi
}

echo "=============================================="
echo "  🚀 記憶體安全的自動化LoRA訓練流程"
echo "=============================================="
echo "開始時間: $(date)"
echo ""

for char_num in "${CHARACTERS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "  📌 Character${char_num} 開始"
    echo "======================================================================"

    # 訓練前清理和檢查
    cleanup_memory
    check_memory

    # 訓練
    echo "⏰ $(date '+%H:%M:%S') - 開始訓練 Character${char_num}..."

    conda run -n ai_env python sd-scripts/train_network.py \
        --config_file configs/character_loras/character${char_num}_config_optimized.toml \
        > /tmp/character${char_num}_training.log 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ $(date '+%H:%M:%S') - Character${char_num} 訓練完成"
    else
        echo "❌ $(date '+%H:%M:%S') - Character${char_num} 訓練失敗！"
        echo "錯誤日誌："
        tail -30 /tmp/character${char_num}_training.log
        cleanup_memory
        continue
    fi

    # 訓練後清理
    cleanup_memory
    echo "⏳ 等待GPU和記憶體冷卻 (15秒)..."
    sleep 15

    # 測試
    echo "🧪 $(date '+%H:%M:%S') - 開始測試 Character${char_num}..."

    LORA_PATH="/mnt/data/ai_data/models/lora/yokai_characters/character${char_num}/yokai_character${char_num}_lora.safetensors"
    OUTPUT_DIR="${EVAL_DIR}/character${char_num}_$(date +%Y%m%d_%H%M%S)"

    if [ -f "$LORA_PATH" ]; then
        conda run -n ai_env python scripts/evaluate_lora.py \
            "$LORA_PATH" \
            --base_model "$BASE_MODEL" \
            --output_dir "$OUTPUT_DIR" \
            --num_samples 8 \
            --seed 42 \
            > /tmp/character${char_num}_eval.log 2>&1

        if [ $? -eq 0 ]; then
            echo "✅ $(date '+%H:%M:%S') - Character${char_num} 測試完成"
            echo "   測試圖片: $OUTPUT_DIR"
        else
            echo "⚠️  $(date '+%H:%M:%S') - Character${char_num} 測試失敗（不影響後續訓練）"
        fi
    else
        echo "⚠️  找不到LoRA檔案: $LORA_PATH"
    fi

    # 測試後清理
    cleanup_memory

    echo "======================================================================"
    echo "  ✅ Character${char_num} 完成"
    echo "======================================================================"
    echo ""
done

echo ""
echo "=============================================="
echo "  🎉 所有角色訓練和測試完成！"
echo "=============================================="
echo "結束時間: $(date)"
echo ""
echo "生成的模型："
ls -lh /mnt/data/ai_data/models/lora/yokai_characters/character*/yokai_character*_lora.safetensors 2>/dev/null
