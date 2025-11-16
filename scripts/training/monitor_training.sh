#!/bin/bash

echo "==================================================================="
echo "監控 LoRA 訓練進度 - iterative_overnight_v5"
echo "==================================================================="
echo ""

while true; do
    clear
    echo "==================================================================="
    echo "時間: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==================================================================="
    echo ""

    # GPU 狀態
    echo "📊 GPU 狀態:"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
        awk -F', ' '{printf "  GPU %s: %s\n  溫度: %s°C | 使用率: %s%% | VRAM: %sMB / %sMB\n\n", $1, $2, $3, $4, $5, $6}'

    # 訓練進程
    echo "🔧 訓練進程:"
    TRAIN_COUNT=$(ps aux | grep "train_network.py" | grep -v grep | wc -l)
    if [ $TRAIN_COUNT -gt 0 ]; then
        # 找到主訓練進程（CPU使用率最高的）
        MAIN_TRAIN=$(ps aux | grep "train_network.py" | grep -v grep | sort -k3 -rn | head -1)
        echo "$MAIN_TRAIN" | awk '{printf "  主進程 PID: %s | CPU: %s%% | MEM: %.1fGB\n", $2, $3, $6/1024/1024}'
        echo "  ✓ 訓練正在運行 (共 $TRAIN_COUNT 個進程，包含 data loaders)"
    else
        echo "  ⚠️  未發現訓練進程"
    fi
    echo ""

    # 檢查是否有已生成的模型
    echo "💾 生成的模型檔案:"
    MODEL_DIR="/mnt/data/ai_data/models/lora/luca/iterative_overnight_v5/iteration_1/luca_human"
    if [ -d "$MODEL_DIR" ]; then
        MODEL_COUNT=$(find "$MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
        if [ $MODEL_COUNT -gt 0 ]; then
            echo "  找到 $MODEL_COUNT 個 checkpoint(s):"
            find "$MODEL_DIR" -name "*.safetensors" -printf "    %f (%kKB) - %TY-%Tm-%Td %TH:%TM\n" | sort
        else
            echo "  尚未生成任何 checkpoint"
        fi
    else
        echo "  輸出目錄尚未創建"
    fi
    echo ""

    # 訓練日誌最後幾行
    echo "📝 最新訓練日誌（最後5行）:"
    LOG_DIR="/mnt/data/ai_data/models/lora/luca/iterative_overnight_v5/logs/iteration_1/luca_human"
    if [ -d "$LOG_DIR" ]; then
        LATEST_LOG=$(find "$LOG_DIR" -name "*.log" -type f 2>/dev/null | sort -r | head -1)
        if [ -n "$LATEST_LOG" ]; then
            tail -5 "$LATEST_LOG" 2>/dev/null | sed 's/^/  /'
        else
            echo "  尚未生成日誌檔案"
        fi
    else
        echo "  日誌目錄尚未創建"
    fi
    echo ""

    echo "==================================================================="
    echo "按 Ctrl+C 停止監控 | 每10秒自動更新"
    echo "==================================================================="

    sleep 10
done
