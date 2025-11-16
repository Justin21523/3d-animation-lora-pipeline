#!/bin/bash

# LoRA Training Monitoring Script
# 多種監控方式，實時追蹤訓練進度

BASE_DIR="/mnt/data/ai_data/models/lora/luca/iterative_overnight_v5"

echo "======================================================================="
echo "LORA TRAINING MONITOR"
echo "======================================================================="
echo ""

# 函數：顯示訓練進度
show_progress() {
    echo "📊 TRAINING PROGRESS:"
    echo "-------------------------------------------------------------------"

    # 顯示已生成的 checkpoint
    echo "✓ Checkpoints generated:"
    find "$BASE_DIR" -name "*.safetensors" -type f -printf "%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n" | sort -r | head -20
    echo ""

    # 顯示當前迭代
    echo "📁 Current iterations:"
    ls -d "$BASE_DIR"/iteration_* 2>/dev/null | while read iter_dir; do
        echo "  $(basename $iter_dir):"
        ls -d "$iter_dir"/*/ 2>/dev/null | while read char_dir; do
            char_name=$(basename "$char_dir")
            checkpoint_count=$(find "$char_dir" -name "*.safetensors" -type f | wc -l)
            echo "    - $char_name: $checkpoint_count checkpoints"
        done
    done
    echo ""
}

# 函數：顯示 GPU 狀態
show_gpu() {
    echo "🖥️  GPU STATUS:"
    echo "-------------------------------------------------------------------"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader,nounits | \
        awk -F',' '{printf "  GPU: %s\n  Temperature: %s°C\n  GPU Util: %s%%\n  Memory Util: %s%%\n  VRAM: %s MB / %s MB\n", $1, $2, $3, $4, $5, $6}'
    echo ""
}

# 函數：顯示進程狀態
show_processes() {
    echo "⚙️  TRAINING PROCESSES:"
    echo "-------------------------------------------------------------------"

    # 主進程
    main_pid=$(ps aux | grep "launch_iterative_training" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$main_pid" ]; then
        echo "  Main process (PID: $main_pid):"
        ps -p "$main_pid" -o pid,etime,%cpu,%mem,stat,cmd --no-headers | \
            awk '{printf "    Runtime: %s | CPU: %s%% | Memory: %s%% | Status: %s\n", $2, $3, $4, $5}'
    else
        echo "  ⚠️  Main process not found"
    fi

    # 訓練進程
    train_pids=$(ps aux | grep "train_network.py" | grep -v grep | awk '{print $2}')
    if [ -n "$train_pids" ]; then
        echo ""
        echo "  Active training processes:"
        echo "$train_pids" | while read pid; do
            ps -p "$pid" -o pid,etime,%cpu,%mem,stat --no-headers | \
                awk '{printf "    PID %s: Runtime=%s CPU=%s%% Memory=%s%% Status=%s\n", $1, $2, $3, $4, $5}'
        done
    fi
    echo ""
}

# 函數：顯示最新日誌
show_logs() {
    echo "📝 RECENT TRAINING LOGS:"
    echo "-------------------------------------------------------------------"

    # 找最新的日誌文件
    latest_log=$(find "$BASE_DIR/logs" -name "*.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -n "$latest_log" ]; then
        echo "  Latest log: $latest_log"
        echo ""
        tail -20 "$latest_log" 2>/dev/null | sed 's/^/    /'
    else
        echo "  No log files found in $BASE_DIR/logs"
    fi
    echo ""
}

# 函數：估算剩餘時間
estimate_time() {
    echo "⏱️  TIME ESTIMATION:"
    echo "-------------------------------------------------------------------"

    # 計算已用時間
    main_pid=$(ps aux | grep "launch_iterative_training" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$main_pid" ]; then
        elapsed=$(ps -p "$main_pid" -o etime --no-headers | tr -d ' ')
        echo "  Elapsed time: $elapsed"
        echo "  Target: 14 hours total"
    fi
    echo ""
}

# 主監控循環
if [ "$1" == "--watch" ]; then
    # 持續監控模式
    while true; do
        clear
        show_progress
        show_gpu
        show_processes
        show_logs
        estimate_time
        echo "======================================================================="
        echo "Press Ctrl+C to exit | Refreshing in 10 seconds..."
        echo "======================================================================="
        sleep 10
    done
else
    # 單次顯示模式
    show_progress
    show_gpu
    show_processes
    show_logs
    estimate_time
    echo "======================================================================="
    echo "Tip: Run with --watch for continuous monitoring"
    echo "======================================================================="
fi
