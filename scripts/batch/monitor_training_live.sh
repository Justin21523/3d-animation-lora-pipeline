#!/bin/bash
# Live Training Monitor - 即時訓練監控腳本

print_header() {
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
}

print_section() {
    echo ""
    echo "──────────────────────────────────────────────────────────────────────────────"
    echo "🔹 $1"
    echo "──────────────────────────────────────────────────────────────────────────────"
}

clear

print_header "📊 SDXL LoRA 訓練即時監控"

echo "⏰ 監控時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. 檢查訓練進程
print_section "訓練進程狀態"
training_pids=$(ps aux | grep "sdxl_train_network.py" | grep -v grep)
if [ -z "$training_pids" ]; then
    echo "❌ 沒有運行中的訓練進程"
else
    echo "$training_pids" | awk '{printf "✅ PID: %s | CPU: %s%% | MEM: %s%% | 運行時間: %s\n", $2, $3, $4, $10}'
fi

# 2. GPU 狀態
print_section "GPU 使用狀態"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader | \
    awk -F', ' '{printf "🎮 GPU %s: %s使用率 | %s / %s VRAM | %s | %s / %s功耗\n", $1, $2, $3, $4, $5, $6, $7}'

# 3. 已保存的 Checkpoints
print_section "已保存的 Checkpoints"
checkpoint_count=0
for char_dir in /mnt/data/ai_data/models/lora_sdxl/*/*/; do
    checkpoints=$(ls "$char_dir"*.safetensors 2>/dev/null)
    if [ ! -z "$checkpoints" ]; then
        char_name=$(basename "$(dirname "$char_dir")")
        count=$(echo "$checkpoints" | wc -l)
        checkpoint_count=$((checkpoint_count + count))
        echo "📦 $char_name: $count checkpoints"
        echo "$checkpoints" | xargs ls -lth | head -3 | awk '{printf "   └─ %s (%s)\n", $9, $5}'
    fi
done

if [ $checkpoint_count -eq 0 ]; then
    echo "⏳ 尚未保存任何 checkpoint（第一個 checkpoint 會在 Epoch 2 保存）"
fi

# 4. TensorBoard 日誌
print_section "TensorBoard 日誌狀態"
log_files=$(find /mnt/data/ai_data/models/lora_sdxl -name "events.out.tfevents*" -mmin -30 2>/dev/null)
if [ -z "$log_files" ]; then
    echo "⚠️  最近 30 分鐘內沒有更新的 TensorBoard 日誌"
else
    echo "$log_files" | while read log_file; do
        size=$(du -h "$log_file" | awk '{print $1}')
        mtime=$(stat -c %y "$log_file" | cut -d'.' -f1)
        echo "📈 $log_file"
        echo "   └─ 大小: $size | 最後更新: $mtime"
    done
fi

# 5. 訓練日誌尾部
print_section "最近訓練輸出（最後 10 行）"
if tmux has-session -t batch_training 2>/dev/null; then
    tmux capture-pane -t batch_training -p | grep -E "epoch|step|loss|%|Training|Checkpoint" | tail -10
else
    echo "⚠️  tmux session 'batch_training' 未運行"
fi

# 6. 快捷指令提示
print_section "快捷指令"
echo "📺 查看即時進度條:    tmux attach -t batch_training"
echo "🔍 啟動 TensorBoard:   conda run -n kohya_ss tensorboard --logdir /mnt/data/ai_data/models/lora_sdxl/orion/orion_identity/logs --port 6006 --bind_all"
echo "⚡ 監控 GPU:           watch -n 2 nvidia-smi"
echo "🔄 重新整理此監控:     bash scripts/batch/monitor_training_live.sh"

print_header "監控完成"
