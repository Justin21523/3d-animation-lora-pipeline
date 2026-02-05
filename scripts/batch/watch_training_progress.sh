#!/bin/bash
#
# Real-time Training Progress with Progress Bar
#

OUTPUT_DIR="${1:-/mnt/data/ai_data/models/lora_sdxl/coco/miguel_identity}"
TOTAL_STEPS=3370  # For Miguel: 337 steps/epoch × 10 epochs
TOTAL_EPOCHS=10

# Get latest TensorBoard log directory
get_latest_log() {
    find "$OUTPUT_DIR/logs" -name "events.out.tfevents.*" -type f 2>/dev/null | sort -r | head -1
}

# Function to draw progress bar
draw_progress_bar() {
    local percent=$1
    local width=50
    local filled=$(( width * percent / 100 ))
    local empty=$(( width - filled ))
    
    printf "["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' '-'
    printf "] %3d%%" "$percent"
}

clear
echo "================================================================================"
echo "📊 SDXL LoRA 訓練即時進度監控"
echo "================================================================================"
echo "輸出目錄: $OUTPUT_DIR"
echo "總步數: $TOTAL_STEPS | 總 Epochs: $TOTAL_EPOCHS"
echo "================================================================================"
echo ""

LAST_STEP=0
START_TIME=$(date +%s)

while true; do
    # Move cursor to line 8
    tput cup 7 0
    tput ed
    
    echo "⏰ 更新時間: $(date '+%H:%M:%S')"
    echo ""
    
    # Get latest tensorboard events
    LOG_FILE=$(get_latest_log)
    
    if [ -n "$LOG_FILE" ]; then
        # Use Python to read TensorBoard data
        METRICS=$(python3 << PYEOF 2>/dev/null
try:
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator("$(dirname "$LOG_FILE")")
    ea.Reload()
    if 'loss/current' in ea.Tags().get('scalars', []):
        events = ea.Scalars('loss/current')
        if events:
            latest = events[-1]
            first = events[0]
            elapsed = latest.wall_time - first.wall_time
            steps_per_sec = latest.step / elapsed if elapsed > 0 else 0
            print(f"{latest.step}|{latest.value:.6f}|{steps_per_sec:.3f}")
except:
    print("0|0|0")
PYEOF
)
        
        IFS='|' read -r CURRENT_STEP CURRENT_LOSS SPEED <<< "$METRICS"
        
        if [ "$CURRENT_STEP" -gt 0 ]; then
            # Calculate progress
            PROGRESS=$(( CURRENT_STEP * 100 / TOTAL_STEPS ))
            CURRENT_EPOCH=$(echo "scale=2; $CURRENT_STEP / ($TOTAL_STEPS / $TOTAL_EPOCHS)" | bc)
            
            # Calculate time
            NOW=$(date +%s)
            ELAPSED=$(( NOW - START_TIME ))
            if [ "$SPEED" != "0" ] && [ $(echo "$SPEED > 0" | bc) -eq 1 ]; then
                REMAINING_STEPS=$(( TOTAL_STEPS - CURRENT_STEP ))
                ETA_SECONDS=$(echo "scale=0; $REMAINING_STEPS / $SPEED" | bc)
                ETA_HOURS=$(( ETA_SECONDS / 3600 ))
                ETA_MINS=$(( (ETA_SECONDS % 3600) / 60 ))
            else
                ETA_HOURS=0
                ETA_MINS=0
            fi
            
            # Display progress
            echo "🔥 訓練進度:"
            echo "  當前步數: $CURRENT_STEP / $TOTAL_STEPS"
            echo "  當前 Epoch: $CURRENT_EPOCH / $TOTAL_EPOCHS"
            echo "  "
            draw_progress_bar "$PROGRESS"
            echo ""
            echo ""
            
            echo "📈 訓練指標:"
            echo "  當前 Loss: $CURRENT_LOSS"
            echo "  訓練速度: $SPEED steps/sec"
            echo "  每步耗時: $(echo "scale=2; 1 / $SPEED" | bc 2>/dev/null || echo "N/A") 秒"
            echo ""
            
            echo "⏱️  時間預估:"
            echo "  已運行時間: $(( ELAPSED / 60 )) 分鐘"
            echo "  預估剩餘時間: ${ETA_HOURS}小時 ${ETA_MINS}分鐘"
            echo "  預估完成時間: $(date -d "+${ETA_SECONDS} seconds" '+%H:%M:%S' 2>/dev/null || echo "計算中...")"
            echo ""
            
            # Next checkpoint
            STEPS_PER_EPOCH=$(( TOTAL_STEPS / TOTAL_EPOCHS ))
            NEXT_CP_EPOCH=$(( (CURRENT_STEP / STEPS_PER_EPOCH / 2 + 1) * 2 ))
            if [ "$NEXT_CP_EPOCH" -le "$TOTAL_EPOCHS" ]; then
                STEPS_TO_CP=$(( NEXT_CP_EPOCH * STEPS_PER_EPOCH - CURRENT_STEP ))
                if [ $(echo "$SPEED > 0" | bc) -eq 1 ]; then
                    TIME_TO_CP=$(echo "scale=0; $STEPS_TO_CP / $SPEED / 60" | bc)
                    echo "💾 下一個 Checkpoint (Epoch $NEXT_CP_EPOCH):"
                    echo "  還需 $STEPS_TO_CP 步"
                    echo "  預估 $TIME_TO_CP 分鐘後"
                fi
            fi
            echo ""
            
            # GPU status
            GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits)
            IFS=',' read -r GPU_UTIL GPU_MEM_USED GPU_MEM_TOTAL GPU_TEMP <<< "$GPU_INFO"
            echo "🖥️  GPU 狀態:"
            echo "  使用率: ${GPU_UTIL}%"
            echo "  VRAM: ${GPU_MEM_USED} / ${GPU_MEM_TOTAL} MB"
            echo "  溫度: ${GPU_TEMP}°C"
            
            LAST_STEP=$CURRENT_STEP
        else
            echo "⏳ 等待訓練開始..."
            echo "  正在初始化模型和緩存 VAE latents..."
        fi
    else
        echo "⏳ 等待 TensorBoard 日誌..."
        echo "  訓練可能正在初始化..."
    fi
    
    echo ""
    echo "================================================================================"
    echo "每 5 秒自動更新 | Ctrl+C 退出"
    
    sleep 5
done
