#!/bin/bash
# Monitor SAM2 Instance Segmentation Progress

OUTPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances"
TOTAL_FRAMES=4323
LOG_FILE="logs/sam2_sampled_$(ls -t logs/sam2_sampled_*.log 2>/dev/null | head -1 | xargs basename)"

echo "=== SAM2 處理進度監控 ==="
echo ""
echo "📁 輸出目錄: $OUTPUT_DIR"
echo "🎯 目標幀數: $TOTAL_FRAMES"
echo ""

# Check if processing is running
if tmux has-session -t luca_sam2 2>/dev/null; then
    echo "✓ tmux 會話 'luca_sam2' 正在運行"
else
    echo "⚠️ tmux 會話 'luca_sam2' 未運行"
    exit 1
fi

# Check GPU status
echo ""
echo "🎮 GPU 狀態:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader | \
    awk -F', ' '{printf "   使用率: %s | 記憶體: %s / %s\n", $1, $3, $4}'

# Count processed instances
if [ -d "$OUTPUT_DIR" ]; then
    CURRENT_COUNT=$(ls "$OUTPUT_DIR" 2>/dev/null | wc -l)
    echo ""
    echo "📊 處理進度:"
    echo "   已生成實例: $CURRENT_COUNT"

    # Estimate frames processed (assuming avg 8 instances per frame)
    FRAMES_PROCESSED=$((CURRENT_COUNT / 8))
    FRAMES_REMAINING=$((TOTAL_FRAMES - FRAMES_PROCESSED))

    PROGRESS=$((CURRENT_COUNT * 100 / (TOTAL_FRAMES * 8)))
    echo "   估計已處理幀: $FRAMES_PROCESSED / $TOTAL_FRAMES"
    echo "   進度: ${PROGRESS}%"

    # Progress bar
    BAR_LENGTH=50
    FILLED=$((PROGRESS * BAR_LENGTH / 100))
    printf "   ["
    printf "%${FILLED}s" | tr ' ' '='
    printf "%$((BAR_LENGTH - FILLED))s" | tr ' ' '-'
    printf "]\n"

    # Show latest files
    echo ""
    echo "📄 最新處理文件:"
    ls -t "$OUTPUT_DIR" | head -3 | sed 's/^/   /'
else
    echo ""
    echo "⚠️ 輸出目錄尚未創建"
fi

# Show recent log output
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "📋 最新日誌 (最後 10 行):"
    tail -10 "$LOG_FILE" 2>/dev/null | grep -v "^$" | sed 's/^/   /'
fi

echo ""
echo "💡 提示: 使用 'tmux attach -t luca_sam2' 查看即時輸出"
echo "💡 使用 'watch -n 30 bash $0' 每 30 秒自動更新"
