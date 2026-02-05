#!/bin/bash
# TensorBoard 啟動腳本

print_header() {
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
}

# 獲取參數
CHARACTER="${1:-orion}"  # 預設 orion
PORT="${2:-6006}"         # 預設 6006

# 檢測角色和電影名稱
case "$CHARACTER" in
    "orion")
        MOVIE="orion"
        CHAR_NAME="orion"
        ;;
    "elio")
        MOVIE="elio"
        CHAR_NAME="elio"
        ;;
    "bryce")
        MOVIE="elio"
        CHAR_NAME="bryce"
        ;;
    "caleb")
        MOVIE="elio"
        CHAR_NAME="caleb"
        ;;
    "alberto")
        MOVIE="luca"
        CHAR_NAME="alberto"
        ;;
    "tyler")
        MOVIE="turning-red"
        CHAR_NAME="tyler"
        ;;
    "miguel")
        MOVIE="coco"
        CHAR_NAME="miguel"
        ;;
    *)
        echo "❌ 未知角色: $CHARACTER"
        echo "✅ 支援的角色: orion, elio, bryce, caleb, alberto, tyler, miguel"
        exit 1
        ;;
esac

LOGDIR="/mnt/data/ai_data/models/lora_sdxl/${MOVIE}/${CHAR_NAME}_identity/logs"

print_header "🚀 啟動 TensorBoard - $CHARACTER"

echo "📂 日誌目錄: $LOGDIR"
echo "🌐 Port: $PORT"
echo ""

# 檢查日誌目錄是否存在
if [ ! -d "$LOGDIR" ]; then
    echo "❌ 錯誤: 日誌目錄不存在"
    echo "   $LOGDIR"
    exit 1
fi

# 檢查是否有日誌檔案
LOG_COUNT=$(find "$LOGDIR" -name "events.out.tfevents*" 2>/dev/null | wc -l)
if [ $LOG_COUNT -eq 0 ]; then
    echo "⚠️  警告: 沒有找到 TensorBoard 日誌檔案"
    echo "   訓練可能尚未開始或日誌尚未生成"
fi

# 檢查 port 是否被佔用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $PORT 已被佔用，嘗試終止舊進程..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null
    sleep 2
fi

echo ""
print_header "🎯 啟動 TensorBoard"
echo ""
echo "請在瀏覽器開啟:"
echo "   🌐 http://localhost:$PORT"
echo ""
echo "按 Ctrl+C 停止 TensorBoard"
echo ""
print_header ""

# 啟動 TensorBoard
conda run -n kohya_ss tensorboard --logdir "$LOGDIR" --port $PORT --bind_all
