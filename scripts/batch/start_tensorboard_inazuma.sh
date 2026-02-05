#!/bin/bash
# TensorBoard launcher for Inazuma Eleven SDXL LoRA runs

set -e

PORT="${1:-6006}"
LOGDIR="/mnt/data/ai_data/models/lora_sdxl/inazuma-eleven"

echo "================================================================================"
echo "🚀 Launching TensorBoard - Inazuma Eleven"
echo "================================================================================"
echo "📂 Logdir: $LOGDIR"
echo "🌐 Port:   $PORT"
echo ""

if [ ! -d "$LOGDIR" ]; then
  echo "❌ ERROR: Logdir not found: $LOGDIR"
  exit 1
fi

echo "Open in browser:"
echo "  http://localhost:$PORT"
echo ""

conda run -n kohya_ss tensorboard --logdir "$LOGDIR" --port "$PORT" --bind_all

