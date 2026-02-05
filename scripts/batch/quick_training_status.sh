#!/bin/bash
# Quick Training Status Check

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader

echo ""
echo "=== Training Processes ==="
ps aux | grep "sdxl_train" | grep -v grep

echo ""
echo "=== Recent Checkpoints (Alberto) ==="
ls -lth /mnt/c/ai_models/lora_sdxl/luca/alberto_seamonster_identity/*.safetensors 2>/dev/null | head -5 || echo "No checkpoints yet"

echo ""
echo "=== Recent Checkpoints (Luca) ==="
ls -lth /mnt/c/ai_models/lora_sdxl/luca/luca_seamonster_identity/*.safetensors 2>/dev/null | head -5 || echo "No checkpoints yet"

echo ""
echo "=== System Memory ==="
free -h | grep "Mem:"
