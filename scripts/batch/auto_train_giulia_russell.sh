#!/bin/bash
# Auto-train Giulia and Russell sequentially
# Ensures Russell starts automatically after Giulia completes

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "自動化訓練: Giulia → Russell"
echo "=========================================="
echo ""
echo "開始時間: $(date)"
echo ""

# Train Giulia
echo ""
echo "=========================================="
echo -e "${BLUE}訓練: Giulia (Luca)${NC}"
echo "配置: configs/training/character_loras_sdxl/luca_giulia_sdxl.toml"
echo "=========================================="

giulia_start=$(date +%s)
echo "Giulia 開始時間: $(date)"
echo ""

if cd /mnt/c/ai_projects/kohya_ss/sd-scripts && \
   conda run -n kohya_ss accelerate launch \
   --num_cpu_threads_per_process=2 \
   sdxl_train_network.py \
   --config_file /mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/luca_giulia_sdxl.toml; then

    giulia_end=$(date +%s)
    giulia_duration=$((giulia_end - giulia_start))
    giulia_minutes=$((giulia_duration / 60))
    giulia_seconds=$((giulia_duration % 60))

    echo ""
    echo -e "${GREEN}✓ Giulia 訓練完成${NC}"
    echo "  耗時: ${giulia_minutes}分 ${giulia_seconds}秒"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Giulia 訓練失敗，中止後續訓練${NC}"
    exit 1
fi

# Wait 5 seconds before starting Russell
echo "等待 5 秒後開始 Russell 訓練..."
sleep 5

# Train Russell
echo ""
echo "=========================================="
echo -e "${BLUE}訓練: Russell (Up)${NC}"
echo "配置: configs/training/character_loras_sdxl/up_russell_sdxl.toml"
echo "=========================================="

russell_start=$(date +%s)
echo "Russell 開始時間: $(date)"
echo ""

if cd /mnt/c/ai_projects/kohya_ss/sd-scripts && \
   conda run -n kohya_ss accelerate launch \
   --num_cpu_threads_per_process=2 \
   sdxl_train_network.py \
   --config_file /mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/up_russell_sdxl.toml; then

    russell_end=$(date +%s)
    russell_duration=$((russell_end - russell_start))
    russell_minutes=$((russell_duration / 60))
    russell_seconds=$((russell_duration % 60))

    echo ""
    echo -e "${GREEN}✓ Russell 訓練完成${NC}"
    echo "  耗時: ${russell_minutes}分 ${russell_seconds}秒"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Russell 訓練失敗${NC}"
    exit 1
fi

# Summary
total_end=$(date +%s)
total_start=$giulia_start
total_duration=$((total_end - total_start))
total_hours=$((total_duration / 3600))
total_minutes=$(((total_duration % 3600) / 60))

echo ""
echo "=========================================="
echo -e "${GREEN}自動化訓練完成${NC}"
echo "=========================================="
echo "Giulia 耗時: ${giulia_minutes}分 ${giulia_seconds}秒"
echo "Russell 耗時: ${russell_minutes}分 ${russell_seconds}秒"
echo "總耗時: ${total_hours}小時 ${total_minutes}分"
echo "結束時間: $(date)"
echo ""
echo -e "${GREEN}✓ Giulia 和 Russell 訓練全部成功!${NC}"
echo ""
echo "輸出檔案:"
echo "  Giulia:"
echo "    /mnt/c/ai_models/lora_sdxl/luca/giulia_identity/giulia_lora_sdxl-000001.safetensors"
echo "    /mnt/c/ai_models/lora_sdxl/luca/giulia_identity/giulia_lora_sdxl.safetensors"
echo "  Russell:"
echo "    /mnt/c/ai_models/lora_sdxl/up/russell_identity/russell_lora_sdxl-000001.safetensors"
echo "    /mnt/c/ai_models/lora_sdxl/up/russell_identity/russell_lora_sdxl.safetensors"
echo ""
