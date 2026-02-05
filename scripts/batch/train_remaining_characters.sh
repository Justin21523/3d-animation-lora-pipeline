#!/bin/bash
# Train remaining 3 characters (Alberto, Giulia, Russell)

set -e

echo "=========================================="
echo "訓練剩餘 3 個角色"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Start time
BATCH_START=$(date +%s)
echo "開始時間: $(date)"
echo ""

# Training counter
completed=0
failed=0

# Remaining configs
declare -a CONFIGS=(
    "configs/training/character_loras_sdxl/luca_alberto_human_sdxl.toml:Alberto Human (Luca)"
    "configs/training/character_loras_sdxl/luca_giulia_sdxl.toml:Giulia (Luca)"
    "configs/training/character_loras_sdxl/up_russell_sdxl.toml:Russell (Up)"
)

# Train each character
for config_info in "${CONFIGS[@]}"; do
    IFS=':' read -r config_path char_name <<< "$config_info"

    echo ""
    echo "=========================================="
    echo -e "${BLUE}訓練: $char_name${NC}"
    echo "配置: $config_path"
    echo "=========================================="

    if [ ! -f "$config_path" ]; then
        echo -e "${RED}錯誤: 配置文件不存在: $config_path${NC}"
        ((failed++))
        continue
    fi

    char_start=$(date +%s)
    echo "角色開始時間: $(date)"
    echo ""

    # Run training
    if cd /mnt/c/ai_projects/kohya_ss/sd-scripts && \
       conda run -n kohya_ss accelerate launch \
       --num_cpu_threads_per_process=2 \
       sdxl_train_network.py \
       --config_file "/mnt/c/ai_projects/3d-animation-lora-pipeline/$config_path"; then

        char_end=$(date +%s)
        char_duration=$((char_end - char_start))
        char_minutes=$((char_duration / 60))
        char_seconds=$((char_duration % 60))

        echo ""
        echo -e "${GREEN}✓ 完成: $char_name${NC}"
        echo "  耗時: ${char_minutes}分 ${char_seconds}秒"
        ((completed++))
    else
        echo ""
        echo -e "${RED}✗ 失敗: $char_name${NC}"
        ((failed++))
    fi

    echo ""
    echo "進度: $completed 完成, $failed 失敗"
    echo ""
done

# Summary
BATCH_END=$(date +%s)
BATCH_DURATION=$((BATCH_END - BATCH_START))
BATCH_HOURS=$((BATCH_DURATION / 3600))
BATCH_MINUTES=$(((BATCH_DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo -e "${GREEN}剩餘角色訓練完成${NC}"
echo "=========================================="
echo "完成: $completed / 3 角色"
echo "失敗: $failed / 3 角色"
echo "總耗時: ${BATCH_HOURS}小時 ${BATCH_MINUTES}分"
echo "結束時間: $(date)"
echo ""

if [ $completed -eq 3 ]; then
    echo -e "${GREEN}✓ 所有角色訓練成功!${NC}"
else
    echo -e "${RED}✗ 部分角色訓練失敗${NC}"
fi
