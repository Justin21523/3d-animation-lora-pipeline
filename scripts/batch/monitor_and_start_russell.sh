#!/bin/bash
# Monitor Giulia training and auto-start Russell when complete
# This script runs in background and polls for Giulia completion

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "自動監控：等待 Giulia 完成後啟動 Russell"
echo "=========================================="
echo ""
echo "監控開始時間: $(date)"
echo ""

# Function to check if Giulia training is still running
check_giulia_running() {
    ps aux | grep -E "sdxl_train_network.*giulia" | grep -v grep > /dev/null
    return $?
}

# Function to check if Giulia checkpoints exist
check_giulia_checkpoints() {
    local checkpoint_dir="/mnt/c/ai_models/lora_sdxl/luca/giulia_identity"
    if [ -f "$checkpoint_dir/giulia_lora_sdxl.safetensors" ] || [ -f "$checkpoint_dir/giulia_lora_sdxl-000002.safetensors" ]; then
        return 0
    fi
    return 1
}

# Poll every 30 seconds
echo "正在監控 Giulia 訓練狀態..."
echo "檢查間隔: 30 秒"
echo ""

while true; do
    if check_giulia_running; then
        echo "[$(date +%H:%M:%S)] Giulia 訓練進行中..."
        sleep 30
    else
        # Giulia process stopped, wait 10 seconds to ensure completion
        echo ""
        echo -e "${YELLOW}Giulia 訓練進程已停止，等待 10 秒確認完成...${NC}"
        sleep 10

        if check_giulia_checkpoints; then
            echo -e "${GREEN}✓ Giulia 訓練已完成，檢測到 checkpoint 檔案${NC}"
            break
        else
            echo -e "${RED}✗ Giulia 訓練進程停止但未找到 checkpoint，可能訓練失敗${NC}"
            exit 1
        fi
    fi
done

echo ""
echo "=========================================="
echo -e "${BLUE}準備啟動 Russell 訓練${NC}"
echo "=========================================="
echo ""

# Wait 5 seconds before starting Russell
echo "等待 5 秒後開始 Russell 訓練..."
sleep 5

# Start Russell training
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
    echo "  結束時間: $(date)"
    echo ""
    echo "輸出檔案:"
    echo "  /mnt/c/ai_models/lora_sdxl/up/russell_identity/russell_lora_sdxl-000001.safetensors (Epoch 1)"
    echo "  /mnt/c/ai_models/lora_sdxl/up/russell_identity/russell_lora_sdxl.safetensors (Epoch 2)"
    echo ""
    echo -e "${GREEN}✓✓✓ 所有角色訓練完成！${NC}"
    echo ""
    echo "完成的角色:"
    echo "  ✓ Barley Lightfoot (Onward)"
    echo "  ✓ Alberto Human (Luca)"
    echo "  ✓ Giulia (Luca)"
    echo "  ✓ Russell (Up)"
else
    echo ""
    echo -e "${RED}✗ Russell 訓練失敗${NC}"
    exit 1
fi
