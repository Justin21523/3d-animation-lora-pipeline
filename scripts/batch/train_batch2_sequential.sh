#!/bin/bash
# Sequential SDXL LoRA Training - Batch 2 (Optimized)
# Trains 4 remaining characters with optimized configuration
# Based on Batch 1 lessons: lower LR, 2 epochs only, light dropout

set -e

echo "=========================================="
echo "SDXL LoRA Training - Batch 2 (Optimized)"
echo "=========================================="
echo "Training 4 characters sequentially with 2 epochs each"
echo "Optimized configuration to prevent overfitting"
echo ""

# Training configurations
declare -a CONFIGS=(
    "configs/training/character_loras_sdxl/onward_barley_lightfoot_sdxl.toml:Barley Lightfoot (Onward)"
    "configs/training/character_loras_sdxl/luca_alberto_human_sdxl.toml:Alberto Human (Luca)"
    "configs/training/character_loras_sdxl/luca_giulia_sdxl.toml:Giulia (Luca)"
    "configs/training/character_loras_sdxl/up_russell_sdxl.toml:Russell (Up)"
)

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Start time
BATCH_START=$(date +%s)
echo "Batch start time: $(date)"
echo ""

# Training counter
completed=0
failed=0

# Train each character
for config_info in "${CONFIGS[@]}"; do
    IFS=':' read -r config_path char_name <<< "$config_info"

    echo ""
    echo "=========================================="
    echo -e "${BLUE}Training: $char_name${NC}"
    echo "Config: $config_path"
    echo "=========================================="

    # Check if config exists
    if [ ! -f "$config_path" ]; then
        echo -e "${RED}ERROR: Config not found: $config_path${NC}"
        ((failed++))
        continue
    fi

    # Start training
    char_start=$(date +%s)
    echo "Character start time: $(date)"
    echo ""

    # Run training using kohya_ss
    if conda run -n kohya_ss accelerate launch \
        --num_cpu_threads_per_process=2 \
        /mnt/c/ai_projects/kohya_ss/sd-scripts/sdxl_train_network.py \
        --config_file "$config_path"; then

        char_end=$(date +%s)
        char_duration=$((char_end - char_start))
        char_minutes=$((char_duration / 60))
        char_seconds=$((char_duration % 60))

        echo ""
        echo -e "${GREEN}✓ Completed: $char_name${NC}"
        echo "  Duration: ${char_minutes}m ${char_seconds}s"
        ((completed++))
    else
        echo ""
        echo -e "${RED}✗ Failed: $char_name${NC}"
        ((failed++))
    fi

    echo ""
    echo "Progress: $completed completed, $failed failed"
    echo ""
done

# End time and summary
BATCH_END=$(date +%s)
BATCH_DURATION=$((BATCH_END - BATCH_START))
BATCH_HOURS=$((BATCH_DURATION / 3600))
BATCH_MINUTES=$(((BATCH_DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo -e "${GREEN}Batch 2 Training Complete${NC}"
echo "=========================================="
echo "Completed: $completed / 4 characters"
echo "Failed: $failed / 4 characters"
echo "Total duration: ${BATCH_HOURS}h ${BATCH_MINUTES}m"
echo "Batch end time: $(date)"
echo ""

if [ $completed -eq 4 ]; then
    echo -e "${GREEN}✓ All characters trained successfully!${NC}"
    echo ""
    echo "Expected outputs (2 checkpoints per character):"
    echo "  /mnt/c/ai_models/lora_sdxl/onward/barley_lightfoot_identity/"
    echo "    ├── barley_lightfoot_lora_sdxl-000001.safetensors (Epoch 1)"
    echo "    └── barley_lightfoot_lora_sdxl-000002.safetensors (Epoch 2)"
    echo "  /mnt/c/ai_models/lora_sdxl/luca/alberto_identity/"
    echo "    ├── alberto_lora_sdxl-000001.safetensors (Epoch 1)"
    echo "    └── alberto_lora_sdxl-000002.safetensors (Epoch 2)"
    echo "  /mnt/c/ai_models/lora_sdxl/luca/giulia_identity/"
    echo "    ├── giulia_lora_sdxl-000001.safetensors (Epoch 1)"
    echo "    └── giulia_lora_sdxl-000002.safetensors (Epoch 2)"
    echo "  /mnt/c/ai_models/lora_sdxl/up/russell_identity/"
    echo "    ├── russell_lora_sdxl-000001.safetensors (Epoch 1)"
    echo "    └── russell_lora_sdxl-000002.safetensors (Epoch 2)"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate all checkpoints to compare Epoch 1 vs Epoch 2"
    echo "  2. Verify optimization prevented overfitting"
    echo "  3. Document results in BATCH2_TRAINING_RESULTS.md"
else
    echo -e "${RED}✗ Some characters failed to train${NC}"
    echo "Please check logs for errors"
fi
