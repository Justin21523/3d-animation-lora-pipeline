#!/bin/bash
# Batch test all SDXL LoRA checkpoints
# Automatically tests epochs 6, 8, 10, and final

set -e  # Exit on error

LORA_DIR="/mnt/data/ai_data/models/lora/luca/sdxl_trial1"
BASE_MODEL="/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
OUTPUT_BASE="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/outputs/lora_comprehensive_test"
SCRIPT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/evaluation/comprehensive_lora_test.py"

echo "=========================================="
echo "SDXL LoRA Checkpoint Batch Testing"
echo "=========================================="
echo ""

# Test parameters
WIDTH=1024
HEIGHT=1024
STEPS=50
CFG=7.5
SEEDS=3

# Checkpoint array: filename, epoch_name
declare -a CHECKPOINTS=(
    "luca_sdxl-000006.safetensors:epoch6"
    "luca_sdxl-000008.safetensors:epoch8"
    "luca_sdxl-000010.safetensors:epoch10"
    "luca_sdxl.safetensors:final"
)

# Test each checkpoint
for checkpoint in "${CHECKPOINTS[@]}"; do
    IFS=':' read -r file epoch <<< "$checkpoint"

    LORA_PATH="$LORA_DIR/$file"
    OUTPUT_DIR="$OUTPUT_BASE/luca_sdxl_$epoch"

    # Check if already tested
    if [ -f "$OUTPUT_DIR/TEST_REPORT.md" ]; then
        echo "✅ $epoch already tested - skipping"
        echo ""
        continue
    fi

    echo "🎨 Testing $epoch: $file"
    echo "   Output: $OUTPUT_DIR"
    echo ""

    # Run test
    conda run -n ai_env python "$SCRIPT" \
        "$LORA_PATH" \
        --base-model "$BASE_MODEL" \
        --sdxl \
        --output-dir "$OUTPUT_DIR" \
        --width $WIDTH \
        --height $HEIGHT \
        --steps $STEPS \
        --cfg-scale $CFG \
        --seeds $SEEDS \
        --device cuda

    echo ""
    echo "✅ $epoch complete!"
    echo "=========================================="
    echo ""
done

echo "🎉 All checkpoints tested!"
echo ""
echo "Results:"
for checkpoint in "${CHECKPOINTS[@]}"; do
    IFS=':' read -r file epoch <<< "$checkpoint"
    OUTPUT_DIR="$OUTPUT_BASE/luca_sdxl_$epoch"

    if [ -f "$OUTPUT_DIR/TEST_REPORT.md" ]; then
        echo "  ✅ $epoch: $OUTPUT_DIR"
    else
        echo "  ❌ $epoch: FAILED"
    fi
done
echo ""
