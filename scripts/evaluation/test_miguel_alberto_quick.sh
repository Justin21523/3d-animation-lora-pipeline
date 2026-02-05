#!/usr/bin/bash
# Quick test for Miguel + Alberto with correct prompts

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/lora_compositions/miguel_alberto_quick_${TIMESTAMP}"

echo "=================================="
echo "Miguel + Alberto Quick Test"
echo "=================================="
echo "Timestamp: $TIMESTAMP"
echo "Output: $OUTPUT_DIR"
echo ""

conda run -n ai_env python /mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/evaluation/sdxl_multi_lora_compositor.py \
  --loras \
    /mnt/c/ai_models/lora_sdxl/coco/miguel_identity/miguel_identity_lora_sdxl-000004.safetensors \
    /mnt/c/ai_models/lora_sdxl/luca/alberto_identity/alberto_identity_lora_sdxl-000002.safetensors \
  --lora-names miguel alberto \
  --weight-combos "1.0,1.0" \
  --prompts-file /mnt/c/ai_projects/3d-animation-lora-pipeline/prompts/composition_tests/miguel_alberto.txt \
  --base-model /mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors \
  --output-dir "$OUTPUT_DIR" \
  --num-samples 2 \
  --steps 20 \
  --guidance-scale 7.5 \
  --seed-start 42 \
  --width 1024 \
  --height 1024

echo ""
echo "=================================="
echo "Test completed!"
echo "Results in: $OUTPUT_DIR"
echo "=================================="
