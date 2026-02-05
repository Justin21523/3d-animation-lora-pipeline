#!/usr/bin/bash
# Quick debug test for multi-LoRA composition

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/lora_compositions/debug_test_${TIMESTAMP}"

echo "=================================="
echo "SDXL Multi-LoRA Debug Test"
echo "=================================="
echo "Timestamp: $TIMESTAMP"
echo "Output: $OUTPUT_DIR"
echo ""

# Test Miguel + Alberto with just ONE weight combination and TWO prompts
conda run -n ai_env python /mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/evaluation/sdxl_multi_lora_compositor.py \
  --loras \
    /mnt/c/ai_models/lora_sdxl/coco/miguel_identity/miguel_identity_lora_sdxl-000004.safetensors \
    /mnt/c/ai_models/lora_sdxl/luca/alberto_identity/alberto_identity_lora_sdxl-000002.safetensors \
  --lora-names miguel alberto \
  --weight-combos "1.0,1.0" \
  --prompts \
    "miguel, a 3d animated boy, pixar style, brown skin, black curly hair, standing, warm lighting" \
    "alberto, a 3d animated boy, pixar style, green scales, sea monster features, smiling, sunny day" \
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
