#!/bin/bash
#
# Retest Luca and Alberto with CORRECT LoRA Paths
# Previous test used wrong paths - LoRAs didn't load!
#

set -e

PROJECT_ROOT="/mnt/c/ai_projects/3d-animation-lora-pipeline"
BASE_MODEL="/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
OUTPUT_BASE="/mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/lora_single_character_tests"

mkdir -p "$OUTPUT_BASE"

timestamp=$(date +%Y%m%d_%H%M%S)

echo "================================================================================"
echo "LUCA & ALBERTO RETEST - CORRECT LORA PATHS"
echo "================================================================================"
echo "Timestamp: $timestamp"
echo ""
echo "Previous issue: Wrong LoRA file paths - LoRAs didn't actually load!"
echo ""
echo "Corrected paths:"
echo "  - Luca:    /mnt/c/ai_models/lora_sdxl/luca/luca_identity/sdxl_trial1/luca_sdxl_RECOMMENDED.safetensors"
echo "  - Alberto: /mnt/c/ai_models/lora_sdxl/luca/alberto_identity/alberto_identity_lora_sdxl-000002.safetensors"
echo ""
echo "================================================================================"
echo ""

# ============================================================================
# Test 1: Alberto Identity LoRA (VERIFIED PATH)
# ============================================================================
echo ""
echo "=== Test 1/2: Alberto (Luca) - WITH CORRECT PATH ==="
echo ""

# Verify file exists
if [ ! -f "/mnt/c/ai_models/lora_sdxl/luca/alberto_identity/alberto_identity_lora_sdxl-000002.safetensors" ]; then
    echo "❌ ERROR: Alberto LoRA file not found!"
    exit 1
fi

echo "✅ Alberto LoRA file verified: 218MB"
echo ""

conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/sdxl_multi_lora_compositor.py" \
    --loras /mnt/c/ai_models/lora_sdxl/luca/alberto_identity/alberto_identity_lora_sdxl-000002.safetensors \
    --lora-names alberto \
    --weight-combos "0.6" "0.8" "1.0" "1.2" \
    --prompts-file "$PROJECT_ROOT/prompts/single_character/alberto_prompts.txt" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_BASE/alberto_RETEST_${timestamp}" \
    --num-samples 3 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 200 \
    --width 1024 \
    --height 1024

echo ""
echo "✅ Alberto retest completed: $OUTPUT_BASE/alberto_RETEST_${timestamp}"
echo ""

# ============================================================================
# Test 2: Luca Identity LoRA (CORRECTED PATH)
# ============================================================================
echo ""
echo "=== Test 2/2: Luca - WITH CORRECTED PATH ==="
echo ""

# Verify file exists
if [ ! -f "/mnt/c/ai_models/lora_sdxl/luca/luca_identity/sdxl_trial1/luca_sdxl_RECOMMENDED.safetensors" ]; then
    echo "❌ ERROR: Luca LoRA file not found!"
    exit 1
fi

echo "✅ Luca LoRA file verified: 871MB (RECOMMENDED version)"
echo ""

conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/sdxl_multi_lora_compositor.py" \
    --loras /mnt/c/ai_models/lora_sdxl/luca/luca_identity/sdxl_trial1/luca_sdxl_RECOMMENDED.safetensors \
    --lora-names luca \
    --weight-combos "0.6" "0.8" "1.0" "1.2" \
    --prompts-file "$PROJECT_ROOT/prompts/single_character/luca_prompts.txt" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_BASE/luca_RETEST_${timestamp}" \
    --num-samples 3 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 400 \
    --width 1024 \
    --height 1024

echo ""
echo "✅ Luca retest completed: $OUTPUT_BASE/luca_RETEST_${timestamp}"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "LUCA & ALBERTO RETEST COMPLETED"
echo "================================================================================"
echo "Results location: $OUTPUT_BASE"
echo ""
echo "Retest summary:"
echo "  - Alberto: 4 weights × 16 prompts × 3 samples = 192 images"
echo "  - Luca:    4 weights × 16 prompts × 3 samples = 192 images"
echo "  - Total: 384 images"
echo ""
echo "Compare with previous (incorrect) results:"
echo "  OLD Alberto: $OUTPUT_BASE/alberto_20251128_114215/ (LoRA didn't load)"
echo "  OLD Luca:    $OUTPUT_BASE/luca_20251128_114215/ (LoRA didn't load)"
echo ""
echo "  NEW Alberto: $OUTPUT_BASE/alberto_RETEST_${timestamp}/ (LoRA LOADED)"
echo "  NEW Luca:    $OUTPUT_BASE/luca_RETEST_${timestamp}/ (LoRA LOADED)"
echo ""
echo "================================================================================"
