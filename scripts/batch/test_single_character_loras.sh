#!/bin/bash
#
# Test Individual SDXL Character LoRAs
# Tests each character separately with different weight configurations
#

set -e

PROJECT_ROOT="/mnt/c/ai_projects/3d-animation-lora-pipeline"
BASE_MODEL="/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
OUTPUT_BASE="/mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/lora_single_character_tests"

# Create output directory
mkdir -p "$OUTPUT_BASE"

timestamp=$(date +%Y%m%d_%H%M%S)

echo "================================================================================"
echo "SDXL SINGLE CHARACTER LORA TESTING"
echo "================================================================================"
echo "Timestamp: $timestamp"
echo ""
echo "Testing strategy: Each character tested individually with multiple weights"
echo "Weight variations: 0.6, 0.8, 1.0, 1.2"
echo "Prompts per character: 16 (diverse poses, expressions, settings)"
echo "Samples per prompt: 3"
echo "================================================================================"
echo ""

# ============================================================================
# Test 1: Miguel Identity LoRA
# ============================================================================
echo ""
echo "=== Test 1/4: Miguel (Coco) ==="
echo ""

conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/sdxl_multi_lora_compositor.py" \
    --loras /mnt/c/ai_models/lora_sdxl/coco/miguel_identity/miguel_identity_lora_sdxl-000004.safetensors \
    --lora-names miguel \
    --weight-combos "0.6" "0.8" "1.0" "1.2" \
    --prompts-file "$PROJECT_ROOT/prompts/single_character/miguel_prompts.txt" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_BASE/miguel_${timestamp}" \
    --num-samples 3 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 100 \
    --width 1024 \
    --height 1024

echo ""
echo "✅ Miguel test completed: $OUTPUT_BASE/miguel_${timestamp}"
echo ""

# ============================================================================
# Test 2: Alberto Identity LoRA
# ============================================================================
echo ""
echo "=== Test 2/4: Alberto (Luca) ==="
echo ""

conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/sdxl_multi_lora_compositor.py" \
    --loras /mnt/c/ai_models/lora_sdxl/luca/alberto_identity/alberto_identity_lora_sdxl-000002.safetensors \
    --lora-names alberto \
    --weight-combos "0.6" "0.8" "1.0" "1.2" \
    --prompts-file "$PROJECT_ROOT/prompts/single_character/alberto_prompts.txt" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_BASE/alberto_${timestamp}" \
    --num-samples 3 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 200 \
    --width 1024 \
    --height 1024

echo ""
echo "✅ Alberto test completed: $OUTPUT_BASE/alberto_${timestamp}"
echo ""

# ============================================================================
# Test 3: Elio Identity LoRA
# ============================================================================
echo ""
echo "=== Test 3/4: Elio ==="
echo ""

conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/sdxl_multi_lora_compositor.py" \
    --loras /mnt/data/training/lora/elio/elio_identity/elio_identity_lora_sdxl-000004.safetensors \
    --lora-names elio \
    --weight-combos "0.6" "0.8" "1.0" "1.2" \
    --prompts-file "$PROJECT_ROOT/prompts/single_character/elio_prompts.txt" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_BASE/elio_${timestamp}" \
    --num-samples 3 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 300 \
    --width 1024 \
    --height 1024

echo ""
echo "✅ Elio test completed: $OUTPUT_BASE/elio_${timestamp}"
echo ""

# ============================================================================
# Test 4: Luca Identity LoRA
# ============================================================================
echo ""
echo "=== Test 4/4: Luca ==="
echo ""

conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/sdxl_multi_lora_compositor.py" \
    --loras /mnt/c/ai_models/lora_sdxl/luca/luca_identity/sdxl_trial1/luca_sdxl_RECOMMENDED.safetensors \
    --lora-names luca \
    --weight-combos "0.6" "0.8" "1.0" "1.2" \
    --prompts-file "$PROJECT_ROOT/prompts/single_character/luca_prompts.txt" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_BASE/luca_${timestamp}" \
    --num-samples 3 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 400 \
    --width 1024 \
    --height 1024

echo ""
echo "✅ Luca test completed: $OUTPUT_BASE/luca_${timestamp}"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "ALL SINGLE CHARACTER TESTS COMPLETED"
echo "================================================================================"
echo "Results location: $OUTPUT_BASE"
echo ""
echo "Test summary:"
echo "  - 4 characters tested: Miguel, Alberto, Elio, Luca"
echo "  - 4 weight variations per character: 0.6, 0.8, 1.0, 1.2"
echo "  - 16 prompts per character (diverse scenarios)"
echo "  - 3 samples per prompt"
echo "  - Total images per character: 192 (4 weights × 16 prompts × 3 samples)"
echo "  - Grand total: 768 images"
echo ""
echo "View results:"
echo "  ls -lh $OUTPUT_BASE/"
echo ""
echo "Generated directories:"
find "$OUTPUT_BASE" -maxdepth 1 -type d -name "*${timestamp}" | sort
echo ""
echo "================================================================================"
