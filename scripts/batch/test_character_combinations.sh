#!/bin/bash
#
# Test SDXL Character LoRA Combinations
# Tests different character pairs with various weight combinations
#

set -e

PROJECT_ROOT="/mnt/c/ai_projects/3d-animation-lora-pipeline"
BASE_MODEL="/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
OUTPUT_BASE="/mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/lora_compositions"

# Create output directory
mkdir -p "$OUTPUT_BASE"

timestamp=$(date +%Y%m%d_%H%M%S)

echo "================================================================================"
echo "SDXL CHARACTER LORA COMPOSITION TESTING"
echo "================================================================================"
echo "Timestamp: $timestamp"
echo ""

# Available LoRAs
MIGUEL_LORA="/mnt/c/ai_models/lora_sdxl/coco/miguel_identity/miguel_identity_lora_sdxl-000004.safetensors"
ALBERTO_LORA="/mnt/c/ai_models/lora_sdxl/luca/alberto_identity/alberto_identity_lora_sdxl-000002.safetensors"
ELIO_LORA="/mnt/data/training/lora/elio/elio_identity/elio_identity_lora_sdxl-000004.safetensors"
LUCA_LORA="/mnt/c/ai_models/lora_sdxl/luca/luca_identity/luca_identity_lora_sdxl-000006.safetensors"

# Prompt files (specific for each combination)
MIGUEL_ALBERTO_PROMPTS="$PROJECT_ROOT/prompts/composition_tests/miguel_alberto.txt"
MIGUEL_ELIO_PROMPTS="$PROJECT_ROOT/prompts/composition_tests/miguel_elio.txt"
MIGUEL_LUCA_PROMPTS="$PROJECT_ROOT/prompts/composition_tests/miguel_luca.txt"
ALBERTO_ELIO_PROMPTS="$PROJECT_ROOT/prompts/composition_tests/alberto_elio.txt"
ALBERTO_LUCA_PROMPTS="$PROJECT_ROOT/prompts/composition_tests/alberto_luca.txt"
ELIO_LUCA_PROMPTS="$PROJECT_ROOT/prompts/composition_tests/elio_luca.txt"

# ============================================================================
# Test 1: Miguel + Alberto (Balanced weights)
# ============================================================================
echo ""
echo "=== Test 1: Miguel + Alberto (Various Weight Combinations) ==="
echo ""

conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/sdxl_multi_lora_compositor.py" \
    --loras "$MIGUEL_LORA" "$ALBERTO_LORA" \
    --lora-names "miguel" "alberto" \
    --weight-combos "1.0,1.0" "1.0,0.8" "0.8,1.0" "0.7,0.7" \
    --prompts-file "$MIGUEL_ALBERTO_PROMPTS" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_BASE/miguel_alberto_${timestamp}" \
    --num-samples 3 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 42 \
    --width 1024 \
    --height 1024

echo ""
echo "✅ Test 1 completed: $OUTPUT_BASE/miguel_alberto_${timestamp}"
echo ""

# ============================================================================
# Test 2: Miguel + Elio
# ============================================================================
echo ""
echo "=== Test 2: Miguel + Elio (Various Weight Combinations) ==="
echo ""

conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/sdxl_multi_lora_compositor.py" \
    --loras "$MIGUEL_LORA" "$ELIO_LORA" \
    --lora-names "miguel" "elio" \
    --weight-combos "1.0,1.0" "1.0,0.8" "0.8,1.0" "0.6,0.6" \
    --prompts-file "$MIGUEL_ELIO_PROMPTS" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_BASE/miguel_elio_${timestamp}" \
    --num-samples 3 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 100 \
    --width 1024 \
    --height 1024

echo ""
echo "✅ Test 2 completed: $OUTPUT_BASE/miguel_elio_${timestamp}"
echo ""

# ============================================================================
# Test 3: Miguel + Luca
# ============================================================================
echo ""
echo "=== Test 3: Miguel + Luca (Various Weight Combinations) ==="
echo ""

conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/sdxl_multi_lora_compositor.py" \
    --loras "$MIGUEL_LORA" "$LUCA_LORA" \
    --lora-names "miguel" "luca" \
    --weight-combos "1.0,1.0" "1.0,0.8" "0.8,1.0" "0.7,0.7" \
    --prompts-file "$MIGUEL_LUCA_PROMPTS" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_BASE/miguel_luca_${timestamp}" \
    --num-samples 3 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 200 \
    --width 1024 \
    --height 1024

echo ""
echo "✅ Test 3 completed: $OUTPUT_BASE/miguel_luca_${timestamp}"
echo ""

# ============================================================================
# Test 4: Alberto + Elio
# ============================================================================
echo ""
echo "=== Test 4: Alberto + Elio (Various Weight Combinations) ==="
echo ""

conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/sdxl_multi_lora_compositor.py" \
    --loras "$ALBERTO_LORA" "$ELIO_LORA" \
    --lora-names "alberto" "elio" \
    --weight-combos "1.0,1.0" "1.0,0.8" "0.8,1.0" "0.75,0.75" \
    --prompts-file "$ALBERTO_ELIO_PROMPTS" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_BASE/alberto_elio_${timestamp}" \
    --num-samples 3 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 300 \
    --width 1024 \
    --height 1024

echo ""
echo "✅ Test 4 completed: $OUTPUT_BASE/alberto_elio_${timestamp}"
echo ""

# ============================================================================
# Test 5: Alberto + Luca
# ============================================================================
echo ""
echo "=== Test 5: Alberto + Luca (Various Weight Combinations) ==="
echo ""

conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/sdxl_multi_lora_compositor.py" \
    --loras "$ALBERTO_LORA" "$LUCA_LORA" \
    --lora-names "alberto" "luca" \
    --weight-combos "1.0,1.0" "1.0,0.8" "0.8,1.0" "0.75,0.75" \
    --prompts-file "$ALBERTO_LUCA_PROMPTS" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_BASE/alberto_luca_${timestamp}" \
    --num-samples 3 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 400 \
    --width 1024 \
    --height 1024

echo ""
echo "✅ Test 5 completed: $OUTPUT_BASE/alberto_luca_${timestamp}"
echo ""

# ============================================================================
# Test 6: Elio + Luca
# ============================================================================
echo ""
echo "=== Test 6: Elio + Luca (Various Weight Combinations) ==="
echo ""

conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/sdxl_multi_lora_compositor.py" \
    --loras "$ELIO_LORA" "$LUCA_LORA" \
    --lora-names "elio" "luca" \
    --weight-combos "1.0,1.0" "1.0,0.8" "0.8,1.0" "0.6,0.6" \
    --prompts-file "$ELIO_LUCA_PROMPTS" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_BASE/elio_luca_${timestamp}" \
    --num-samples 3 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 500 \
    --width 1024 \
    --height 1024

echo ""
echo "✅ Test 6 completed: $OUTPUT_BASE/elio_luca_${timestamp}"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "ALL COMPOSITION TESTS COMPLETED"
echo "================================================================================"
echo "Results location: $OUTPUT_BASE"
echo ""
echo "View results:"
echo "  ls -lh $OUTPUT_BASE/"
echo ""
echo "Generated directories:"
find "$OUTPUT_BASE" -maxdepth 1 -type d -name "*${timestamp}" | sort
echo ""
echo "================================================================================"
