#!/bin/bash
# Quick launcher for Trial 3.5 LoRA comprehensive testing
# Tests the best checkpoint (epoch 18) with extensive prompts

set -e

echo "======================================================================="
echo "Trial 3.5 LoRA Comprehensive Testing"
echo "======================================================================="
echo ""

# Configuration
LORA_PATH="/mnt/data/ai_data/models/lora/luca/trial35/luca_trial35.safetensors"
BASE_MODEL="runwayml/stable-diffusion-v1-5"
OUTPUT_DIR="outputs/lora_testing/trial35_comprehensive_$(date +%Y%m%d_%H%M%S)"

# Verify LoRA exists
if [ ! -f "$LORA_PATH" ]; then
    echo "❌ ERROR: LoRA not found at: $LORA_PATH"
    exit 1
fi

echo "✅ LoRA found: $(basename $LORA_PATH)"
echo "📁 Output: $OUTPUT_DIR"
echo ""

# Check GPU
if ! nvidia-smi &>/dev/null; then
    echo "⚠️  WARNING: nvidia-smi not available, falling back to CPU"
    DEVICE="cpu"
else
    echo "🎮 GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
    DEVICE="cuda"
fi

# Activate environment
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "ai_env" ]; then
    echo "🔄 Switching to ai_env conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate ai_env
fi

# Run comprehensive test
echo "======================================================================="
echo "🚀 Starting Comprehensive LoRA Testing"
echo "======================================================================="
echo ""
echo "Parameters:"
echo "  - Seeds per prompt: 3"
echo "  - Inference steps: 30"
echo "  - CFG scale: 7.5"
echo "  - Resolution: 512x512"
echo "  - Categories: 9 (portraits, full_body, angles, etc.)"
echo ""
echo "This will generate ~135 test images (45 prompts × 3 seeds)"
echo "Estimated time: 15-20 minutes on RTX 3090"
echo ""

read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Run testing
python scripts/evaluation/comprehensive_lora_test.py \
    "$LORA_PATH" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --lora-scale 1.0 \
    --seeds 3 \
    --steps 30 \
    --cfg-scale 7.5 \
    --width 512 \
    --height 512

echo ""
echo "======================================================================="
echo "✅ Testing Complete!"
echo "======================================================================="
echo ""
echo "📁 Results Location:"
echo "   $OUTPUT_DIR"
echo ""
echo "📄 Files Generated:"
echo "   - TEST_REPORT.md         : Comprehensive quality report"
echo "   - test_results.json      : Detailed results data"
echo "   - grids/                 : Comparison grids by category"
echo "   - grids/master_comparison.png : Overview of all categories"
echo "   - images/                : Individual test images by category"
echo ""
echo "📝 Next Steps:"
echo "   1. Review the TEST_REPORT.md file"
echo "   2. Check the comparison grids in grids/ folder"
echo "   3. Inspect individual images if needed"
echo "   4. If quality is good → Proceed to SDXL training"
echo "   5. If issues found → Document and adjust training"
echo ""
echo "🚀 SDXL Training:"
echo "   bash scripts/training/start_sdxl_16gb_training.sh"
echo ""
echo "======================================================================="
