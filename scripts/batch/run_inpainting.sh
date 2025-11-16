#!/bin/bash
# Quick Start Script for Inpainting Luca Instances
# Run this after SAM2 completes

set -e

echo "🎨 Luca Inpainting Quick Start"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
INPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances"
OUTPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/instances_inpainted"
CONFIG="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/inpainting/luca_prompts.json"

# Check if SAM2 instances exist
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${YELLOW}⚠️  Input directory not found: $INPUT_DIR${NC}"
    echo "   Please wait for SAM2 to complete first."
    exit 1
fi

# Count instances
INSTANCE_COUNT=$(ls "$INPUT_DIR" 2>/dev/null | wc -l)
echo -e "${GREEN}✓ Found $INSTANCE_COUNT instances${NC}"
echo ""

# Ask user for method
echo "Select inpainting method:"
echo "  1) OpenCV (fast, basic quality) - RECOMMENDED for testing"
echo "  2) LaMa (balanced speed/quality) - requires additional setup"
echo "  3) Stable Diffusion (slow, best quality) - requires additional setup"
echo ""
read -p "Enter choice (1-3) [default: 1]: " METHOD_CHOICE
METHOD_CHOICE=${METHOD_CHOICE:-1}

case $METHOD_CHOICE in
    1)
        METHOD="cv"
        THRESHOLD=0.05
        echo -e "${GREEN}Selected: OpenCV (fast)${NC}"
        ;;
    2)
        METHOD="lama"
        THRESHOLD=0.15
        echo -e "${GREEN}Selected: LaMa (balanced)${NC}"
        echo -e "${YELLOW}Note: LaMa may require fixing dependency conflicts${NC}"
        ;;
    3)
        METHOD="sd"
        THRESHOLD=0.20
        echo -e "${GREEN}Selected: Stable Diffusion (high quality)${NC}"
        echo -e "${YELLOW}Note: SD requires large VRAM and may take many hours${NC}"
        ;;
    *)
        echo "Invalid choice, using OpenCV"
        METHOD="cv"
        THRESHOLD=0.05
        ;;
esac

echo ""
echo "Parameters:"
echo "  Method: $METHOD"
echo "  Occlusion threshold: $THRESHOLD"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

read -p "Proceed? (y/n) [default: y]: " CONFIRM
CONFIRM=${CONFIRM:-y}

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting inpainting..."
echo ""

# Run inpainting
conda run -n ai_env python scripts/generic/enhancement/inpaint_occlusions.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --method "$METHOD" \
    --occlusion-threshold "$THRESHOLD" \
    --device cuda

echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ Inpainting complete!${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR/inpainted/"
echo "Report: $OUTPUT_DIR/inpainting_report.json"
echo ""
echo "Next steps:"
echo "  1. Review report: cat $OUTPUT_DIR/inpainting_report.json"
echo "  2. Check sample results: ls $OUTPUT_DIR/inpainted/ | head -20"
echo "  3. Run identity clustering next"
echo "═══════════════════════════════════════════════════════════"
