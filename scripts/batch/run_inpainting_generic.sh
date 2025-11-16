#!/bin/bash
# Generic Inpainting Script for Any Pixar/3D Animation Film
# Supports multiple projects with automatic config loading

set -e

echo "🎨 3D Animation Inpainting - Generic Pipeline"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ask for project name
echo -e "${BLUE}Enter project/film name (e.g., 'luca', 'toy_story', 'finding_nemo'):${NC}"
read -p "> " PROJECT_NAME

if [ -z "$PROJECT_NAME" ]; then
    echo -e "${YELLOW}⚠️  No project name provided. Exiting.${NC}"
    exit 1
fi

# Construct paths based on project name
BASE_DATA_DIR="/mnt/data/ai_data/datasets/3d-anime"
INPUT_DIR="${BASE_DATA_DIR}/${PROJECT_NAME}/instances_sampled/instances"
OUTPUT_DIR="${BASE_DATA_DIR}/${PROJECT_NAME}/instances_inpainted"

echo ""
echo "Project: $PROJECT_NAME"
echo "Input:   $INPUT_DIR"
echo "Output:  $OUTPUT_DIR"
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${YELLOW}⚠️  Input directory not found: $INPUT_DIR${NC}"
    echo ""
    echo "Please verify:"
    echo "  1. Project name is correct"
    echo "  2. SAM2 instance segmentation has completed"
    echo "  3. Path structure follows: /mnt/data/ai_data/datasets/3d-anime/{project}/instances_sampled/instances"
    echo ""
    read -p "Continue anyway? (y/n) [default: n]: " FORCE_CONTINUE
    if [ "$FORCE_CONTINUE" != "y" ] && [ "$FORCE_CONTINUE" != "Y" ]; then
        exit 1
    fi
else
    # Count instances
    INSTANCE_COUNT=$(ls "$INPUT_DIR" 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found $INSTANCE_COUNT instances${NC}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Select inpainting method:"
echo "═══════════════════════════════════════════════════════════"
echo "  1) OpenCV (fast, basic quality)"
echo "     - Speed: Very fast (<1s per image)"
echo "     - Quality: Basic"
echo "     - Best for: Small occlusions, quick testing"
echo ""
echo "  2) LaMa (balanced, recommended)"
echo "     - Speed: Fast (1-2s per image)"
echo "     - Quality: Excellent"
echo "     - Best for: General use, production"
echo "     - Note: May need dependency fixes"
echo ""
echo "  3) Stable Diffusion (slow, best quality)"
echo "     - Speed: Slow (5-10s per image)"
echo "     - Quality: Best"
echo "     - Best for: High quality needs, severe occlusions"
echo "     - Note: Requires 8GB+ VRAM, dependency fixes"
echo ""
read -p "Enter choice (1-3) [default: 1]: " METHOD_CHOICE
METHOD_CHOICE=${METHOD_CHOICE:-1}

case $METHOD_CHOICE in
    1)
        METHOD="cv"
        THRESHOLD=0.05
        echo -e "${GREEN}✓ Selected: OpenCV (fast)${NC}"
        ;;
    2)
        METHOD="lama"
        THRESHOLD=0.15
        echo -e "${GREEN}✓ Selected: LaMa (balanced)${NC}"
        echo -e "${YELLOW}⚠️  Note: LaMa may require fixing dependency conflicts${NC}"
        ;;
    3)
        METHOD="sd"
        THRESHOLD=0.20
        USE_AUTO_DETECT="yes"
        echo -e "${GREEN}✓ Selected: Stable Diffusion (high quality)${NC}"
        echo -e "${YELLOW}⚠️  Note: SD requires large VRAM and may take many hours${NC}"
        echo ""
        echo "Enable character auto-detection? (requires character config)"
        read -p "(y/n) [default: y]: " AUTO_DETECT_CONFIRM
        if [ "$AUTO_DETECT_CONFIRM" != "n" ] && [ "$AUTO_DETECT_CONFIRM" != "N" ]; then
            USE_AUTO_DETECT="yes"
        else
            USE_AUTO_DETECT="no"
        fi
        ;;
    *)
        echo "Invalid choice, using OpenCV"
        METHOD="cv"
        THRESHOLD=0.05
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Configuration Summary:"
echo "═══════════════════════════════════════════════════════════"
echo "  Project: $PROJECT_NAME"
echo "  Method: $METHOD"
echo "  Occlusion threshold: $THRESHOLD"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
if [ "$USE_AUTO_DETECT" == "yes" ]; then
    echo "  Auto-detect characters: Yes"
    echo "  Config: configs/inpainting/${PROJECT_NAME}_prompts.json"
fi
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

# Build command
CMD="conda run -n ai_env python scripts/generic/enhancement/inpaint_occlusions.py"
CMD="$CMD --input-dir \"$INPUT_DIR\""
CMD="$CMD --output-dir \"$OUTPUT_DIR\""
CMD="$CMD --method $METHOD"
CMD="$CMD --occlusion-threshold $THRESHOLD"
CMD="$CMD --device cuda"

# Add project-specific config if using SD with auto-detect
if [ "$USE_AUTO_DETECT" == "yes" ]; then
    CMD="$CMD --project $PROJECT_NAME"
    CMD="$CMD --auto-detect-character"
fi

# Execute
echo "Executing: $CMD"
echo ""
eval $CMD

echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ Inpainting complete!${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Results:"
echo "  Output dir: $OUTPUT_DIR/inpainted/"
echo "  Report: $OUTPUT_DIR/inpainting_report.json"
echo ""
echo "Next steps:"
echo "  1. Review report:"
echo "     cat $OUTPUT_DIR/inpainting_report.json"
echo ""
echo "  2. Check sample results:"
echo "     ls $OUTPUT_DIR/inpainted/ | head -20"
echo ""
echo "  3. Run identity clustering:"
echo "     python scripts/generic/clustering/character_clustering.py \\"
echo "       --input-dir $OUTPUT_DIR/inpainted \\"
echo "       --output-dir ${BASE_DATA_DIR}/${PROJECT_NAME}/clustered"
echo ""
echo "═══════════════════════════════════════════════════════════"
