#!/bin/bash
#
# Complete Luca Background Reprocessing Pipeline
# ================================================
#
# This script performs a complete re-segmentation and inpainting workflow:
# 1. SAM2 instance segmentation with optimized parameters
# 2. LaMa inpainting with 20px mask dilation
# 3. Quality validation and organization
#
# Author: Claude Code
# Date: 2025-11-16

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "🎬 Luca Background Reprocessing Pipeline"
echo "======================================================================"
echo

# Configuration
FRAMES_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/frames"
SAM2_OUTPUT="/mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2"
FINAL_BACKGROUNDS="/mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_lama_v2"
LOG_DIR="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/logs"

# Create directories
mkdir -p "$SAM2_OUTPUT"
mkdir -p "$FINAL_BACKGROUNDS"
mkdir -p "$LOG_DIR"

# Timestamped log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/luca_reprocess_$TIMESTAMP.log"

echo "📁 Directories:"
echo "   Frames: $FRAMES_DIR"
echo "   SAM2 Output: $SAM2_OUTPUT"
echo "   Final Backgrounds: $FINAL_BACKGROUNDS"
echo "   Log: $LOG_FILE"
echo

# Check if frames exist
FRAME_COUNT=$(ls -1 "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l)
if [ $FRAME_COUNT -eq 0 ]; then
    echo -e "${RED}❌ No frames found in $FRAMES_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found $FRAME_COUNT frames${NC}"
echo

#################################################################
# STEP 1: SAM2 Instance Segmentation with Optimized Parameters
#################################################################

echo "======================================================================"
echo "STEP 1: SAM2 Instance Segmentation"
echo "======================================================================"
echo

echo "🔧 Configuration Changes:"
echo "   • points_per_side: 20 → 32 (better character detection)"
echo "   • pred_iou_thresh: 0.76 → 0.70 (capture more instances)"
echo "   • stability_score_thresh: 0.86 → 0.80 (include partial occlusions)"
echo "   • mask_dilation: 10px → 20px (complete character coverage)"
echo

echo "⏱️  Estimated time: 2-4 hours for $FRAME_COUNT frames"
echo "   (Average: 3-5 seconds per frame)"
echo

read -p "Press Enter to start SAM2 segmentation (or Ctrl+C to cancel)..."

# Run SAM2 segmentation
conda run -n ai_env python /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/generic/segmentation/instance_segmentation.py \
    "$FRAMES_DIR" \
    --output-dir "$SAM2_OUTPUT" \
    --model sam2_hiera_large \
    --device cuda \
    --min-size 4096 \
    --save-masks \
    --save-backgrounds \
    --context-mode transparent \
    2>&1 | tee -a "$LOG_FILE"

SAM2_EXIT_CODE=${PIPESTATUS[0]}

if [ $SAM2_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}❌ SAM2 segmentation failed with exit code $SAM2_EXIT_CODE${NC}"
    echo "   Check log: $LOG_FILE"
    exit 1
fi

echo
echo -e "${GREEN}✅ SAM2 segmentation complete${NC}"
echo

# Check output
BG_COUNT=$(ls -1 "$SAM2_OUTPUT/backgrounds"/*.jpg 2>/dev/null | wc -l || echo 0)
MASK_COUNT=$(ls -1 "$SAM2_OUTPUT/masks"/*.png 2>/dev/null | wc -l || echo 0)

echo "📊 SAM2 Output:"
echo "   Backgrounds: $BG_COUNT"
echo "   Masks: $MASK_COUNT"
echo

if [ $BG_COUNT -eq 0 ] || [ $MASK_COUNT -eq 0 ]; then
    echo -e "${RED}❌ SAM2 output validation failed${NC}"
    echo "   Expected: backgrounds and masks"
    echo "   Got: $BG_COUNT backgrounds, $MASK_COUNT masks"
    exit 1
fi

#################################################################
# STEP 2: LaMa Inpainting with 20px Mask Dilation
#################################################################

echo "======================================================================"
echo "STEP 2: LaMa Inpainting (20px mask dilation)"
echo "======================================================================"
echo

echo "🎨 Using full LaMa model (not OpenCV fallback)"
echo "   • Method: LaMa (big-lama)"
echo "   • Mask dilation: 20px"
echo "   • Batch size: 8"
echo

echo "⏱️  Estimated time: 3-4 hours for $BG_COUNT backgrounds"
echo "   (Average: 2-3 seconds per image)"
echo

read -p "Press Enter to start LaMa inpainting (or Ctrl+C to cancel)..."

# Run LaMa inpainting
conda run -n ai_env python /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/generic/inpainting/sam2_background_inpainting.py \
    --sam2-dir "$SAM2_OUTPUT" \
    --output-dir "$FINAL_BACKGROUNDS" \
    --method lama \
    --batch-size 8 \
    --device cuda \
    --mask-dilate 20 \
    2>&1 | tee -a "$LOG_FILE"

LAMA_EXIT_CODE=${PIPESTATUS[0]}

if [ $LAMA_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}❌ LaMa inpainting failed with exit code $LAMA_EXIT_CODE${NC}"
    echo "   Check log: $LOG_FILE"
    exit 1
fi

echo
echo -e "${GREEN}✅ LaMa inpainting complete${NC}"
echo

# Check output
FINAL_COUNT=$(ls -1 "$FINAL_BACKGROUNDS"/*.jpg 2>/dev/null | wc -l || echo 0)
TOTAL_SIZE=$(du -sh "$FINAL_BACKGROUNDS" 2>/dev/null | cut -f1 || echo "0")

echo "📊 Final Output:"
echo "   Backgrounds: $FINAL_COUNT"
echo "   Total size: $TOTAL_SIZE"
echo

#################################################################
# STEP 3: Validation and Summary
#################################################################

echo "======================================================================"
echo "STEP 3: Validation & Summary"
echo "======================================================================"
echo

# Check metadata
if [ -f "$FINAL_BACKGROUNDS/inpainting_metadata.json" ]; then
    echo "📄 Metadata:"
    cat "$FINAL_BACKGROUNDS/inpainting_metadata.json" | jq '.' 2>/dev/null || cat "$FINAL_BACKGROUNDS/inpainting_metadata.json"
    echo
fi

# Success rate
if [ $FINAL_COUNT -eq $BG_COUNT ]; then
    echo -e "${GREEN}✅ All backgrounds processed successfully (100%)${NC}"
elif [ $FINAL_COUNT -gt 0 ]; then
    SUCCESS_RATE=$(echo "scale=2; $FINAL_COUNT * 100 / $BG_COUNT" | bc)
    echo -e "${YELLOW}⚠️  Processed $FINAL_COUNT / $BG_COUNT backgrounds ($SUCCESS_RATE%)${NC}"
else
    echo -e "${RED}❌ No backgrounds were successfully processed${NC}"
    exit 1
fi

echo
echo "======================================================================"
echo "✅ PIPELINE COMPLETE"
echo "======================================================================"
echo
echo "📁 Output Location: $FINAL_BACKGROUNDS"
echo "📄 Log File: $LOG_FILE"
echo
echo "🎯 Next Steps:"
echo "   1. Review sample backgrounds for quality"
echo "   2. Organize by scene type (indoor/outdoor/underwater)"
echo "   3. Create background LoRA training configuration"
echo
echo "======================================================================"
