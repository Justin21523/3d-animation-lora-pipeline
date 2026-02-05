#!/bin/bash
# Quick Test SDXL Caption Expansion
#
# Tests caption expansion on a small sample to verify setup before full batch
#
# Usage:
#   bash scripts/batch/quick_test_sdxl_expansion.sh alberto
#   bash scripts/batch/quick_test_sdxl_expansion.sh elio
#
# Author: LLMProvider Tooling
# Date: 2025-11-22

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BASE_DIR="/mnt/data/ai_data/datasets/3d-anime"
TEST_FILES=5  # Number of files to test

# Character mappings (character_id -> film/training_path/style)
declare -A CHARACTER_FILMS=(
    ["alberto"]="luca"
    ["giulia"]="luca"
    ["ian"]="onward"
    ["barley"]="onward"
    ["tyler"]="turning_red"
    ["russell"]="up"
    ["orion"]="orion"
    ["elio"]="elio"
    ["bryce"]="elio"
    ["caleb"]="elio"
    ["glordon"]="elio"
    ["miguel"]="coco"
)

declare -A CHARACTER_STYLES=(
    ["alberto"]="pixar"
    ["giulia"]="pixar"
    ["ian"]="pixar"
    ["barley"]="pixar"
    ["tyler"]="pixar"
    ["russell"]="pixar"
    ["orion"]="dreamworks"
    ["elio"]="pixar"
    ["bryce"]="pixar"
    ["caleb"]="pixar"
    ["glordon"]="pixar"
    ["miguel"]="pixar"
)

declare -A CHARACTER_NAMES=(
    ["alberto"]="alberto scorfano"
    ["giulia"]="giulia marcovaldo"
    ["ian"]="ian lightfoot"
    ["barley"]="barley lightfoot"
    ["tyler"]="tyler"
    ["russell"]="russell"
    ["orion"]="orion"
    ["elio"]="elio solis"
    ["bryce"]="bryce markwell"
    ["caleb"]="caleb"
    ["glordon"]="glordon"
    ["miguel"]="miguel rivera"
)

# Function to display usage
usage() {
    echo "Usage: $0 <character_id>"
    echo ""
    echo "Supported characters:"
    echo "  Luca:         alberto, giulia"
    echo "  Onward:       ian, barley"
    echo "  Turning Red:  tyler"
    echo "  Up:           russell"
    echo "  Orion:        orion"
    echo "  Elio:         elio, bryce, caleb, glordon"
    echo "  Coco:         miguel"
    echo ""
    echo "Example:"
    echo "  $0 alberto"
    echo "  $0 elio"
    exit 1
}

# Check arguments
if [ $# -eq 0 ]; then
    usage
fi

CHARACTER_ID=$1

# Validate character
if [ -z "${CHARACTER_FILMS[$CHARACTER_ID]}" ]; then
    echo -e "${RED}Error: Unknown character '$CHARACTER_ID'${NC}"
    usage
fi

# Get character info
FILM="${CHARACTER_FILMS[$CHARACTER_ID]}"
STYLE="${CHARACTER_STYLES[$CHARACTER_ID]}"
CHARACTER_NAME="${CHARACTER_NAMES[$CHARACTER_ID]}"

# Determine training data path
case "$FILM" in
    "luca")
        TRAINING_PATH="${CHARACTER_ID}_identity"
        ;;
    "onward")
        TRAINING_PATH="${CHARACTER_ID}_lightfoot_identity"
        ;;
    "elio"|"coco"|"orion"|"turning_red"|"up")
        TRAINING_PATH="${CHARACTER_ID}_identity"
        ;;
    *)
        echo -e "${RED}Error: Unknown film '$FILM'${NC}"
        exit 1
        ;;
esac

# Paths
SD15_DIR="$BASE_DIR/$FILM/lora_data/training_data/$TRAINING_PATH"
SDXL_DIR="$BASE_DIR/$FILM/lora_data/training_data_sdxl/${TRAINING_PATH}_test"

# Display test info
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}SDXL Caption Expansion - Quick Test${NC}"
echo -e "${GREEN}=========================================${NC}"
echo "Character:    $CHARACTER_NAME ($CHARACTER_ID)"
echo "Film:         $FILM"
echo "Style:        $STYLE"
echo "Test files:   $TEST_FILES"
echo ""
echo "SD1.5 dir:    $SD15_DIR"
echo "SDXL test:    $SDXL_DIR"
echo -e "${GREEN}=========================================${NC}"
echo ""

# Check if SD1.5 data exists
if [ ! -d "$SD15_DIR" ]; then
    echo -e "${RED}Error: SD1.5 training data not found at $SD15_DIR${NC}"
    exit 1
fi

# Count caption files
CAPTION_COUNT=$(find "$SD15_DIR" -name "*.txt" | wc -l)
if [ $CAPTION_COUNT -eq 0 ]; then
    echo -e "${RED}Error: No caption files found in $SD15_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found $CAPTION_COUNT caption files in SD1.5 directory"
echo ""

# Check API key
if [ -z "$LLM_VENDOR_API_KEY" ]; then
    echo -e "${RED}Error: LLM_VENDOR_API_KEY not set${NC}"
    echo "Set it with: export LLM_VENDOR_API_KEY='your-api-key-here'"
    exit 1
fi

echo -e "${GREEN}✓${NC} LLM_VENDOR_API_KEY is set"
echo ""

# Run expansion test
echo -e "${YELLOW}Running caption expansion test (${TEST_FILES} files)...${NC}"
echo ""

conda run -n ai_env python scripts/generic/training/sdxl_caption_expander.py \
    --input-dir "$SD15_DIR" \
    --output-dir "$SDXL_DIR" \
    --character-name "$CHARACTER_NAME" \
    --style "$STYLE" \
    --max-files $TEST_FILES

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}✓ Test completed successfully!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""

    # Display sample results
    echo -e "${YELLOW}Sample expanded captions:${NC}"
    echo ""

    cd "$SDXL_DIR"
    for f in $(ls *.txt | head -3); do
        echo -e "${YELLOW}=== $f ===${NC}"
        cat "$f"
        echo ""
    done

    # Display metadata
    if [ -f "sdxl_expansion_metadata.json" ]; then
        echo -e "${YELLOW}Expansion statistics:${NC}"
        python3 << 'EOF'
import json
with open("sdxl_expansion_metadata.json") as f:
    data = json.load(f)
    stats = data['statistics']
    print(f"  Processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Avg original length: {stats['avg_orig_length']:.1f} tokens")
    print(f"  Avg expanded length: {stats['avg_expanded_length']:.1f} tokens")
    print(f"  Estimated cost: ${stats['estimated_cost_usd']:.3f}")
EOF
    fi

    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "  1. Review the expanded captions above"
    echo "  2. If quality is good, run full expansion:"
    echo "     conda run -n ai_env python scripts/generic/training/sdxl_caption_expander.py \\"
    echo "       --input-dir \"$SD15_DIR\" \\"
    echo "       --output-dir \"$BASE_DIR/$FILM/lora_data/training_data_sdxl/$TRAINING_PATH\" \\"
    echo "       --character-name \"$CHARACTER_NAME\" \\"
    echo "       --style \"$STYLE\""
    echo ""
    echo "  3. Or run batch expansion for all characters:"
    echo "     conda run -n ai_env python scripts/batch/expand_all_sdxl_captions.py --execute"
    echo ""

else
    echo ""
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}✗ Test failed${NC}"
    echo -e "${RED}=========================================${NC}"
    echo ""
    echo "Check the error messages above and fix the issue before running full expansion."
    exit 1
fi
