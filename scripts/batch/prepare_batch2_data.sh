#!/bin/bash
# Prepare Batch 2 SDXL Training Data
# Copies training data from training_data/ to training_data_sdxl/ with correct repeat structure

set -e

echo "=========================================="
echo "Preparing Batch 2 SDXL Training Data"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Character configurations: name, movie, images, repeats
declare -a CHARACTERS=(
    "barley_lightfoot:onward:254:4"
    "alberto:luca:509:2"
    "giulia:luca:546:2"
    "russell:up:243:4"
)

# Process each character
for char_config in "${CHARACTERS[@]}"; do
    IFS=':' read -r char_name movie expected_images repeats <<< "$char_config"

    echo -e "${YELLOW}Processing: $char_name from $movie${NC}"
    echo "  Expected images: $expected_images"
    echo "  Repeats: $repeats"

    # Define paths
    source_dir="/mnt/data/datasets/general/$movie/lora_data/training_data/${char_name}_identity"
    target_base="/mnt/data/datasets/general/$movie/lora_data/training_data_sdxl/${char_name}_identity"
    target_dir="$target_base/${repeats}_${char_name}"

    echo "  Source: $source_dir"
    echo "  Target: $target_dir"

    # Check if source exists
    if [ ! -d "$source_dir" ]; then
        echo -e "${RED}ERROR: Source directory not found: $source_dir${NC}"
        exit 1
    fi

    # Count source images
    source_images=$(find "$source_dir" -name "*.png" | wc -l)
    echo "  Found $source_images images in source"

    if [ "$source_images" -ne "$expected_images" ]; then
        echo -e "${YELLOW}WARNING: Expected $expected_images images but found $source_images${NC}"
    fi

    # Create target directory
    echo "  Creating target directory..."
    mkdir -p "$target_dir"

    # Find all subdirectories with images (e.g., 10_barley_lightfoot/)
    subdirs=$(find "$source_dir" -mindepth 1 -maxdepth 1 -type d)

    if [ -z "$subdirs" ]; then
        # No subdirectories, copy from source_dir directly
        echo "  Copying images and captions from main directory..."
        cp -v "$source_dir"/*.png "$target_dir/" 2>/dev/null || echo "  No PNG files in main directory"
        cp -v "$source_dir"/*.txt "$target_dir/" 2>/dev/null || echo "  No TXT files in main directory"
    else
        # Copy from all subdirectories
        echo "  Found subdirectories, copying from all subdirs..."
        for subdir in $subdirs; do
            subdir_name=$(basename "$subdir")
            echo "    Copying from $subdir_name/"
            cp -v "$subdir"/*.png "$target_dir/" 2>/dev/null || echo "      No PNG files"
            cp -v "$subdir"/*.txt "$target_dir/" 2>/dev/null || echo "      No TXT files"
        done
    fi

    # Verify copied files
    target_images=$(find "$target_dir" -name "*.png" | wc -l)
    target_captions=$(find "$target_dir" -name "*.txt" ! -name "*metadata*" | wc -l)

    echo "  Target images: $target_images"
    echo "  Target captions: $target_captions"

    if [ "$target_images" -ne "$target_captions" ]; then
        echo -e "${RED}ERROR: Image-caption mismatch! Images: $target_images, Captions: $target_captions${NC}"
        exit 1
    fi

    # Calculate training steps
    steps_per_epoch=$((target_images * repeats))
    total_steps=$((steps_per_epoch * 2))  # 2 epochs

    echo -e "${GREEN}✓ Success:${NC}"
    echo "    $target_images images with matching captions"
    echo "    Steps/epoch: $steps_per_epoch"
    echo "    Total steps (2 epochs): $total_steps"
    echo ""
done

echo "=========================================="
echo -e "${GREEN}All Batch 2 data prepared successfully!${NC}"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Barley Lightfoot: 4 repeats → ~1016 steps/epoch"
echo "  Alberto (human):  2 repeats → ~1018 steps/epoch"
echo "  Giulia:           2 repeats → ~1092 steps/epoch"
echo "  Russell:          4 repeats → ~972 steps/epoch"
echo ""
echo "Next step: Run sequential training script"
echo "  bash scripts/batch/train_batch2_sequential.sh"
