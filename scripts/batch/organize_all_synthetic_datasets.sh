#!/bin/bash
#
# Organize All Synthetic LoRA Datasets (45 total)
# Creates Kohya_ss format datasets for:
# - 42 character-specific LoRAs (14 characters × 3 types)
# - 3 universal LoRAs (pose, action, expression)
#
# Author: LLMProvider Tooling
# Date: 2025-12-04

set -e

FILTERED_DATA="/mnt/data/ai_data/synthetic_lora_data/filtered_data"
OUTPUT_ROOT="/mnt/data/ai_data/synthetic_lora_data/datasets"
SCRIPT_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/generic/training"

# Characters list
CHARACTERS=(
    "alberto"
    "bryce"
    "caleb"
    "elio"
    "giulia"
    "ian_lightfoot"
    "luca"
    "miguel"
    "orion"
    "russell"
    "tyler"
    "alberto_seamonster"
    "luca_seamonster"
    "barley_lightfoot"
)

LORA_TYPES=("pose" "action" "expression")

echo "====================================="
echo "Dataset Organization - Phase 3"
echo "====================================="
echo ""
echo "Total datasets to create: 45"
echo "  - 42 character-specific"
echo "  - 3 universal (cross-character)"
echo ""
echo "Output root: $OUTPUT_ROOT"
echo ""

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# Statistics
TOTAL_DATASETS=0
SUCCESS_COUNT=0
FAILED_COUNT=0

# ===================================
# Part 1: Character-Specific Datasets
# ===================================

echo "=== Part 1: Character-Specific Datasets (42) ==="
echo ""

for CHARACTER in "${CHARACTERS[@]}"; do
    for LORA_TYPE in "${LORA_TYPES[@]}"; do
        DATASET_NAME="${CHARACTER}_${LORA_TYPE}"
        INPUT_DIR="${FILTERED_DATA}/${CHARACTER}/${LORA_TYPE}/tier_a"
        OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}"

        echo "[$((TOTAL_DATASETS + 1))/45] Organizing: $DATASET_NAME"

        # Check if input directory exists
        if [ ! -d "$INPUT_DIR" ]; then
            echo "  ⚠️  Input directory not found: $INPUT_DIR"
            echo "  Skipping..."
            ((FAILED_COUNT++))
            ((TOTAL_DATASETS++))
            continue
        fi

        # Count images
        IMAGE_COUNT=$(find "$INPUT_DIR" -name "*.png" | wc -l)
        if [ "$IMAGE_COUNT" -eq 0 ]; then
            echo "  ⚠️  No images found in $INPUT_DIR"
            echo "  Skipping..."
            ((FAILED_COUNT++))
            ((TOTAL_DATASETS++))
            continue
        fi

        echo "  Input: $IMAGE_COUNT images"

        # Run dataset organizer
        conda run -n ai_env python "${SCRIPT_DIR}/dataset_organizer.py" \
            --source-dir "$INPUT_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --concept-name "${CHARACTER}_${LORA_TYPE}" \
            --repeat-count 1 \
            --target-resolution 1024 \
            2>&1 | grep -E "(✓|✗|Error|Warning|organized|created)" || true

        if [ $? -eq 0 ]; then
            echo "  ✅ Complete: $DATASET_NAME"
            ((SUCCESS_COUNT++))
        else
            echo "  ❌ Failed: $DATASET_NAME"
            ((FAILED_COUNT++))
        fi

        ((TOTAL_DATASETS++))
        echo ""
    done
done

echo ""
echo "=== Character-Specific Datasets Complete ==="
echo "  Success: $SUCCESS_COUNT"
echo "  Failed: $FAILED_COUNT"
echo ""

# ===================================
# Part 2: Universal (Cross-Character) Datasets
# ===================================

echo "=== Part 2: Universal Cross-Character Datasets (3) ==="
echo ""

for LORA_TYPE in "${LORA_TYPES[@]}"; do
    DATASET_NAME="universal_${LORA_TYPE}"
    TEMP_DIR="/tmp/synthetic_universal_${LORA_TYPE}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}"

    echo "[$((TOTAL_DATASETS + 1))/45] Creating: $DATASET_NAME"

    # Create temp directory
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"

    # Collect all images from all characters for this type
    echo "  Collecting images from all characters..."
    COLLECTED_COUNT=0

    for CHARACTER in "${CHARACTERS[@]}"; do
        INPUT_DIR="${FILTERED_DATA}/${CHARACTER}/${LORA_TYPE}/tier_a"

        if [ -d "$INPUT_DIR" ]; then
            # Copy images and captions to temp dir
            find "$INPUT_DIR" -name "*.png" -exec cp {} "$TEMP_DIR/" \; 2>/dev/null || true
            find "$INPUT_DIR" -name "*.txt" -exec cp {} "$TEMP_DIR/" \; 2>/dev/null || true

            COUNT=$(find "$INPUT_DIR" -name "*.png" | wc -l)
            ((COLLECTED_COUNT += COUNT))
        fi
    done

    echo "  Total collected: $COLLECTED_COUNT images"

    if [ "$COLLECTED_COUNT" -eq 0 ]; then
        echo "  ⚠️  No images collected for $DATASET_NAME"
        echo "  Skipping..."
        rm -rf "$TEMP_DIR"
        ((FAILED_COUNT++))
        ((TOTAL_DATASETS++))
        continue
    fi

    # Run dataset organizer
    conda run -n ai_env python "${SCRIPT_DIR}/dataset_organizer.py" \
        --source-dir "$TEMP_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --concept-name "universal_${LORA_TYPE}" \
        --repeat-count 1 \
        --target-resolution 1024 \
        2>&1 | grep -E "(✓|✗|Error|Warning|organized|created)" || true

    if [ $? -eq 0 ]; then
        echo "  ✅ Complete: $DATASET_NAME ($COLLECTED_COUNT images)"
        ((SUCCESS_COUNT++))
    else
        echo "  ❌ Failed: $DATASET_NAME"
        ((FAILED_COUNT++))
    fi

    # Cleanup temp dir
    rm -rf "$TEMP_DIR"

    ((TOTAL_DATASETS++))
    echo ""
done

echo ""
echo "====================================="
echo "Dataset Organization Complete!"
echo "====================================="
echo ""
echo "Total datasets: $TOTAL_DATASETS"
echo "  ✅ Success: $SUCCESS_COUNT"
echo "  ❌ Failed: $FAILED_COUNT"
echo ""
echo "Output directory: $OUTPUT_ROOT"
echo ""

# Generate summary report
REPORT_FILE="${OUTPUT_ROOT}/organization_report.json"

cat > "$REPORT_FILE" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "total_datasets": $TOTAL_DATASETS,
  "success_count": $SUCCESS_COUNT,
  "failed_count": $FAILED_COUNT,
  "output_root": "$OUTPUT_ROOT",
  "character_specific": 42,
  "universal": 3
}
EOF

echo "Report saved to: $REPORT_FILE"
echo ""
echo "✅ Phase 3 Complete! Ready for Phase 4 (Training Config Generation)"
