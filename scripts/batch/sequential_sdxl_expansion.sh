#!/bin/bash
# Sequential SDXL Caption Expansion
# 逐個角色處理，避免批次處理卡住問題
#
# Author: LLMProvider Tooling
# Date: 2025-11-22

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Characters to process (順序執行)
CHARACTERS=(
    "giulia:giulia marcovaldo:luca:pixar"
    "russell:russell:up:pixar"
    "orion:orion:orion:dreamworks"
    "elio:elio solis:elio:pixar"
    "bryce:bryce markwell:elio:pixar"
    "caleb:caleb:elio:pixar"
    "glordon:glordon:elio:pixar"
    "miguel:miguel rivera:coco:pixar"
)

BASE_DIR="/mnt/data/ai_data/datasets/3d-anime"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/sequential_sdxl_expansion_$TIMESTAMP.log"

echo "================================================================" | tee -a "$MAIN_LOG"
echo "Sequential SDXL Caption Expansion Started" | tee -a "$MAIN_LOG"
echo "================================================================" | tee -a "$MAIN_LOG"
echo "Timestamp: $TIMESTAMP" | tee -a "$MAIN_LOG"
echo "Processing ${#CHARACTERS[@]} characters" | tee -a "$MAIN_LOG"
echo "================================================================" | tee -a "$MAIN_LOG"

SUCCESSFUL=0
FAILED=0
SKIPPED=0

for char_config in "${CHARACTERS[@]}"; do
    IFS=':' read -r char_id char_name film style <<< "$char_config"

    echo "" | tee -a "$MAIN_LOG"
    echo "================================================================" | tee -a "$MAIN_LOG"
    echo "Processing: $char_id ($film)" | tee -a "$MAIN_LOG"
    echo "================================================================" | tee -a "$MAIN_LOG"

    # Determine input/output paths based on film
    if [ "$film" = "luca" ]; then
        INPUT_DIR="$BASE_DIR/luca/lora_data/training_data/${char_id}_identity"
        OUTPUT_DIR="$BASE_DIR/luca/lora_data/training_data_sdxl/${char_id}_identity"
    elif [ "$film" = "onward" ]; then
        INPUT_DIR="$BASE_DIR/onward/lora_data/training_data/${char_id}_lightfoot_identity"
        OUTPUT_DIR="$BASE_DIR/onward/lora_data/training_data_sdxl/${char_id}_lightfoot_identity"
    elif [ "$film" = "up" ]; then
        INPUT_DIR="$BASE_DIR/up/lora_data/training_data/${char_id}_identity"
        OUTPUT_DIR="$BASE_DIR/up/lora_data/training_data_sdxl/${char_id}_identity"
    elif [ "$film" = "orion" ]; then
        INPUT_DIR="$BASE_DIR/orion/lora_data/training_data/${char_id}_identity"
        OUTPUT_DIR="$BASE_DIR/orion/lora_data/training_data_sdxl/${char_id}_identity"
    elif [ "$film" = "elio" ]; then
        INPUT_DIR="$BASE_DIR/elio/lora_data/training_data/${char_id}_identity"
        OUTPUT_DIR="$BASE_DIR/elio/lora_data/training_data_sdxl/${char_id}_identity"
    elif [ "$film" = "coco" ]; then
        INPUT_DIR="$BASE_DIR/coco/lora_data/training_data/${char_id}_identity"
        OUTPUT_DIR="$BASE_DIR/coco/lora_data/training_data_sdxl/${char_id}_identity"
    else
        echo "❌ Unknown film: $film" | tee -a "$MAIN_LOG"
        ((FAILED++))
        continue
    fi

    # Check if input exists
    if [ ! -d "$INPUT_DIR" ]; then
        echo "⚠️  Input directory not found: $INPUT_DIR" | tee -a "$MAIN_LOG"
        ((SKIPPED++))
        continue
    fi

    # Count source captions
    TOTAL_CAPTIONS=$(find "$INPUT_DIR" -name "*.txt" 2>/dev/null | wc -l)
    if [ "$TOTAL_CAPTIONS" -eq 0 ]; then
        echo "⚠️  No captions found in $INPUT_DIR" | tee -a "$MAIN_LOG"
        ((SKIPPED++))
        continue
    fi

    # Check if already completed
    if [ -d "$OUTPUT_DIR" ]; then
        EXISTING_CAPTIONS=$(find "$OUTPUT_DIR" -name "*.txt" 2>/dev/null | wc -l)
        METADATA_FILE="$OUTPUT_DIR/sdxl_expansion_metadata.json"

        if [ "$EXISTING_CAPTIONS" -eq "$TOTAL_CAPTIONS" ] && [ -f "$METADATA_FILE" ]; then
            echo "✅ Already completed: $char_id ($EXISTING_CAPTIONS/$TOTAL_CAPTIONS)" | tee -a "$MAIN_LOG"
            ((SUCCESSFUL++))
            continue
        else
            echo "🔄 Partial completion: $char_id ($EXISTING_CAPTIONS/$TOTAL_CAPTIONS)" | tee -a "$MAIN_LOG"
        fi
    fi

    # Run expansion for this character
    CHAR_LOG="$LOG_DIR/sdxl_${char_id}_$TIMESTAMP.log"
    echo "▶️  Starting expansion..." | tee -a "$MAIN_LOG"
    echo "   Input: $INPUT_DIR" | tee -a "$MAIN_LOG"
    echo "   Output: $OUTPUT_DIR" | tee -a "$MAIN_LOG"
    echo "   Log: $CHAR_LOG" | tee -a "$MAIN_LOG"

    if conda run -n ai_env python "$PROJECT_ROOT/scripts/generic/training/sdxl_caption_expander.py" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --character-name "$char_name" \
        --style "$style" \
        2>&1 | tee "$CHAR_LOG"; then

        # Verify completion
        FINAL_COUNT=$(find "$OUTPUT_DIR" -name "*.txt" 2>/dev/null | wc -l)
        if [ "$FINAL_COUNT" -eq "$TOTAL_CAPTIONS" ]; then
            echo "✅ Success: $char_id ($FINAL_COUNT/$TOTAL_CAPTIONS)" | tee -a "$MAIN_LOG"
            ((SUCCESSFUL++))
        else
            echo "⚠️  Incomplete: $char_id ($FINAL_COUNT/$TOTAL_CAPTIONS)" | tee -a "$MAIN_LOG"
            ((FAILED++))
        fi
    else
        echo "❌ Failed: $char_id" | tee -a "$MAIN_LOG"
        ((FAILED++))
    fi

    # Wait between characters to avoid rate limits
    echo "Waiting 10 seconds before next character..." | tee -a "$MAIN_LOG"
    sleep 10
done

echo "" | tee -a "$MAIN_LOG"
echo "================================================================" | tee -a "$MAIN_LOG"
echo "Sequential SDXL Expansion Complete" | tee -a "$MAIN_LOG"
echo "================================================================" | tee -a "$MAIN_LOG"
echo "Successful: $SUCCESSFUL" | tee -a "$MAIN_LOG"
echo "Failed: $FAILED" | tee -a "$MAIN_LOG"
echo "Skipped: $SKIPPED" | tee -a "$MAIN_LOG"
echo "================================================================" | tee -a "$MAIN_LOG"

exit 0
