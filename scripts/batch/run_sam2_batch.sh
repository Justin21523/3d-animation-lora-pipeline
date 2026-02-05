#!/bin/bash
# SAM2 Batch Processing Script for 2D Animation Projects
# Processes all episodes sequentially for maximum GPU utilization

set -e

SCRIPT_DIR="/mnt/c/ai_projects/2d-animation-lora-pipeline"
SAM2_SCRIPT="$SCRIPT_DIR/scripts/segmentation/instance_segmentation.py"

# Configuration
MODEL="sam2_hiera_large"
MIN_SIZE=4096
CONTEXT_MODE="all"  # transparent, context, blurred versions
DEVICE="cuda"

process_project() {
    local PROJECT=$1
    local BASE_DIR="/mnt/data/datasets/general/$PROJECT"

    echo "=============================================="
    echo "Processing project: $PROJECT"
    echo "=============================================="

    # Find all episode directories
    if [ -d "$BASE_DIR/frames" ]; then
        for EPISODE_DIR in "$BASE_DIR/frames"/*/; do
            EPISODE=$(basename "$EPISODE_DIR")

            # Skip if not a directory or no frames
            [ ! -d "$EPISODE_DIR" ] && continue
            FRAME_COUNT=$(ls "$EPISODE_DIR"/*.jpg 2>/dev/null | wc -l)
            [ "$FRAME_COUNT" -eq 0 ] && continue

            OUTPUT_DIR="$BASE_DIR/instances/$EPISODE"

            # Check if already processed
            if [ -d "$OUTPUT_DIR/instances" ]; then
                EXISTING=$(ls "$OUTPUT_DIR/instances"/*.png 2>/dev/null | wc -l)
                if [ "$EXISTING" -gt 0 ]; then
                    echo "[$PROJECT/$EPISODE] Already has $EXISTING instances, checking if complete..."
                    # Simple heuristic: if we have at least 50% of frames processed, skip
                    if [ "$EXISTING" -gt $((FRAME_COUNT / 2)) ]; then
                        echo "[$PROJECT/$EPISODE] Looks complete, skipping..."
                        continue
                    fi
                fi
            fi

            echo ""
            echo ">>> Processing $PROJECT/$EPISODE ($FRAME_COUNT frames)"
            echo ">>> Output: $OUTPUT_DIR"
            echo ""

            # Create output directory
            mkdir -p "$OUTPUT_DIR"

            # Run SAM2
            cd "$SCRIPT_DIR"
            conda run -n ai_env python "$SAM2_SCRIPT" \
                "$EPISODE_DIR" \
                --output-dir "$OUTPUT_DIR" \
                --model "$MODEL" \
                --min-size "$MIN_SIZE" \
                --save-masks \
                --context-mode "$CONTEXT_MODE" \
                --device "$DEVICE"

            echo ""
            echo ">>> Completed $PROJECT/$EPISODE"
            echo ""
        done
    fi
}

# Main execution
echo "SAM2 Batch Processing Started: $(date)"
echo ""

# Process specified projects or all
if [ $# -gt 0 ]; then
    for PROJECT in "$@"; do
        process_project "$PROJECT"
    done
else
    # Default: process wylde-pak and gumbell
    process_project "wylde-pak"
    process_project "gumbell"
fi

echo ""
echo "=============================================="
echo "SAM2 Batch Processing Completed: $(date)"
echo "=============================================="
