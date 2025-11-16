#!/bin/bash
# Automated Background LoRA Data Preparation Workflow
# Phase 1: CPU-intensive tasks (run in background during GPU optimization)

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Configuration
INPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/segmented/background"
INPAINTED_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_clean_cpu"
DEDUP_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_dedup"
LOG_DIR="/tmp/background_lora_prep"

mkdir -p "$LOG_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Background LoRA Data Preparation - Phase 1 (CPU)        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "This script will:"
echo "  1. Inpaint backgrounds (remove character remnants) - CPU version"
echo "  2. Deduplicate images (pHash)"
echo "  3. Generate statistics report"
echo ""
echo "Running in background to avoid interfering with GPU optimization..."
echo ""

# Step 1: Check if background data exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Background directory not found: $INPUT_DIR"
    echo "Please run layered segmentation first."
    exit 1
fi

NUM_BACKGROUNDS=$(ls "$INPUT_DIR"/*.png 2>/dev/null | wc -l)
echo "📊 Found $NUM_BACKGROUNDS background images"

if [ "$NUM_BACKGROUNDS" -eq 0 ]; then
    echo "❌ Error: No background images found!"
    exit 1
fi

# Step 2: Background Inpainting (CPU version)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/3: Background Inpainting (CPU - OpenCV telea)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "This will remove character remnants from backgrounds..."
echo "Estimated time: 30-60 minutes"
echo ""

# Run with low priority (nice) to avoid interfering with GPU training
nice -n 19 python3 "$PROJECT_ROOT/scripts/generic/inpainting/background_inpainting.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$INPAINTED_DIR" \
    --method telea \
    --device cpu \
    --log-file "$LOG_DIR/background_inpainting.log" 2>&1 | tee -a "$LOG_DIR/workflow.log"

if [ $? -ne 0 ]; then
    echo "❌ Inpainting failed! Check log: $LOG_DIR/background_inpainting.log"
    exit 1
fi

echo "✅ Inpainting completed!"

# Step 3: Deduplication
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/3: Image Deduplication (pHash)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Removing duplicate backgrounds..."
echo "Estimated time: 10-20 minutes"
echo ""

nice -n 19 python3 "$PROJECT_ROOT/scripts/generic/quality/deduplicate_images.py" \
    --input-dir "$INPAINTED_DIR" \
    --output-dir "$DEDUP_DIR" \
    --method phash \
    --threshold 12 \
    --keep-strategy best \
    --log-file "$LOG_DIR/deduplication.log" 2>&1 | tee -a "$LOG_DIR/workflow.log"

if [ $? -ne 0 ]; then
    echo "❌ Deduplication failed! Check log: $LOG_DIR/deduplication.log"
    exit 1
fi

echo "✅ Deduplication completed!"

# Step 4: Generate Statistics
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/3: Dataset Statistics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

FINAL_COUNT=$(ls "$DEDUP_DIR"/*.png 2>/dev/null | wc -l)
REMOVED_COUNT=$((NUM_BACKGROUNDS - FINAL_COUNT))
DEDUP_RATE=$(echo "scale=1; $REMOVED_COUNT * 100 / $NUM_BACKGROUNDS" | bc)

# Generate summary report
REPORT_FILE="$DEDUP_DIR/preparation_summary.txt"
cat > "$REPORT_FILE" <<EOF
╔════════════════════════════════════════════════════════════╗
║        Background LoRA Data Preparation Summary            ║
╚════════════════════════════════════════════════════════════╝

Date: $(date '+%Y-%m-%d %H:%M:%S')

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Data Statistics

Original backgrounds:     $NUM_BACKGROUNDS
After inpainting:         $(ls "$INPAINTED_DIR"/*.png 2>/dev/null | wc -l)
After deduplication:      $FINAL_COUNT
Removed duplicates:       $REMOVED_COUNT
Deduplication rate:       $DEDUP_RATE%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📂 Output Directories

Inpainted backgrounds:    $INPAINTED_DIR
Deduplicated backgrounds: $DEDUP_DIR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Next Steps

✅ Phase 1 (CPU tasks) completed!

⏸️  Waiting for GPU optimization to complete...

📝 When optimization finishes:
   1. Run scene clustering (CLIP embeddings)
   2. Generate captions (VLM)
   3. Train Background LoRA

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Estimated time saved: 2-3 hours (by running CPU tasks in parallel)

EOF

cat "$REPORT_FILE"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                ✅ Phase 1 Completed!                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary report saved to: $REPORT_FILE"
echo ""
echo "Background data is ready for clustering and caption generation"
echo "(to be done when GPU optimization completes)"
echo ""

# Save metadata
METADATA_FILE="$DEDUP_DIR/preparation_metadata.json"
cat > "$METADATA_FILE" <<EOF
{
  "phase": "Phase 1 - CPU Tasks",
  "completion_time": "$(date '+%Y-%m-%d %H:%M:%S')",
  "input_dir": "$INPUT_DIR",
  "inpainted_dir": "$INPAINTED_DIR",
  "output_dir": "$DEDUP_DIR",
  "statistics": {
    "original_backgrounds": $NUM_BACKGROUNDS,
    "final_backgrounds": $FINAL_COUNT,
    "removed_duplicates": $REMOVED_COUNT,
    "deduplication_rate_percent": $DEDUP_RATE
  },
  "next_steps": [
    "Run scene clustering (CLIP embeddings)",
    "Generate captions (VLM)",
    "Train Background LoRA"
  ]
}
EOF

echo "Metadata saved to: $METADATA_FILE"
echo ""
echo "✨ All Phase 1 tasks completed successfully!"
