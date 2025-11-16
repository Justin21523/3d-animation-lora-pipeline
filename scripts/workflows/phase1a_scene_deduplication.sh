#!/bin/bash
# Phase 1a: Scene Frame Deduplication (Pure CPU)
# Reduces 4323 frames to ~1000-1500 unique scenes for future background extraction

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Configuration
INPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/frames_sampled"
OUTPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/frames_deduplicated"
LOG_DIR="/tmp/scene_dedup_phase1a"

mkdir -p "$LOG_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Phase 1a: Scene Frame Deduplication (CPU Only)          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Purpose: Reduce 4323 frames to unique scenes before background extraction"
echo ""
echo "This will:"
echo "  1. Deduplicate scene frames using pHash"
echo "  2. Generate statistics and quality report"
echo "  3. Prepare for Phase 2 (character detection + background extraction)"
echo ""
echo "Running in background with low priority (nice -n 19)..."
echo "Estimated time: 15-25 minutes"
echo ""

# Step 1: Check if input data exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

NUM_FRAMES=$(ls "$INPUT_DIR"/*.jpg 2>/dev/null | wc -l)
echo "📊 Found $NUM_FRAMES scene frames"

if [ "$NUM_FRAMES" -eq 0 ]; then
    echo "❌ Error: No frames found!"
    exit 1
fi

# Step 2: Scene Deduplication
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/2: Scene Frame Deduplication (pHash)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Removing duplicate/near-duplicate frames from same scenes..."
echo ""

# Use stricter threshold (8) for full scenes vs individual instances (12)
# Lower threshold = more strict (fewer duplicates pass through)
cd "$PROJECT_ROOT" && nice -n 19 /home/b0979/.conda/envs/ai_env/bin/python scripts/generic/preprocessing/deduplicate.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --method phash \
    --phash-threshold 8 \
    --keep-mode best 2>&1 | tee -a "$LOG_DIR/workflow.log"

if [ $? -ne 0 ]; then
    echo "❌ Deduplication failed! Check log: $LOG_DIR/deduplication.log"
    exit 1
fi

echo "✅ Deduplication completed!"

# Step 3: Generate Statistics
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/2: Statistics and Quality Report"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

FINAL_COUNT=$(ls "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l)
REMOVED_COUNT=$((NUM_FRAMES - FINAL_COUNT))
DEDUP_RATE=$(echo "scale=1; $REMOVED_COUNT * 100 / $NUM_FRAMES" | bc)

# Generate summary report
REPORT_FILE="$OUTPUT_DIR/phase1a_summary.txt"
cat > "$REPORT_FILE" <<EOF
╔════════════════════════════════════════════════════════════╗
║        Phase 1a: Scene Deduplication Summary              ║
╚════════════════════════════════════════════════════════════╝

Date: $(date '+%Y-%m-%d %H:%M:%S')

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Data Statistics

Original frames:          $NUM_FRAMES
Unique scenes (dedupe):   $FINAL_COUNT
Removed duplicates:       $REMOVED_COUNT
Deduplication rate:       $DEDUP_RATE%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📂 Output Directory

Deduplicated frames:      $OUTPUT_DIR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Next Steps (Phase 2 - requires GPU)

✅ Phase 1a completed!

⏳ When GPU optimization completes or reaches Trial 30+:

📝 Phase 2 Tasks:
   1. Character detection (YOLOv8/SAM2)
   2. Character mask generation
   3. Background inpainting (LaMa)
   4. Scene clustering (CLIP embeddings)
   5. Caption generation (VLM)
   6. Train Background LoRA

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Efficiency Gain

By deduplicating first:
- Reduced processing load by $DEDUP_RATE%
- Saved ~$((REMOVED_COUNT * 5 / 60)) minutes in future GPU processing
- Better quality training data (unique scenes only)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 Quality Notes

- Used pHash with threshold=8 (strict for full scenes)
- Kept best quality version when duplicates found
- Preserved original frame naming for traceability

EOF

cat "$REPORT_FILE"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                ✅ Phase 1a Completed!                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary report saved to: $REPORT_FILE"
echo ""
echo "Unique scenes ready for Phase 2 processing"
echo "(Phase 2 will require GPU for character detection + inpainting)"
echo ""

# Save metadata
METADATA_FILE="$OUTPUT_DIR/phase1a_metadata.json"
cat > "$METADATA_FILE" <<EOF
{
  "phase": "Phase 1a - Scene Deduplication (CPU)",
  "completion_time": "$(date '+%Y-%m-%d %H:%M:%S')",
  "input_dir": "$INPUT_DIR",
  "output_dir": "$OUTPUT_DIR",
  "statistics": {
    "original_frames": $NUM_FRAMES,
    "unique_scenes": $FINAL_COUNT,
    "removed_duplicates": $REMOVED_COUNT,
    "deduplication_rate_percent": $DEDUP_RATE
  },
  "parameters": {
    "method": "phash",
    "threshold": 8,
    "keep_strategy": "best"
  },
  "next_steps": [
    "Character detection (YOLOv8/SAM2)",
    "Character mask generation",
    "Background inpainting (LaMa)",
    "Scene clustering (CLIP)",
    "Caption generation (VLM)",
    "Train Background LoRA"
  ]
}
EOF

echo "Metadata saved to: $METADATA_FILE"
echo ""
echo "✨ Phase 1a tasks completed successfully!"
