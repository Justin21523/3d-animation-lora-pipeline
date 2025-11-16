#!/usr/bin/env bash
#
# Fully Automated Overnight Pipeline - Phase 1-4 Auto-Execution
#
# This script automatically executes all 4 phases and sends completion notification
#
# Usage:
#   nohup bash scripts/workflows/auto_overnight_pipeline.sh > /tmp/overnight_pipeline.log 2>&1 &
#

set -e  # Exit on error

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
DATA_ROOT="/mnt/data/ai_data/datasets/3d-anime/luca"
LOG_FILE="/tmp/overnight_pipeline_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Send notification (can be extended with email/webhook)
notify() {
    local title="$1"
    local message="$2"
    log "════════════════════════════════════════════════════════════"
    log "  NOTIFICATION: $title"
    log "════════════════════════════════════════════════════════════"
    log "$message"
    log "════════════════════════════════════════════════════════════"

    # Write notification file
    echo "$title" > /tmp/pipeline_notification.txt
    echo "$message" >> /tmp/pipeline_notification.txt
    echo "Time: $(date)" >> /tmp/pipeline_notification.txt
}

# Check if process completed successfully
wait_for_completion() {
    local pid=$1
    local phase_name="$2"
    local start_time=$(date +%s)

    log "⏳ Waiting for $phase_name (PID: $pid) to complete..."

    while kill -0 $pid 2>/dev/null; do
        sleep 60  # Check every minute
        elapsed=$(($(date +%s) - start_time))
        log "  Still running... (${elapsed}s elapsed)"
    done

    wait $pid
    exit_code=$?

    elapsed=$(($(date +%s) - start_time))

    if [ $exit_code -eq 0 ]; then
        log "✅ $phase_name completed successfully in ${elapsed}s"
        return 0
    else
        log "❌ $phase_name failed with exit code $exit_code"
        notify "Pipeline Failed" "$phase_name failed after ${elapsed}s"
        exit 1
    fi
}

log "════════════════════════════════════════════════════════════"
log "  AUTOMATED OVERNIGHT PIPELINE START"
log "════════════════════════════════════════════════════════════"
log "Log file: $LOG_FILE"
log ""

# ==================== Phase 1: Face Recognition Filter ====================
log "🔍 Phase 1: Face Recognition Filter"
log "════════════════════════════════════════════════════════════"

PHASE1_START=$(date +%s)

conda run -n ai_env python "$PROJECT_ROOT/scripts/generic/filtering/reference_based_frame_filter.py" \
    --input-dir "$DATA_ROOT/frames" \
    --reference-dir "$DATA_ROOT/training_ready/1_luca" \
    --output-dir "$DATA_ROOT/luca_frames_filtered" \
    --similarity-threshold 0.5 \
    --batch-size 128 \
    --device cuda \
    --save-metadata \
    >> "$LOG_FILE" 2>&1 &

PHASE1_PID=$!
wait_for_completion $PHASE1_PID "Phase 1"

PHASE1_END=$(date +%s)
PHASE1_DURATION=$((PHASE1_END - PHASE1_START))

FILTERED_COUNT=$(find "$DATA_ROOT/luca_frames_filtered" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
log "📊 Phase 1 Results: $FILTERED_COUNT frames filtered in ${PHASE1_DURATION}s"

if [ $FILTERED_COUNT -eq 0 ]; then
    notify "Pipeline Failed" "Phase 1 produced 0 filtered frames!"
    exit 1
fi

# ==================== Phase 2: Optimized SAM2 Segmentation ====================
log ""
log "🎯 Phase 2: Optimized SAM2 Segmentation"
log "════════════════════════════════════════════════════════════"

PHASE2_START=$(date +%s)

# Check if SAM2 is available, if not skip to Phase 3
if conda run -n ai_env python -c "import sam2" 2>/dev/null; then
    log "✓ SAM2 available, starting segmentation..."

    conda run -n ai_env python "$PROJECT_ROOT/scripts/generic/segmentation/optimized_sam2_batch.py" \
        --input-dir "$DATA_ROOT/luca_frames_filtered" \
        --output-dir "$DATA_ROOT/luca_instances_sam2" \
        --model sam2_hiera_large \
        --encoder-batch-size 16 \
        --decoder-batch-size 48 \
        --device cuda \
        >> "$LOG_FILE" 2>&1 &

    PHASE2_PID=$!
    wait_for_completion $PHASE2_PID "Phase 2"

    INSTANCE_COUNT=$(find "$DATA_ROOT/luca_instances_sam2" -name "*_inst*.png" 2>/dev/null | wc -l)
    log "📊 Phase 2 Results: $INSTANCE_COUNT instances extracted"

    PHASE2_INPUT="$DATA_ROOT/luca_instances_sam2"
else
    log "⚠️ SAM2 not available, using filtered frames directly"
    INSTANCE_COUNT=$FILTERED_COUNT
    PHASE2_INPUT="$DATA_ROOT/luca_frames_filtered"
fi

PHASE2_END=$(date +%s)
PHASE2_DURATION=$((PHASE2_END - PHASE2_START))

# ==================== Phase 3: Intelligent Processing ====================
log ""
log "🧠 Phase 3: Intelligent Frame Processing"
log "════════════════════════════════════════════════════════════"

PHASE3_START=$(date +%s)

conda run -n ai_env python "$PROJECT_ROOT/scripts/data_curation/intelligent_frame_processor.py" \
    "$PHASE2_INPUT" \
    --output-dir "$DATA_ROOT/luca_intelligent_candidates" \
    --device cuda \
    >> "$LOG_FILE" 2>&1 &

PHASE3_PID=$!
wait_for_completion $PHASE3_PID "Phase 3"

CANDIDATE_COUNT=0
for strategy in keep_full segment create_occlusion enhance_segment; do
    if [ -d "$DATA_ROOT/luca_intelligent_candidates/$strategy/images" ]; then
        count=$(find "$DATA_ROOT/luca_intelligent_candidates/$strategy/images" -name "*.png" 2>/dev/null | wc -l)
        CANDIDATE_COUNT=$((CANDIDATE_COUNT + count))
        log "  $strategy: $count images"
    fi
done

PHASE3_END=$(date +%s)
PHASE3_DURATION=$((PHASE3_END - PHASE3_START))
log "📊 Phase 3 Results: $CANDIDATE_COUNT candidates in ${PHASE3_DURATION}s"

# ==================== Phase 4: AI Quality Assessment + Curation ====================
log ""
log "⭐ Phase 4: AI Quality Assessment + Smart Curation"
log "════════════════════════════════════════════════════════════"

PHASE4_START=$(date +%s)

# Step 4a: AI Quality Assessment
log "📊 Step 4a: AI Quality Assessment..."
conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/ai_quality_assessor.py" \
    "$DATA_ROOT/luca_intelligent_candidates" \
    --output "$DATA_ROOT/luca_quality_scores.json" \
    --batch \
    --device cuda \
    >> "$LOG_FILE" 2>&1 &

PHASE4A_PID=$!
wait_for_completion $PHASE4A_PID "Phase 4a (Quality Assessment)"

# Step 4b: Smart Curation
log ""
log "🎯 Step 4b: Smart Curation..."
conda run -n ai_env python "$PROJECT_ROOT/scripts/data_curation/intelligent_dataset_curator.py" \
    "$DATA_ROOT/luca_intelligent_candidates" \
    --output-dir "$DATA_ROOT/luca_training_final" \
    --target-size 400 \
    --quality-scores "$DATA_ROOT/luca_quality_scores.json" \
    >> "$LOG_FILE" 2>&1 &

PHASE4B_PID=$!
wait_for_completion $PHASE4B_PID "Phase 4b (Curation)"

FINAL_COUNT=$(find "$DATA_ROOT/luca_training_final" -name "*.png" 2>/dev/null | wc -l)

PHASE4_END=$(date +%s)
PHASE4_DURATION=$((PHASE4_END - PHASE4_START))
log "📊 Phase 4 Results: $FINAL_COUNT final images in ${PHASE4_DURATION}s"

# ==================== Pipeline Complete ====================
PIPELINE_END=$(date +%s)
TOTAL_DURATION=$((PIPELINE_END - PHASE1_START))

log ""
log "════════════════════════════════════════════════════════════"
log "  ✅ PIPELINE COMPLETE!"
log "════════════════════════════════════════════════════════════"
log ""
log "📊 Pipeline Summary:"
log "────────────────────────────────────────"
log "  Phase 1: Face Filter"
log "    14,410 frames → $FILTERED_COUNT Luca frames"
log "    Time: ${PHASE1_DURATION}s"
log ""
log "  Phase 2: SAM2 Segmentation"
log "    $FILTERED_COUNT frames → $INSTANCE_COUNT instances"
log "    Time: ${PHASE2_DURATION}s"
log ""
log "  Phase 3: Intelligent Processing"
log "    $INSTANCE_COUNT instances → $CANDIDATE_COUNT candidates"
log "    Time: ${PHASE3_DURATION}s"
log ""
log "  Phase 4: AI Quality + Curation"
log "    $CANDIDATE_COUNT candidates → $FINAL_COUNT best images"
log "    Time: ${PHASE4_DURATION}s"
log ""
log "📈 Total Time: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 3600))h $(($TOTAL_DURATION % 3600 / 60))m)"
log ""
log "📁 Final Output:"
log "  Location: $DATA_ROOT/luca_training_final"
log "  Images: $FINAL_COUNT"
log "  Quality Scores: $DATA_ROOT/luca_quality_scores.json"
log ""

# Send completion notification
notify "Pipeline Complete!" "Successfully processed 14,410 frames → $FINAL_COUNT training images in $(($TOTAL_DURATION / 3600))h $(($TOTAL_DURATION % 3600 / 60))m.

Final dataset ready at: $DATA_ROOT/luca_training_final

Next steps:
1. Review quality scores and samples
2. Generate training config
3. Start LoRA training (Trial 3.8)"

log "🎯 Next Steps:"
log "  1. Review: cat $DATA_ROOT/luca_quality_scores.json"
log "  2. Samples: ls $DATA_ROOT/luca_training_final/*.png | head -20"
log "  3. Train: Generate Trial 3.8 config and start training"
log ""
log "════════════════════════════════════════════════════════════"
