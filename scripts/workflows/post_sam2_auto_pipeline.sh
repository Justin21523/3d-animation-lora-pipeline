#!/usr/bin/env bash
#
# Post-SAM2 Automatic Pipeline
#
# Waits for SAM2 to complete, then automatically runs:
# 1. Verification of outputs
# 2. Intelligent Processing (using instances_context/)
# 3. Quality Curation (→ 400 best images)
# 4. Summary report
#
# Usage: bash scripts/workflows/post_sam2_auto_pipeline.sh

set -e

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
DATA_ROOT="/mnt/data/ai_data/datasets/3d-anime/luca"

SAM2_OUTPUT="${DATA_ROOT}/luca_instances_sam2"
INSTANCES_CONTEXT="${SAM2_OUTPUT}/instances_context"
INTELLIGENT_OUTPUT="${DATA_ROOT}/luca_intelligent_candidates"
FINAL_OUTPUT="${DATA_ROOT}/luca_curated_400"

LOG_FILE="${DATA_ROOT}/post_sam2_pipeline.log"

echo "════════════════════════════════════════════════════════════"
echo "  Post-SAM2 Automatic Pipeline"
echo "  Using instances_context/ (with background)"
echo "════════════════════════════════════════════════════════════"
echo "" | tee -a "$LOG_FILE"

# ==================== Wait for SAM2 to Complete ====================
echo "⏳ Waiting for SAM2 segmentation to complete..."
echo "   Process ID: Monitoring instance_segmentation.py"
echo ""

while ps aux | grep -q "[i]nstance_segmentation.py"; do
    PROCESSED=$(find "${SAM2_OUTPUT}/instances" -name "scene*_inst0.png" 2>/dev/null | wc -l)
    TOTAL=5143
    PERCENTAGE=$(echo "scale=1; $PROCESSED * 100 / $TOTAL" | bc)

    echo "[$(date '+%H:%M:%S')] Progress: $PROCESSED / $TOTAL frames ($PERCENTAGE%)" | tee -a "$LOG_FILE"
    sleep 300  # Check every 5 minutes
done

echo "" | tee -a "$LOG_FILE"
echo "✅ SAM2 Segmentation Complete!" | tee -a "$LOG_FILE"
echo "   Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== Verify SAM2 Outputs ====================
echo "════════════════════════════════════════════════════════════"
echo "📊 Verifying SAM2 Outputs"
echo "════════════════════════════════════════════════════════════"
echo "" | tee -a "$LOG_FILE"

INSTANCES_COUNT=$(find "${SAM2_OUTPUT}/instances" -name "*.png" 2>/dev/null | wc -l)
CONTEXT_COUNT=$(find "${INSTANCES_CONTEXT}" -name "*.png" 2>/dev/null | wc -l)
BACKGROUNDS_COUNT=$(find "${SAM2_OUTPUT}/backgrounds" -name "*.png" 2>/dev/null | wc -l)
MASKS_COUNT=$(find "${SAM2_OUTPUT}/masks" -name "*.png" 2>/dev/null | wc -l)

echo "SAM2 Output Summary:" | tee -a "$LOG_FILE"
echo "  instances/          $INSTANCES_COUNT files" | tee -a "$LOG_FILE"
echo "  instances_context/  $CONTEXT_COUNT files ⭐" | tee -a "$LOG_FILE"
echo "  backgrounds/        $BACKGROUNDS_COUNT files" | tee -a "$LOG_FILE"
echo "  masks/              $MASKS_COUNT files" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ ! -d "$INSTANCES_CONTEXT" ] || [ "$CONTEXT_COUNT" -eq 0 ]; then
    echo "❌ Error: instances_context/ not found or empty!" | tee -a "$LOG_FILE"
    echo "   Expected: instances with background context" | tee -a "$LOG_FILE"
    echo "   Pipeline aborted." | tee -a "$LOG_FILE"
    exit 1
fi

echo "✅ Verification passed. Using instances_context/ for next phase." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== Phase 2: Intelligent Processing ====================
echo "════════════════════════════════════════════════════════════"
echo "🧠 Phase 2: Intelligent Frame Processing"
echo "════════════════════════════════════════════════════════════"
echo "  Input:    $INSTANCES_CONTEXT" | tee -a "$LOG_FILE"
echo "  Output:   $INTELLIGENT_OUTPUT" | tee -a "$LOG_FILE"
echo "  Strategy: 4 approaches (keep_full, segment, occlusion, enhance)" | tee -a "$LOG_FILE"
echo "  Expected: ~8,000 diverse candidates" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

START_TIME=$(date +%s)

echo "🚀 Starting intelligent processing..." | tee -a "$LOG_FILE"

conda run -n ai_env python "$PROJECT_ROOT/scripts/data_curation/intelligent_frame_processor.py" \
    "$INSTANCES_CONTEXT" \
    --output-dir "$INTELLIGENT_OUTPUT" \
    --decision-config "$PROJECT_ROOT/configs/stages/intelligent_processing/decision_thresholds.yaml" \
    --strategy-config "$PROJECT_ROOT/configs/stages/intelligent_processing/strategy_configs.yaml" \
    --device cuda 2>&1 | tee -a "$LOG_FILE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "" | tee -a "$LOG_FILE"
echo "✅ Phase 2 Complete!" | tee -a "$LOG_FILE"
echo "   Time elapsed: $((ELAPSED / 60)) minutes" | tee -a "$LOG_FILE"

# Count candidates by strategy
CANDIDATE_COUNT=0
echo "" | tee -a "$LOG_FILE"
echo "Strategy Distribution:" | tee -a "$LOG_FILE"
for strategy in keep_full segment create_occlusion enhance_segment; do
    if [ -d "$INTELLIGENT_OUTPUT/$strategy/images" ]; then
        count=$(find "$INTELLIGENT_OUTPUT/$strategy/images" -name "*.png" 2>/dev/null | wc -l)
        CANDIDATE_COUNT=$((CANDIDATE_COUNT + count))
        printf "  %-20s %6d images\n" "$strategy:" "$count" | tee -a "$LOG_FILE"
    fi
done
echo "  ────────────────────────────────" | tee -a "$LOG_FILE"
printf "  %-20s %6d images\n" "Total:" "$CANDIDATE_COUNT" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== Phase 3: Quality Curation ====================
echo "════════════════════════════════════════════════════════════"
echo "⭐ Phase 3: Quality Curation"
echo "════════════════════════════════════════════════════════════"
echo "  Input:    $INTELLIGENT_OUTPUT ($CANDIDATE_COUNT candidates)" | tee -a "$LOG_FILE"
echo "  Output:   $FINAL_OUTPUT" | tee -a "$LOG_FILE"
echo "  Target:   400 best images" | tee -a "$LOG_FILE"
echo "  Method:   Quality scoring + diversity balancing + deduplication" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

START_TIME=$(date +%s)

echo "🚀 Starting quality curation..." | tee -a "$LOG_FILE"

conda run -n ai_env python "$PROJECT_ROOT/scripts/data_curation/intelligent_dataset_curator.py" \
    "$INTELLIGENT_OUTPUT" \
    --output-dir "$FINAL_OUTPUT" \
    --target-size 400 \
    --decision-config "$PROJECT_ROOT/configs/stages/intelligent_processing/decision_thresholds.yaml" 2>&1 | tee -a "$LOG_FILE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

FINAL_COUNT=$(find "$FINAL_OUTPUT/images" -name "*.png" 2>/dev/null | wc -l)

echo "" | tee -a "$LOG_FILE"
echo "✅ Phase 3 Complete!" | tee -a "$LOG_FILE"
echo "   Final training set: $FINAL_COUNT images" | tee -a "$LOG_FILE"
echo "   Time elapsed: $((ELAPSED / 60)) minutes" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ==================== Final Summary ====================
echo "════════════════════════════════════════════════════════════"
echo "  ✅ PIPELINE COMPLETE!"
echo "════════════════════════════════════════════════════════════"
echo "" | tee -a "$LOG_FILE"

echo "📊 Complete Pipeline Summary:" | tee -a "$LOG_FILE"
echo "────────────────────────────────────────" | tee -a "$LOG_FILE"
echo "  Phase 1: Face Recognition Filter" | tee -a "$LOG_FILE"
echo "    14,410 frames → 5,143 Luca frames" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  Phase 2: SAM2 Segmentation" | tee -a "$LOG_FILE"
echo "    5,143 frames → $CONTEXT_COUNT instances (with context)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  Phase 3: Intelligent Processing" | tee -a "$LOG_FILE"
echo "    $CONTEXT_COUNT instances → $CANDIDATE_COUNT diverse candidates" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  Phase 4: Quality Curation" | tee -a "$LOG_FILE"
echo "    $CANDIDATE_COUNT candidates → $FINAL_COUNT best images" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "📁 Final Training Dataset:" | tee -a "$LOG_FILE"
echo "────────────────────────────────────────" | tee -a "$LOG_FILE"
echo "  Location: $FINAL_OUTPUT" | tee -a "$LOG_FILE"
echo "  Images:   $FINAL_COUNT" | tee -a "$LOG_FILE"
if [ -f "$FINAL_OUTPUT/curation_report.json" ]; then
    echo "  Report:   $FINAL_OUTPUT/curation_report.json" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

echo "🎯 Next Steps:" | tee -a "$LOG_FILE"
echo "────────────────────────────────────────" | tee -a "$LOG_FILE"
echo "  1. Review curation report and sample images" | tee -a "$LOG_FILE"
echo "  2. (Optional) Generate captions if needed" | tee -a "$LOG_FILE"
echo "  3. Create LoRA training config" | tee -a "$LOG_FILE"
echo "  4. Start LoRA training (Trial 3.7)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "📝 Full log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "════════════════════════════════════════════════════════════"

echo ""
echo "🎉 All phases completed successfully!"
echo "   Final dataset ready for LoRA training."
echo ""
