#!/usr/bin/env bash
#
# Optimized 3D Character LoRA Pipeline - Project-Agnostic
# Face Filter + Optimized SAM2 + AI Curation
#
# Pipeline Overview:
# 1. Face Recognition Filter (14,410 → ~2,500 frames) [2-3 hours]
# 2. Optimized SAM2 Segmentation (~2,500 → ~5,000 instances) [2-3 hours]
# 3. Intelligent Processing (~5,000 → ~8,000 candidates) [2-3 hours]
# 4. AI Quality Assessment + Curation (~8,000 → 400 best) [1-2 hours]
#
# Total estimated time: 7-11 hours
#
# Usage:
#   bash scripts/workflows/optimized_luca_pipeline.sh [project]
#
# Examples:
#   bash scripts/workflows/optimized_luca_pipeline.sh              # Use luca project
#   bash scripts/workflows/optimized_luca_pipeline.sh alberto      # Use alberto project
#

set -e  # Exit on error

# ============================================================
# Project Configuration
# ============================================================

PROJECT="${1:-luca}"  # Default to luca if no argument provided
PROJECT_CONFIG="configs/projects/${PROJECT}.yaml"

# Verify project config exists
if [ ! -f "$PROJECT_CONFIG" ]; then
    echo "❌ Error: Project config not found: $PROJECT_CONFIG"
    echo "Available projects in configs/projects/:"
    ls -1 configs/projects/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//' || echo "  (none found)"
    exit 1
fi

# Read project configuration from YAML
PROJECT_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$PROJECT_CONFIG'))['project']['name'])")
BASE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$PROJECT_CONFIG'))['paths']['base_dir'])")

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
DATA_ROOT="${BASE_DIR}"

echo "════════════════════════════════════════════════════════════"
echo "  Optimized ${PROJECT_NAME^^} LoRA Pipeline"
echo "  Face Filter + Optimized SAM2 + AI Curation"
echo "════════════════════════════════════════════════════════════"
echo ""

# Display initial status
echo "📊 Initial Status:"
echo "────────────────────────────────────────"
TOTAL_FRAMES=$(find "$DATA_ROOT/frames" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
REFERENCE_COUNT=$(find "$DATA_ROOT/training_ready/1_${PROJECT_NAME}" -name "*.png" 2>/dev/null | wc -l)
echo "  Total frames:      $TOTAL_FRAMES"
echo "  Reference images:  $REFERENCE_COUNT (curated ${PROJECT_NAME})"
echo "  Target:            400 best training images"
echo ""

# ==================== Phase 1: Face Recognition Filter ====================
echo "════════════════════════════════════════════════════════════"
echo "🔍 Phase 1: Face Recognition Filter"
echo "════════════════════════════════════════════════════════════"
echo "  Input:    $DATA_ROOT/frames ($TOTAL_FRAMES frames)"
echo "  Method:   ArcFace face recognition with reference matching"
echo "  Output:   $DATA_ROOT/${PROJECT_NAME}_frames_filtered"
echo "  Expected: ~2,500 ${PROJECT_NAME} frames (~17% match rate)"
echo "  Time:     ~2-3 hours"
echo ""

read -p "Start Phase 1 (Face Recognition Filter)? [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "🚀 Starting face recognition filter..."
    echo ""

    START_TIME=$(date +%s)

    conda run -n ai_env python "$PROJECT_ROOT/scripts/generic/filtering/reference_based_frame_filter.py" \
        --input-dir "$DATA_ROOT/frames" \
        --reference-dir "$DATA_ROOT/training_ready/1_${PROJECT_NAME}" \
        --output-dir "$DATA_ROOT/${PROJECT_NAME}_frames_filtered" \
        --similarity-threshold 0.6 \
        --device cuda \
        --save-metadata

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    FILTERED_COUNT=$(find "$DATA_ROOT/${PROJECT_NAME}_frames_filtered" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)

    echo ""
    echo "✅ Phase 1 Complete!"
    echo "   Filtered frames: $FILTERED_COUNT"
    echo "   Match rate: $(echo "scale=1; $FILTERED_COUNT * 100 / $TOTAL_FRAMES" | bc)%"
    echo "   Time elapsed: $((ELAPSED / 60)) minutes"
    echo ""
else
    echo "⏭️  Skipping Phase 1"
    echo ""
fi

# Verify Phase 1 output
if [ ! -d "$DATA_ROOT/${PROJECT_NAME}_frames_filtered" ] || [ -z "$(ls -A $DATA_ROOT/${PROJECT_NAME}_frames_filtered)" ]; then
    echo "❌ Phase 1 output not found. Please run Phase 1 first."
    exit 1
fi

FILTERED_COUNT=$(find "$DATA_ROOT/${PROJECT_NAME}_frames_filtered" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)

# ==================== Phase 2: Optimized SAM2 Segmentation ====================
echo "════════════════════════════════════════════════════════════"
echo "🎯 Phase 2: Optimized SAM2 Segmentation"
echo "════════════════════════════════════════════════════════════"
echo "  Input:    $DATA_ROOT/${PROJECT_NAME}_frames_filtered ($FILTERED_COUNT frames)"
echo "  Method:   SAM2 with optimized batching"
echo "            - Image Encoder batch: 8 frames"
echo "            - Mask Decoder batch: 32 prompts"
echo "  Output:   $DATA_ROOT/${PROJECT_NAME}_instances_sam2"
echo "  Expected: ~2-3 instances per frame = ~5,000 total instances"
echo "  Time:     ~2-3 hours (vs 10-20 hours without optimization!)"
echo ""

read -p "Start Phase 2 (Optimized SAM2 Segmentation)? [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "🚀 Starting optimized SAM2 segmentation..."
    echo ""

    START_TIME=$(date +%s)

    conda run -n ai_env python "$PROJECT_ROOT/scripts/generic/segmentation/optimized_sam2_batch.py" \
        --input-dir "$DATA_ROOT/${PROJECT_NAME}_frames_filtered" \
        --output-dir "$DATA_ROOT/${PROJECT_NAME}_instances_sam2" \
        --model sam2_hiera_large \
        --device cuda \
        --encoder-batch-size 8 \
        --decoder-batch-size 32 \
        --points-per-side 32 \
        --min-area 100 \
        --save-masks

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    INSTANCE_COUNT=$(find "$DATA_ROOT/${PROJECT_NAME}_instances_sam2" -name "*_inst*.png" 2>/dev/null | wc -l)

    echo ""
    echo "✅ Phase 2 Complete!"
    echo "   Instances extracted: $INSTANCE_COUNT"
    echo "   Avg per frame: $(echo "scale=1; $INSTANCE_COUNT / $FILTERED_COUNT" | bc)"
    echo "   Time elapsed: $((ELAPSED / 60)) minutes"
    echo "   Speed: $(echo "scale=1; $FILTERED_COUNT / ($ELAPSED / 60)" | bc) frames/min"
    echo ""
else
    echo "⏭️  Skipping Phase 2"
    echo ""
fi

# Verify Phase 2 output
if [ ! -d "$DATA_ROOT/${PROJECT_NAME}_instances_sam2" ] || [ -z "$(ls -A $DATA_ROOT/${PROJECT_NAME}_instances_sam2)" ]; then
    echo "❌ Phase 2 output not found. Please run Phase 2 first."
    exit 1
fi

INSTANCE_COUNT=$(find "$DATA_ROOT/${PROJECT_NAME}_instances_sam2" -name "*_inst*.png" 2>/dev/null | wc -l)

# ==================== Phase 3: Intelligent Processing ====================
echo "════════════════════════════════════════════════════════════"
echo "🧠 Phase 3: Intelligent Frame Processing"
echo "════════════════════════════════════════════════════════════"
echo "  Input:    $DATA_ROOT/${PROJECT_NAME}_instances_sam2 ($INSTANCE_COUNT instances)"
echo "  Method:   4 strategies (keep_full, segment, occlusion, enhance)"
echo "  Output:   $DATA_ROOT/${PROJECT_NAME}_intelligent_candidates"
echo "  Expected: ~8,000 diverse candidates"
echo "  Time:     ~2-3 hours"
echo ""

read -p "Start Phase 3 (Intelligent Processing)? [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "🚀 Starting intelligent processing..."
    echo ""

    START_TIME=$(date +%s)

    conda run -n ai_env python "$PROJECT_ROOT/scripts/data_curation/intelligent_frame_processor.py" \
        "$DATA_ROOT/${PROJECT_NAME}_instances_sam2" \
        --output-dir "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates" \
        --device cuda

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    CANDIDATE_COUNT=0
    for strategy in keep_full segment create_occlusion enhance_segment; do
        if [ -d "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates/$strategy/images" ]; then
            count=$(find "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates/$strategy/images" -name "*.png" 2>/dev/null | wc -l)
            CANDIDATE_COUNT=$((CANDIDATE_COUNT + count))
            echo "   $strategy: $count images"
        fi
    done

    echo ""
    echo "✅ Phase 3 Complete!"
    echo "   Total candidates: $CANDIDATE_COUNT"
    echo "   Time elapsed: $((ELAPSED / 60)) minutes"
    echo ""
else
    echo "⏭️  Skipping Phase 3"
    echo ""
fi

# Verify Phase 3 output
if [ ! -d "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates" ]; then
    echo "❌ Phase 3 output not found. Please run Phase 3 first."
    exit 1
fi

# ==================== Phase 4: AI Quality Assessment + Curation ====================
echo "════════════════════════════════════════════════════════════"
echo "⭐ Phase 4: AI Quality Assessment + Smart Curation"
echo "════════════════════════════════════════════════════════════"
echo "  Input:    $DATA_ROOT/${PROJECT_NAME}_intelligent_candidates"
echo "  Method:   7 AI models + diversity balancing + deduplication"
echo "            - LAION Aesthetics (trained weights ✓)"
echo "            - Face Quality (InsightFace)"
echo "            - BRISQUE (technical quality)"
echo "            - + Traditional CV metrics"
echo "  Output:   $DATA_ROOT/${PROJECT_NAME}_training_final (400 best images)"
echo "  Time:     ~1-2 hours"
echo ""

TARGET_SIZE=400
read -p "Target dataset size (default: 400): " USER_SIZE
if [ ! -z "$USER_SIZE" ]; then
    TARGET_SIZE=$USER_SIZE
fi

read -p "Start Phase 4 (AI Quality + Curation)? [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "🚀 Starting AI quality assessment and curation..."
    echo ""

    START_TIME=$(date +%s)

    # Step 4a: AI Quality Assessment
    echo "📊 Step 4a: AI Quality Assessment..."
    conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/ai_quality_assessor.py" \
        "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates" \
        --output "$DATA_ROOT/${PROJECT_NAME}_quality_scores.json" \
        --batch \
        --device cuda

    # Step 4b: Smart Curation
    echo ""
    echo "🎯 Step 4b: Smart Curation..."
    conda run -n ai_env python "$PROJECT_ROOT/scripts/data_curation/intelligent_dataset_curator.py" \
        "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates" \
        --output-dir "$DATA_ROOT/${PROJECT_NAME}_training_final" \
        --target-size $TARGET_SIZE \
        --quality-scores "$DATA_ROOT/${PROJECT_NAME}_quality_scores.json"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    FINAL_COUNT=$(find "$DATA_ROOT/${PROJECT_NAME}_training_final" -name "*.png" 2>/dev/null | wc -l)

    echo ""
    echo "✅ Phase 4 Complete!"
    echo "   Final training set: $FINAL_COUNT images"
    echo "   Time elapsed: $((ELAPSED / 60)) minutes"
    echo ""
else
    echo "⏭️  Skipping Phase 4"
    echo ""
fi

# ==================== Summary ====================
echo "════════════════════════════════════════════════════════════"
echo "  ✅ OPTIMIZED PIPELINE COMPLETE FOR ${PROJECT_NAME^^}!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 Pipeline Summary:"
echo "────────────────────────────────────────"
echo "  Phase 1: Face Recognition Filter"
FILTERED_COUNT=$(find "$DATA_ROOT/${PROJECT_NAME}_frames_filtered" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
echo "    $TOTAL_FRAMES frames → $FILTERED_COUNT ${PROJECT_NAME} frames"
echo ""
echo "  Phase 2: Optimized SAM2 Segmentation"
INSTANCE_COUNT=$(find "$DATA_ROOT/${PROJECT_NAME}_instances_sam2" -name "*_inst*.png" 2>/dev/null | wc -l)
echo "    $FILTERED_COUNT frames → $INSTANCE_COUNT instances"
echo ""
echo "  Phase 3: Intelligent Processing"
CANDIDATE_COUNT=0
for strategy in keep_full segment create_occlusion enhance_segment; do
    if [ -d "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates/$strategy/images" ]; then
        count=$(find "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates/$strategy/images" -name "*.png" 2>/dev/null | wc -l)
        CANDIDATE_COUNT=$((CANDIDATE_COUNT + count))
    fi
done
echo "    $INSTANCE_COUNT instances → $CANDIDATE_COUNT diverse candidates"
echo ""
echo "  Phase 4: AI Quality + Curation"
if [ -d "$DATA_ROOT/${PROJECT_NAME}_training_final" ]; then
    FINAL_COUNT=$(find "$DATA_ROOT/${PROJECT_NAME}_training_final" -name "*.png" 2>/dev/null | wc -l)
    echo "    $CANDIDATE_COUNT candidates → $FINAL_COUNT best images"
fi
echo ""
echo "📁 Final Output:"
echo "────────────────────────────────────────"
echo "  Location: $DATA_ROOT/${PROJECT_NAME}_training_final"
echo "  Images:   $(find "$DATA_ROOT/${PROJECT_NAME}_training_final" -name "*.png" 2>/dev/null | wc -l)"
if [ -f "$DATA_ROOT/${PROJECT_NAME}_quality_scores.json" ]; then
    echo "  Scores:   $DATA_ROOT/${PROJECT_NAME}_quality_scores.json"
fi
echo ""
echo "🎯 Next Steps:"
echo "────────────────────────────────────────"
echo "  1. Review quality scores and curation report"
echo "  2. Inspect sample images from final dataset"
echo "  3. Generate training config for ${PROJECT_NAME}"
echo "  4. Start LoRA training"
echo "  5. Evaluate checkpoints"
echo ""
echo "════════════════════════════════════════════════════════════"
