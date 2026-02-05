#!/bin/bash
# Character LoRA Data Processing Pipeline (Universal)
# Pipeline: Characters Inpainting → Manual Grouping → Augmentation → Expression Detection → Update Expression LoRA
#
# Usage:
#   bash scripts/batch/process_new_movies_pipeline.sh [movie1] [movie2] ... [--stage N]
#   bash scripts/batch/process_new_movies_pipeline.sh onward turning-red up --stage 1
#   bash scripts/batch/process_new_movies_pipeline.sh all --stage 3

set -e

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
DATA_ROOT="/mnt/data/ai_data/datasets/3d-anime"
LOG_DIR="$PROJECT_ROOT/logs/character_pipeline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"; }
info() { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"; }

# Available films
AVAILABLE_FILMS=($(ls -d "$DATA_ROOT"/*/ 2>/dev/null | xargs -n 1 basename | grep -v "cross_character"))

# Parse arguments
MOVIES=()
STAGE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        all)
            MOVIES=("${AVAILABLE_FILMS[@]}")
            shift
            ;;
        *)
            MOVIES+=("$1")
            shift
            ;;
    esac
done

# If no movies specified, show usage
if [ ${#MOVIES[@]} -eq 0 ]; then
    echo "Usage: $0 [movie1] [movie2] ... [--stage N]"
    echo ""
    echo "Available films: ${AVAILABLE_FILMS[@]}"
    echo ""
    echo "Stages:"
    echo "  1  - Characters Inpainting + Manual Grouping prompt"
    echo "  2  - Manual Grouping prompt only"
    echo "  3  - Augmentation + Expression Detection + Captions + Update LoRA"
    echo "  4  - Expression Detection + Captions + Update LoRA"
    echo "  5  - Expression Captions + Update LoRA"
    echo "  6  - Update Expression LoRA only"
    echo ""
    echo "Examples:"
    echo "  $0 onward turning-red up --stage 1"
    echo "  $0 all --stage 3"
    exit 1
fi

log "=========================================="
log "Character LoRA Data Processing Pipeline"
log "Movies: ${MOVIES[@]}"
log "Stage: $STAGE"
log "Started: $(date)"
log "=========================================="

# ============================================================================
# STAGE 1: Characters Inpainting (PowerPaint)
# ============================================================================
stage_1_characters_inpainting() {
    log ""
    log "=== STAGE 1: Characters Inpainting ==="
    log ""

    for movie in "${MOVIES[@]}"; do
        info "--- Processing $movie ---"

        INPUT_DIR="$DATA_ROOT/$movie/lora_data/characters"
        OUTPUT_DIR="$DATA_ROOT/$movie/lora_data/characters_inpainted"
        BACKGROUND_DIR="$DATA_ROOT/$movie/backgrounds_lama_v2"

        # Count characters
        char_count=$(find "$INPUT_DIR" -name "*.png" 2>/dev/null | wc -l)
        info "Characters to inpaint: $char_count"

        if [ $char_count -eq 0 ]; then
            warn "No characters found in $INPUT_DIR, skipping..."
            continue
        fi

        # Background directory check removed - not needed for character inpainting
        # Character inpainting uses the alpha channel from character PNGs

        log "Running LaMa batch optimized inpainting for $movie..."
        conda run -n ai_env python scripts/generic/inpainting/lama_batch_optimized.py \
            --input-dir "$INPUT_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --flat-input \
            --batch-size 8 \
            --device cuda \
            > "$LOG_DIR/inpaint_${movie}_${TIMESTAMP}.log" 2>&1 &

        log "✅ Started inpainting for $movie (PID: $!)"
        sleep 2
    done

    log ""
    info "⏳ Waiting for all inpainting tasks to complete..."
    info "Monitor with: tail -f $LOG_DIR/inpaint_*_${TIMESTAMP}.log"

    # Wait for completion
    while pgrep -f "lama_batch_optimized.py" > /dev/null; do
        sleep 30
    done

    log "✅ STAGE 1 Complete: Characters Inpainting"
}

# ============================================================================
# STAGE 2: Manual Character Grouping Reminder
# ============================================================================
stage_2_manual_grouping() {
    echo ""
    echo "=== STAGE 2: Manual Character Grouping ==="
    echo ""
    echo "📝 MANUAL STEP REQUIRED:"
    echo ""
    echo "Please review and group the inpainted characters:"
    echo ""

    for movie in "${MOVIES[@]}"; do
        inpainted_dir="$BASE_DIR/$movie/lora_data/characters_inpainted"
        clusters_dir="$BASE_DIR/$movie/lora_data/character_clusters"

        if [ -d "$inpainted_dir" ]; then
            count=$(find "$inpainted_dir" -name "*.png" 2>/dev/null | wc -l)
            echo "  $movie: $count characters in $inpainted_dir"
            echo "    → Group into: $clusters_dir/[character_name]/"
        fi
    done

    echo ""
    echo "After grouping, run: bash scripts/batch/process_new_movies_pipeline.sh --stage 3"
    echo ""
    read -p "Press Enter after you have completed manual grouping..."
}

# ============================================================================
# STAGE 3: Augmentation (Ensure 200+ images per character)
# ============================================================================
stage_3_augmentation() {
    echo ""
    echo "=== STAGE 3: Character Dataset Augmentation ==="
    echo ""

    for movie in "${MOVIES[@]}"; do
        echo "--- Processing $movie ---"

        clusters_dir="$BASE_DIR/$movie/lora_data/character_clusters"
        augmented_dir="$BASE_DIR/$movie/lora_data/characters_augmented"

        if [ ! -d "$clusters_dir" ]; then
            echo "⚠️  Character clusters not found, skipping..."
            continue
        fi

        # Find all character directories
        for char_dir in "$clusters_dir"/*; do
            if [ -d "$char_dir" ]; then
                char_name=$(basename "$char_dir")
                char_count=$(find "$char_dir" -name "*.png" 2>/dev/null | wc -l)

                echo "  $char_name: $char_count images"

                if [ $char_count -lt 200 ]; then
                    target_count=200
                    echo "    → Augmenting to $target_count images..."

                    conda run -n ai_env python scripts/generic/training/augment_small_clusters.py \
                        --input-dir "$char_dir" \
                        --output-dir "$augmented_dir/$char_name" \
                        --target-count $target_count \
                        --augment-3d \
                        > "$LOG_DIR/augment_${movie}_${char_name}_${TIMESTAMP}.log" 2>&1 &
                else
                    echo "    ✅ Already has enough images"
                    # Just copy to augmented dir
                    mkdir -p "$augmented_dir/$char_name"
                    cp -r "$char_dir"/* "$augmented_dir/$char_name/"
                fi
            fi
        done
    done

    echo ""
    echo "⏳ Waiting for augmentation tasks..."
    wait

    echo "✅ STAGE 3 Complete: Augmentation"
}

# ============================================================================
# STAGE 4: Expression Detection
# ============================================================================
stage_4_expression_detection() {
    echo ""
    echo "=== STAGE 4: Expression Detection ==="
    echo ""

    for movie in "${MOVIES[@]}"; do
        echo "--- Processing $movie ---"

        augmented_dir="$BASE_DIR/$movie/lora_data/characters_augmented"
        expressions_dir="$BASE_DIR/$movie/lora_data/expressions_classified"

        if [ ! -d "$augmented_dir" ]; then
            echo "⚠️  Augmented characters not found, skipping..."
            continue
        fi

        # Run CLIP expression classification for each character
        for char_dir in "$augmented_dir"/*; do
            if [ -d "$char_dir" ]; then
                char_name=$(basename "$char_dir")

                echo "  Detecting expressions for $char_name..."

                conda run -n ai_env python scripts/generic/training/clip_expression_classifier.py \
                    --input-dir "$char_dir" \
                    --output-dir "$expressions_dir/$char_name" \
                    --threshold 0.25 \
                    --batch-size 32 \
                    > "$LOG_DIR/expr_${movie}_${char_name}_${TIMESTAMP}.log" 2>&1 &
            fi
        done
    done

    echo ""
    echo "⏳ Waiting for expression detection..."
    wait

    echo "✅ STAGE 4 Complete: Expression Detection"
}

# ============================================================================
# STAGE 5: Generate Expression Captions
# ============================================================================
stage_5_expression_captions() {
    echo ""
    echo "=== STAGE 5: Expression Caption Generation ==="
    echo ""

    for movie in "${MOVIES[@]}"; do
        echo "--- Processing $movie ---"

        expressions_dir="$BASE_DIR/$movie/lora_data/expressions_classified"
        captions_dir="$BASE_DIR/$movie/lora_data/expressions_captioned"

        if [ ! -d "$expressions_dir" ]; then
            echo "⚠️  Expression classifications not found, skipping..."
            continue
        fi

        # Generate captions for each character's expressions
        for char_dir in "$expressions_dir"/*; do
            if [ -d "$char_dir" ]; then
                char_name=$(basename "$char_dir")

                echo "  Generating captions for $char_name..."

                conda run -n ai_env python scripts/generic/training/generate_expression_captions.py \
                    --input-dir "$char_dir" \
                    --output-dir "$captions_dir/$char_name" \
                    --character-name "$char_name" \
                    --movie "$movie" \
                    > "$LOG_DIR/caption_${movie}_${char_name}_${TIMESTAMP}.log" 2>&1 &
            fi
        done
    done

    echo ""
    echo "⏳ Waiting for caption generation..."
    wait

    echo "✅ STAGE 5 Complete: Expression Captions"
}

# ============================================================================
# STAGE 6: Update Cross-Character Expression LoRA Dataset
# ============================================================================
stage_6_update_expression_lora() {
    echo ""
    echo "=== STAGE 6: Update Cross-Character Expression LoRA ==="
    echo ""

    EXPR_LORA_DIR="$BASE_DIR/cross_character_expression_lora"

    for movie in "${MOVIES[@]}"; do
        echo "--- Adding $movie expressions ---"

        captions_dir="$BASE_DIR/$movie/lora_data/expressions_captioned"

        if [ ! -d "$captions_dir" ]; then
            echo "⚠️  Expression captions not found, skipping..."
            continue
        fi

        # Copy expression data to cross-character dataset
        for char_dir in "$captions_dir"/*; do
            if [ -d "$char_dir" ]; then
                char_name=$(basename "$char_dir")

                echo "  Adding $char_name expressions..."

                # Copy to expression LoRA training dataset
                for expr_type in happy sad angry surprised neutral fear disgust; do
                    src_dir="$char_dir/$expr_type"
                    dst_dir="$EXPR_LORA_DIR/training_data/$expr_type"

                    if [ -d "$src_dir" ]; then
                        count=$(find "$src_dir" -name "*.png" 2>/dev/null | wc -l)
                        if [ $count -gt 0 ]; then
                            mkdir -p "$dst_dir"
                            cp "$src_dir"/* "$dst_dir/" 2>/dev/null || true
                            echo "    Added $count $expr_type expressions"
                        fi
                    fi
                done
            fi
        done
    done

    echo ""
    echo "✅ STAGE 6 Complete: Expression LoRA Dataset Updated"
    echo ""
    echo "Expression LoRA training data location:"
    echo "  $EXPR_LORA_DIR/training_data/"
}

# ============================================================================
# Main Execution
# ============================================================================
main() {
    STAGE=${1:-1}

    case $STAGE in
        1)
            stage_1_characters_inpainting
            stage_2_manual_grouping
            ;;
        2)
            stage_2_manual_grouping
            ;;
        3)
            stage_3_augmentation
            stage_4_expression_detection
            stage_5_expression_captions
            stage_6_update_expression_lora
            ;;
        4)
            stage_4_expression_detection
            stage_5_expression_captions
            stage_6_update_expression_lora
            ;;
        5)
            stage_5_expression_captions
            stage_6_update_expression_lora
            ;;
        6)
            stage_6_update_expression_lora
            ;;
        all)
            stage_1_characters_inpainting
            echo ""
            echo "⚠️  PIPELINE PAUSED AT STAGE 2"
            echo "Please complete manual grouping, then run:"
            echo "  bash $0 3"
            ;;
        *)
            echo "Usage: $0 [stage]"
            echo "Stages:"
            echo "  1  - Characters Inpainting + Manual Grouping prompt"
            echo "  2  - Manual Grouping prompt only"
            echo "  3  - Augmentation + Expression Detection + Captions + Update LoRA"
            echo "  4  - Expression Detection + Captions + Update LoRA"
            echo "  5  - Expression Captions + Update LoRA"
            echo "  6  - Update Expression LoRA only"
            echo "  all - Run stages 1-2, then pause for manual grouping"
            exit 1
            ;;
    esac

    echo ""
    echo "=========================================="
    echo "Pipeline execution completed"
    echo "Finished: $(date)"
    echo "=========================================="
}

main "$@"
