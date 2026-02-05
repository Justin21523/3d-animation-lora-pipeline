#!/bin/bash
# Resume Training Pipeline: Russell, Tyler, Barley, Giulia
# This script continues from where the previous training stopped

# Safer error handling - don't exit on first error
set +e  # DISABLED set -e for robustness
trap 'echo "[ERROR] Script interrupted at line $LINENO"' ERR

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
KOHYA_ROOT="/mnt/c/AI_LLM_projects/kohya_ss/sd-scripts"
LOG_DIR="$PROJECT_ROOT/logs/training"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

BASE_MODEL="/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors"

mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/resume_russell_to_giulia_${TIMESTAMP}.log"

# Logging functions
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "$MAIN_LOG"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $*" | tee -a "$MAIN_LOG"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$MAIN_LOG" >&2
}

log_section() {
    echo "" | tee -a "$MAIN_LOG"
    echo "==========================================" | tee -a "$MAIN_LOG"
    echo "$*" | tee -a "$MAIN_LOG"
    echo "==========================================" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
}

# Characters to train (starting from Russell)
declare -a CHARACTERS=(
    "russell:configs/training/character_loras/up_russell_identity.toml:/mnt/data/ai_data/models/lora/up/russell_identity:russell"
    "tyler:configs/training/character_loras/turning-red_tyler_identity.toml:/mnt/data/ai_data/models/lora/turning-red/tyler_identity:tyler"
    "barley:configs/training/character_loras/onward_barley_lightfoot_identity.toml:/mnt/data/ai_data/models/lora/onward/barley_lightfoot_identity:barley_lightfoot"
    "giulia:configs/training/character_loras/luca_giulia_identity.toml:/mnt/data/ai_data/models/lora/luca/giulia_identity:giulia"
)

log_section "Resumed Training Pipeline: Russell → Giulia"
log_info "Started: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "Total characters: ${#CHARACTERS[@]}"
log_info "Previous completed: Orion ✅, Ian ✅"
log_info ""

TOTAL_CHARS=${#CHARACTERS[@]}
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# Main training loop
for char_entry in "${CHARACTERS[@]}"; do
    IFS=':' read -r CHAR_NAME CONFIG_FILE OUTPUT_DIR PROMPT_NAME <<< "$char_entry"

    CHAR_NUM=$((COMPLETED + FAILED + 1))

    log_section "[$CHAR_NUM/$TOTAL_CHARS] Processing: $CHAR_NAME"

    CONFIG_PATH="$PROJECT_ROOT/$CONFIG_FILE"
    EVAL_DIR="$OUTPUT_DIR/evaluations_${TIMESTAMP}"

    # Verify config
    if [ ! -f "$CONFIG_PATH" ]; then
        log_error "Config not found: $CONFIG_PATH"
        ((FAILED++))
        continue
    fi

    log_info "Configuration: $CONFIG_FILE"
    log_info "Output directory: $OUTPUT_DIR"
    echo ""

    # ==========================================
    # Stage 1: LoRA Training
    # ==========================================

    log_section "Stage 1: LoRA Training - $CHAR_NAME"

    TRAIN_LOG="$LOG_DIR/train_${CHAR_NAME}_${TIMESTAMP}.log"

    log_info "Starting Kohya training..."
    log_info "  Config: $CONFIG_PATH"
    log_info "  Log: $TRAIN_LOG"
    echo ""

    # Run training (don't use && to avoid script exit on error)
    cd "$KOHYA_ROOT"
    conda run -n kohya_ss accelerate launch --num_cpu_threads_per_process=4 train_network.py \
        --config_file="$CONFIG_PATH" \
        2>&1 | tee "$TRAIN_LOG"

    TRAIN_EXIT_CODE=$?
    cd "$PROJECT_ROOT"

    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        log_success "Training completed: $CHAR_NAME"

        # Count checkpoints
        CHECKPOINT_COUNT=$(find "$OUTPUT_DIR" -name "*.safetensors" -type f | wc -l)
        log_info "Generated $CHECKPOINT_COUNT checkpoint(s)"
    else
        log_error "Training failed: $CHAR_NAME (exit code: $TRAIN_EXIT_CODE)"
        log_error "Check log: $TRAIN_LOG"
        ((FAILED++))
        log_info "Continuing to next character despite failure..."
        continue
    fi

    # ==========================================
    # Stage 2: Checkpoint Evaluation
    # ==========================================

    log_section "Stage 2: Checkpoint Evaluation - $CHAR_NAME"

    if [ "$CHECKPOINT_COUNT" -eq 0 ]; then
        log_error "No checkpoints found for $CHAR_NAME"
        ((FAILED++))
        continue
    fi

    log_info "Evaluating all $CHECKPOINT_COUNT checkpoint(s)..."
    echo ""

    EVAL_LOG="$LOG_DIR/eval_${CHAR_NAME}_${TIMESTAMP}.log"
    PROMPTS_FILE="$PROJECT_ROOT/prompts/lora_testing/${PROMPT_NAME}_identity_test.txt"

    if [ ! -f "$PROMPTS_FILE" ]; then
        log_info "Creating test prompts for $CHAR_NAME..."
        mkdir -p "$PROJECT_ROOT/prompts/lora_testing"

        cat > "$PROMPTS_FILE" <<EOF
$PROMPT_NAME, a 3d animated character, pixar style, portrait, neutral expression, studio lighting
$PROMPT_NAME, a 3d animated character, pixar style, full body, standing, natural lighting
$PROMPT_NAME, a 3d animated character, pixar style, close-up face, smiling, soft lighting
$PROMPT_NAME, a 3d animated character, pixar style, three-quarter view, dramatic lighting
$PROMPT_NAME, a 3d animated character, pixar style, side profile, cinematic lighting
$PROMPT_NAME, a 3d animated character, pixar style, outdoor scene, golden hour lighting
EOF
        log_success "Test prompts created: $PROMPTS_FILE"
    fi

    # Run evaluation
    log_info "Running checkpoint evaluation script..."

    conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/test_lora_checkpoints.py" \
        "$OUTPUT_DIR" \
        --base-model "$BASE_MODEL" \
        --output-dir "$EVAL_DIR" \
        --prompts-file "$PROMPTS_FILE" \
        --num-variations 4 \
        --steps 30 \
        --cfg-scale 7.5 \
        --seed 42 \
        --device cuda \
        2>&1 | tee "$EVAL_LOG"

    EVAL_EXIT_CODE=$?

    if [ $EVAL_EXIT_CODE -eq 0 ]; then
        log_success "Evaluation completed: $CHAR_NAME"
    else
        log_error "Evaluation failed: $CHAR_NAME (exit code: $EVAL_EXIT_CODE)"
        log_error "Check log: $EVAL_LOG"
        # Don't increment FAILED - training was successful
    fi

    # ==========================================
    # Stage 3: Best Checkpoint Analysis
    # ==========================================

    log_section "Stage 3: Best Checkpoint Analysis - $CHAR_NAME"

    EVAL_REPORT="$EVAL_DIR/evaluation_report.json"

    if [ -f "$EVAL_REPORT" ]; then
        log_info "Evaluation report found: $EVAL_REPORT"

        # Try to extract best checkpoint info
        BEST_INFO=$(conda run -n ai_env python -c "
import json
import sys

try:
    with open('$EVAL_REPORT', 'r') as f:
        report = json.load(f)

    if 'best_checkpoint' in report:
        best = report['best_checkpoint']
        print(f\"Checkpoint: {best['name']}\")
        print(f\"Epoch: {best.get('epoch', 'N/A')}\")
        print(f\"Score: {best.get('score', 'N/A')}\")
    elif 'checkpoints' in report and len(report['checkpoints']) > 0:
        last_ckpt = report['checkpoints'][-1]
        print(f\"Checkpoint: {last_ckpt['name']}\")
        print(f\"Epoch: {last_ckpt.get('epoch', 'Final')}\")
        print(f\"Score: N/A (using last checkpoint)\")
    else:
        print('No checkpoint information found')
        sys.exit(1)
except Exception as e:
    print(f'Error reading report: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1) || log_error "Could not parse evaluation report"

        if [ $? -eq 0 ]; then
            log_success "Best checkpoint identified:"
            echo "$BEST_INFO" | while read line; do
                log_info "  $line"
            done
        fi
    else
        log_info "Evaluation report not found (check manually)"
        log_info "Location: $EVAL_DIR"
    fi

    echo ""
    log_success "✅ $CHAR_NAME pipeline completed"
    ((COMPLETED++))

    # Always continue to next character
    if [ $CHAR_NUM -lt $TOTAL_CHARS ]; then
        echo "" | tee -a "$MAIN_LOG"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
        log_info "Proceeding to next character..."
        sleep 5  # Increased sleep for GPU cooldown
    fi
done

# Final Summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

log_section "Resumed Training Pipeline Complete"

log_info "Total characters: $TOTAL_CHARS"
log_success "Successfully completed: $COMPLETED"
log_error "Failed: $FAILED"
echo ""
log_info "Total elapsed time: ${HOURS}h ${MINS}m"
log_info "Main log: $MAIN_LOG"
echo ""

if [ $FAILED -eq 0 ]; then
    log_section "🎉 All 4 Characters Trained Successfully!"
    log_info ""
    log_info "Combined with previous: Orion ✅ Ian ✅ + Russell ✅ Tyler ✅ Barley ✅ Giulia ✅"
    log_info ""
    log_info "Total 6-character pipeline: COMPLETE"
    exit 0
else
    log_section "⚠️  Pipeline completed with $FAILED failure(s)"
    log_info "Check individual logs for details"
    exit 1
fi
