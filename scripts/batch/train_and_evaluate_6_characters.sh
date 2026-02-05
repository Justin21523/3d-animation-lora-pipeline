#!/bin/bash
# Comprehensive Training and Evaluation Pipeline for 6 Character Identity LoRAs
# Pipeline: Train → Auto-Evaluate All Checkpoints → Select Best → Next Character
#
# Characters: Orion, Ian, Russell, Tyler, Barley, Giulia
# Each character is fully trained and evaluated before proceeding to the next
#
# Features:
# - Training with automatic sample generation every 2 epochs
# - Comprehensive checkpoint evaluation after training completes
# - Best checkpoint selection based on quality metrics
# - Detailed logs and reports for each character

set -e

# ==========================================
# Configuration
# ==========================================

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
KOHYA_ROOT="/mnt/c/AI_LLM_projects/kohya_ss/sd-scripts"
LOG_DIR="$PROJECT_ROOT/logs/training"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Base model for evaluation
BASE_MODEL="/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors"

# Create log directory
mkdir -p "$LOG_DIR"

# Main log file
MAIN_LOG="$LOG_DIR/train_evaluate_6chars_${TIMESTAMP}.log"

# ==========================================
# Logging Functions
# ==========================================

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

# ==========================================
# Character Configurations
# ==========================================

# Array format: char_name:config_file:output_dir:prompts_file
declare -a CHARACTERS=(
    # "orion:configs/training/character_loras/orion_orion_identity.toml:/mnt/data/ai_data/models/lora/orion/orion_identity:orion"  # ✅ COMPLETED
    "ian:configs/training/character_loras/onward_ian_lightfoot_identity.toml:/mnt/data/ai_data/models/lora/onward/ian_lightfoot_identity:ian_lightfoot"
    "russell:configs/training/character_loras/up_russell_identity.toml:/mnt/data/ai_data/models/lora/up/russell_identity:russell"
    "tyler:configs/training/character_loras/turning-red_tyler_identity.toml:/mnt/data/ai_data/models/lora/turning-red/tyler_identity:tyler"
    "barley:configs/training/character_loras/onward_barley_lightfoot_identity.toml:/mnt/data/ai_data/models/lora/onward/barley_lightfoot_identity:barley_lightfoot"
    "giulia:configs/training/character_loras/luca_giulia_identity.toml:/mnt/data/ai_data/models/lora/luca/giulia_identity:giulia"
)

# ==========================================
# Pipeline Start
# ==========================================

log_section "6-Character LoRA Training & Evaluation Pipeline"
log_info "Started: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "Total characters: ${#CHARACTERS[@]}"
log_info ""
log_info "Pipeline stages for each character:"
log_info "  1. LoRA Training (with sample generation every 2 epochs)"
log_info "  2. Comprehensive checkpoint evaluation"
log_info "  3. Best checkpoint selection"
log_info "  4. Proceed to next character"
echo ""

# Statistics
TOTAL_CHARS=${#CHARACTERS[@]}
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# ==========================================
# Main Training Loop
# ==========================================

for char_entry in "${CHARACTERS[@]}"; do
    IFS=':' read -r CHAR_NAME CONFIG_FILE OUTPUT_DIR PROMPT_NAME <<< "$char_entry"

    CHAR_NUM=$((COMPLETED + FAILED + 1))

    log_section "[$CHAR_NUM/$TOTAL_CHARS] Processing: $CHAR_NAME"

    # Full paths
    CONFIG_PATH="$PROJECT_ROOT/$CONFIG_FILE"
    EVAL_DIR="$OUTPUT_DIR/evaluations_${TIMESTAMP}"

    # Verify config exists
    if [ ! -f "$CONFIG_PATH" ]; then
        log_error "Config not found: $CONFIG_PATH"
        ((FAILED++))
        continue
    fi

    log_info "Configuration: $CONFIG_FILE"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Evaluation directory: $EVAL_DIR"
    echo ""

    # ==========================================
    # Stage 1: LoRA Training
    # ==========================================

    log_section "Stage 1: LoRA Training - $CHAR_NAME"

    TRAIN_LOG="$LOG_DIR/train_${CHAR_NAME}_${TIMESTAMP}.log"

    log_info "Starting Kohya training..."
    log_info "  Config: $CONFIG_PATH"
    log_info "  Log: $TRAIN_LOG"
    log_info "  Note: Sample images will be generated every 2 epochs"
    echo ""

    # Run training
    if cd "$KOHYA_ROOT" && conda run -n kohya_ss accelerate launch --num_cpu_threads_per_process=4 train_network.py \
        --config_file="$CONFIG_PATH" \
        2>&1 | tee "$TRAIN_LOG"; then

        log_success "Training completed: $CHAR_NAME"

        # Count checkpoints
        CHECKPOINT_COUNT=$(find "$OUTPUT_DIR" -name "*.safetensors" -type f | wc -l)
        log_info "Generated $CHECKPOINT_COUNT checkpoint(s)"
    else
        log_error "Training failed: $CHAR_NAME"
        log_error "Check log: $TRAIN_LOG"
        ((FAILED++))
        cd "$PROJECT_ROOT"
        continue
    fi

    # Return to project root
    cd "$PROJECT_ROOT"

    # ==========================================
    # Stage 2: Checkpoint Evaluation
    # ==========================================

    log_section "Stage 2: Comprehensive Checkpoint Evaluation - $CHAR_NAME"

    # Check if checkpoints exist
    if [ "$CHECKPOINT_COUNT" -eq 0 ]; then
        log_error "No checkpoints found for $CHAR_NAME"
        ((FAILED++))
        continue
    fi

    log_info "Evaluating all $CHECKPOINT_COUNT checkpoint(s)..."
    log_info "  Base model: $BASE_MODEL"
    log_info "  Output: $EVAL_DIR"
    echo ""

    # Evaluation parameters
    EVAL_LOG="$LOG_DIR/eval_${CHAR_NAME}_${TIMESTAMP}.log"

    # Create evaluation prompts on-the-fly if not exists
    PROMPTS_FILE="$PROJECT_ROOT/prompts/lora_testing/${PROMPT_NAME}_identity_test.txt"

    if [ ! -f "$PROMPTS_FILE" ]; then
        log_info "Creating test prompts for $CHAR_NAME..."
        mkdir -p "$PROJECT_ROOT/prompts/lora_testing"

        # Generate basic test prompts
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

    # Run comprehensive evaluation
    log_info "Running checkpoint evaluation script..."

    if conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/test_lora_checkpoints.py" \
        "$OUTPUT_DIR" \
        --base-model "$BASE_MODEL" \
        --output-dir "$EVAL_DIR" \
        --prompts-file "$PROMPTS_FILE" \
        --num-variations 4 \
        --steps 30 \
        --cfg-scale 7.5 \
        --seed 42 \
        --device cuda \
        2>&1 | tee "$EVAL_LOG"; then

        log_success "Evaluation completed: $CHAR_NAME"
    else
        log_error "Evaluation failed: $CHAR_NAME"
        log_error "Check log: $EVAL_LOG"
        ((FAILED++))
        continue
    fi

    # ==========================================
    # Stage 3: Best Checkpoint Selection
    # ==========================================

    log_section "Stage 3: Best Checkpoint Analysis - $CHAR_NAME"

    # Check if evaluation report exists
    EVAL_REPORT="$EVAL_DIR/evaluation_report.json"

    if [ -f "$EVAL_REPORT" ]; then
        log_info "Evaluation report found: $EVAL_REPORT"

        # Extract best checkpoint info (using Python)
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
        # Fallback: use last checkpoint if no explicit best
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
" 2>&1)

        if [ $? -eq 0 ]; then
            log_success "Best checkpoint identified:"
            echo "$BEST_INFO" | while read line; do
                log_info "  $line"
            done
        else
            log_error "Could not determine best checkpoint"
            log_info "Manual review recommended: $EVAL_DIR"
        fi
    else
        log_info "Evaluation report not found (check evaluation directory manually)"
        log_info "Location: $EVAL_DIR"
    fi

    echo ""
    log_success "✅ $CHAR_NAME pipeline completed successfully"
    ((COMPLETED++))

    # Separator before next character
    if [ $CHAR_NUM -lt $TOTAL_CHARS ]; then
        echo "" | tee -a "$MAIN_LOG"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
        log_info "Proceeding to next character..."
        sleep 2
    fi
done

# ==========================================
# Final Summary
# ==========================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

log_section "Pipeline Complete - Final Summary"

log_info "Total characters: $TOTAL_CHARS"
log_success "Successfully completed: $COMPLETED"
log_error "Failed: $FAILED"
echo ""
log_info "Total elapsed time: ${HOURS}h ${MINS}m"
log_info "Main log: $MAIN_LOG"
echo ""

if [ $FAILED -eq 0 ]; then
    log_section "🎉 All Characters Trained and Evaluated Successfully!"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Review evaluation results in each character's evaluation directory"
    log_info "  2. Use the best checkpoints for inference/testing"
    log_info "  3. Consider fine-tuning based on evaluation feedback"
    exit 0
else
    log_section "⚠️  Pipeline completed with $FAILED failure(s)"
    log_info ""
    log_info "Check individual training/evaluation logs for details"
    log_info "Failed characters require manual intervention"
    exit 1
fi
