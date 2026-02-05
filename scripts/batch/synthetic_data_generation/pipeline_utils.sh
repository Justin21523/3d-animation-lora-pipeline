#!/usr/bin/bash
"""
Pipeline Utility Functions Library
===================================

Shared utility functions for batch synthetic data generation pipeline.

Provides:
- Logging functions
- Checkpoint management
- GPU monitoring and recovery
- Retry logic
- Pipeline phase implementations

Author: LLMProvider Tooling
Date: 2025-11-30
"""

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1" | tee -a "$STATUS_LOG"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$ERROR_LOG" "$STATUS_LOG"
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1" | tee -a "$STATUS_LOG"
}

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

is_task_completed() {
    local task_key="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "\"$task_key\": *\"completed\"" "$CHECKPOINT_FILE" 2>/dev/null
        return $?
    fi
    return 1
}

mark_task_completed() {
    local task_key="$1"
    mkdir -p "$(dirname "$CHECKPOINT_FILE")"

    if [ ! -f "$CHECKPOINT_FILE" ]; then
        echo "{}" > "$CHECKPOINT_FILE"
    fi

    python3 << EOF
import json
try:
    with open("$CHECKPOINT_FILE", "r") as f:
        data = json.load(f)
except:
    data = {}

data["$task_key"] = "completed"
data["${task_key}_timestamp"] = "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

with open("$CHECKPOINT_FILE", "w") as f:
    json.dump(data, f, indent=2)
EOF

    log_info "Task completed: $task_key"
}

# ============================================================================
# GPU MONITORING
# ============================================================================

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found - GPU unavailable"
        return 1
    fi

    if ! nvidia-smi &> /dev/null; then
        log_error "GPU not responding"
        return 1
    fi

    local free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free_mem" -lt 2048 ]; then
        log_warning "Low GPU memory: ${free_mem}MB free"
    fi

    return 0
}

recover_gpu() {
    log_warning "Attempting GPU recovery..."

    pkill -9 -f "python.*diffusers" 2>/dev/null || true
    pkill -9 -f "python.*torch" 2>/dev/null || true
    sleep 5

    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --gpu-reset 2>/dev/null || true
    fi

    sleep $GPU_RECOVERY_DELAY

    if check_gpu; then
        log_info "GPU recovery successful"
        return 0
    else
        log_error "GPU recovery failed"
        return 1
    fi
}

# ============================================================================
# RETRY LOGIC
# ============================================================================

execute_with_retry() {
    local max_attempts="$1"
    shift
    local description="$1"
    shift
    local command=("$@")

    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        log_info "Executing: $description (attempt $attempt/$max_attempts)"

        if ! check_gpu; then
            log_warning "GPU check failed, attempting recovery..."
            if ! recover_gpu; then
                log_error "Cannot recover GPU, skipping task: $description"
                return 1
            fi
        fi

        if "${command[@]}"; then
            log_info "Success: $description"
            return 0
        else
            local exit_code=$?
            log_error "Failed: $description (exit code: $exit_code)"

            if [ $attempt -lt $max_attempts ]; then
                log_info "Retrying in ${RETRY_DELAY}s... ($((max_attempts - attempt)) attempts left)"
                sleep $RETRY_DELAY

                if [ $((attempt % 2)) -eq 0 ]; then
                    recover_gpu
                fi
            fi
        fi

        attempt=$((attempt + 1))
    done

    log_error "All retry attempts exhausted for: $description"
    return 1
}

# ============================================================================
# PIPELINE PHASE 1: VOCABULARY GENERATION
# ============================================================================

run_phase_1_vocabulary_generation() {
    echo ""
    echo "========================================================================"
    echo "PHASE 1: GENERATING PROMPT VOCABULARIES"
    echo "========================================================================"
    log_info "Starting Phase 1: Vocabulary Generation"

    for i in "${!CHARACTERS[@]}"; do
        local char="${CHARACTERS[$i]}"
        local lora_file="${LORA_FILES[$i]}"

        local CHAR_WORKSPACE="$WORKSPACE_DIR/generated_data/$char"
        mkdir -p "$CHAR_WORKSPACE"

        for lora_type in "${LORA_TYPES[@]}"; do
            mkdir -p "$CHAR_WORKSPACE/$lora_type"

            local task_key="vocab_${char}_${lora_type}"

            if is_task_completed "$task_key"; then
                log_info "Skipping completed task: $task_key"
                continue
            fi

            log_info "Generating $lora_type vocabulary for $char..."

            execute_with_retry $MAX_RETRIES \
                "Vocabulary generation: $char $lora_type" \
                conda run -n ai_env python "$PROJECT_ROOT/scripts/generic/training/orchestration/vocabulary_generator.py" \
                    --character-name "$char" \
                    --character-description "A 3D animated character from Pixar-style animation" \
                    --lora-type "$lora_type" \
                    --num-prompts $NUM_PROMPTS_PER_TYPE \
                    --output-file "$CHAR_WORKSPACE/$lora_type/prompts.json" \
                    --use-templates \
                    --template-variations 5 \
                    2>&1 | tee "$WORKSPACE_DIR/logs/${char}_${lora_type}_vocab.log"

            if [ $? -eq 0 ]; then
                mark_task_completed "$task_key"
            else
                log_error "Failed to generate vocabulary for $char $lora_type after $MAX_RETRIES attempts"
            fi
        done
    done

    log_info "Phase 1 complete: Vocabulary Generation"
}

# ============================================================================
# PIPELINE PHASE 2: IMAGE GENERATION
# ============================================================================

run_phase_2_image_generation() {
    echo ""
    echo "========================================================================"
    echo "PHASE 2: LARGE-SCALE IMAGE GENERATION"
    echo "========================================================================"
    log_info "Starting Phase 2: Image Generation"

    TOTAL_GENERATED=0
    TOTAL_FAILED=0

    for i in "${!CHARACTERS[@]}"; do
        local char="${CHARACTERS[$i]}"
        local lora_file="${LORA_FILES[$i]}"

        log_info "Processing character: $char (LoRA: $(basename "$lora_file"))"

        local CHAR_WORKSPACE="$WORKSPACE_DIR/generated_data/$char"

        for lora_type in "${LORA_TYPES[@]}"; do
            local task_key="generation_${char}_${lora_type}"

            if is_task_completed "$task_key"; then
                log_info "Skipping completed generation: $task_key"
                local count=$(find "$CHAR_WORKSPACE/$lora_type/generated" -name "*.png" 2>/dev/null | wc -l)
                TOTAL_GENERATED=$((TOTAL_GENERATED + count))
                continue
            fi

            log_info "Generating $lora_type images for $char (target: $((NUM_PROMPTS_PER_TYPE * IMAGES_PER_PROMPT)) images)..."

            if [ ! -f "$CHAR_WORKSPACE/$lora_type/prompts.json" ]; then
                log_error "Prompts file not found: $CHAR_WORKSPACE/$lora_type/prompts.json"
                TOTAL_FAILED=$((TOTAL_FAILED + 1))
                continue
            fi

            execute_with_retry $MAX_RETRIES \
                "Image generation: $char $lora_type" \
                conda run -n ai_env python "$PROJECT_ROOT/scripts/generic/training/batch_image_generator.py" \
                    --prompts-file "$CHAR_WORKSPACE/$lora_type/prompts.json" \
                    --base-model "$BASE_MODEL_PATH" \
                    --lora-paths "$lora_file" \
                    --lora-scales 1.0 \
                    --output-dir "$CHAR_WORKSPACE/$lora_type/generated" \
                    --num-images-per-prompt $IMAGES_PER_PROMPT \
                    --num-inference-steps $NUM_INFERENCE_STEPS \
                    --guidance-scale $GUIDANCE_SCALE \
                    --device cuda \
                    --seed $((42 + i * 10 + $(echo "$lora_type" | md5sum | cut -c1-2 | tr -d '\n'))) \
                    --use-random-seeds \
                    --save-prompts \
                    2>&1 | tee "$WORKSPACE_DIR/logs/${char}_${lora_type}_generation.log"

            if [ $? -eq 0 ]; then
                mark_task_completed "$task_key"
                local count=$(find "$CHAR_WORKSPACE/$lora_type/generated" -name "*.png" 2>/dev/null | wc -l)
                TOTAL_GENERATED=$((TOTAL_GENERATED + count))
                log_info "Generated $count images for $char $lora_type (total so far: $TOTAL_GENERATED)"
            else
                log_error "Failed to generate images for $char $lora_type after $MAX_RETRIES attempts"
                TOTAL_FAILED=$((TOTAL_FAILED + 1))
            fi

            sleep 5
        done
    done

    log_info "Phase 2 complete: Generated $TOTAL_GENERATED images ($TOTAL_FAILED tasks failed)"
}

# ============================================================================
# PIPELINE PHASE 3: QUALITY FILTERING
# ============================================================================

run_phase_3_quality_filtering() {
    echo ""
    echo "========================================================================"
    echo "PHASE 3: QUALITY FILTERING"
    echo "========================================================================"
    log_info "Starting Phase 3: Quality Filtering"

    TOTAL_FILTERED=0

    for char in "${CHARACTERS[@]}"; do
        local CHAR_WORKSPACE="$WORKSPACE_DIR/generated_data/$char"

        for lora_type in "${LORA_TYPES[@]}"; do
            local task_key="filtering_${char}_${lora_type}"

            if is_task_completed "$task_key"; then
                log_info "Skipping completed filtering: $task_key"
                local count=$(find "$CHAR_WORKSPACE/$lora_type/filtered" -name "*.png" 2>/dev/null | wc -l)
                TOTAL_FILTERED=$((TOTAL_FILTERED + count))
                continue
            fi

            if [ ! -d "$CHAR_WORKSPACE/$lora_type/generated" ]; then
                log_warning "No generated images found for $char $lora_type, skipping filtering"
                continue
            fi

            local gen_count=$(find "$CHAR_WORKSPACE/$lora_type/generated" -name "*.png" 2>/dev/null | wc -l)

            if [ $gen_count -eq 0 ]; then
                log_warning "No images to filter for $char $lora_type"
                continue
            fi

            log_info "Filtering $gen_count images for $char $lora_type..."

            execute_with_retry $MAX_RETRIES \
                "Quality filtering: $char $lora_type" \
                conda run -n ai_env python "$PROJECT_ROOT/scripts/generic/quality/image_quality_filter.py" \
                    --input-dir "$CHAR_WORKSPACE/$lora_type/generated" \
                    --output-dir "$CHAR_WORKSPACE/$lora_type/filtered" \
                    --blur-threshold 80 \
                    --min-resolution 512 \
                    --brightness-range 0.2 0.9 \
                    --contrast-threshold 0.3 \
                    --enable-nsfw-filter \
                    --enable-face-detection \
                    --min-face-size 64 \
                    --enable-deduplication \
                    --similarity-threshold 0.95 \
                    --device cuda \
                    --batch-size 16 \
                    --save-report \
                    2>&1 | tee "$WORKSPACE_DIR/logs/${char}_${lora_type}_filtering.log"

            if [ $? -eq 0 ]; then
                mark_task_completed "$task_key"
                local count=$(find "$CHAR_WORKSPACE/$lora_type/filtered" -name "*.png" 2>/dev/null | wc -l)
                TOTAL_FILTERED=$((TOTAL_FILTERED + count))
                local retention=$((count * 100 / gen_count))
                log_info "Filtered: $count/$gen_count images retained (${retention}%)"
            else
                log_error "Failed to filter images for $char $lora_type after $MAX_RETRIES attempts"
            fi
        done
    done

    log_info "Phase 3 complete: $TOTAL_FILTERED images after filtering"
}

# ============================================================================
# PIPELINE PHASE 4: DATASET ORGANIZATION
# ============================================================================

run_phase_4_dataset_organization() {
    echo ""
    echo "========================================================================"
    echo "PHASE 4: DATASET ORGANIZATION"
    echo "========================================================================"
    log_info "Starting Phase 4: Dataset Organization"

    for char in "${CHARACTERS[@]}"; do
        local CHAR_WORKSPACE="$WORKSPACE_DIR/generated_data/$char"

        for lora_type in "${LORA_TYPES[@]}"; do
            local task_key="organization_${char}_${lora_type}"

            if is_task_completed "$task_key"; then
                log_info "Skipping completed organization: $task_key"
                continue
            fi

            if [ ! -d "$CHAR_WORKSPACE/$lora_type/filtered" ]; then
                log_warning "No filtered images for $char $lora_type, skipping organization"
                continue
            fi

            local filtered_count=$(find "$CHAR_WORKSPACE/$lora_type/filtered" -name "*.png" 2>/dev/null | wc -l)

            if [ $filtered_count -eq 0 ]; then
                log_warning "No filtered images to organize for $char $lora_type"
                continue
            fi

            log_info "Organizing $filtered_count images for $char $lora_type..."

            execute_with_retry $MAX_RETRIES \
                "Dataset organization: $char $lora_type" \
                conda run -n ai_env python "$PROJECT_ROOT/scripts/generic/training/preparers/dataset_preparer.py" \
                    --input-dir "$CHAR_WORKSPACE/$lora_type/filtered" \
                    --output-dir "$WORKSPACE_DIR/datasets/${char}_${lora_type}" \
                    --character-name "$char" \
                    --lora-type "$lora_type" \
                    --repeat-count 10 \
                    --min-resolution 512 \
                    --max-resolution 2048 \
                    --target-resolution 1024 \
                    --resize-if-needed \
                    --copy-images \
                    --generate-captions \
                    --caption-style "3d_animation" \
                    --save-metadata \
                    2>&1 | tee "$WORKSPACE_DIR/logs/${char}_${lora_type}_organization.log"

            if [ $? -eq 0 ]; then
                mark_task_completed "$task_key"
                log_info "Organized dataset for $char $lora_type"
            else
                log_error "Failed to organize dataset for $char $lora_type after $MAX_RETRIES attempts"
            fi
        done
    done

    log_info "Phase 4 complete: Dataset Organization"
}

# ============================================================================
# FINAL SUMMARY
# ============================================================================

generate_final_summary() {
    echo ""
    echo "========================================================================"
    echo "🎉 BATCH SYNTHETIC DATA GENERATION COMPLETE"
    echo "========================================================================"
    echo "End time: $(date)"
    echo ""

    local total_tasks=0
    local completed_tasks=0

    if [ -f "$CHECKPOINT_FILE" ]; then
        total_tasks=$(grep -c "\"vocab_\|generation_\|filtering_\|organization_" "$CHECKPOINT_FILE" 2>/dev/null || echo 0)
        completed_tasks=$(grep -c "\"completed\"" "$CHECKPOINT_FILE" 2>/dev/null || echo 0)
    fi

    local total_generated=$(find "$WORKSPACE_DIR/generated_data" -name "*.png" -path "*/generated/*" 2>/dev/null | wc -l)
    local total_filtered=$(find "$WORKSPACE_DIR/generated_data" -name "*.png" -path "*/filtered/*" 2>/dev/null | wc -l)

    echo "📊 Pipeline Statistics:"
    echo "   ├─ Characters processed: ${#CHARACTERS[@]}"
    echo "   ├─ LoRA types: ${#LORA_TYPES[@]}"
    echo "   ├─ Total images generated: $total_generated"
    echo "   ├─ Total images filtered: $total_filtered"
    if [ $total_generated -gt 0 ]; then
        local retention=$((total_filtered * 100 / total_generated))
        echo "   ├─ Retention rate: ${retention}%"
    fi
    echo "   ├─ Tasks completed: $completed_tasks"
    echo "   └─ Completion rate: $((completed_tasks * 100 / (${#CHARACTERS[@]} * ${#LORA_TYPES[@]} * 4)))%"
    echo ""
    echo "📁 Output Locations:"
    echo "   ├─ Generated data: $WORKSPACE_DIR/generated_data/"
    echo "   ├─ Filtered data: $WORKSPACE_DIR/generated_data/*/*/filtered/"
    echo "   ├─ Datasets: $WORKSPACE_DIR/datasets/"
    echo "   ├─ Logs: $WORKSPACE_DIR/logs/"
    echo "   └─ Checkpoint: $CHECKPOINT_FILE"
    echo ""
    echo "📋 Logs:"
    echo "   ├─ Status log: $STATUS_LOG"
    echo "   └─ Error log: $ERROR_LOG"
    echo ""

    if [ -f "$ERROR_LOG" ]; then
        local error_count=$(wc -l < "$ERROR_LOG")
        if [ $error_count -gt 0 ]; then
            echo "⚠️  $error_count errors logged - review $ERROR_LOG"
        fi
    fi

    echo "========================================================================"
}
