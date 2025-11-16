#!/usr/bin/bash
#
# 3D Character Dataset Preparation & LoRA Training - Complete Workflow (Project-Agnostic)
# =======================================================================================
#
# This script runs the complete automated pipeline from SAM2 results to trained LoRA model.
#
# Usage:
#   ./run_luca_dataset_pipeline.sh [project]     # project defaults to "luca" if not specified
#
# Examples:
#   ./run_luca_dataset_pipeline.sh              # Use luca project
#   ./run_luca_dataset_pipeline.sh alberto      # Use alberto project
#
# Stages:
# 1. Face pre-filtering (ArcFace vs reference images)
# 2. Quality filtering (3D-specific metrics)
# 3. Comprehensive augmentation (10k-15k for manual review)
# 4. Diversity-based auto-selection (400 images)
# 5. Caption generation (Qwen2-VL with detailed prompts)
# 6. Training data preparation (Kohya_ss format)
# 7. LoRA training (optimized parameters)
# 8. (Optional) Parameter optimization
#
# Author: Claude Code
# Date: 2025-11-15
# Version: 2.0 (Project-agnostic)

set -e  # Exit on error
set -u  # Exit on undefined variable

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

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
CONDA_ENV="ai_env"
CONFIG_FILE="configs/projects/${PROJECT}_dataset_prep_v2.yaml"

# Paths (project-specific)
SAM2_RESULTS="${BASE_DIR}/${PROJECT_NAME}_instances_sam2"
TRAINING_OUTPUT="/mnt/data/ai_data/training_data/${PROJECT_NAME}_pixar_400"
LORA_OUTPUT="/mnt/data/ai_data/models/lora/${PROJECT_NAME}/trial_sam2_400"

# Logging
LOG_DIR="${PROJECT_ROOT}/logs/${PROJECT_NAME}_pipeline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

# ============================================================
# Helper Functions
# ============================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log_stage() {
    echo "" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "$1" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
}

check_conda_env() {
    if ! conda env list | grep -q "^${CONDA_ENV} "; then
        log "ERROR: Conda environment '${CONDA_ENV}' not found"
        exit 1
    fi
}

check_sam2_results() {
    if [ ! -d "${SAM2_RESULTS}" ]; then
        log "ERROR: SAM2 results directory not found: ${SAM2_RESULTS}"
        exit 1
    fi

    local instance_count=$(find "${SAM2_RESULTS}/instances" -name "*.png" 2>/dev/null | wc -l)
    log "✓ Found ${instance_count} SAM2 instances for ${PROJECT_NAME}"
}

# ============================================================
# Main Pipeline Stages
# ============================================================

run_dataset_preparation() {
    log_stage "DATASET PREPARATION PIPELINE - ${PROJECT_NAME^^}"

    log "Starting automated dataset preparation..."
    log "  Project: ${PROJECT_NAME}"
    log "  Config: ${CONFIG_FILE}"
    log "  SAM2 input: ${SAM2_RESULTS}"
    log "  Training output: ${TRAINING_OUTPUT}"

    cd "${PROJECT_ROOT}"

    # Use the project-agnostic pipeline with project config
    conda run -n "${CONDA_ENV}" python scripts/projects/luca/pipelines/luca_dataset_pipeline_simplified.py \
        --config "${CONFIG_FILE}" \
        --project-config "${PROJECT_CONFIG}" \
        2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log "✓ Dataset preparation completed successfully for ${PROJECT_NAME}"
    else
        log "✗ Dataset preparation failed for ${PROJECT_NAME}"
        exit 1
    fi
}

check_training_data() {
    log "Checking training data readiness for ${PROJECT_NAME}..."

    local image_count=$(find "${TRAINING_OUTPUT}/10_${PROJECT_NAME}_human" -name "*.png" 2>/dev/null | wc -l)
    local caption_count=$(find "${TRAINING_OUTPUT}/10_${PROJECT_NAME}_human" -name "*.txt" 2>/dev/null | wc -l)

    log "  Images: ${image_count}"
    log "  Captions: ${caption_count}"

    if [ ${image_count} -lt 100 ]; then
        log "WARNING: Only ${image_count} images found. Expected ~400"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    if [ ${caption_count} -lt ${image_count} ]; then
        log "WARNING: Caption count (${caption_count}) < image count (${image_count})"
    fi
}

run_lora_training() {
    log_stage "LORA TRAINING - ${PROJECT_NAME^^} (Optimized Parameters)"

    log "Starting LoRA training with optimized parameters for ${PROJECT_NAME}..."

    # Check if training config exists
    TRAINING_CONFIG="${PROJECT_ROOT}/configs/training/${PROJECT_NAME}_sam2_optimized.toml"

    if [ ! -f "${TRAINING_CONFIG}" ]; then
        log "Creating training configuration for ${PROJECT_NAME}..."
        create_training_config
    fi

    # Find Kohya_ss installation
    KOHYA_DIR=$(find /mnt -type d -name "kohya_ss" 2>/dev/null | head -1)

    if [ -z "${KOHYA_DIR}" ]; then
        log "ERROR: Kohya_ss directory not found"
        log "Please install Kohya_ss first"
        exit 1
    fi

    log "Using Kohya_ss: ${KOHYA_DIR}"

    cd "${KOHYA_DIR}"

    conda run -n kohya_ss python train_network.py \
        --config_file "${TRAINING_CONFIG}" \
        2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log "✓ LoRA training completed successfully for ${PROJECT_NAME}"
        log "  Model saved to: ${LORA_OUTPUT}"
    else
        log "✗ LoRA training failed for ${PROJECT_NAME}"
        exit 1
    fi
}

create_training_config() {
    log "Generating training configuration for ${PROJECT_NAME}..."

    mkdir -p "$(dirname "${TRAINING_CONFIG}")"

    cat > "${TRAINING_CONFIG}" <<EOF
# ${PROJECT_NAME} LoRA Training - SAM2 Optimized
# Project-agnostic configuration
# Using 400 auto-selected diverse images from SAM2 multi-instance pipeline

[model]
pretrained_model_name_or_path = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors"
vae = ""

[network]
network_module = "networks.lora"
network_dim = 64
network_alpha = 32
network_dropout = 0.1

[training]
# Data
train_data_dir = "${TRAINING_OUTPUT}"
output_dir = "${LORA_OUTPUT}"
output_name = "${PROJECT_NAME}_sam2_400"
resolution = "512,512"

# Optimization (optimized parameters)
learning_rate = 8e-5
text_encoder_lr = 5e-5
unet_lr = 8e-5
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100

optimizer_type = "AdamW"

# Training duration
max_train_epochs = 16
train_batch_size = 4
gradient_accumulation_steps = 2

# Regularization
min_snr_gamma = 5.0
noise_offset = 0.05

# Data augmentation (3D-safe, already applied in pipeline)
random_crop = false
color_aug = false
flip_aug = false

# Saving
save_every_n_epochs = 2
save_precision = "fp16"
mixed_precision = "fp16"

# Logging
logging_dir = "${LORA_OUTPUT}/logs"
log_with = "tensorboard"

# Performance
max_data_loader_n_workers = 8
persistent_data_loader_workers = true
seed = 42

# Miscellaneous
caption_extension = ".txt"
keep_tokens = 0
shuffle_caption = false
EOF

    log "✓ Training configuration created: ${TRAINING_CONFIG}"
}

run_parameter_optimization() {
    log_stage "PARAMETER OPTIMIZATION - ${PROJECT_NAME^^}"

    log "Starting parameter search for ${PROJECT_NAME}..."

    # This would call the parameter optimization script
    # For now, just log that it's available

    log "Parameter optimization script: scripts/training/hyperparameter_optimization.py"
    log "This will search for optimal training parameters"
    log "Run separately if needed: conda run -n ai_env python scripts/training/hyperparameter_optimization.py --project ${PROJECT_NAME}"
}

generate_summary_report() {
    log_stage "PIPELINE SUMMARY - ${PROJECT_NAME^^}"

    local duration=$SECONDS

    log "Pipeline completed in $(($duration / 3600))h $(($duration % 3600 / 60))m $(($duration % 60))s"
    log ""
    log "Output locations for ${PROJECT_NAME}:"
    log "  Face matched instances: ${BASE_DIR}/${PROJECT_NAME}_face_matched"
    log "  Quality filtered: ${BASE_DIR}/${PROJECT_NAME}_quality_filtered"
    log "  Augmented dataset (10k-15k): ${BASE_DIR}/${PROJECT_NAME}_augmented_comprehensive"
    log "  Auto-selected 400: ${BASE_DIR}/${PROJECT_NAME}_curated_400"
    log "  Training data: ${TRAINING_OUTPUT}"
    log "  LoRA model: ${LORA_OUTPUT}"
    log ""
    log "Next steps:"
    log "  1. Review augmented dataset: ${BASE_DIR}/${PROJECT_NAME}_augmented_comprehensive"
    log "  2. (Optional) Manual selection for alternative training set"
    log "  3. Test LoRA checkpoints: python scripts/evaluation/test_lora_checkpoints.py ${LORA_OUTPUT}"
    log "  4. (Optional) Run parameter optimization for tuning"
}

# ============================================================
# Main Execution
# ============================================================

main() {
    log_stage "${PROJECT_NAME^^} DATASET PREPARATION & LORA TRAINING PIPELINE"

    log "Starting automated pipeline for project: ${PROJECT_NAME}"
    log "Timestamp: ${TIMESTAMP}"

    # Pre-flight checks
    log "Running pre-flight checks..."
    check_conda_env
    check_sam2_results

    # Stage 1-6: Dataset Preparation
    run_dataset_preparation

    # Check training data
    check_training_data

    # Stage 7: LoRA Training
    read -p "Proceed with LoRA training for ${PROJECT_NAME}? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_lora_training
    else
        log "Skipping LoRA training for ${PROJECT_NAME}"
    fi

    # Stage 8: Parameter Optimization (optional)
    # run_parameter_optimization  # Commented out - run separately if needed

    # Generate summary
    generate_summary_report

    log "="*80
    log "✓ PIPELINE COMPLETE FOR ${PROJECT_NAME^^}"
    log "="*80
}

# Run main function
main "$@"
