#!/usr/bin/bash
"""
Batch Synthetic Data Generation Pipeline (Universal)
=====================================================

A fault-tolerant pipeline for generating large-scale synthetic training data
using existing identity LoRAs.

Features:
- Configurable via YAML or command-line arguments
- Checkpoint/Resume capability
- Automatic retry on failures (up to N attempts)
- GPU health monitoring and recovery
- Comprehensive error logging
- Works with any set of identity LoRAs

Usage:
  bash batch_synthetic_data_pipeline.sh --config <config.yaml>
  bash batch_synthetic_data_pipeline.sh --lora-dir <dir> --output-dir <dir> [options]

Author: LLMProvider Tooling
Date: 2025-11-30
Version: 1.0.0
"""

# ============================================================================
# DEFAULT CONFIGURATION (can be overridden by config file or CLI args)
# ============================================================================

# Paths
IDENTITY_LORAS_DIR=""
BASE_MODEL_PATH="/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
WORKSPACE_DIR=""
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Generation settings
NUM_PROMPTS_PER_TYPE=50
IMAGES_PER_PROMPT=10
NUM_INFERENCE_STEPS=30
GUIDANCE_SCALE=7.5

# Retry and recovery settings
MAX_RETRIES=3
RETRY_DELAY=60
GPU_RECOVERY_DELAY=120

# LoRA types to generate
LORA_TYPES=("pose" "expression" "action")

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

show_help() {
    cat << EOF
Batch Synthetic Data Generation Pipeline

Usage:
  $(basename "$0") --config <config.yaml>
  $(basename "$0") --lora-dir <dir> --output-dir <dir> [options]

Required (if not using --config):
  --lora-dir DIR          Directory containing identity LoRAs (*.safetensors)
  --output-dir DIR        Output directory for generated data

Optional:
  --config FILE           YAML config file (overrides all other options)
  --base-model FILE       Path to base SDXL model (default: sd_xl_base_1.0.safetensors)
  --num-prompts N         Number of prompts per type (default: 50)
  --images-per-prompt N   Images to generate per prompt (default: 10)
  --inference-steps N     Number of inference steps (default: 30)
  --guidance-scale N      Guidance scale (default: 7.5)
  --max-retries N         Maximum retry attempts (default: 3)
  --lora-types TYPE,...   Comma-separated LoRA types (default: pose,expression,action)
  --resume                Resume from checkpoint if available
  --help                  Show this help message

Examples:
  # Using config file
  $(basename "$0") --config configs/batch/synthetic_data_generation.yaml

  # Direct command line
  $(basename "$0") \\
    --lora-dir /mnt/c/ai_models/lora_sdxl/BEST_CHECKPOINTS_COLLECTION \\
    --output-dir /mnt/c/ai_projects/synthetic_data_output \\
    --num-prompts 100 \\
    --images-per-prompt 15

EOF
}

# Parse arguments
RESUME_MODE=false
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --lora-dir)
            IDENTITY_LORAS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            WORKSPACE_DIR="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS_PER_TYPE="$2"
            shift 2
            ;;
        --images-per-prompt)
            IMAGES_PER_PROMPT="$2"
            shift 2
            ;;
        --inference-steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --guidance-scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --lora-types)
            IFS=',' read -ra LORA_TYPES <<< "$2"
            shift 2
            ;;
        --resume)
            RESUME_MODE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Load config file if provided
if [ -n "$CONFIG_FILE" ]; then
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found: $CONFIG_FILE"
        exit 1
    fi

    # Use Python to parse YAML config
    eval "$(python3 << 'EOF'
import sys
import yaml

try:
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    # Print bash variable assignments
    if 'identity_loras_dir' in config:
        print(f"IDENTITY_LORAS_DIR='{config['identity_loras_dir']}'")
    if 'base_model_path' in config:
        print(f"BASE_MODEL_PATH='{config['base_model_path']}'")
    if 'workspace_dir' in config:
        print(f"WORKSPACE_DIR='{config['workspace_dir']}'")
    if 'num_prompts_per_type' in config:
        print(f"NUM_PROMPTS_PER_TYPE={config['num_prompts_per_type']}")
    if 'images_per_prompt' in config:
        print(f"IMAGES_PER_PROMPT={config['images_per_prompt']}")
    if 'num_inference_steps' in config:
        print(f"NUM_INFERENCE_STEPS={config['num_inference_steps']}")
    if 'guidance_scale' in config:
        print(f"GUIDANCE_SCALE={config['guidance_scale']}")
    if 'max_retries' in config:
        print(f"MAX_RETRIES={config['max_retries']}")
    if 'lora_types' in config:
        types = ' '.join(config['lora_types'])
        print(f"LORA_TYPES=({types})")

except Exception as e:
    print(f"echo 'Error parsing config: {e}' >&2", file=sys.stderr)
    sys.exit(1)
EOF
"$CONFIG_FILE")"
fi

# Validate required parameters
if [ -z "$IDENTITY_LORAS_DIR" ]; then
    echo "Error: --lora-dir is required (or specify in config file)"
    show_help
    exit 1
fi

if [ -z "$WORKSPACE_DIR" ]; then
    echo "Error: --output-dir is required (or specify in config file)"
    show_help
    exit 1
fi

if [ ! -d "$IDENTITY_LORAS_DIR" ]; then
    echo "Error: LoRA directory not found: $IDENTITY_LORAS_DIR"
    exit 1
fi

if [ ! -f "$BASE_MODEL_PATH" ]; then
    echo "Error: Base model not found: $BASE_MODEL_PATH"
    exit 1
fi

# ============================================================================
# SETUP
# ============================================================================

# Create workspace structure
mkdir -p "$WORKSPACE_DIR"/{logs,checkpoints,generated_data,filtered_data,datasets}

CHECKPOINT_FILE="$WORKSPACE_DIR/checkpoints/pipeline_progress.json"
ERROR_LOG="$WORKSPACE_DIR/logs/errors.log"
STATUS_LOG="$WORKSPACE_DIR/logs/status.log"

# Discover available identity LoRAs
mapfile -t LORA_FILES < <(find "$IDENTITY_LORAS_DIR" -name "*.safetensors" -type f | sort)

if [ ${#LORA_FILES[@]} -eq 0 ]; then
    echo "Error: No LoRA files found in $IDENTITY_LORAS_DIR"
    exit 1
fi

# Extract character names from LoRA filenames
CHARACTERS=()
for lora_file in "${LORA_FILES[@]}"; do
    # Extract base name and remove common prefixes/suffixes
    basename=$(basename "$lora_file" .safetensors)
    # Remove common prefixes like "BEST_", "final_", etc.
    char_name=$(echo "$basename" | sed -E 's/^(BEST_|final_|v[0-9]+_)//gi' | sed -E 's/_(lora|sdxl)$//gi')
    CHARACTERS+=("$char_name")
done

# ============================================================================
# SOURCE UTILITY FUNCTIONS FROM SHARED LIBRARY
# ============================================================================

source "$PROJECT_ROOT/scripts/batch/synthetic_data_generation/pipeline_utils.sh"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

echo "========================================================================"
echo "🚀 BATCH SYNTHETIC DATA GENERATION PIPELINE"
echo "========================================================================"
echo "Configuration:"
echo "  LoRA Directory:    $IDENTITY_LORAS_DIR"
echo "  Output Directory:  $WORKSPACE_DIR"
echo "  Base Model:        $BASE_MODEL_PATH"
echo "  Characters Found:  ${#CHARACTERS[@]}"
echo "  LoRA Types:        ${LORA_TYPES[*]}"
echo "  Prompts per Type:  $NUM_PROMPTS_PER_TYPE"
echo "  Images per Prompt: $IMAGES_PER_PROMPT"
echo "  Max Retries:       $MAX_RETRIES"
echo "  Resume Mode:       $RESUME_MODE"
echo ""
echo "Expected Output:    $(( ${#CHARACTERS[@]} * ${#LORA_TYPES[@]} * NUM_PROMPTS_PER_TYPE * IMAGES_PER_PROMPT )) images"
echo "========================================================================"
echo ""

log_info "Pipeline started with ${#CHARACTERS[@]} characters"

# Initial GPU check
if ! check_gpu; then
    log_error "Initial GPU check failed"
    exit 1
fi

# Execute pipeline phases
run_phase_1_vocabulary_generation
run_phase_2_image_generation
run_phase_3_quality_filtering
run_phase_4_dataset_organization

# Final summary
generate_final_summary

log_info "Pipeline execution completed"
