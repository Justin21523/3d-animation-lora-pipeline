#!/usr/bin/env bash
# Batch Character LoRA Training Script
# Train all character identity LoRAs for a specified film
#
# Usage: bash scripts/batch/train_character_loras.sh <film_name>
# Example: bash scripts/batch/train_character_loras.sh luca

set -euo pipefail

# Configuration
FILM_NAME="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/lora_training"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# AI Warehouse paths
WAREHOUSE_ROOT="/mnt/data/ai_data"
DATASETS_ROOT="$WAREHOUSE_ROOT/datasets/3d-anime"
LORA_OUTPUT_ROOT="$WAREHOUSE_ROOT/training_data/loras"

# Training configuration paths
CONFIG_DIR="$PROJECT_ROOT/configs/training"
BASE_CONFIG="$CONFIG_DIR/lora_base_config.yaml"
CHARACTER_PRESET="$CONFIG_DIR/character_lora_preset.yaml"

# Kohya sd-scripts path (adjust if needed)
KOHYA_SCRIPTS="/mnt/data/ai_data/training_repos/kohya_ss"

# ========================================
# Helper Functions
# ========================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$LOG_FILE" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $*" | tee -a "$LOG_FILE"
}

# ========================================
# Validation
# ========================================

if [ -z "$FILM_NAME" ]; then
    echo "Usage: $0 <film_name>"
    echo "Example: $0 luca"
    echo ""
    echo "Available films:"
    ls -1 "$DATASETS_ROOT" | grep -v "cross_character"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_character_loras_${FILM_NAME}_${TIMESTAMP}.log"

log_info "=========================================="
log_info "Character LoRA Training Batch Script"
log_info "=========================================="
log_info "Film: $FILM_NAME"
log_info "Timestamp: $TIMESTAMP"
log_info ""

# Check if film directory exists
FILM_DIR="$DATASETS_ROOT/$FILM_NAME"
if [ ! -d "$FILM_DIR" ]; then
    log_error "Film directory not found: $FILM_DIR"
    exit 1
fi

# Check if character config exists
CHAR_CONFIG="$PROJECT_ROOT/configs/characters/${FILM_NAME}_characters.yaml"
if [ ! -f "$CHAR_CONFIG" ]; then
    log_error "Character config not found: $CHAR_CONFIG"
    exit 1
fi

log_success "Found film directory: $FILM_DIR"
log_success "Found character config: $CHAR_CONFIG"

# ========================================
# Extract Character List from YAML Config
# ========================================

log_info "Extracting character list from config..."

# Use Python to parse YAML and extract character IDs
CHARACTERS=$(conda run -n ai_env python -c "
import yaml
import sys

config_file = '$CHAR_CONFIG'
try:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if 'characters' in config:
        for char in config['characters']:
            print(char['id'])
except Exception as e:
    print(f'Error parsing config: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

if [ -z "$CHARACTERS" ]; then
    log_error "No characters found in config file"
    exit 1
fi

CHAR_COUNT=$(echo "$CHARACTERS" | wc -l)
log_success "Found $CHAR_COUNT characters to train"
echo "$CHARACTERS" | while read -r char; do
    log_info "  - $char"
done
echo ""

# ========================================
# Training Configuration
# ========================================

# Training hyperparameters (from character_lora_preset.yaml)
MAX_TRAIN_EPOCHS=15
SAVE_EVERY_N_EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=1.5e-4
NETWORK_DIM=128
NETWORK_ALPHA=64

log_info "Training Configuration:"
log_info "  - Max epochs: $MAX_TRAIN_EPOCHS"
log_info "  - Save interval: $SAVE_EVERY_N_EPOCHS epochs"
log_info "  - Batch size: $BATCH_SIZE"
log_info "  - Learning rate: $LEARNING_RATE"
log_info "  - Network dim: $NETWORK_DIM"
log_info "  - Network alpha: $NETWORK_ALPHA"
echo ""

# ========================================
# Training Loop
# ========================================

TRAINED_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0

for CHAR_ID in $CHARACTERS; do
    log_info "=========================================="
    log_info "Processing character: $CHAR_ID"
    log_info "=========================================="

    # Check if character dataset exists
    CHAR_DATASET="$FILM_DIR/lora_data/characters_inpainted/$CHAR_ID"

    if [ ! -d "$CHAR_DATASET" ]; then
        log_error "Character dataset not found: $CHAR_DATASET"
        log_info "Skipping $CHAR_ID"
        ((SKIPPED_COUNT++))
        continue
    fi

    # Count images
    IMAGE_COUNT=$(find "$CHAR_DATASET" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l)
    log_info "Found $IMAGE_COUNT images for $CHAR_ID"

    if [ "$IMAGE_COUNT" -lt 100 ]; then
        log_error "Insufficient images ($IMAGE_COUNT < 100 minimum)"
        log_info "Skipping $CHAR_ID"
        ((SKIPPED_COUNT++))
        continue
    fi

    # Create output directory
    OUTPUT_DIR="$LORA_OUTPUT_ROOT/${FILM_NAME}/${CHAR_ID}_identity"
    mkdir -p "$OUTPUT_DIR"

    # Generate training config file (TOML format for Kohya)
    TOML_CONFIG="$OUTPUT_DIR/${CHAR_ID}_training_config.toml"

    log_info "Generating training config: $TOML_CONFIG"

    # Create TOML config using Python
    conda run -n ai_env python -c "
import yaml

# Load base config and character preset
with open('$BASE_CONFIG', 'r') as f:
    base_config = yaml.safe_load(f)
with open('$CHARACTER_PRESET', 'r') as f:
    char_preset = yaml.safe_load(f)

# Override with character-specific values
output_name = f'${FILM_NAME}_{CHAR_ID}_identity_lora'

# Generate TOML (simplified - production version should use full config)
toml_content = f'''
[model_arguments]
pretrained_model_name_or_path = \"{base_config['base_model']['path']}\"
v2 = false
v_parameterization = false

[dataset_arguments]
resolution = {base_config['training']['resolution']}
batch_size = $BATCH_SIZE
enable_bucket = {str(base_config['training']['enable_bucket']).lower()}
min_bucket_reso = {base_config['training']['min_bucket_reso']}
max_bucket_reso = {base_config['training']['max_bucket_reso']}

[dataset_arguments.datasets.0.subsets.0]
image_dir = \"$CHAR_DATASET\"
num_repeats = 1
shuffle_caption = true
keep_tokens = 2
caption_extension = \".txt\"

[training_arguments]
output_dir = \"$OUTPUT_DIR\"
output_name = \"{output_name}\"
save_precision = \"fp16\"
save_model_as = \"safetensors\"

max_train_epochs = $MAX_TRAIN_EPOCHS
save_every_n_epochs = $SAVE_EVERY_N_EPOCHS

learning_rate = $LEARNING_RATE
unet_lr = $LEARNING_RATE
text_encoder_lr = {$LEARNING_RATE / 2}
lr_scheduler = \"{base_config['optimizer']['lr_scheduler']}\"
lr_warmup_steps = {base_config['optimizer']['lr_warmup_steps']}

optimizer_type = \"{base_config['optimizer']['type']}\"

mixed_precision = \"{base_config['training']['mixed_precision']}\"
gradient_checkpointing = {str(base_config['training']['gradient_checkpointing']).lower()}
gradient_accumulation_steps = 1
max_grad_norm = {base_config['training']['max_grad_norm']}

xformers = {str(base_config['training']['xformers']).lower()}
cache_latents = {str(base_config['training']['cache_latents']).lower()}
cache_latents_to_disk = {str(base_config['training']['cache_latents_to_disk']).lower()}

[network_arguments]
network_module = \"{base_config['network']['network_module']}\"
network_dim = $NETWORK_DIM
network_alpha = $NETWORK_ALPHA
conv_dim = {char_preset['network']['conv_dim']}
conv_alpha = {char_preset['network']['conv_alpha']}
network_dropout = 0.0

[logging_arguments]
log_with = \"{base_config['logging']['log_with']}\"
logging_dir = \"{base_config['logging']['logging_dir']}\"
log_prefix = \"{output_name}\"

[sample_prompts]
sample_every_n_epochs = $SAVE_EVERY_N_EPOCHS
sample_sampler = \"euler_a\"
'''

with open('$TOML_CONFIG', 'w') as f:
    f.write(toml_content)
print('Config written successfully')
" 2>&1 | tee -a "$LOG_FILE"

    if [ ! -f "$TOML_CONFIG" ]; then
        log_error "Failed to generate training config"
        ((FAILED_COUNT++))
        continue
    fi

    log_success "Training config generated"

    # Launch training
    log_info "Starting LoRA training for $CHAR_ID..."
    log_info "Output directory: $OUTPUT_DIR"

    TRAIN_LOG="$LOG_DIR/${FILM_NAME}_${CHAR_ID}_training_${TIMESTAMP}.log"

    # Run Kohya training script
    if conda run -n ai_env python "$KOHYA_SCRIPTS/train_network.py" \
        --config_file "$TOML_CONFIG" \
        2>&1 | tee "$TRAIN_LOG"; then
        log_success "Training completed for $CHAR_ID"
        ((TRAINED_COUNT++))
    else
        log_error "Training failed for $CHAR_ID (see $TRAIN_LOG)"
        ((FAILED_COUNT++))
    fi

    echo ""
done

# ========================================
# Summary
# ========================================

log_info "=========================================="
log_info "Training Summary"
log_info "=========================================="
log_info "Total characters: $CHAR_COUNT"
log_success "Successfully trained: $TRAINED_COUNT"
log_error "Failed: $FAILED_COUNT"
log_info "Skipped: $SKIPPED_COUNT"
log_info ""
log_info "Output directory: $LORA_OUTPUT_ROOT/$FILM_NAME"
log_info "Log file: $LOG_FILE"
log_info "=========================================="

exit 0
