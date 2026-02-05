#!/bin/bash
# Restart Endou Mamoru training with fixed config

set -e

echo "=================================================="
echo "Restarting Endou Mamoru SDXL LoRA Training"
echo "Fixed: NaN loss issue (removed min_snr_gamma, added gradient clipping)"
echo "=================================================="
echo ""

# Paths
LORA_DIR="/mnt/data/ai_data/models/lora_sdxl/inazuma-eleven/endou_mamoru_identity"
CONFIG_FILE="/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/inazuma_endou_mamoru_sdxl.toml"
KOHYA_ROOT="/mnt/c/ai_projects/kohya_ss/sd-scripts"
CONDA_ENV="kohya_ss"
LOG_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/logs"

# Create backup of broken checkpoints
echo "📦 Creating backup of broken checkpoints..."
BACKUP_DIR="$LORA_DIR/broken_nan_checkpoints_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if ls $LORA_DIR/*.safetensors 2>/dev/null; then
    mv $LORA_DIR/*.safetensors "$BACKUP_DIR/"
    echo "✓ Moved broken checkpoints to: $BACKUP_DIR"
else
    echo "✓ No existing checkpoints to backup"
fi

# Clean up TensorBoard logs
if [ -d "$LORA_DIR/logs" ]; then
    echo "🧹 Cleaning TensorBoard logs..."
    rm -rf "$LORA_DIR/logs"
    echo "✓ Logs cleaned"
fi

echo ""
echo "🚀 Starting training with fixed config..."
echo "   Config: $CONFIG_FILE"
echo "   Key fixes:"
echo "   - Learning rate: 8e-5 → 5e-5"
echo "   - Added max_grad_norm = 1.0"
echo "   - Removed min_snr_gamma (was causing NaN)"
echo "   - Disabled xformers (not available)"
echo ""

# Change to Kohya directory
cd "$KOHYA_ROOT"

# Generate log filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/inazuma_endou_mamoru_sdxl_training_FIXED_${TIMESTAMP}.log"

# Run training
echo "📝 Logging to: $LOG_FILE"
echo ""

conda run -n "$CONDA_ENV" python sdxl_train_network.py \
    --config_file "$CONFIG_FILE" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================="
echo "✅ Training completed!"
echo "=================================================="
echo ""
echo "Checkpoints saved to: $LORA_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Check log for loss values (should NOT be NaN)"
echo "  2. Test the final checkpoint"
echo "  3. If successful, update other character configs"
