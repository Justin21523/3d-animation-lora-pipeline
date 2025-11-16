#!/bin/bash
# Test LoRA training with TOML configuration file
# This script tests that TOML configs work correctly with kohya_ss environment

set -e

echo "========================================================================"
echo "TOML CONFIGURATION TEST - LoRA Training"
echo "========================================================================"
echo "Environment: kohya_ss"
echo "Character: luca_human"
echo "Duration: 1 epoch (quick test)"
echo "========================================================================"
echo ""

# Configuration
CONFIG_FILE="/mnt/data/ai_data/models/lora/luca/test_toml_run/training_config.toml"
SD_SCRIPTS_DIR="/mnt/c/AI_LLM_projects/ai_warehouse/sd-scripts"
OUTPUT_DIR="/mnt/data/ai_data/models/lora/luca/test_toml_run"

# Create output directory
mkdir -p "$OUTPUT_DIR/logs"

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "✓ Config file: $CONFIG_FILE"
echo "✓ Output directory: $OUTPUT_DIR"
echo ""

# Display configuration
echo "Configuration preview:"
echo "----------------------------------------"
head -20 "$CONFIG_FILE"
echo "... (see full file for complete config)"
echo "----------------------------------------"
echo ""

echo "Starting training with TOML config..."
echo "========================================================================"
echo ""

# Run training with TOML config file
conda run -n kohya_ss python "$SD_SCRIPTS_DIR/train_network.py" \
    --config_file "$CONFIG_FILE"

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ TOML CONFIG TEST PASSED"
    echo "========================================================================"
    echo ""
    echo "Outputs:"
    ls -lh "$OUTPUT_DIR"/*.safetensors 2>/dev/null || echo "  (checkpoint still saving...)"
    echo ""
    echo "✓ TOML configuration works correctly!"
    echo "✓ You can now use TOML templates for all future training."
else
    echo "✗ TOML CONFIG TEST FAILED"
    echo "========================================================================"
    echo "Exit code: $EXIT_CODE"
    echo "Check logs in: $OUTPUT_DIR/logs/"
    exit 1
fi
