#!/bin/bash
# ============================================================================
# Generate Training Configs for All Synthetic LoRAs
# ============================================================================
#
# Generates 42 training configs (14 characters × 3 types) using templates.
# Uses training_config_generator.py with type-specific templates.
#
# Output: Populated TOML configs in /mnt/c/ai_models/lora_sdxl/{char}/{type}/
#
# Author: LLMProvider Tooling
# Date: 2025-12-04
# ============================================================================

set -euo pipefail

# Configuration
DATASET_ROOT="/mnt/data/ai_data/synthetic_lora_data/datasets"
OUTPUT_ROOT="/mnt/c/ai_models/lora_sdxl"
TEMPLATE_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training"
LOG_DIR="/mnt/data/ai_data/synthetic_lora_data/logs"

# Characters
CHARACTERS=(
    alberto
    bryce
    caleb
    elio
    giulia
    ian_lightfoot
    luca
    miguel
    orion
    russell
    tyler
    alberto_seamonster
    luca_seamonster
    barley_lightfoot
)

# LoRA types
TYPES=(pose action expression)

# Create log directory
mkdir -p "$LOG_DIR"

echo "============================================="
echo "Generating Synthetic LoRA Training Configs"
echo "============================================="
echo ""
echo "Total configs to generate: $((${#CHARACTERS[@]} * ${#TYPES[@]}))"
echo ""

# Counter
total_generated=0
total_failed=0

# Generate configs
for type in "${TYPES[@]}"; do
    echo "----------------------------------------"
    echo "Type: ${type^^}"
    echo "----------------------------------------"

    template="${TEMPLATE_DIR}/${type}_lora_sdxl_template.toml"

    if [ ! -f "$template" ]; then
        echo "⚠️  Template not found: $template"
        continue
    fi

    for char in "${CHARACTERS[@]}"; do
        dataset_dir="${DATASET_ROOT}/${char}_${type}"
        output_dir="${OUTPUT_ROOT}/${char}/${type}"

        # Check if dataset exists
        if [ ! -d "$dataset_dir" ]; then
            echo "⚠️  Dataset not found: $dataset_dir, skipping ${char} ${type}"
            ((total_failed++))
            continue
        fi

        echo -n "Generating ${char} ${type}... "

        # Create output directory
        mkdir -p "$output_dir"

        # Copy template and fill in paths
        config_file="${output_dir}/config.toml"

        # Use sed to fill in template values
        sed -e "s|^output_dir = \".*\"|output_dir = \"${output_dir}\"|" \
            -e "s|^output_name = \".*\"|output_name = \"${char}_${type}_lora\"|" \
            -e "s|^logging_dir = \".*\"|logging_dir = \"${LOG_DIR}/${char}_${type}\"|" \
            -e "s|^train_data_dir = \".*\"|train_data_dir = \"${dataset_dir}\"|" \
            "$template" > "$config_file"

        if [ $? -eq 0 ]; then
            echo "✓"
            ((total_generated++))
        else
            echo "✗ Failed"
            ((total_failed++))
        fi
    done

    echo ""
done

# Summary
echo "============================================="
echo "Config Generation Complete"
echo "============================================="
echo "Total generated: $total_generated"
echo "Total failed: $total_failed"
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", ($total_generated / ($total_generated + $total_failed)) * 100}")%"
echo ""
echo "Configs saved to: $OUTPUT_ROOT"
echo ""
echo "Next step: Train LoRAs using train_all_synthetic_loras.sh"
echo "============================================="
