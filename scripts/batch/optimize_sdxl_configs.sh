#!/bin/bash
#
# SDXL Config Optimization Script
# Applies performance optimizations to all SDXL training configs
#
# Changes:
# 1. cache_latents_to_disk: true → false (RAM caching, faster)
# 2. max_data_loader_n_workers: 2 → 6 (better CPU utilization)
# 3. min_bucket_reso: 640 → 768 (reduce bucketing overhead)
# 4. max_bucket_reso: 1536 → 1280 (reduce bucketing overhead)
# 5. bucket_reso_steps: 64 → 128 (fewer buckets)

set -e

CONFIG_DIR="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl"
BACKUP_DIR="${CONFIG_DIR}/_backups_$(date +%Y%m%d_%H%M%S)"

echo "========================================="
echo "SDXL Config Optimization Script"
echo "========================================="
echo ""

# Create backup directory
echo "📁 Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Find all TOML files
configs=($(find "$CONFIG_DIR" -maxdepth 1 -name "*.toml" -type f))
total=${#configs[@]}

echo "✅ Found $total config files to optimize"
echo ""

# Backup and optimize each file
for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    filename=$(basename "$config")
    num=$((i + 1))

    echo "[$num/$total] Processing: $filename"

    # Backup original
    cp "$config" "$BACKUP_DIR/$filename"
    echo "  ✅ Backed up to: $BACKUP_DIR/$filename"

    # Apply optimizations using sed
    sed -i 's/^cache_latents_to_disk = true/cache_latents_to_disk = false/' "$config"
    sed -i 's/^max_data_loader_n_workers = 2/max_data_loader_n_workers = 6/' "$config"
    sed -i 's/^min_bucket_reso = 640/min_bucket_reso = 768/' "$config"
    sed -i 's/^max_bucket_reso = 1536/max_bucket_reso = 1280/' "$config"
    sed -i 's/^bucket_reso_steps = 64/bucket_reso_steps = 128/' "$config"

    echo "  ✅ Applied optimizations"
    echo ""
done

echo "========================================="
echo "✅ Optimization Complete!"
echo "========================================="
echo ""
echo "📊 Summary:"
echo "  - Total configs optimized: $total"
echo "  - Backups saved to: $BACKUP_DIR"
echo ""
echo "🚀 Key optimizations:"
echo "  ✅ cache_latents_to_disk: true → false"
echo "  ✅ max_data_loader_n_workers: 2 → 6"
echo "  ✅ min_bucket_reso: 640 → 768"
echo "  ✅ max_bucket_reso: 1536 → 1280"
echo "  ✅ bucket_reso_steps: 64 → 128"
echo ""
echo "⚡ Expected improvements:"
echo "  - Epoch time: 66min → 35-45min (-30-45%)"
echo "  - Training per character: 11h → 6-7.5h (-40%)"
echo ""
