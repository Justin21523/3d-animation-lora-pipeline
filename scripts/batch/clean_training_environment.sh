#!/bin/bash
#
# Clean Training Environment Script
# Removes old cached latents and training artifacts
#

set -e

echo "========================================="
echo "Training Environment Cleanup"
echo "========================================="
echo ""

# Base directories
DATASET_BASE="/mnt/data/ai_data/datasets/3d-anime"
MODELS_BASE="/mnt/data/ai_data/models/lora_sdxl"

total_deleted=0

echo "🧹 Cleaning cached latents from training data directories..."
echo ""

# Find and delete .npz files (cached latents)
for movie in coco elio luca onward orion turning-red up; do
    training_dir="$DATASET_BASE/$movie/lora_data/training_data_sdxl"

    if [ -d "$training_dir" ]; then
        echo "Checking: $movie"

        # Count .npz files
        npz_count=$(find "$training_dir" -name "*.npz" 2>/dev/null | wc -l)

        if [ "$npz_count" -gt 0 ]; then
            echo "  Found $npz_count cached latent files"
            find "$training_dir" -name "*.npz" -delete
            echo "  ✅ Deleted"
            total_deleted=$((total_deleted + npz_count))
        else
            echo "  ℹ️  No cached latents found"
        fi
    fi
done

echo ""
echo "========================================="
echo "✅ Cleanup Complete!"
echo "========================================="
echo ""
echo "📊 Summary:"
echo "  - Total cached latent files deleted: $total_deleted"
echo ""
echo "💡 Why clean cached latents?"
echo "  - Old latents were cached to disk (slow)"
echo "  - New config caches to RAM (fast)"
echo "  - Must regenerate latents with new config"
echo ""
