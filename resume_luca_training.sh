#!/bin/bash
#
# Resume Luca Sea Monster Training After Reboot
# Location: /mnt/c/ai_projects/3d-animation-lora-pipeline/resume_luca_training.sh
#

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  Resuming Luca Sea Monster LoRA Training"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check NVIDIA driver
echo "🔍 Checking NVIDIA driver..."
if nvidia-smi > /dev/null 2>&1; then
    echo "✅ NVIDIA driver OK"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "❌ NVIDIA driver issue detected!"
    echo "Please fix driver before training"
    exit 1
fi
echo ""

# Check memory protection
echo "🔍 Checking memory protection..."
if pgrep -f "memory-watchdog" > /dev/null; then
    echo "✅ Memory protection active"
else
    echo "⚠️  Memory protection not running, starting..."
    ~/.local/bin/system-protection-start
fi
echo ""

# Check training data
echo "🔍 Checking training data..."
DATA_DIR="/mnt/data/datasets/general/luca/sdxl_seamonster_training/luca_seamonster_prepared/4_luca_seamonster"
IMAGE_COUNT=$(ls -1 "$DATA_DIR"/*.png 2>/dev/null | wc -l)
echo "Found $IMAGE_COUNT training images"

if [ "$IMAGE_COUNT" -ne 255 ]; then
    echo "⚠️  Expected 255 images, found $IMAGE_COUNT"
fi
echo ""

# Start training
echo "🚀 Starting Luca sea monster training..."
echo "Config: configs/training/character_loras_sdxl/luca_luca_seamonster_sdxl.toml"
echo "Output: /mnt/c/ai_models/lora_sdxl/luca/luca_seamonster_identity/"
echo ""
echo "Training will:"
echo "  - Run for 10 epochs"
echo "  - Save checkpoints every 2 epochs"
echo "  - Use 255 images × 4 repeats = 1020 steps/epoch"
echo ""
read -p "Press Enter to start training (Ctrl+C to cancel)..."
echo ""

cd /mnt/c/ai_projects/kohya_ss/sd-scripts

conda run -n kohya_ss accelerate launch \
  --num_cpu_threads_per_process=2 \
  sdxl_train_network.py \
  --config_file /mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/luca_luca_seamonster_sdxl.toml

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Training Complete!"
echo "════════════════════════════════════════════════════════════════"
