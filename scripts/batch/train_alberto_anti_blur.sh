#!/bin/bash
# Alberto Anti-Blur Training Launcher
# Optimized for training data with motion blur artifacts

set -e

echo "🔧 Alberto Anti-Blur LoRA Training"
echo "=================================="
echo ""

# Configuration
CONFIG_FILE="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/luca_alberto_identity_sdxl.toml"
OUTPUT_DIR="/mnt/data/ai_data/models/lora_sdxl/luca/alberto_identity"
BACKUP_DIR="${OUTPUT_DIR}/backup_$(date +%Y%m%d_%H%M%S)"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "📋 Training Configuration:"
echo "   Config: $CONFIG_FILE"
echo "   Output: $OUTPUT_DIR"
echo ""

# Backup existing checkpoints
if ls "$OUTPUT_DIR"/*.safetensors 1> /dev/null 2>&1; then
    echo "💾 Backing up existing checkpoints..."
    mkdir -p "$BACKUP_DIR"
    mv "$OUTPUT_DIR"/*.safetensors "$BACKUP_DIR/" 2>/dev/null || true
    echo "   Backed up to: $BACKUP_DIR"
    echo ""
fi

# Show optimized parameters
echo "🎯 Anti-Blur Optimizations:"
echo "   • Learning Rate: 0.00004 (reduced from 0.0001)"
echo "   • Network Dim: 32 (reduced from 64)"
echo "   • Min SNR Gamma: 8.0 (increased from 5.0)"
echo "   • Noise Offset: 0.15 (increased from 0.05)"
echo "   • Max Epochs: 6 (save every epoch)"
echo ""

# Check GPU
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Confirm before starting
read -p "▶️  Start training? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled"
    exit 0
fi

echo ""
echo "🚀 Starting training..."
echo "   Monitor samples in: ${OUTPUT_DIR}/sample/"
echo "   View logs in: ${OUTPUT_DIR}/logs/"
echo ""
echo "⏱️  Expected time: ~2-3 hours for 6 epochs"
echo ""
echo "💡 Tips:"
echo "   • Check sample images every epoch"
echo "   • Best quality likely at epoch 2-3"
echo "   • Stop early if you see watercolor artifacts"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start training
conda run -n kohya_ss accelerate launch \
  --num_cpu_threads_per_process=2 \
  sd-scripts/sdxl_train_network.py \
  --config_file="$CONFIG_FILE"

TRAIN_EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "📊 Generated checkpoints:"
    ls -lh "$OUTPUT_DIR"/*.safetensors 2>/dev/null || echo "   No checkpoints found"
    echo ""
    echo "🖼️  Sample images:"
    ls -lt "$OUTPUT_DIR"/sample/*.png 2>/dev/null | head -5 || echo "   No samples found"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Review sample images for each epoch"
    echo "   2. Test all checkpoints with evaluation script"
    echo "   3. Select the cleanest checkpoint (likely epoch 2-3)"
    echo "   4. Compare with previous blurry results"
    echo ""
    echo "🧪 To evaluate all checkpoints:"
    echo "   conda run -n ai_env python scripts/evaluation/sdxl_lora_evaluator.py \\"
    echo "     --lora-dir $OUTPUT_DIR \\"
    echo "     --base-model /home/b0979/models/sdxl/sd_xl_base_1.0.safetensors \\"
    echo "     --output-dir /mnt/data/ai_data/lora_evaluation/alberto_anti_blur \\"
    echo "     --device cuda"
    echo ""
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
    echo ""
    echo "🔍 Check logs:"
    echo "   tail -100 $OUTPUT_DIR/logs/*.log"
    echo ""
    exit $TRAIN_EXIT_CODE
fi
