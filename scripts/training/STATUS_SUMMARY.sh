#!/bin/bash
# Training Pipeline Status Summary
# DO NOT PUT IN /tmp - This is a project script

echo "========================================="
echo "LUCA LORA PIPELINE - STATUS SUMMARY"
echo "========================================="
echo ""

# Caption Generation Status
echo "1. CAPTION GENERATION"
if tmux has-session -t caption_luca 2>/dev/null; then
    echo "   Status: ✓ Running (tmux: caption_luca)"
    CAPTIONS=$(find /mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset_v2_smart/luca_human/captions -name "*.txt" 2>/dev/null | wc -l)
    echo "   Progress: $CAPTIONS / 372 ($((CAPTIONS*100/372))%)"
    REMAINING=$((372-CAPTIONS))
    echo "   Remaining: $REMAINING captions"
else
    echo "   Status: Not running"
fi
echo "   Monitor: bash /tmp/monitor_caption.sh"
echo "   Attach: tmux attach -t caption_luca"
echo ""

# Training Environment
echo "2. TRAINING ENVIRONMENT"
echo "   ✓ ai_env: Caption generation (Qwen2-VL)"
echo "   ✓ kohya_ss: LoRA training (Kohya SS sd-scripts)"
echo "   ✓ sd-scripts: /mnt/c/AI_LLM_projects/ai_warehouse/sd-scripts"
echo "   ✓ Base model: v1-5-pruned-emaonly.safetensors (4GB)"
echo "   ✓ NumPy: 1.26.4 (kohya_ss env fixed)"
echo ""

# Training Configuration
echo "3. TRAINING CONFIGURATION"
echo "   ✓ Config: configs/training/luca_human.toml"
echo "   ✓ Based on: docs/training_optimization/ research"
echo "   ✓ Optimizations:"
echo "      - Learning rate: 8e-5 (防止特徵漂移)"
echo "      - Text Encoder LR: 6e-5 (加強身份學習)"
echo "      - Network dim: 96 (3D角色最佳，固定不增長)"
echo "      - Epochs: 10 (防止過度訓練)"
echo "      - Batch size: 8 (effective: 16 with gradient_accumulation)"
echo "   ✓ RTX 5080: NO xformers, AdamW optimizer"
echo ""

# Dataset
echo "4. DATASET"
IMAGE_COUNT=$(find /mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset_v2_smart/luca_human/images -name "*.png" 2>/dev/null | wc -l)
CAPTION_COUNT=$(find /mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset_v2_smart/luca_human/captions -name "*.txt" 2>/dev/null | wc -l)
echo "   Images: $IMAGE_COUNT"
echo "   Captions: $CAPTION_COUNT"
if [ "$CAPTION_COUNT" -ge "$IMAGE_COUNT" ]; then
    echo "   Status: ✓ Complete"
else
    echo "   Status: ⏳ In progress ($((IMAGE_COUNT-CAPTION_COUNT)) remaining)"
fi
echo ""

# Next Steps
echo "5. NEXT STEPS"
if [ "$CAPTION_COUNT" -lt "$IMAGE_COUNT" ]; then
    echo "   ⏳ Wait for caption generation ($CAPTION_COUNT/$IMAGE_COUNT)"
    echo "   ⏱️  Est. time: ~$((((IMAGE_COUNT-CAPTION_COUNT))/2)) min"
else
    echo "   ✅ Ready to train!"
    echo "   🚀 Command:"
    echo "      bash scripts/generic/training/launch_lora_training.sh \\"
    echo "        --config configs/training/luca_human.toml \\"
    echo "        --tmux lora_luca"
fi
echo ""
echo "========================================="
