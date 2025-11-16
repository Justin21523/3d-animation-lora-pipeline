#!/usr/bin/env bash
#
# Trial 3.6 - Complete Improvement Pipeline
#
# Executes:
#   1. Aggressive data augmentation (2604 → 10,400 images)
#   2. Training with optimized parameters (18 epochs)
#   3. Automated evaluation
#   4. Comparison with Trial 3.5
#

set -e  # Exit on error

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
TEMPORAL_EXPANDED="/mnt/data/ai_data/datasets/3d-anime/luca/training_temporal_expanded"
FINAL_DATASET="/mnt/data/ai_data/datasets/3d-anime/luca/training_final_v1"
OUTPUT_DIR="/mnt/data/ai_data/models/lora/luca/trial_3.6_final"

echo "════════════════════════════════════════════════════════════"
echo "  Trial 3.6 - Complete Improvement Pipeline"
echo "════════════════════════════════════════════════════════════"
echo ""

# ==================== Phase 1: Data Augmentation ====================
echo "📊 Phase 1: Aggressive Data Augmentation"
echo "────────────────────────────────────────"
echo "  Input:  $TEMPORAL_EXPANDED (2604 images)"
echo "  Output: $FINAL_DATASET (~10,400 images)"
echo ""

if [ -d "$FINAL_DATASET" ]; then
    echo "⚠️  Final dataset already exists. Skip or overwrite?"
    read -p "Press [S]kip or [O]verwrite: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        echo "Removing existing dataset..."
        rm -rf "$FINAL_DATASET"
    elif [[ $REPLY =~ ^[Ss]$ ]]; then
        echo "Skipping augmentation..."
    else
        echo "Invalid input. Exiting."
        exit 1
    fi
fi

if [ ! -d "$FINAL_DATASET" ]; then
    echo "🔄 Running aggressive augmentation..."
    python3 "$PROJECT_ROOT/scripts/data_curation/aggressive_augmentation.py" \
        "$TEMPORAL_EXPANDED" \
        --output-dir "$FINAL_DATASET" \
        --samples-per-image 4

    echo "✅ Augmentation complete!"
    echo ""
else
    echo "✅ Dataset already exists"
    echo ""
fi

# Verify dataset size
FINAL_COUNT=$(find "$FINAL_DATASET" \( -name "*.png" -o -name "*.jpg" \) | wc -l)
echo "📊 Final dataset size: $FINAL_COUNT images"
echo ""

# ==================== Phase 2: Training ====================
echo "🎓 Phase 2: LoRA Training"
echo "────────────────────────────────────────"
echo "  Config: $PROJECT_ROOT/configs/training/luca_trial3.6_final.toml"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: 18 (estimated 4-5 hours)"
echo ""

read -p "Start training? [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "🚀 Starting training..."

    # Check if kohya_ss is set up
    if [ ! -d "/path/to/kohya_ss" ]; then
        echo "⚠️  Please update kohya_ss path in this script"
        echo "Current: /path/to/kohya_ss"
        read -p "Enter kohya_ss directory: " KOHYA_DIR
    else
        KOHYA_DIR="/path/to/kohya_ss"
    fi

    cd "$KOHYA_DIR"

    # Create training config if not exists
    CONFIG_FILE="$PROJECT_ROOT/configs/training/luca_trial3.6_final.toml"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "❌ Config file not found: $CONFIG_FILE"
        echo "Please create it first (see Trial 3.6 docs)"
        exit 1
    fi

    # Run training
    conda run -n kohya_ss python train_network.py \
        --config_file "$CONFIG_FILE" \
        > /tmp/trial_3.6_training.log 2>&1 &

    TRAIN_PID=$!
    echo "✅ Training started (PID: $TRAIN_PID)"
    echo "📄 Log file: /tmp/trial_3.6_training.log"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f /tmp/trial_3.6_training.log"
    echo ""
    echo "Waiting for training to complete..."
    echo "(This will take 4-5 hours - you can Ctrl+C and come back later)"

    # Wait for training
    wait $TRAIN_PID
    echo "✅ Training complete!"
    echo ""
else
    echo "Skipping training."
    echo ""
fi

# ==================== Phase 3: Evaluation ====================
echo "📈 Phase 3: Automated Evaluation"
echo "────────────────────────────────────────"
echo "  Testing all checkpoints..."
echo ""

if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR/*.safetensors 2>/dev/null)" ]; then
    read -p "Start evaluation? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        conda run -n kohya_ss python "$PROJECT_ROOT/scripts/evaluation/test_lora_checkpoints.py" \
            "$OUTPUT_DIR" \
            --base-model "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors" \
            --output-dir "$OUTPUT_DIR/evaluation" \
            --prompts-file "$PROJECT_ROOT/prompts/luca/luca_human_prompts.json" \
            --num-variations 4 \
            --device cuda

        echo "✅ Evaluation complete!"
        echo ""
    fi
else
    echo "⚠️  No checkpoints found. Train first."
    echo ""
fi

# ==================== Phase 4: Comparison ====================
echo "📊 Phase 4: Comparison with Trial 3.5"
echo "────────────────────────────────────────"

TRIAL_35_EVAL="/mnt/data/ai_data/models/lora/luca/trial_3.5/evaluation"
TRIAL_36_EVAL="$OUTPUT_DIR/evaluation"

if [ -d "$TRIAL_35_EVAL" ] && [ -d "$TRIAL_36_EVAL" ]; then
    read -p "Generate comparison report? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        echo "📊 Generating comparison..."

        # TODO: Implement comparison script
        echo "⚠️  Comparison script not yet implemented"
        echo "Manual comparison:"
        echo "  Trial 3.5: $TRIAL_35_EVAL"
        echo "  Trial 3.6: $TRIAL_36_EVAL"
        echo ""
    fi
else
    echo "⚠️  Missing evaluation results"
    echo ""
fi

# ==================== Summary ====================
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Pipeline Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📁 Results:"
echo "  Dataset:     $FINAL_DATASET ($FINAL_COUNT images)"
echo "  Model:       $OUTPUT_DIR"
echo "  Evaluation:  $OUTPUT_DIR/evaluation"
echo ""
echo "🎯 Next Steps:"
echo "  1. Review evaluation results"
echo "  2. Compare with Trial 3.5"
echo "  3. Select best checkpoint"
echo "  4. Test in production"
echo ""
echo "📚 Documentation:"
echo "  $PROJECT_ROOT/docs/guides/TRIAL_3.6_FINAL_PLAN.md"
echo ""
