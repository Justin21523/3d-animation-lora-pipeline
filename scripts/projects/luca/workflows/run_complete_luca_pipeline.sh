#!/usr/bin/env bash
#
# Complete 3D Character LoRA Pipeline - Project-Agnostic
# Face Recognition Approach
#
# Pipeline:
# 1. Face Recognition Filter (14,410 → ~2,500 character frames)
# 2. Intelligent Processing (~2,500 → ~4,400 diverse candidates)
# 3. Quality Curation (~4,400 → 400 best images)
# 4. Training with curated dataset
#
# Usage:
#   bash scripts/workflows/run_complete_luca_pipeline.sh [project]
#
# Examples:
#   bash scripts/workflows/run_complete_luca_pipeline.sh              # Use luca project
#   bash scripts/workflows/run_complete_luca_pipeline.sh alberto      # Use alberto project
#

set -e  # Exit on error

# ============================================================
# Project Configuration
# ============================================================

PROJECT="${1:-luca}"  # Default to luca if no argument provided
PROJECT_CONFIG="configs/projects/${PROJECT}.yaml"

# Verify project config exists
if [ ! -f "$PROJECT_CONFIG" ]; then
    echo "❌ Error: Project config not found: $PROJECT_CONFIG"
    echo "Available projects in configs/projects/:"
    ls -1 configs/projects/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//' || echo "  (none found)"
    exit 1
fi

# Read project configuration from YAML
PROJECT_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$PROJECT_CONFIG'))['project']['name'])")
BASE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$PROJECT_CONFIG'))['paths']['base_dir'])")

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
DATA_ROOT="${BASE_DIR}"

echo "════════════════════════════════════════════════════════════"
echo "  Complete ${PROJECT_NAME^^} LoRA Pipeline (Face Recognition)"
echo "════════════════════════════════════════════════════════════"
echo ""

# ==================== Phase 1: Face Recognition Filter ====================
echo "🔍 Phase 1: Face Recognition Filter"
echo "────────────────────────────────────"
echo "  Input:  $DATA_ROOT/frames (14,410 frames)"
echo "  Method: Face recognition to find all ${PROJECT_NAME} appearances"
echo "  Output: $DATA_ROOT/${PROJECT_NAME}_filtered (~2,500 frames)"
echo ""

read -p "Run face recognition filter? [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "🚀 Starting face recognition..."

    # Use existing curated frames as reference
    REFERENCE_DIR="$DATA_ROOT/training_ready/1_${PROJECT_NAME}"

    if [ ! -d "$REFERENCE_DIR" ]; then
        echo "❌ Reference directory not found: $REFERENCE_DIR"
        echo "Please ensure you have curated reference images of ${PROJECT_NAME}"
        exit 1
    fi

    # Run face recognition filter
    python3 "$PROJECT_ROOT/scripts/generic/clustering/face_identity_clustering.py" \
        --input-dir "$DATA_ROOT/frames" \
        --reference-dir "$REFERENCE_DIR" \
        --output-dir "$DATA_ROOT/${PROJECT_NAME}_filtered" \
        --similarity-threshold 0.6 \
        --device cuda

    echo "✅ Face recognition complete!"

    # Count results
    FILTERED_COUNT=$(find "$DATA_ROOT/${PROJECT_NAME}_filtered" -name "*.png" -o -name "*.jpg" | wc -l)
    echo "📊 Found $FILTERED_COUNT ${PROJECT_NAME} frames"
    echo ""
else
    echo "⏭️  Skipping face recognition"
    echo ""
fi

# Verify filtered frames exist
if [ ! -d "$DATA_ROOT/${PROJECT_NAME}_filtered" ] || [ -z "$(ls -A $DATA_ROOT/${PROJECT_NAME}_filtered)" ]; then
    echo "❌ No filtered frames found. Please run Phase 1 first."
    exit 1
fi

# ==================== Phase 2: Intelligent Processing ====================
echo "🧠 Phase 2: Intelligent Frame Processing"
echo "────────────────────────────────────"
echo "  Input:  $DATA_ROOT/${PROJECT_NAME}_filtered (~2,500 frames)"
echo "  Method: 4 strategies (keep_full, segment, occlusion, enhance)"
echo "  Output: $DATA_ROOT/${PROJECT_NAME}_intelligent_candidates (~4,400 images)"
echo ""

read -p "Run intelligent processing? [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "🚀 Starting intelligent processing..."

    python3 "$PROJECT_ROOT/scripts/data_curation/intelligent_frame_processor.py" \
        "$DATA_ROOT/${PROJECT_NAME}_filtered" \
        --output-dir "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates" \
        --device cuda

    echo "✅ Intelligent processing complete!"

    # Count results
    CANDIDATE_COUNT=0
    for strategy in keep_full segment create_occlusion enhance_segment; do
        if [ -d "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates/$strategy/images" ]; then
            count=$(find "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates/$strategy/images" -name "*.png" | wc -l)
            CANDIDATE_COUNT=$((CANDIDATE_COUNT + count))
            echo "   $strategy: $count images"
        fi
    done
    echo "📊 Total candidates: $CANDIDATE_COUNT images"
    echo ""
else
    echo "⏭️  Skipping intelligent processing"
    echo ""
fi

# Verify candidates exist
if [ ! -d "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates" ]; then
    echo "❌ No candidates found. Please run Phase 2 first."
    exit 1
fi

# ==================== Phase 3: Quality Curation ====================
echo "⭐ Phase 3: Quality Curation"
echo "────────────────────────────────────"
echo "  Input:  $DATA_ROOT/${PROJECT_NAME}_intelligent_candidates (~4,400 images)"
echo "  Method: Quality scoring + diversity balancing + deduplication"
echo "  Output: $DATA_ROOT/${PROJECT_NAME}_training_final (400 best images)"
echo ""

TARGET_SIZE=400
read -p "Target dataset size (default: 400): " USER_SIZE
if [ ! -z "$USER_SIZE" ]; then
    TARGET_SIZE=$USER_SIZE
fi

read -p "Run quality curation? [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "🚀 Starting quality curation (target: $TARGET_SIZE images)..."

    python3 "$PROJECT_ROOT/scripts/data_curation/intelligent_dataset_curator.py" \
        "$DATA_ROOT/${PROJECT_NAME}_intelligent_candidates" \
        --output-dir "$DATA_ROOT/${PROJECT_NAME}_training_final" \
        --target-size $TARGET_SIZE

    echo "✅ Quality curation complete!"

    # Verify final count
    FINAL_COUNT=$(find "$DATA_ROOT/${PROJECT_NAME}_training_final/images" -name "*.png" | wc -l)
    echo "📊 Final training set: $FINAL_COUNT images"
    echo ""
else
    echo "⏭️  Skipping quality curation"
    echo ""
fi

# Verify final dataset exists
if [ ! -d "$DATA_ROOT/${PROJECT_NAME}_training_final" ]; then
    echo "❌ No final dataset found. Please run Phase 3 first."
    exit 1
fi

# ==================== Phase 4: Training Configuration ====================
echo "🎓 Phase 4: Training Setup"
echo "────────────────────────────────────"
echo "  Dataset: $DATA_ROOT/${PROJECT_NAME}_training_final"
echo "  Config:  $PROJECT_ROOT/configs/training/${PROJECT_NAME}_curated.toml"
echo ""

read -p "Generate training config? [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    # Get final image count for config
    FINAL_COUNT=$(find "$DATA_ROOT/${PROJECT_NAME}_training_final/images" -name "*.png" | wc -l)

    # Calculate optimal epochs (aim for ~500-800 steps per epoch)
    BATCH_SIZE=4
    STEPS_PER_EPOCH=$((FINAL_COUNT / BATCH_SIZE))
    OPTIMAL_EPOCHS=$((2400 / STEPS_PER_EPOCH))  # Target ~2400 total steps

    # Create training config
    CONFIG_FILE="$PROJECT_ROOT/configs/training/${PROJECT_NAME}_curated.toml"

    cat > "$CONFIG_FILE" <<EOF
# ${PROJECT_NAME} LoRA Training - Face-Recognized + Intelligently Processed + Curated Dataset
# Generated: $(date)

[model]
pretrained_model_name_or_path = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors"
v2 = false
v_parameterization = false

# Network configuration (increased capacity for better quality)
network_module = "networks.lora"
network_dim = 64
network_alpha = 32
network_dropout = 0.1

[training]
# Dataset
train_data_dir = "$DATA_ROOT/${PROJECT_NAME}_training_final"
resolution = "512,512"
batch_size = $BATCH_SIZE
enable_bucket = true
min_bucket_reso = 320
max_bucket_reso = 960

# Epochs (optimized for dataset size)
max_train_epochs = $OPTIMAL_EPOCHS
save_every_n_epochs = 2

# Learning rate
learning_rate = 6e-5
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 200

# Optimizer
optimizer_type = "AdamW8bit"
optimizer_args = []

# Regularization
min_snr_gamma = 5.0
noise_offset = 0.05
random_crop = false  # Don't crop - images are already well-composed
color_aug = false    # Don't augment color - 3D materials should stay consistent

# Mixed precision
mixed_precision = "fp16"
full_fp16 = false

[output]
output_dir = "/mnt/data/ai_data/models/lora/${PROJECT_NAME}/curated"
output_name = "${PROJECT_NAME}_curated"
save_model_as = "safetensors"

logging_dir = "/mnt/data/ai_data/models/lora/${PROJECT_NAME}/curated/logs"
log_prefix = "${PROJECT_NAME}_curated"

[caption]
# Captions already generated by intelligent processor
caption_extension = ".txt"
shuffle_caption = true
keep_tokens = 0

[other]
seed = 42
clip_skip = 2
max_token_length = 77
EOF

    echo "✅ Training config generated: $CONFIG_FILE"
    echo ""
    echo "📊 Training Parameters:"
    echo "   Dataset size:   $FINAL_COUNT images"
    echo "   Batch size:     $BATCH_SIZE"
    echo "   Steps/epoch:    $STEPS_PER_EPOCH"
    echo "   Epochs:         $OPTIMAL_EPOCHS"
    echo "   Total steps:    ~$((STEPS_PER_EPOCH * OPTIMAL_EPOCHS))"
    echo ""
else
    echo "⏭️  Skipping config generation"
    echo ""
fi

# ==================== Phase 5: Training ====================
echo "🎓 Phase 5: LoRA Training (Optional)"
echo "────────────────────────────────────"
echo "  Config: $PROJECT_ROOT/configs/training/${PROJECT_NAME}_curated.toml"
echo "  Output: /mnt/data/ai_data/models/lora/${PROJECT_NAME}/curated"
echo ""

read -p "Start training now? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting LoRA training..."

    # Check kohya_ss path
    KOHYA_PATH="/mnt/c/AI_LLM_projects/kohya_ss"
    if [ ! -d "$KOHYA_PATH" ]; then
        echo "❌ Kohya SS not found at: $KOHYA_PATH"
        read -p "Enter kohya_ss path: " KOHYA_PATH
    fi

    cd "$KOHYA_PATH"

    # Run training
    CONFIG_FILE="$PROJECT_ROOT/configs/training/${PROJECT_NAME}_curated.toml"

    conda run -n kohya_ss python train_network.py \
        --config_file "$CONFIG_FILE" \
        > /tmp/${PROJECT_NAME}_training.log 2>&1 &

    TRAIN_PID=$!
    echo "✅ Training started (PID: $TRAIN_PID)"
    echo "📄 Log file: /tmp/${PROJECT_NAME}_training.log"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f /tmp/${PROJECT_NAME}_training.log"
    echo ""
else
    echo "⏭️  Training skipped"
    echo ""
    echo "To train later, run:"
    echo "  cd /path/to/kohya_ss"
    echo "  conda run -n kohya_ss python train_network.py \\"
    echo "    --config_file $PROJECT_ROOT/configs/training/${PROJECT_NAME}_curated.toml"
    echo ""
fi

# ==================== Summary ====================
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Pipeline Complete for ${PROJECT_NAME^^}!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 Pipeline Summary:"
echo "  Phase 1: Face Recognition"
echo "    14,410 frames → $(find "$DATA_ROOT/${PROJECT_NAME}_filtered" -name "*.png" 2>/dev/null | wc -l) ${PROJECT_NAME} frames"
echo ""
echo "  Phase 2: Intelligent Processing"
echo "    4 strategies applied"
echo "    ~4,400 diverse candidates generated"
echo ""
echo "  Phase 3: Quality Curation"
echo "    $(find "$DATA_ROOT/${PROJECT_NAME}_training_final/images" -name "*.png" 2>/dev/null | wc -l) best images selected"
echo ""
echo "📁 Final Training Dataset:"
echo "  Location: $DATA_ROOT/${PROJECT_NAME}_training_final"
echo "  Images:   $DATA_ROOT/${PROJECT_NAME}_training_final/images/"
echo "  Captions: $DATA_ROOT/${PROJECT_NAME}_training_final/captions/"
echo "  Report:   $DATA_ROOT/${PROJECT_NAME}_training_final/curation_report.json"
echo ""
echo "🎯 Next Steps:"
echo "  1. Review curation report"
echo "  2. Inspect sample images"
echo "  3. Start training (if not already started)"
echo "  4. Evaluate checkpoints when training completes"
echo ""
echo "📚 Documentation:"
echo "  - $PROJECT_ROOT/MODEL_INTEGRATION_COMPLETE.md"
echo "  - $PROJECT_ROOT/docs/guides/INTELLIGENT_FRAME_PROCESSING.md"
echo ""
