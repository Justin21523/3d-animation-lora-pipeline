#!/usr/bin/env bash
# Start SDXL LoRA Training on 16GB VRAM GPU
# Optimized configuration with memory-efficient settings
# ==============================================================================

set -e

echo "======================================================================"
echo "SDXL LoRA Training - 16GB VRAM Optimized"
echo "======================================================================"
echo "Configuration: 8-bit AdamW + Gradient Checkpointing + BF16"
echo "Expected VRAM usage: 14-15GB peak"
echo "Training time estimate: ~5-6 hours (410 images, 20 epochs)"
echo "======================================================================"
echo ""

# Paths
KOHYA_DIR="/mnt/c/AI_LLM_projects/kohya_ss"
CONFIG_FILE="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/sdxl_16gb_optimized.toml"
DATASET_PREP_SCRIPT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/training/prepare_kohya_dataset.sh"
LOG_DIR="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/logs"
LOG_FILE="${LOG_DIR}/luca_sdxl_training_$(date +%Y%m%d_%H%M%S).log"
TMUX_SESSION="sdxl_luca_training"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Dataset paths (adjust these for your character)
SOURCE_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/luca_final_data"
OUTPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/luca_sdxl_training"
CHARACTER_NAME="luca"
REPEAT=10

# Step 1: Check GPU VRAM
echo "Step 1: Checking GPU VRAM..."
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$((VRAM / 1024))
echo "  Detected VRAM: ${VRAM_GB}GB"

if [ "$VRAM_GB" -lt 14 ]; then
    echo "  ⚠️  Warning: Less than 14GB VRAM detected"
    echo "  Consider reducing batch size or using SD1.5 instead"
    read -p "  Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "  ✓ VRAM sufficient for SDXL training"
fi
echo ""

# Step 2: Stop any previous training
echo "Step 2: Checking for previous training processes..."
if pgrep -f "sdxl_train_network.py" > /dev/null; then
    echo "  Stopping previous SDXL training..."
    pkill -f "sdxl_train_network.py" || true
    sleep 3
    echo "  ✓ Previous training stopped"
else
    echo "  ✓ No previous training running"
fi
echo ""

# Step 3: Prepare Kohya dataset (if not already prepared)
echo "Step 3: Preparing Kohya training dataset..."
if [ ! -d "${OUTPUT_DIR}/${REPEAT}_${CHARACTER_NAME}" ]; then
    echo "  Creating SDXL dataset from source..."
    chmod +x "$DATASET_PREP_SCRIPT"
    bash "$DATASET_PREP_SCRIPT" \
        --source-dir "$SOURCE_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --repeat "$REPEAT" \
        --name "$CHARACTER_NAME" \
        --validate
else
    echo "  ✓ Dataset already prepared: ${OUTPUT_DIR}/${REPEAT}_${CHARACTER_NAME}"
    IMAGE_COUNT=$(ls "${OUTPUT_DIR}/${REPEAT}_${CHARACTER_NAME}"/*.png 2>/dev/null | wc -l)
    echo "    Images: $IMAGE_COUNT"
fi
echo ""

# Step 4: Verify Kohya directory
echo "Step 4: Verifying Kohya_ss installation..."
if [ ! -d "$KOHYA_DIR" ]; then
    echo "❌ Error: Kohya_ss directory not found: $KOHYA_DIR"
    exit 1
fi

if [ ! -f "${KOHYA_DIR}/sd-scripts/sdxl_train_network.py" ]; then
    echo "❌ Error: sdxl_train_network.py not found in Kohya_ss"
    exit 1
fi

echo "  ✓ Kohya_ss directory: $KOHYA_DIR"
echo ""

# Step 5: Verify SDXL base model
echo "Step 5: Checking SDXL base model..."
BASE_MODEL="/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"

if [ ! -f "$BASE_MODEL" ]; then
    echo "  ⚠️  SDXL base model not found: $BASE_MODEL"
    echo "  Expected file: sd_xl_base_1.0.safetensors (6.5GB)"
    echo "  Please download from:"
    echo "  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    echo ""
    echo "  Or run:"
    echo "  cd /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/"
    echo "  wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    exit 1
else
    MODEL_SIZE=$(ls -lh "$BASE_MODEL" | awk '{print $5}')
    echo "  ✓ SDXL base model found: $BASE_MODEL"
    echo "    Size: $MODEL_SIZE"
fi
echo ""

# Step 6: Verify training config
echo "Step 6: Verifying training configuration..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Training config not found: $CONFIG_FILE"
    exit 1
fi

echo "  ✓ Training config: $CONFIG_FILE"
echo ""

# Step 7: Clear GPU cache
echo "Step 7: Clearing GPU cache..."
python3 << EOF
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("  ✓ GPU cache cleared")
else:
    print("  ⚠️  CUDA not available")
EOF
echo ""

# Step 8: Check and manage tmux session
echo "Step 8: Setting up tmux session..."
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "  ⚠️  Existing tmux session found: $TMUX_SESSION"
    echo "  Killing existing session..."
    tmux kill-session -t "$TMUX_SESSION"
    sleep 2
    echo "  ✓ Old session terminated"
fi

echo "  Creating new tmux session: $TMUX_SESSION"
tmux new-session -d -s "$TMUX_SESSION"
echo "  ✓ Tmux session created"
echo ""

echo "======================================================================"
echo "📺 Tmux Session Information"
echo "======================================================================"
echo "Session name: $TMUX_SESSION"
echo ""
echo "To attach to session (view live output):"
echo "  tmux attach -t $TMUX_SESSION"
echo ""
echo "To detach from session (keep it running):"
echo "  Press: Ctrl+B, then D"
echo ""
echo "To kill session (stop training):"
echo "  tmux kill-session -t $TMUX_SESSION"
echo "======================================================================"
echo ""

# Step 9: Navigate to Kohya directory
cd "$KOHYA_DIR"

echo "======================================================================"
echo "Starting SDXL Training with 16GB VRAM Optimizations..."
echo "======================================================================"
echo "Log file: $LOG_FILE"
echo ""
echo "Key Optimizations Enabled:"
echo "  ✓ 8-bit AdamW optimizer (saves ~40% VRAM)"
echo "  ✓ Gradient checkpointing (saves ~30% VRAM)"
echo "  ✓ Mixed precision BF16 (saves ~25% VRAM)"
echo "  ✓ Batch size 1 + Gradient accumulation 8"
echo ""
echo "Training Parameters:"
echo "  - Learning rate: 0.0001"
echo "  - Network dim/alpha: 128/96"
echo "  - Epochs: 20"
echo "  - Resolution: 1024x1024"
echo "  - Checkpoints: Every 2 epochs"
echo ""
echo "Expected Results:"
echo "  - VRAM usage: 14-15GB peak"
echo "  - Training time: ~5-6 hours"
echo "  - Visual quality: Significantly better than SD1.5"
echo "======================================================================"
echo ""

# Send commands to tmux session for training
echo "Sending training commands to tmux session..."

# Command 1: Navigate to Kohya directory
tmux send-keys -t "$TMUX_SESSION" "cd $KOHYA_DIR" C-m
sleep 1

# Command 2: Activate conda environment
tmux send-keys -t "$TMUX_SESSION" "source /opt/miniconda3/etc/profile.d/conda.sh" C-m
tmux send-keys -t "$TMUX_SESSION" "conda activate kohya_ss" C-m
sleep 2

# Command 3: Clear GPU cache in training environment
tmux send-keys -t "$TMUX_SESSION" "python3 -c 'import torch; torch.cuda.empty_cache(); print(\"✓ Training env GPU cache cleared\")'" C-m
sleep 1

# Command 4: Start training with logging (SDXL-specific script)
TRAIN_CMD="accelerate launch --num_cpu_threads_per_process=2 ./sd-scripts/sdxl_train_network.py --config_file=$CONFIG_FILE 2>&1 | tee $LOG_FILE"
tmux send-keys -t "$TMUX_SESSION" "$TRAIN_CMD" C-m

echo "  ✓ Training commands sent to tmux session"
echo ""

# Wait a moment for training to initialize
sleep 10

# Show initial training log
echo "======================================================================"
echo "📋 Training Initialization (First 30 lines)"
echo "======================================================================"
tail -30 "$LOG_FILE" 2>/dev/null || echo "Waiting for log output..."
echo ""

echo "======================================================================"
echo "✅ SDXL Training Started Successfully in tmux!"
echo "======================================================================"
echo ""
echo "📊 Training Information:"
echo "   - Session: $TMUX_SESSION"
echo "   - Log file: $LOG_FILE"
echo "   - Output: /mnt/data/ai_data/models/lora/luca/sdxl_trial1"
echo "   - Duration: ~5-6 hours (20 epochs)"
echo ""
echo "======================================================================"
echo "📺 Monitoring Commands"
echo "======================================================================"
echo ""
echo "1. Attach to tmux session (live view):"
echo "   tmux attach -t $TMUX_SESSION"
echo ""
echo "2. View training log:"
echo "   tail -f $LOG_FILE"
echo ""
echo "3. Monitor GPU usage:"
echo "   watch -n 5 nvidia-smi"
echo ""
echo "4. Check training progress (in another terminal):"
echo "   ls -lh /mnt/data/ai_data/models/lora/luca/sdxl_trial1/"
echo ""
echo "5. Stop training:"
echo "   tmux kill-session -t $TMUX_SESSION"
echo ""
echo "======================================================================"
echo "📝 After Training Completes"
echo "======================================================================"
echo ""
echo "1. Review checkpoints:"
echo "   ls -lh /mnt/data/ai_data/models/lora/luca/sdxl_trial1/"
echo ""
echo "2. Check validation samples (generated every 2 epochs):"
echo "   ls /mnt/data/ai_data/models/lora/luca/sdxl_trial1/sample/"
echo ""
echo "3. Run evaluation to find best checkpoint:"
echo "   conda run -n ai_env python scripts/evaluation/sota_lora_evaluator.py \\"
echo "     --evaluate-samples \\"
echo "     --lora-dir /mnt/data/ai_data/models/lora/luca/sdxl_trial1 \\"
echo "     --sample-dir /mnt/data/ai_data/models/lora/luca/sdxl_trial1/sample \\"
echo "     --output-dir /mnt/data/ai_data/models/lora/luca/sdxl_trial1/evaluation \\"
echo "     --device cuda"
echo ""
echo "======================================================================"
echo ""
echo "🎯 Training is now running in tmux session: $TMUX_SESSION"
echo "💡 Use 'tmux attach -t $TMUX_SESSION' to view live progress"
echo ""
echo "======================================================================"
