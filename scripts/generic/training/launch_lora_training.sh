#!/bin/bash
# Launch LoRA Training for Luca Characters
# Wrapper script for kohya_ss sd-scripts training

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default paths
SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-/mnt/c/AI_LLM_projects/ai_warehouse/sd-scripts}"
CONDA_ENV="${CONDA_ENV:-ai_env}"

# Usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config FILE         Path to training config TOML file (required)"
    echo "  --sd-scripts DIR      Path to sd-scripts directory (default: $SD_SCRIPTS_DIR)"
    echo "  --conda-env NAME      Conda environment name (default: $CONDA_ENV)"
    echo "  --tmux SESSION        Run in tmux session with given name"
    echo "  --dry-run             Print command without executing"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Train single character"
    echo "  $0 --config configs/luca/luca_human.toml"
    echo ""
    echo "  # Train in background tmux session"
    echo "  $0 --config configs/luca/alberto_human.toml --tmux lora_alberto"
    echo ""
    exit 1
}

# Parse arguments
CONFIG_FILE=""
TMUX_SESSION=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --sd-scripts)
            SD_SCRIPTS_DIR="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --tmux)
            TMUX_SESSION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate config file
if [ -z "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: --config is required${NC}"
    usage
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Validate sd-scripts directory
if [ ! -d "$SD_SCRIPTS_DIR" ]; then
    echo -e "${RED}Error: sd-scripts directory not found: $SD_SCRIPTS_DIR${NC}"
    echo -e "${YELLOW}Please install kohya_ss sd-scripts or specify correct path with --sd-scripts${NC}"
    exit 1
fi

TRAIN_SCRIPT="$SD_SCRIPTS_DIR/train_network.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo -e "${RED}Error: train_network.py not found in $SD_SCRIPTS_DIR${NC}"
    exit 1
fi

# Get character name from config
CHARACTER=$(basename "$CONFIG_FILE" .toml)

# Print header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           LUCA LORA TRAINING - LAUNCH SCRIPT                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Character:${NC}    $CHARACTER"
echo -e "${GREEN}Config:${NC}       $CONFIG_FILE"
echo -e "${GREEN}sd-scripts:${NC}   $SD_SCRIPTS_DIR"
echo -e "${GREEN}Conda Env:${NC}    $CONDA_ENV"

# Build training command
TRAIN_CMD="conda run -n $CONDA_ENV python \"$TRAIN_SCRIPT\" --config_file=\"$CONFIG_FILE\""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    echo -e "${GREEN}GPU:${NC}          $GPU_INFO"
else
    echo -e "${YELLOW}Warning: nvidia-smi not found, cannot verify GPU${NC}"
fi

echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Parse config for summary
if command -v grep &> /dev/null; then
    EPOCHS=$(grep "max_train_epochs" "$CONFIG_FILE" | awk -F'=' '{print $2}' | tr -d ' ')
    LR=$(grep "^learning_rate" "$CONFIG_FILE" | awk -F'=' '{print $2}' | tr -d ' ')
    IMAGE_DIR=$(grep "image_dir" "$CONFIG_FILE" | awk -F'=' '{print $2}' | tr -d ' "')

    if [ -n "$IMAGE_DIR" ] && [ -d "$IMAGE_DIR" ]; then
        NUM_IMAGES=$(find "$IMAGE_DIR" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
        echo -e "${BLUE}Dataset:${NC}      $NUM_IMAGES images"
    fi

    [ -n "$EPOCHS" ] && echo -e "${BLUE}Epochs:${NC}       $EPOCHS"
    [ -n "$LR" ] && echo -e "${BLUE}Learning Rate:${NC} $LR"
fi

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Dry run
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}[DRY RUN] Would execute:${NC}"
    echo "$TRAIN_CMD"
    exit 0
fi

# Execute training
if [ -n "$TMUX_SESSION" ]; then
    # Run in tmux
    echo -e "${GREEN}Starting training in tmux session: $TMUX_SESSION${NC}"
    echo -e "${YELLOW}Use 'tmux attach -t $TMUX_SESSION' to view progress${NC}"
    echo ""

    tmux new-session -d -s "$TMUX_SESSION" "$TRAIN_CMD"

    echo -e "${GREEN}✓ Training started in background${NC}"
    echo ""
    echo -e "${BLUE}Useful commands:${NC}"
    echo -e "  tmux attach -t $TMUX_SESSION    # Attach to session"
    echo -e "  tmux kill-session -t $TMUX_SESSION  # Stop training"
    echo ""
else
    # Run in foreground
    echo -e "${GREEN}Starting training...${NC}"
    echo ""
    eval "$TRAIN_CMD"
fi

echo -e "${GREEN}✓ Training complete!${NC}"
