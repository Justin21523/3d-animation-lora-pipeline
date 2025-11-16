#!/bin/bash
# Quick Launch Script for Iterative LoRA Optimization
# Universal launcher for any 3D animation character training

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PRESET_CONFIG="$PROJECT_ROOT/configs/optimization_presets.yaml"

# Usage
usage() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     ITERATIVE LORA OPTIMIZATION - QUICK LAUNCHER           ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo -e "${YELLOW}Required:${NC}"
    echo "  --characters NAMES      Character names (space-separated)"
    echo "  --dataset-dir PATH      Curated dataset directory"
    echo "  --base-model PATH       Base SD model path"
    echo "  --output-dir PATH       Output directory for results"
    echo ""
    echo -e "${YELLOW}Optional:${NC}"
    echo "  --sd-scripts PATH       Path to sd-scripts (default: auto-detect)"
    echo "  --strategy NAME         Optimization strategy (default: conservative)"
    echo "                          Options: conservative, aggressive, fine_tune, exploration"
    echo "  --schedule NAME         Training schedule (default: overnight)"
    echo "                          Options: overnight, weekend, quick, extended"
    echo "  --time-limit HOURS      Max training time in hours (default: 14)"
    echo "  --max-iterations N      Max iterations per character (default: 5)"
    echo "  --tmux SESSION          Run in tmux session"
    echo "  --dry-run               Print config and exit"
    echo "  --help                  Show this help"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo ""
    echo -e "${GREEN}# Basic usage (Luca + Alberto overnight)${NC}"
    echo "  $0 \\"
    echo "    --characters luca_human alberto_human \\"
    echo "    --dataset-dir /mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset \\"
    echo "    --base-model /path/to/sd-v1-5 \\"
    echo "    --output-dir /mnt/data/ai_data/models/lora/luca/iterative"
    echo ""
    echo -e "${GREEN}# Aggressive 14-hour optimization${NC}"
    echo "  $0 \\"
    echo "    --characters luca_human alberto_human \\"
    echo "    --dataset-dir /path/to/dataset \\"
    echo "    --base-model /path/to/model \\"
    echo "    --output-dir /path/to/output \\"
    echo "    --strategy aggressive \\"
    echo "    --schedule overnight"
    echo ""
    echo -e "${GREEN}# Run in background tmux${NC}"
    echo "  $0 --characters character_name ... --tmux lora_optimization"
    echo ""
    exit 1
}

# Parse arguments
CHARACTERS=()
DATASET_DIR=""
BASE_MODEL=""
OUTPUT_DIR=""
SD_SCRIPTS_DIR=""
STRATEGY="conservative"
SCHEDULE="overnight"
TIME_LIMIT=""
MAX_ITERATIONS=""
TMUX_SESSION=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --characters)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                CHARACTERS+=("$1")
                shift
            done
            ;;
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --sd-scripts)
            SD_SCRIPTS_DIR="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --schedule)
            SCHEDULE="$2"
            shift 2
            ;;
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
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

# Validate required arguments
if [ ${#CHARACTERS[@]} -eq 0 ]; then
    echo -e "${RED}Error: --characters is required${NC}"
    usage
fi

if [ -z "$DATASET_DIR" ]; then
    echo -e "${RED}Error: --dataset-dir is required${NC}"
    usage
fi

if [ -z "$BASE_MODEL" ]; then
    echo -e "${RED}Error: --base-model is required${NC}"
    usage
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo -e "${RED}Error: --output-dir is required${NC}"
    usage
fi

# Auto-detect sd-scripts if not provided
if [ -z "$SD_SCRIPTS_DIR" ]; then
    POSSIBLE_PATHS=(
        "/mnt/c/AI_LLM_projects/ai_warehouse/sd-scripts"
        "$HOME/sd-scripts"
        "$PROJECT_ROOT/../sd-scripts"
    )

    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/train_network.py" ]; then
            SD_SCRIPTS_DIR="$path"
            break
        fi
    done

    if [ -z "$SD_SCRIPTS_DIR" ]; then
        echo -e "${RED}Error: Could not auto-detect sd-scripts. Please specify with --sd-scripts${NC}"
        exit 1
    fi
fi

# Validate paths
if [ ! -d "$DATASET_DIR" ]; then
    echo -e "${RED}Error: Dataset directory not found: $DATASET_DIR${NC}"
    exit 1
fi

if [ ! -d "$SD_SCRIPTS_DIR" ]; then
    echo -e "${RED}Error: sd-scripts directory not found: $SD_SCRIPTS_DIR${NC}"
    exit 1
fi

# Load schedule defaults if not overridden
if [ -z "$TIME_LIMIT" ] || [ -z "$MAX_ITERATIONS" ]; then
    case $SCHEDULE in
        overnight)
            TIME_LIMIT="${TIME_LIMIT:-14}"
            MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
            ;;
        weekend)
            TIME_LIMIT="${TIME_LIMIT:-48}"
            MAX_ITERATIONS="${MAX_ITERATIONS:-8}"
            ;;
        quick)
            TIME_LIMIT="${TIME_LIMIT:-4}"
            MAX_ITERATIONS="${MAX_ITERATIONS:-2}"
            ;;
        extended)
            TIME_LIMIT="${TIME_LIMIT:-72}"
            MAX_ITERATIONS="${MAX_ITERATIONS:-10}"
            ;;
        *)
            TIME_LIMIT="${TIME_LIMIT:-14}"
            MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
            ;;
    esac
fi

# Print configuration
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║        ITERATIVE LORA OPTIMIZATION - CONFIGURATION         ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Characters:${NC}       ${CHARACTERS[*]}"
echo -e "${BLUE}Dataset:${NC}          $DATASET_DIR"
echo -e "${BLUE}Base Model:${NC}       $BASE_MODEL"
echo -e "${BLUE}Output:${NC}           $OUTPUT_DIR"
echo -e "${BLUE}sd-scripts:${NC}       $SD_SCRIPTS_DIR"
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Strategy:${NC}         $STRATEGY"
echo -e "${BLUE}Schedule:${NC}         $SCHEDULE"
echo -e "${BLUE}Time Limit:${NC}       ${TIME_LIMIT}h"
echo -e "${BLUE}Max Iterations:${NC}   $MAX_ITERATIONS"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Verify datasets exist
echo -e "${GREEN}Verifying datasets...${NC}"
for char in "${CHARACTERS[@]}"; do
    char_dir="$DATASET_DIR/$char"
    if [ ! -d "$char_dir" ]; then
        echo -e "  ${RED}✗${NC} $char: Directory not found"
        exit 1
    fi

    img_count=$(find "$char_dir/images" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
    if [ "$img_count" -eq 0 ]; then
        echo -e "  ${RED}✗${NC} $char: No images found"
        exit 1
    fi

    echo -e "  ${GREEN}✓${NC} $char: $img_count images"
done
echo ""

# Build command
OPTIMIZER_SCRIPT="$SCRIPT_DIR/iterative_lora_optimizer.py"

CMD="conda run -n ai_env python \"$OPTIMIZER_SCRIPT\" \
  --characters ${CHARACTERS[*]} \
  --dataset-dir \"$DATASET_DIR\" \
  --base-model \"$BASE_MODEL\" \
  --output-dir \"$OUTPUT_DIR\" \
  --sd-scripts \"$SD_SCRIPTS_DIR\" \
  --max-iterations $MAX_ITERATIONS \
  --time-limit $TIME_LIMIT"

# Dry run
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}[DRY RUN] Would execute:${NC}"
    echo "$CMD"
    echo ""
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save configuration
CONFIG_FILE="$OUTPUT_DIR/launch_config.json"
cat > "$CONFIG_FILE" <<EOF
{
  "characters": [$(printf '"%s",' "${CHARACTERS[@]}" | sed 's/,$//')],
  "dataset_dir": "$DATASET_DIR",
  "base_model": "$BASE_MODEL",
  "output_dir": "$OUTPUT_DIR",
  "sd_scripts_dir": "$SD_SCRIPTS_DIR",
  "strategy": "$STRATEGY",
  "schedule": "$SCHEDULE",
  "time_limit_hours": $TIME_LIMIT,
  "max_iterations": $MAX_ITERATIONS,
  "started_at": "$(date -Iseconds)"
}
EOF

echo -e "${GREEN}Configuration saved: $CONFIG_FILE${NC}"
echo ""

# Execute
if [ -n "$TMUX_SESSION" ]; then
    echo -e "${GREEN}Starting optimization in tmux session: ${TMUX_SESSION}${NC}"
    echo -e "${YELLOW}Use 'tmux attach -t $TMUX_SESSION' to view progress${NC}"
    echo ""

    # Create logging wrapper
    LOG_FILE="$OUTPUT_DIR/optimization.log"
    TMUX_CMD="$CMD 2>&1 | tee '$LOG_FILE'"

    tmux new-session -d -s "$TMUX_SESSION" "bash -c \"$TMUX_CMD\""

    echo -e "${GREEN}✓ Optimization started in background${NC}"
    echo ""
    echo -e "${BLUE}Useful commands:${NC}"
    echo -e "  tmux attach -t $TMUX_SESSION          # View live progress"
    echo -e "  tail -f $LOG_FILE    # Monitor log file"
    echo -e "  tmux kill-session -t $TMUX_SESSION    # Stop optimization"
    echo ""
else
    echo -e "${GREEN}Starting optimization...${NC}"
    echo ""
    eval "$CMD"
fi

echo -e "${GREEN}✓ Launch complete!${NC}"
