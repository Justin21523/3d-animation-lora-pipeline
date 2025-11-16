#!/bin/bash
# Real-time Caption Generation Progress Monitor

TRAINING_DATA_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/training_data"
CLUSTERED_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/clustered_enhanced"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to count files
count_captions() {
    local dir="$1"
    find "$dir" -name "*.txt" 2>/dev/null | wc -l
}

count_images() {
    local dir="$1"
    find "$dir" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l
}

# Main monitoring loop
while true; do
    clear
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║         LUCA CAPTION GENERATION - PROGRESS MONITOR           ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Check if process is running
    if ps aux | grep -q "[q]wen_caption_generator.py"; then
        echo -e "${GREEN}✓ Process Status: RUNNING${NC}"

        # GPU status
        gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
        if [ -n "$gpu_info" ]; then
            gpu_util=$(echo "$gpu_info" | cut -d',' -f1 | xargs)
            gpu_mem=$(echo "$gpu_info" | cut -d',' -f2 | xargs)
            gpu_total=$(echo "$gpu_info" | cut -d',' -f3 | xargs)
            echo -e "${GREEN}✓ GPU Utilization: ${gpu_util}% | Memory: ${gpu_mem}MB / ${gpu_total}MB${NC}"
        fi
    else
        echo -e "${RED}✗ Process Status: NOT RUNNING${NC}"
    fi
    echo ""

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Character Progress:${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    total_input=0
    total_output=0

    # Iterate through all character directories
    for cluster_dir in "$CLUSTERED_DIR"/*/ ; do
        if [ -d "$cluster_dir" ]; then
            char_name=$(basename "$cluster_dir")

            # Skip noise and special directories
            if [[ "$char_name" == "noise" ]] || [[ "$char_name" == "__pycache__" ]]; then
                continue
            fi

            input_count=$(count_images "$cluster_dir")
            total_input=$((total_input + input_count))

            output_dir="$TRAINING_DATA_DIR/$char_name/captions"
            if [ -d "$output_dir" ]; then
                output_count=$(count_captions "$output_dir")
                total_output=$((total_output + output_count))

                # Calculate percentage
                if [ "$input_count" -gt 0 ]; then
                    percent=$((output_count * 100 / input_count))
                else
                    percent=0
                fi

                # Progress bar
                bar_length=30
                filled=$((percent * bar_length / 100))
                bar=$(printf "█%.0s" $(seq 1 $filled))
                empty=$(printf "░%.0s" $(seq 1 $((bar_length - filled))))

                if [ "$output_count" -eq "$input_count" ]; then
                    echo -e "  ${GREEN}✓${NC} $char_name: [${bar}${empty}] ${output_count}/${input_count} (${percent}%)"
                elif [ "$output_count" -gt 0 ]; then
                    echo -e "  ${YELLOW}⟳${NC} $char_name: [${bar}${empty}] ${output_count}/${input_count} (${percent}%)"
                else
                    echo -e "  ${RED}○${NC} $char_name: [${empty}] 0/${input_count} (0%)"
                fi
            else
                echo -e "  ${RED}○${NC} $char_name: Pending... (${input_count} images)"
            fi
        fi
    done

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Overall progress
    if [ "$total_input" -gt 0 ]; then
        overall_percent=$((total_output * 100 / total_input))
    else
        overall_percent=0
    fi

    echo ""
    echo -e "${BLUE}📊 Overall Progress: ${total_output} / ${total_input} (${overall_percent}%)${NC}"

    # Estimate remaining time
    if [ "$total_output" -gt 0 ]; then
        remaining=$((total_input - total_output))
        # Assume 2.5 seconds per image
        est_seconds=$((remaining * 25 / 10))
        est_minutes=$((est_seconds / 60))
        echo -e "${BLUE}⏱️  Estimated time remaining: ~${est_minutes} minutes${NC}"
    fi

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "Recent files (last 5):"
    find "$TRAINING_DATA_DIR" -name "*.txt" -type f -mmin -5 2>/dev/null | tail -5 | while read file; do
        char=$(echo "$file" | awk -F'/' '{print $(NF-2)}')
        filename=$(basename "$file")
        echo -e "  ${GREEN}→${NC} $char: $filename"
    done

    echo ""
    echo -e "${BLUE}Press Ctrl+C to exit monitor${NC}"

    # Wait 5 seconds before refresh
    sleep 5
done
