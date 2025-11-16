#!/bin/bash
# Real-time Training Progress Monitor with ETA
# Shows live training progress, steps, epochs, and estimated time remaining
# Usage: bash watch_training_progress.sh

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SESSION_NAME="sdxl_luca_training_safe"
OUTPUT_DIR="/mnt/data/ai_data/models/lora/luca/sdxl_trial1"

# Training parameters (from config)
TOTAL_EPOCHS=12
IMAGES=400  # Approximate
BATCH_SIZE=1
GRADIENT_ACCUM=6

# Calculate total steps
STEPS_PER_EPOCH=$((IMAGES / (BATCH_SIZE * GRADIENT_ACCUM)))
TOTAL_STEPS=$((STEPS_PER_EPOCH * TOTAL_EPOCHS))

while true; do
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║        SDXL LoRA Training - Real-time Progress Monitor       ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo

    # Current time
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC}"
    echo

    # ========================================
    # GPU Status
    # ========================================
    echo -e "${GREEN}═══ GPU Status ═══${NC}"

    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits)

    if [[ -n "$gpu_info" ]]; then
        IFS=',' read -r gpu_util vram_used vram_total temp power <<< "$gpu_info"

        # Calculate VRAM percentage
        vram_pct=$(awk "BEGIN {printf \"%.1f\", ($vram_used/$vram_total)*100}")

        # Color coding
        if (( $(echo "$gpu_util < 50" | bc -l) )); then
            util_color=$YELLOW
        else
            util_color=$GREEN
        fi

        if (( $(echo "$temp > 80" | bc -l) )); then
            temp_color=$RED
        elif (( $(echo "$temp > 70" | bc -l) )); then
            temp_color=$YELLOW
        else
            temp_color=$GREEN
        fi

        echo -e "  GPU Utilization: ${util_color}${gpu_util}%${NC}"
        echo -e "  VRAM: ${vram_used} MB / ${vram_total} MB (${vram_pct}%)"
        echo -e "  Temperature: ${temp_color}${temp}°C${NC}"
        echo -e "  Power Draw: ${power} W"
    else
        echo -e "  ${RED}Unable to query GPU${NC}"
    fi
    echo

    # ========================================
    # System RAM
    # ========================================
    echo -e "${GREEN}═══ System Memory ═══${NC}"

    ram_info=$(free -h | awk 'NR==2{printf "%s / %s (%.1f%%) | Available: %s", $3, $2, $3*100/$2, $7}')
    echo -e "  RAM: $ram_info"

    swap_used=$(free -h | awk 'NR==3{print $3}')
    if [[ "$swap_used" != "0B" ]] && [[ "$swap_used" != "0" ]]; then
        echo -e "  ${YELLOW}⚠ Swap in use: $swap_used${NC}"
    fi
    echo

    # ========================================
    # Training Progress
    # ========================================
    echo -e "${GREEN}═══ Training Progress ═══${NC}"

    # Check if session is running
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "  ${RED}✗ Training session NOT running!${NC}"
        echo -e "  Session name: $SESSION_NAME"
        echo
        echo -e "${YELLOW}Press Ctrl+C to exit${NC}"
        sleep 5
        continue
    fi

    echo -e "  ${GREEN}✓${NC} Training session: $SESSION_NAME"
    echo

    # Try to extract progress from tmux pane
    training_output=$(tmux capture-pane -t "$SESSION_NAME" -p 2>/dev/null | grep -E "steps:|epoch" | tail -5)

    # Parse the latest progress line
    current_step=""
    current_epoch=""

    if [[ -n "$training_output" ]]; then
        # Try to find step info (format: "steps: 45%|████▌     | 1234/2730 [12:34<56:78, 1.23s/it]")
        step_line=$(echo "$training_output" | grep "steps:" | tail -1)

        if [[ -n "$step_line" ]]; then
            # Extract current/total steps
            if [[ "$step_line" =~ ([0-9]+)/([0-9]+) ]]; then
                current_step=${BASH_REMATCH[1]}
                total_steps_detected=${BASH_REMATCH[2]}
                TOTAL_STEPS=$total_steps_detected  # Update total steps from actual output
            fi

            # Extract percentage
            if [[ "$step_line" =~ ([0-9]+)% ]]; then
                progress_pct=${BASH_REMATCH[1]}
            fi

            # Extract time info [elapsed<remaining, speed]
            if [[ "$step_line" =~ \[([0-9:]+)\<([0-9:]+),\ ([0-9.]+[a-z/]+)\] ]]; then
                elapsed_str=${BASH_REMATCH[1]}
                remaining_str=${BASH_REMATCH[2]}
                speed_str=${BASH_REMATCH[3]}
            fi
        fi

        # Try to find epoch info
        epoch_line=$(echo "$training_output" | grep -i "epoch" | tail -1)
        if [[ "$epoch_line" =~ epoch[[:space:]]*([0-9]+) ]]; then
            current_epoch=${BASH_REMATCH[1]}
        fi
    fi

    # Display parsed progress
    if [[ -n "$current_step" ]] && [[ -n "$TOTAL_STEPS" ]]; then
        echo -e "${CYAN}  Progress:${NC}"
        echo -e "    Steps: ${current_step} / ${TOTAL_STEPS} (${progress_pct}%)"

        if [[ -n "$current_epoch" ]]; then
            echo -e "    Epoch: ${current_epoch} / ${TOTAL_EPOCHS}"
        fi

        if [[ -n "$elapsed_str" ]] && [[ -n "$remaining_str" ]]; then
            echo -e "    Elapsed: ${elapsed_str} | Remaining: ${remaining_str}"
            echo -e "    Speed: ${speed_str}"
        fi

        # Progress bar
        bar_width=50
        filled=$((progress_pct * bar_width / 100))
        empty=$((bar_width - filled))

        bar=$(printf "%${filled}s" | tr ' ' '█')
        bar+=$(printf "%${empty}s" | tr ' ' '░')

        echo -e "    [${bar}] ${progress_pct}%"
    else
        # Fallback: show raw output
        echo -e "${CYAN}  Latest training output:${NC}"
        if [[ -n "$training_output" ]]; then
            echo "$training_output" | sed 's/^/    /'
        else
            echo -e "    ${YELLOW}No progress info found yet (training may be initializing)${NC}"
        fi
    fi
    echo

    # ========================================
    # Checkpoints
    # ========================================
    echo -e "${GREEN}═══ Checkpoints ═══${NC}"

    checkpoint_count=$(find "$OUTPUT_DIR" -name "*.safetensors" -type f 2>/dev/null | wc -l)
    echo -e "  Total checkpoints saved: ${checkpoint_count}"

    if [[ $checkpoint_count -gt 0 ]]; then
        echo -e "${CYAN}  Latest 3:${NC}"
        ls -lht "$OUTPUT_DIR"/*.safetensors 2>/dev/null | head -3 | \
            awk '{printf "    %s %s  %-6s  %s\n", $6, $7, $5, $9}' | \
            sed 's|'"$OUTPUT_DIR"'/||'

        # Check checkpoint age
        latest_ckpt=$(find "$OUTPUT_DIR" -name "*.safetensors" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')

        if [[ -n "$latest_ckpt" ]]; then
            ckpt_mtime=$(stat -c %Y "$latest_ckpt" 2>/dev/null)
            current_time=$(date +%s)
            ckpt_age=$((current_time - ckpt_mtime))
            ckpt_age_min=$((ckpt_age / 60))

            if [[ $ckpt_age_min -gt 30 ]]; then
                echo -e "  ${RED}⚠ WARNING: No new checkpoint for ${ckpt_age_min} minutes!${NC}"
            else
                echo -e "  ${GREEN}✓${NC} Latest checkpoint: ${ckpt_age_min} minutes ago"
            fi
        fi
    else
        echo -e "  ${YELLOW}No checkpoints saved yet${NC}"
    fi
    echo

    # ========================================
    # Sample Images
    # ========================================
    sample_dir="$OUTPUT_DIR/sample"
    if [[ -d "$sample_dir" ]]; then
        sample_count=$(find "$sample_dir" -name "*.png" 2>/dev/null | wc -l)

        if [[ $sample_count -gt 0 ]]; then
            echo -e "${GREEN}═══ Sample Images ═══${NC}"
            echo -e "  Total samples: ${sample_count}"
            echo -e "${CYAN}  Latest 3:${NC}"
            ls -lt "$sample_dir"/*.png 2>/dev/null | head -3 | \
                awk '{printf "    %s %s  %s\n", $6, $7, $9}' | \
                sed 's|'"$sample_dir"'/||'
            echo
        fi
    fi

    # ========================================
    # Training Session Info
    # ========================================
    echo -e "${GREEN}═══ Session Info ═══${NC}"

    # Session uptime
    session_info=$(tmux list-sessions 2>/dev/null | grep "$SESSION_NAME")
    if [[ -n "$session_info" ]]; then
        echo -e "  Session: ${SESSION_NAME}"
        echo "$session_info" | sed 's/^/  /'
    fi
    echo

    # ========================================
    # Footer
    # ========================================
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  Refreshing every 5 seconds... Press Ctrl+C to exit${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"

    sleep 5
done
