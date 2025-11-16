#!/bin/bash
# Quick Training Progress Monitor
# Usage: bash monitor_training_progress.sh
# Press Ctrl+C to exit

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

while true; do
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           SDXL LoRA Training Progress Monitor                 ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo

    # GPU Status
    echo -e "${GREEN}=== GPU Status ===${NC}"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader | \
        awk -F', ' '{printf "  GPU Util: %s | VRAM: %s / %s | Temp: %s | Power: %s\n", $1, $2, $3, $4, $5}'
    echo

    # System RAM
    echo -e "${GREEN}=== System RAM ===${NC}"
    free -h | awk 'NR==2{printf "  Used: %s / %s (%.1f%%) | Available: %s\n", $3, $2, $3*100/$2, $7}'

    # Check if swap is being used
    swap_used=$(free -h | awk 'NR==3{print $3}')
    if [[ "$swap_used" != "0B" ]]; then
        echo -e "  ${YELLOW}⚠ Swap in use: $swap_used${NC}"
    fi
    echo

    # Training Progress
    echo -e "${GREEN}=== Training Status ===${NC}"
    if tmux has-session -t sdxl_luca_training_safe 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Training session is running"

        # Try to extract progress from tmux
        latest_output=$(tmux capture-pane -t sdxl_luca_training_safe -p | grep -E "steps:|epoch" | tail -5)
        if [[ -n "$latest_output" ]]; then
            echo -e "${CYAN}  Latest training output:${NC}"
            echo "$latest_output" | sed 's/^/    /'
        fi
    else
        echo -e "  ${RED}✗${NC} Training session NOT running"
    fi
    echo

    # Checkpoints
    echo -e "${GREEN}=== Latest Checkpoints ===${NC}"
    ls -lht /mnt/data/ai_data/models/lora/luca/sdxl_trial1/*.safetensors 2>/dev/null | head -3 | \
        awk '{printf "  %s %s  %s  %s\n", $6, $7, $5, $9}' | \
        sed 's|/mnt/data/ai_data/models/lora/luca/sdxl_trial1/||'

    # Checkpoint age check
    latest_ckpt=$(find /mnt/data/ai_data/models/lora/luca/sdxl_trial1 -name "*.safetensors" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
    if [[ -n "$latest_ckpt" ]]; then
        ckpt_age=$(($(date +%s) - $(stat -c %Y "$latest_ckpt" 2>/dev/null)))
        ckpt_age_min=$((ckpt_age / 60))
        echo -e "  Latest checkpoint age: ${ckpt_age_min} minutes ago"

        if [[ $ckpt_age_min -gt 30 ]]; then
            echo -e "  ${YELLOW}⚠ Warning: No new checkpoint for >30 minutes${NC}"
        fi
    fi
    echo

    # Sample Images
    echo -e "${GREEN}=== Latest Sample Images ===${NC}"
    sample_count=$(find /mnt/data/ai_data/models/lora/luca/sdxl_trial1/sample -name "*.png" 2>/dev/null | wc -l)
    echo -e "  Total samples: $sample_count"

    if [[ $sample_count -gt 0 ]]; then
        ls -lt /mnt/data/ai_data/models/lora/luca/sdxl_trial1/sample/*.png 2>/dev/null | head -3 | \
            awk '{printf "  %s %s  %s\n", $6, $7, $9}' | \
            sed 's|/mnt/data/ai_data/models/lora/luca/sdxl_trial1/sample/||'
    fi
    echo

    # Top Memory Processes
    echo -e "${GREEN}=== Top 5 Memory Consumers ===${NC}"
    ps aux --sort=-%mem | head -6 | tail -5 | \
        awk '{printf "  %-8s %6s%%  %8.1f MB  %s\n", $1, $4, $6/1024, $11}'
    echo

    # Update info
    echo -e "${CYAN}Last updated: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${CYAN}Refreshing every 10 seconds... (Ctrl+C to exit)${NC}"

    sleep 10
done
