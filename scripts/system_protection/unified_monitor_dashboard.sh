#!/bin/bash
# Unified System Monitor Dashboard for AI Training
# Shows RAM, GPU, Training Status, and OOM Protection Status

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Get status with color
get_status_icon() {
    local value=$1
    local warn=$2
    local crit=$3

    if [ "$value" -lt "$warn" ]; then
        echo "✓"
    elif [ "$value" -lt "$crit" ]; then
        echo "⚠"
    else
        echo "🚨"
    fi
}

get_status_color() {
    local value=$1
    local warn=$2
    local crit=$3

    if [ "$value" -lt "$warn" ]; then
        echo "$GREEN"
    elif [ "$value" -lt "$crit" ]; then
        echo "$YELLOW"
    else
        echo "$RED"
    fi
}

while true; do
    clear

    echo -e "${BOLD}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║          AI Training System - Unified Monitoring Dashboard                     ║${NC}"
    echo -e "${BOLD}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # ========================================
    # SYSTEM MEMORY
    # ========================================
    echo -e "${BOLD}${CYAN}💾 SYSTEM MEMORY${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    MEM_INFO=$(free -h | awk '/^Mem:/ {print $2,$3,$7}')
    read -r MEM_TOTAL MEM_USED MEM_AVAIL <<< "$MEM_INFO"
    MEM_PCT=$(free | awk '/^Mem:/ {printf "%.0f", ($3/$2)*100}')
    SWAP_PCT=$(free | awk '/^Swap:/ {if ($2>0) printf "%.0f", ($3/$2)*100; else print "0"}')

    MEM_COLOR=$(get_status_color $MEM_PCT 80 90)
    MEM_ICON=$(get_status_icon $MEM_PCT 80 90)
    SWAP_COLOR=$(get_status_color $SWAP_PCT 50 80)
    SWAP_ICON=$(get_status_icon $SWAP_PCT 50 80)

    printf "RAM:  ${MEM_COLOR}%-4s${NC} │ Used: %6s / %6s │ Available: %6s │ Usage: ${MEM_COLOR}%3d%%${NC}\n" \
           "$MEM_ICON" "$MEM_USED" "$MEM_TOTAL" "$MEM_AVAIL" "$MEM_PCT"

    SWAP_INFO=$(free -h | awk '/^Swap:/ {print $2,$3}')
    read -r SWAP_TOTAL SWAP_USED <<< "$SWAP_INFO"
    printf "Swap: ${SWAP_COLOR}%-4s${NC} │ Used: %6s / %6s │                    │ Usage: ${SWAP_COLOR}%3d%%${NC}\n" \
           "$SWAP_ICON" "$SWAP_USED" "$SWAP_TOTAL" "$SWAP_PCT"

    # Memory bar
    BAR_LEN=60
    MEM_FILLED=$((MEM_PCT * BAR_LEN / 100))
    MEM_BAR=$(printf "%${MEM_FILLED}s" | tr ' ' '█')
    MEM_EMPTY=$(printf "%$((BAR_LEN - MEM_FILLED))s" | tr ' ' '░')
    echo -e "      [${MEM_COLOR}${MEM_BAR}${NC}${MEM_EMPTY}]"

    echo ""

    # ========================================
    # GPU STATUS
    # ========================================
    echo -e "${BOLD}${MAGENTA}🎮 GPU STATUS${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)

        while IFS=',' read -r gpu_id gpu_name temp util mem_used mem_total; do
            gpu_name=$(echo $gpu_name | xargs)
            temp=$(echo $temp | xargs)
            util=$(echo $util | xargs)
            mem_used=$(echo $mem_used | xargs)
            mem_total=$(echo $mem_total | xargs)
            mem_pct=$(awk "BEGIN {printf \"%.0f\", ($mem_used/$mem_total)*100}")

            # Color coding
            TEMP_COLOR=$(get_status_color $temp 70 85)
            TEMP_ICON=$(get_status_icon $temp 70 85)
            UTIL_COLOR=$(get_status_color $util 90 97)
            GMEM_COLOR=$(get_status_color $mem_pct 85 95)
            GMEM_ICON=$(get_status_icon $mem_pct 85 95)

            printf "GPU %s: %s\n" "$gpu_id" "$gpu_name"
            printf "  Temp: ${TEMP_COLOR}%-4s %3d°C${NC} │ Utilization: ${UTIL_COLOR}%3d%%${NC} │ Memory: ${GMEM_COLOR}%-4s %5dMB/%5dMB (%3d%%)${NC}\n" \
                   "$TEMP_ICON" "$temp" "$util" "$GMEM_ICON" "$mem_used" "$mem_total" "$mem_pct"

            # GPU memory bar
            GPU_FILLED=$((mem_pct * BAR_LEN / 100))
            GPU_BAR=$(printf "%${GPU_FILLED}s" | tr ' ' '█')
            GPU_EMPTY=$(printf "%$((BAR_LEN - GPU_FILLED))s" | tr ' ' '░')
            echo -e "         [${GMEM_COLOR}${GPU_BAR}${NC}${GPU_EMPTY}]"

        done <<< "$GPU_INFO"
    else
        echo "  No NVIDIA GPU detected"
    fi

    echo ""

    # ========================================
    # TRAINING PROCESSES
    # ========================================
    echo -e "${BOLD}${YELLOW}🚀 TRAINING PROCESSES${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    TRAIN_PROCS=$(ps aux --sort=-%mem | grep -E "python.*(train|kohya|accelerate)" | grep -v grep | head -5)

    if [ -n "$TRAIN_PROCS" ]; then
        echo "$TRAIN_PROCS" | awk '{
            printf "  %-8s │ CPU: %5s%% │ MEM: %5s%% │ TIME: %8s │ %s\n",
                   $2, $3, $4, $10, substr($0, index($0,$11))
        }' | head -5
    else
        echo "  No active training processes"
    fi

    echo ""

    # ========================================
    # OOM PROTECTION STATUS
    # ========================================
    echo -e "${BOLD}${GREEN}🛡️  OOM PROTECTION STATUS${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check watchdogs
    MEM_WATCHDOG=$(pgrep -f memory_watchdog.sh)
    GPU_WATCHDOG=$(pgrep -f gpu_watchdog.sh)

    if [ -n "$MEM_WATCHDOG" ]; then
        echo -e "  ${GREEN}✓${NC} Memory Watchdog:  Running (PID: $MEM_WATCHDOG)"
    else
        echo -e "  ${RED}✗${NC} Memory Watchdog:  Not running"
    fi

    if [ -n "$GPU_WATCHDOG" ]; then
        echo -e "  ${GREEN}✓${NC} GPU Watchdog:     Running (PID: $GPU_WATCHDOG)"
    else
        echo -e "  ${RED}✗${NC} GPU Watchdog:     Not running"
    fi

    # Check OOM scores for training processes
    TRAIN_PIDS=$(pgrep -f "python.*train")
    if [ -n "$TRAIN_PIDS" ]; then
        echo -e "  ${GREEN}✓${NC} Training OOM Protection: Active"
        for pid in $TRAIN_PIDS; do
            if [ -f /proc/$pid/oom_score_adj ]; then
                SCORE=$(cat /proc/$pid/oom_score_adj 2>/dev/null)
                if [ "$SCORE" -lt 0 ]; then
                    echo -e "      PID $pid: ${GREEN}Protected (score: $SCORE)${NC}"
                fi
            fi
        done
    fi

    echo ""

    # ========================================
    # TOP MEMORY CONSUMERS
    # ========================================
    echo -e "${BOLD}${BLUE}📊 TOP MEMORY CONSUMERS${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    ps aux --sort=-%mem | head -6 | tail -5 | awk '{
        cmd = $11
        for(i=12; i<=NF && length(cmd)<50; i++) cmd = cmd " " $i
        if(length(cmd) > 50) cmd = substr(cmd, 1, 47) "..."
        printf "  %5s%% │ %7s MB │ %-50s\n", $4, int($6/1024), cmd
    }'

    echo ""

    # ========================================
    # RECENT ALERTS
    # ========================================
    echo -e "${BOLD}${YELLOW}⚠️  RECENT ALERTS (Last 5)${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f /tmp/memory_watchdog.log ]; then
        tail -5 /tmp/memory_watchdog.log 2>/dev/null | sed 's/^/  /' || echo "  No alerts"
    else
        echo "  No alert log found"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') │ Refreshing every 5s │ Ctrl+C to exit"
    echo ""

    sleep 5
done
