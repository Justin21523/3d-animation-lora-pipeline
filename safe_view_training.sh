#!/bin/bash
# Safe Training Log Viewer
# Views training progress WITHOUT entering the tmux session
# Cannot accidentally interrupt training

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SESSION_NAME="sdxl_luca_training_safe"

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║            Safe Training Log Viewer (Read-Only)               ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "${GREEN}✓ This viewer is SAFE - cannot interrupt training${NC}"
echo -e "${YELLOW}Press Ctrl+C to exit viewer (training continues)${NC}"
echo
echo -e "Monitoring session: ${CYAN}${SESSION_NAME}${NC}"
echo -e "Refreshing every 3 seconds..."
echo
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo

while true; do
    # Clear screen
    tput clear

    # Header
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                  Training Progress (Read-Only)                ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC}"
    echo

    # Check if session exists
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${RED}✗ Training session NOT running!${NC}"
        echo -e "Session name: $SESSION_NAME"
        sleep 5
        continue
    fi

    # Capture and display training output
    tmux capture-pane -t "$SESSION_NAME" -p 2>/dev/null | tail -35

    echo
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Refreshing every 3 seconds... Press Ctrl+C to exit (training continues)${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"

    sleep 3
done
