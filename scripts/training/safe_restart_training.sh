#!/bin/bash
# Safe Training Restart Script with GPU Cleanup
# Combines Option A (GPU cleanup) + Option B (optimized config)
# Created: 2025-11-15

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KOHYA_ROOT="/mnt/c/AI_LLM_projects/kohya_ss"

# Training settings
SESSION_NAME="sdxl_luca_training_safe"
CONFIG_FILE="$REPO_ROOT/configs/training/sdxl_16gb_stable.toml"
OUTPUT_DIR="/mnt/data/ai_data/models/lora/luca/sdxl_trial1"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_banner() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Safe Training Restart with GPU Cleanup & Optimization     ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo
}

# ============================================================================
# Step 1: Cleanup Old Sessions
# ============================================================================

cleanup_old_sessions() {
    log_info "Step 1: Cleaning up old training sessions..."

    # Kill old SDXL training sessions
    for session in sdxl_luca_training sdxl_luca_training_safe; do
        if tmux has-session -t "$session" 2>/dev/null; then
            log_warning "Killing existing session: $session"
            tmux kill-session -t "$session"
        fi
    done

    log_success "Old sessions cleaned"
}

# ============================================================================
# Step 2: Wait for GPU to Clear
# ============================================================================

wait_for_gpu_clear() {
    log_info "Step 2: Waiting for GPU to completely clear..."

    local max_wait=60
    local waited=0

    while [[ $waited -lt $max_wait ]]; do
        local gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)

        if [[ $gpu_procs -eq 0 ]]; then
            log_success "GPU cleared (no processes running)"
            return 0
        fi

        echo -n "."
        sleep 2
        waited=$((waited + 2))
    done

    log_warning "GPU still has processes after ${max_wait}s, continuing anyway"
}

# ============================================================================
# Step 3: Check Latest Checkpoint
# ============================================================================

check_latest_checkpoint() {
    log_info "Step 3: Checking for existing checkpoints..."

    local latest=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.safetensors" -type f -printf '%T@ %p\n' 2>/dev/null | \
        sort -rn | head -1 | awk '{print $2}')

    if [[ -n "$latest" ]]; then
        log_success "Found latest checkpoint: $(basename "$latest")"
        echo "$latest"
        return 0
    else
        log_warning "No checkpoints found, will start from scratch"
        echo ""
        return 1
    fi
}

# ============================================================================
# Step 4: Verify Optimized Config
# ============================================================================

verify_config() {
    log_info "Step 4: Verifying optimized configuration..."

    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Config file not found: $CONFIG_FILE"
        log_info "Please run the configuration creation first"
        return 1
    fi

    # Display key settings
    log_info "Configuration settings:"
    grep -E "gradient_checkpointing|gradient_accumulation_steps|train_batch_size|max_train_epochs" "$CONFIG_FILE" | \
        sed 's/^/  /'

    log_success "Configuration verified"
}

# ============================================================================
# Step 5: Create Training Session
# ============================================================================

start_training() {
    log_info "Step 5: Starting training in tmux session..."

    # Create new session
    tmux new-session -d -s "$SESSION_NAME"

    # Build training command
    local train_cmd="cd '$KOHYA_ROOT' && "
    train_cmd+="conda run -n kohya_ss "
    train_cmd+="accelerate launch --num_cpu_threads_per_process=2 "
    train_cmd+="./sd-scripts/sdxl_train_network.py "
    train_cmd+="--config_file='$CONFIG_FILE'"

    # Send command to session
    tmux send-keys -t "$SESSION_NAME" "$train_cmd" C-m

    log_success "Training started in session: $SESSION_NAME"
    log_info "To view training:"
    echo -e "  ${CYAN}tmux attach -t $SESSION_NAME${NC}"
    echo -e "  ${CYAN}(Press Ctrl+B then D to detach)${NC}"
}

# ============================================================================
# Step 6: Launch Monitor
# ============================================================================

launch_monitor() {
    local monitor_script="$REPO_ROOT/scripts/monitoring/training_health_monitor.sh"

    if [[ ! -f "$monitor_script" ]]; then
        log_warning "Monitor script not found, skipping"
        return 1
    fi

    log_info "Step 6: Launching health monitor..."

    # Start monitor in background
    nohup bash "$monitor_script" \
        --session "$SESSION_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --interval 300 \
        --max-restarts 3 \
        > "$REPO_ROOT/logs/training_monitor/monitor_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

    local monitor_pid=$!
    log_success "Monitor started (PID: $monitor_pid)"
    log_info "Monitor logs: $REPO_ROOT/logs/training_monitor/"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    print_banner

    # Execute steps
    cleanup_old_sessions
    wait_for_gpu_clear
    check_latest_checkpoint
    verify_config || exit 1

    echo
    log_info "Ready to start training"
    log_warning "This will use the STABLE configuration (no gradient_checkpointing)"
    echo
    read -p "Continue? (y/n): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cancelled by user"
        exit 0
    fi

    start_training
    sleep 5
    launch_monitor

    echo
    log_success "========================================="
    log_success "Training Successfully Restarted!"
    log_success "========================================="
    log_info "Session: $SESSION_NAME"
    log_info "Config: $CONFIG_FILE"
    log_info "Output: $OUTPUT_DIR"
    echo
    log_info "Monitor commands:"
    echo -e "  ${CYAN}# View training${NC}"
    echo -e "  ${CYAN}tmux attach -t $SESSION_NAME${NC}"
    echo -e ""
    echo -e "  ${CYAN}# Check GPU${NC}"
    echo -e "  ${CYAN}watch -n 5 nvidia-smi${NC}"
    echo -e ""
    echo -e "  ${CYAN}# View checkpoints${NC}"
    echo -e "  ${CYAN}ls -lht $OUTPUT_DIR/*.safetensors${NC}"
    echo
}

main
