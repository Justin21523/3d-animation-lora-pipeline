#!/bin/bash
# Training Health Monitor & Auto-Recovery System
# Monitors GPU training processes and automatically handles failures
# Created: 2025-11-15

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Monitoring settings
CHECK_INTERVAL=300  # Check every 5 minutes
MAX_GPU_TEMP=85     # Maximum safe GPU temperature (°C)
MAX_VRAM_USAGE=95   # Maximum VRAM usage percentage
MIN_GPU_UTIL=5      # Minimum GPU utilization to consider "alive"
HANG_TIMEOUT=1800   # Consider hung if no progress for 30 minutes

# Auto-recovery settings
ENABLE_AUTO_RESTART=true
MAX_RESTART_ATTEMPTS=3
RESTART_DELAY=60    # Wait 60 seconds before restart

# Notification settings
ENABLE_NOTIFICATIONS=true
LOG_DIR="$REPO_ROOT/logs/training_monitor"

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
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} [INFO] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} [✓] $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} [⚠] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} [✗] $1" | tee -a "$LOG_FILE"
}

send_notification() {
    local level=$1
    local message=$2

    # Log to file
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" >> "$LOG_DIR/notifications.log"

    # Desktop notification (if available)
    if [[ "$ENABLE_NOTIFICATIONS" == "true" ]] && command -v notify-send &> /dev/null; then
        notify-send "Training Monitor - $level" "$message"
    fi
}

# ============================================================================
# GPU Health Check Functions
# ============================================================================

get_gpu_temperature() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null || echo "0"
}

get_gpu_utilization() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/ %//'
}

get_gpu_memory_usage() {
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
        awk '{printf "%.1f", ($1/$2)*100}'
}

get_gpu_power_draw() {
    nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null
}

check_gpu_health() {
    local temp=$(get_gpu_temperature)
    local util=$(get_gpu_utilization)
    local vram=$(get_gpu_memory_usage)

    local health_status="OK"
    local warnings=""

    # Temperature check
    if (( $(echo "$temp > $MAX_GPU_TEMP" | bc -l) )); then
        health_status="WARNING"
        warnings="$warnings Temperature too high: ${temp}°C (max: ${MAX_GPU_TEMP}°C). "
    fi

    # VRAM check
    if (( $(echo "$vram > $MAX_VRAM_USAGE" | bc -l) )); then
        health_status="WARNING"
        warnings="$warnings VRAM usage too high: ${vram}% (max: ${MAX_VRAM_USAGE}%). "
    fi

    # Utilization check (detect if training is stuck)
    if (( $(echo "$util < $MIN_GPU_UTIL" | bc -l) )); then
        health_status="WARNING"
        warnings="$warnings GPU utilization too low: ${util}% (min: ${MIN_GPU_UTIL}%). "
    fi

    echo "$health_status|$warnings|$temp|$util|$vram"
}

# ============================================================================
# Training Process Detection
# ============================================================================

find_training_processes() {
    # Look for common training process patterns
    ps aux | grep -E "train_network\.py|sdxl_train|train\.py|accelerate launch" | \
        grep -v grep | \
        awk '{print $2}'
}

get_process_info() {
    local pid=$1

    if [[ ! -d "/proc/$pid" ]]; then
        echo "NOT_RUNNING"
        return 1
    fi

    # Get process command
    local cmd=$(ps -p "$pid" -o cmd= 2>/dev/null || echo "")

    # Get CPU and memory usage
    local stats=$(ps -p "$pid" -o %cpu,%mem,etime 2>/dev/null | tail -1)

    echo "$cmd|$stats"
}

# ============================================================================
# Checkpoint Detection
# ============================================================================

find_latest_checkpoint() {
    local output_dir=$1

    # Find most recent .safetensors file
    local latest=$(find "$output_dir" -maxdepth 1 -name "*.safetensors" -type f -printf '%T@ %p\n' 2>/dev/null | \
        sort -rn | head -1 | awk '{print $2}')

    if [[ -n "$latest" ]]; then
        echo "$latest"
        return 0
    else
        echo ""
        return 1
    fi
}

get_checkpoint_age() {
    local checkpoint=$1

    if [[ ! -f "$checkpoint" ]]; then
        echo "-1"
        return 1
    fi

    local now=$(date +%s)
    local mtime=$(stat -c %Y "$checkpoint" 2>/dev/null || echo "0")
    local age=$((now - mtime))

    echo "$age"
}

# ============================================================================
# Hang Detection
# ============================================================================

check_training_progress() {
    local output_dir=$1
    local last_checkpoint_file="$LOG_DIR/last_checkpoint_time.txt"

    # Find latest checkpoint
    local latest_checkpoint=$(find_latest_checkpoint "$output_dir")

    if [[ -z "$latest_checkpoint" ]]; then
        log_warning "No checkpoints found in $output_dir"
        return 1
    fi

    # Get checkpoint age
    local age=$(get_checkpoint_age "$latest_checkpoint")

    # Check if training is hung (no new checkpoint for HANG_TIMEOUT seconds)
    if [[ $age -gt $HANG_TIMEOUT ]]; then
        log_error "Training appears hung: No checkpoint for $((age/60)) minutes"
        log_error "Latest checkpoint: $(basename "$latest_checkpoint")"
        return 2  # Hung
    else
        log_success "Training active: Latest checkpoint $(basename "$latest_checkpoint") is $((age/60)) minutes old"
        return 0  # Active
    fi
}

# ============================================================================
# Auto-Recovery Functions
# ============================================================================

restart_training() {
    local session_name=$1
    local training_script=$2
    local config_file=$3
    local output_dir=$4

    log_info "Attempting to restart training in session: $session_name"

    # Find latest checkpoint for resume
    local latest_checkpoint=$(find_latest_checkpoint "$output_dir")

    if [[ -n "$latest_checkpoint" ]]; then
        log_info "Found checkpoint for resume: $latest_checkpoint"
    fi

    # Kill existing session if it exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        log_info "Killing existing session: $session_name"
        tmux kill-session -t "$session_name"
        sleep 5
    fi

    # Clear CUDA cache by waiting
    log_info "Waiting ${RESTART_DELAY}s for GPU to clear..."
    sleep "$RESTART_DELAY"

    # Create new training session
    log_info "Creating new training session..."

    tmux new-session -d -s "$session_name"

    # Build training command
    local train_cmd="cd '$REPO_ROOT' && conda run -n kohya_ss accelerate launch"

    if [[ -n "$latest_checkpoint" ]]; then
        # TODO: Add resume logic here
        # This depends on your training script's resume capability
        train_cmd="$train_cmd $training_script --config_file=$config_file"
    else
        train_cmd="$train_cmd $training_script --config_file=$config_file"
    fi

    tmux send-keys -t "$session_name" "$train_cmd" C-m

    log_success "Training restarted in session: $session_name"
    send_notification "SUCCESS" "Training auto-restarted: $session_name"
}

# ============================================================================
# Main Monitoring Loop
# ============================================================================

monitor_training() {
    local session_name=${1:-"sdxl_luca_training"}
    local output_dir=${2:-"/mnt/data/ai_data/models/lora/luca/sdxl_trial1"}
    local training_script=${3:-"./sd-scripts/sdxl_train_network.py"}
    local config_file=${4:-"/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/sdxl_16gb_optimized.toml"}

    local restart_count=0

    log_info "========================================="
    log_info "Training Health Monitor Started"
    log_info "========================================="
    log_info "Session: $session_name"
    log_info "Output: $output_dir"
    log_info "Check interval: ${CHECK_INTERVAL}s"
    log_info "Auto-restart: $ENABLE_AUTO_RESTART (max: $MAX_RESTART_ATTEMPTS)"
    log_info "========================================="

    while true; do
        log_info "Running health check..."

        # Check GPU health
        local gpu_health=$(check_gpu_health)
        IFS='|' read -r health_status warnings temp util vram <<< "$gpu_health"

        log_info "GPU Status: Temp=${temp}°C, Util=${util}%, VRAM=${vram}%"

        if [[ "$health_status" == "WARNING" ]]; then
            log_warning "GPU health warning: $warnings"
            send_notification "WARNING" "GPU health issue: $warnings"
        fi

        # Check if session exists
        if ! tmux has-session -t "$session_name" 2>/dev/null; then
            log_error "Training session not found: $session_name"

            if [[ "$ENABLE_AUTO_RESTART" == "true" ]] && [[ $restart_count -lt $MAX_RESTART_ATTEMPTS ]]; then
                log_warning "Attempting auto-restart (attempt $((restart_count + 1))/$MAX_RESTART_ATTEMPTS)"
                restart_training "$session_name" "$training_script" "$config_file" "$output_dir"
                restart_count=$((restart_count + 1))
            else
                log_error "Max restart attempts reached or auto-restart disabled"
                send_notification "ERROR" "Training failed and cannot auto-restart"
                break
            fi
        else
            # Check training progress
            check_training_progress "$output_dir"
            local progress_status=$?

            if [[ $progress_status -eq 2 ]]; then
                # Training is hung
                log_error "Training detected as HUNG"

                if [[ "$ENABLE_AUTO_RESTART" == "true" ]] && [[ $restart_count -lt $MAX_RESTART_ATTEMPTS ]]; then
                    log_warning "Restarting hung training (attempt $((restart_count + 1))/$MAX_RESTART_ATTEMPTS)"
                    restart_training "$session_name" "$training_script" "$config_file" "$output_dir"
                    restart_count=$((restart_count + 1))
                fi
            elif [[ $progress_status -eq 0 ]]; then
                # Training is active - reset restart count
                restart_count=0
            fi
        fi

        # Wait for next check
        log_info "Next check in ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
    done
}

# ============================================================================
# Usage & Argument Parsing
# ============================================================================

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Monitors GPU training processes and provides automatic recovery.

OPTIONS:
    --session NAME          Tmux session name (default: sdxl_luca_training)
    --output-dir DIR        Training output directory
    --training-script PATH  Training script path
    --config-file PATH      Training config file path
    --interval SECONDS      Check interval (default: $CHECK_INTERVAL)
    --max-restarts N        Maximum restart attempts (default: $MAX_RESTART_ATTEMPTS)
    --disable-auto-restart  Disable automatic restart
    -h, --help             Show this help message

EXAMPLES:
    # Basic monitoring
    $0 --session sdxl_luca_training

    # Custom interval
    $0 --interval 600

    # Monitor without auto-restart
    $0 --disable-auto-restart

EOF
}

# Defaults
SESSION_NAME="sdxl_luca_training"
OUTPUT_DIR="/mnt/data/ai_data/models/lora/luca/sdxl_trial1"
TRAINING_SCRIPT="./sd-scripts/sdxl_train_network.py"
CONFIG_FILE="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/sdxl_16gb_optimized.toml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --session)
            SESSION_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --training-script)
            TRAINING_SCRIPT="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        --max-restarts)
            MAX_RESTART_ATTEMPTS="$2"
            shift 2
            ;;
        --disable-auto-restart)
            ENABLE_AUTO_RESTART=false
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Main Entry Point
# ============================================================================

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/monitor_$(date +%Y%m%d_%H%M%S).log"

# Start monitoring
monitor_training "$SESSION_NAME" "$OUTPUT_DIR" "$TRAINING_SCRIPT" "$CONFIG_FILE"
