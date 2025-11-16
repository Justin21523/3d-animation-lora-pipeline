#!/bin/bash
# Batch Frame Extraction Script
# Processes multiple 3D animation films in parallel using tmux sessions
# Created: 2025-11-15

set -euo pipefail

# ============================================================================
# Configuration & Defaults
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXTRACTOR_SCRIPT="$REPO_ROOT/scripts/generic/video/universal_frame_extractor.py"

# Default settings
DEFAULT_WORKERS_PER_PROJECT=12
DEFAULT_MODE="hybrid"
DEFAULT_RETRY=3
DEFAULT_SCENE_THRESHOLD=30.0
DEFAULT_FRAMES_PER_SCENE=10
DEFAULT_INTERVAL_SECONDS=2.0
DEFAULT_QUALITY="high"

# Paths
RAW_VIDEOS_DIR="/mnt/c/raw_videos"
WAREHOUSE_ROOT="/mnt/data/ai_data"
DATASETS_DIR="$WAREHOUSE_ROOT/datasets/3d-anime"
LOG_DIR="$REPO_ROOT/logs/frame_extraction"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_banner() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║        Batch Frame Extraction for 3D Animation Pipeline       ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo
}

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

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Batch frame extraction from multiple 3D animation films using tmux sessions.

OPTIONS:
    --projects PROJECTS         Comma-separated list of project names (required)
                               Example: "onward,orion"

    --workers-per-project N     Number of workers per project (default: $DEFAULT_WORKERS_PER_PROJECT)

    --mode MODE                 Extraction mode: scene, interval, or hybrid (default: $DEFAULT_MODE)

    --scene-threshold N         Scene detection threshold for scene/hybrid modes (default: $DEFAULT_SCENE_THRESHOLD)

    --frames-per-scene N        Frames to extract per scene (default: $DEFAULT_FRAMES_PER_SCENE)

    --interval-seconds N        Interval in seconds for interval/hybrid modes (default: $DEFAULT_INTERVAL_SECONDS)

    --quality QUALITY           JPEG quality: low (85), medium (90), high (95) (default: $DEFAULT_QUALITY)

    --retry N                   Number of retry attempts on failure (default: $DEFAULT_RETRY)

    --force                     Force re-processing even if results exist

    --monitor                   Launch monitoring session after starting extractions

    --dry-run                   Print configuration without executing

    -h, --help                  Show this help message

EXAMPLES:
    # Basic usage with two projects
    $0 --projects "onward,orion"

    # Custom workers and monitoring
    $0 --projects "onward,orion" --workers-per-project 16 --monitor

    # Scene-based extraction with custom threshold
    $0 --projects "onward" --mode scene --scene-threshold 25.0

    # Force re-processing with retry
    $0 --projects "orion" --force --retry 5

NOTES:
    - Each project runs in a separate tmux session: frame_extraction_{project}
    - Logs are saved to: $LOG_DIR/{project}_{timestamp}.log
    - Video files must be in: $RAW_VIDEOS_DIR/{project}/
    - Supported formats: .mp4, .mkv, .avi, .ts, .m2ts, .mov, .wmv, .flv
    - CPU usage is automatically monitored and adjusted if needed

EOF
}

# ============================================================================
# Resource Management
# ============================================================================

get_cpu_cores() {
    nproc 2>/dev/null || echo "4"
}

get_cpu_usage() {
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d% -f1
}

calculate_optimal_workers() {
    local total_cores=$1
    local num_projects=$2
    local requested_workers=$3

    # Calculate based on 75% CPU usage (leave 25% for system + GPU tasks)
    local available_cores=$((total_cores * 3 / 4))
    local optimal_per_project=$((available_cores / num_projects))

    # Use minimum of requested and optimal
    if [[ $requested_workers -le $optimal_per_project ]]; then
        echo "$requested_workers"
    else
        # Send warning to stderr to avoid contaminating the return value
        log_warning "Requested $requested_workers workers exceeds optimal $optimal_per_project, using optimal" >&2
        echo "$optimal_per_project"
    fi
}

check_memory_available() {
    local free_mem_gb=$(free -g | awk '/^Mem:/{print $7}')
    if [[ $free_mem_gb -lt 8 ]]; then
        log_warning "Low available memory: ${free_mem_gb}GB. Consider reducing workers."
        return 1
    fi
    return 0
}

# ============================================================================
# Video Discovery
# ============================================================================

find_video_directory() {
    local project=$1
    local project_dir="$RAW_VIDEOS_DIR/$project"

    # Check if project directory exists
    if [[ ! -d "$project_dir" ]]; then
        log_error "Project directory not found: $project_dir"
        return 1
    fi

    # Check if directory contains video files
    local video_count=$(find "$project_dir" -maxdepth 1 -type f \( \
        -iname "*.mp4" -o \
        -iname "*.mkv" -o \
        -iname "*.avi" -o \
        -iname "*.ts" -o \
        -iname "*.m2ts" -o \
        -iname "*.mov" -o \
        -iname "*.wmv" -o \
        -iname "*.flv" \
    \) | wc -l)

    if [[ $video_count -eq 0 ]]; then
        log_error "No video files found in $project_dir"
        return 1
    fi

    echo "$project_dir"
    return 0
}

# ============================================================================
# Checkpoint & Resume
# ============================================================================

check_existing_results() {
    local output_dir=$1
    local results_file="$output_dir/extraction_results.json"

    if [[ -f "$results_file" ]]; then
        # Parse JSON to get completion status
        local total_frames=$(jq -r '.total_frames // 0' "$results_file" 2>/dev/null || echo "0")
        if [[ "$total_frames" -gt 0 ]]; then
            return 0  # Results exist
        fi
    fi
    return 1  # No valid results
}

# ============================================================================
# Main Extraction Function
# ============================================================================

run_extraction() {
    local project=$1
    local video_dir=$2
    local output_dir=$3
    local workers=$4
    local mode=$5
    local scene_threshold=$6
    local frames_per_scene=$7
    local interval_seconds=$8
    local jpeg_quality=$9
    local log_file="${10}"

    log_info "Starting extraction for $project"
    log_info "  Video directory: $video_dir"
    log_info "  Output: $output_dir"
    log_info "  Workers: $workers"
    log_info "  Mode: $mode"

    # Build command arguments
    local cmd="conda run -n ai_env python \"$EXTRACTOR_SCRIPT\""
    cmd="$cmd \"$video_dir\""  # Positional argument
    cmd="$cmd --output-dir \"$output_dir\""
    cmd="$cmd --mode \"$mode\""
    cmd="$cmd --workers $workers"
    cmd="$cmd --jpeg-quality $jpeg_quality"
    cmd="$cmd --episode-pattern none"  # Single film, preserve full filename

    # Add mode-specific parameters
    if [[ "$mode" == "scene" || "$mode" == "hybrid" ]]; then
        cmd="$cmd --scene-threshold $scene_threshold"
        cmd="$cmd --frames-per-scene $frames_per_scene"
    fi

    if [[ "$mode" == "interval" || "$mode" == "hybrid" ]]; then
        cmd="$cmd --interval-seconds $interval_seconds"
    fi

    # Execute with logging
    echo "Command: $cmd" >> "$log_file"
    echo "Started at: $(date)" >> "$log_file"

    if eval "$cmd" >> "$log_file" 2>&1; then
        echo "Completed at: $(date)" >> "$log_file"
        return 0
    else
        echo "Failed at: $(date)" >> "$log_file"
        return 1
    fi
}

# ============================================================================
# Tmux Session Management
# ============================================================================

create_tmux_session() {
    local session_name=$1
    local project=$2
    local video_dir=$3
    local output_dir=$4
    local workers=$5
    local mode=$6
    local scene_threshold=$7
    local frames_per_scene=$8
    local interval_seconds=$9
    local jpeg_quality=${10}
    local log_file=${11}
    local max_retries=${12}

    # Check if session already exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        log_warning "Tmux session '$session_name' already exists. Attaching to existing session."
        return 0
    fi

    # Create new detached session
    tmux new-session -d -s "$session_name"

    # Build command with optional parameters
    local cmd_params="--scene-threshold $scene_threshold --frames-per-scene $frames_per_scene"
    if [[ "$mode" == "interval" || "$mode" == "hybrid" ]]; then
        cmd_params="$cmd_params --interval-seconds $interval_seconds"
    fi

    # Change to repo directory
    tmux send-keys -t "$session_name" "cd '$REPO_ROOT'" C-m

    # Start retry loop
    tmux send-keys -t "$session_name" "for attempt in \$(seq 1 $max_retries); do" C-m
    tmux send-keys -t "$session_name" "  echo '========================================='" C-m
    tmux send-keys -t "$session_name" "  echo \"Attempt \\\$attempt of $max_retries\"" C-m
    tmux send-keys -t "$session_name" "  echo '========================================='" C-m

    # Execute extraction command
    tmux send-keys -t "$session_name" "  if conda run -n ai_env python '$EXTRACTOR_SCRIPT' '$video_dir' --output-dir '$output_dir' --mode '$mode' --workers $workers --jpeg-quality $jpeg_quality --episode-pattern none $cmd_params; then" C-m
    tmux send-keys -t "$session_name" "    echo '✓ Extraction completed successfully'" C-m
    tmux send-keys -t "$session_name" "    break" C-m
    tmux send-keys -t "$session_name" "  else" C-m
    tmux send-keys -t "$session_name" "    echo \"✗ Attempt \\\$attempt failed\"" C-m
    tmux send-keys -t "$session_name" "    if [[ \\\$attempt -lt $max_retries ]]; then" C-m
    tmux send-keys -t "$session_name" "      echo 'Waiting 30 seconds before retry...'" C-m
    tmux send-keys -t "$session_name" "      sleep 30" C-m
    tmux send-keys -t "$session_name" "    else" C-m
    tmux send-keys -t "$session_name" "      echo '✗ All retry attempts exhausted'" C-m
    tmux send-keys -t "$session_name" "    fi" C-m
    tmux send-keys -t "$session_name" "  fi" C-m
    tmux send-keys -t "$session_name" "done" C-m
    tmux send-keys -t "$session_name" "echo 'Press Enter to close session...'" C-m
    tmux send-keys -t "$session_name" "read" C-m

    log_success "Created tmux session: $session_name"
}

kill_extraction_sessions() {
    local projects=("$@")

    for project in "${projects[@]}"; do
        local session_name="frame_extraction_$project"
        if tmux has-session -t "$session_name" 2>/dev/null; then
            tmux kill-session -t "$session_name"
            log_info "Killed session: $session_name"
        fi
    done
}

# ============================================================================
# Main Processing Logic
# ============================================================================

process_projects() {
    local projects=("$@")
    local num_projects=${#projects[@]}
    local total_cores=$(get_cpu_cores)

    log_info "System resources:"
    log_info "  CPU cores: $total_cores ($(($total_cores / 2)) physical)"
    log_info "  Requested workers per project: $WORKERS_PER_PROJECT"

    # Calculate optimal workers
    WORKERS_PER_PROJECT=$(calculate_optimal_workers "$total_cores" "$num_projects" "$WORKERS_PER_PROJECT")
    log_info "  Adjusted workers per project: $WORKERS_PER_PROJECT"

    # Check memory
    check_memory_available || log_warning "Proceed with caution due to low memory"

    # Create log directory
    mkdir -p "$LOG_DIR"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local success_count=0
    local skip_count=0
    local fail_count=0

    for project in "${projects[@]}"; do
        log_info "Processing project: $project"

        # Define paths
        local output_dir="$DATASETS_DIR/$project/frames"
        local log_file="$LOG_DIR/${project}_${timestamp}.log"

        # Check if already processed
        if [[ "$FORCE" != "true" ]] && check_existing_results "$output_dir"; then
            log_success "⏭️  Results already exist for $project, skipping (use --force to re-process)"
            skip_count=$((skip_count + 1))
            continue
        fi

        # Find video directory
        local video_dir
        if ! video_dir=$(find_video_directory "$project"); then
            log_error "Cannot process $project: video directory not found"
            fail_count=$((fail_count + 1))
            continue
        fi

        log_info "Found video directory: $video_dir"

        # Create output directory
        mkdir -p "$output_dir"

        # Dry run mode
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would create tmux session: frame_extraction_$project"
            log_info "[DRY RUN] Command: python $EXTRACTOR_SCRIPT \"$video_dir\" --output-dir \"$output_dir\" --mode $MODE --workers $WORKERS_PER_PROJECT --jpeg-quality 95"
            success_count=$((success_count + 1))
            continue
        fi

        # Convert quality to jpeg_quality value
        local jpeg_quality=95
        case "$QUALITY" in
            low) jpeg_quality=85 ;;
            medium) jpeg_quality=90 ;;
            high) jpeg_quality=95 ;;
            *) jpeg_quality=95 ;;
        esac

        # Create tmux session
        local session_name="frame_extraction_$project"
        if create_tmux_session \
            "$session_name" \
            "$project" \
            "$video_dir" \
            "$output_dir" \
            "$WORKERS_PER_PROJECT" \
            "$MODE" \
            "$SCENE_THRESHOLD" \
            "$FRAMES_PER_SCENE" \
            "$INTERVAL_SECONDS" \
            "$jpeg_quality" \
            "$log_file" \
            "$MAX_RETRY"; then

            success_count=$((success_count + 1))
            log_success "Started processing: $project"
            log_info "  Tmux session: $session_name"
            log_info "  Log file: $log_file"
        else
            fail_count=$((fail_count + 1))
            log_error "Failed to start processing: $project"
        fi

        # Small delay between session creation
        sleep 2
    done

    echo
    log_info "========================================="
    log_info "Batch Processing Summary"
    log_info "========================================="
    log_info "Total projects: $num_projects"
    log_success "Started: $success_count"
    log_warning "Skipped: $skip_count"
    log_error "Failed: $fail_count"
    echo

    if [[ $success_count -gt 0 ]] && [[ "$DRY_RUN" != "true" ]]; then
        log_info "Active tmux sessions:"
        for project in "${projects[@]}"; do
            local session_name="frame_extraction_$project"
            if tmux has-session -t "$session_name" 2>/dev/null; then
                echo "  • $session_name"
            fi
        done
        echo
        log_info "To attach to a session:"
        echo "  tmux attach -t frame_extraction_{project}"
        echo
        log_info "To list all sessions:"
        echo "  tmux ls"
        echo
        log_info "Log files location:"
        echo "  $LOG_DIR"
        echo
    fi
}

# ============================================================================
# Argument Parsing
# ============================================================================

PROJECTS=""
WORKERS_PER_PROJECT=$DEFAULT_WORKERS_PER_PROJECT
MODE=$DEFAULT_MODE
SCENE_THRESHOLD=$DEFAULT_SCENE_THRESHOLD
FRAMES_PER_SCENE=$DEFAULT_FRAMES_PER_SCENE
INTERVAL_SECONDS=$DEFAULT_INTERVAL_SECONDS
QUALITY=$DEFAULT_QUALITY
MAX_RETRY=$DEFAULT_RETRY
FORCE="false"
MONITOR="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --projects)
            PROJECTS="$2"
            shift 2
            ;;
        --workers-per-project)
            WORKERS_PER_PROJECT="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --scene-threshold)
            SCENE_THRESHOLD="$2"
            shift 2
            ;;
        --frames-per-scene)
            FRAMES_PER_SCENE="$2"
            shift 2
            ;;
        --interval-seconds)
            INTERVAL_SECONDS="$2"
            shift 2
            ;;
        --quality)
            QUALITY="$2"
            shift 2
            ;;
        --retry)
            MAX_RETRY="$2"
            shift 2
            ;;
        --force)
            FORCE="true"
            shift
            ;;
        --monitor)
            MONITOR="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    print_banner

    # Validate required arguments
    if [[ -z "$PROJECTS" ]]; then
        log_error "Missing required argument: --projects"
        echo
        print_usage
        exit 1
    fi

    # Convert comma-separated projects to array
    IFS=',' read -ra PROJECT_ARRAY <<< "$PROJECTS"

    # Validate mode
    if [[ "$MODE" != "scene" && "$MODE" != "interval" && "$MODE" != "hybrid" ]]; then
        log_error "Invalid mode: $MODE (must be scene, interval, or hybrid)"
        exit 1
    fi

    # Process projects
    process_projects "${PROJECT_ARRAY[@]}"

    # Launch monitoring if requested
    if [[ "$MONITOR" == "true" ]] && [[ "$DRY_RUN" != "true" ]]; then
        local monitor_script="$REPO_ROOT/scripts/monitoring/monitor_frame_extraction.sh"
        if [[ -f "$monitor_script" ]]; then
            log_info "Launching monitoring script..."
            bash "$monitor_script" --projects "$PROJECTS" &
        else
            log_warning "Monitoring script not found: $monitor_script"
        fi
    fi
}

# Run main function
main
