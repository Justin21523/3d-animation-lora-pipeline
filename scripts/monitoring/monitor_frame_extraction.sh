#!/bin/bash
# Frame Extraction Progress Monitoring Script
# Monitors active tmux sessions and displays real-time progress
# Created: 2025-11-15

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/frame_extraction"
DATASETS_DIR="/mnt/data/ai_data/datasets/3d-anime"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Refresh interval (seconds)
REFRESH_INTERVAL=10

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         Frame Extraction Progress Monitor                     ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${BLUE}Updated: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo
}

get_session_status() {
    local session_name=$1

    if tmux has-session -t "$session_name" 2>/dev/null; then
        # Check if any panes in session are still running
        local panes=$(tmux list-panes -t "$session_name" -F "#{pane_pid}" 2>/dev/null || echo "")
        if [[ -n "$panes" ]]; then
            echo "RUNNING"
        else
            echo "IDLE"
        fi
    else
        echo "NOT_FOUND"
    fi
}

parse_log_progress() {
    local log_file=$1

    if [[ ! -f "$log_file" ]]; then
        echo "0|0|N/A"
        return
    fi

    # Try to extract progress information from log
    # Look for patterns like "Processing frame 1234/5678" or similar
    local total_frames=$(grep -oP 'total.*?(\d+).*?frames' "$log_file" | tail -1 | grep -oP '\d+' || echo "0")
    local processed_frames=$(grep -oP 'Extracted.*?(\d+).*?frames' "$log_file" | tail -1 | grep -oP '\d+' || echo "0")

    # Calculate percentage
    local percentage="N/A"
    if [[ "$total_frames" -gt 0 ]]; then
        percentage=$(awk "BEGIN {printf \"%.1f\", ($processed_frames / $total_frames) * 100}")
    fi

    echo "$processed_frames|$total_frames|$percentage"
}

get_extraction_results() {
    local project=$1
    local results_file="$DATASETS_DIR/$project/frames/extraction_results.json"

    if [[ ! -f "$results_file" ]]; then
        echo "0|0|N/A|N/A"
        return
    fi

    # Parse JSON results
    local total_frames=$(jq -r '.total_frames // 0' "$results_file" 2>/dev/null || echo "0")
    local successful_episodes=$(jq -r '.successful_episodes // 0' "$results_file" 2>/dev/null || echo "0")
    local total_episodes=$(jq -r '.total_episodes // 0' "$results_file" 2>/dev/null || echo "0")

    local status="COMPLETE"
    if [[ "$total_frames" -eq 0 ]]; then
        status="INCOMPLETE"
    fi

    echo "$total_frames|$successful_episodes|$total_episodes|$status"
}

get_log_tail() {
    local log_file=$1
    local lines=${2:-5}

    if [[ -f "$log_file" ]]; then
        tail -n "$lines" "$log_file" 2>/dev/null || echo "No log data"
    else
        echo "Log file not found"
    fi
}

get_elapsed_time() {
    local log_file=$1

    if [[ ! -f "$log_file" ]]; then
        echo "N/A"
        return
    fi

    # Extract start time from log
    local start_time=$(grep "Started at:" "$log_file" | head -1 | sed 's/Started at: //')
    if [[ -z "$start_time" ]]; then
        echo "N/A"
        return
    fi

    local start_epoch=$(date -d "$start_time" +%s 2>/dev/null || echo "0")
    local current_epoch=$(date +%s)
    local elapsed=$((current_epoch - start_epoch))

    # Format as HH:MM:SS
    printf "%02d:%02d:%02d" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
}

estimate_time_remaining() {
    local processed=$1
    local total=$2
    local elapsed_seconds=$3

    if [[ "$processed" -eq 0 ]] || [[ "$total" -eq 0 ]]; then
        echo "N/A"
        return
    fi

    local rate=$(awk "BEGIN {printf \"%.2f\", $processed / $elapsed_seconds}")
    local remaining=$((total - processed))
    local eta_seconds=$(awk "BEGIN {printf \"%.0f\", $remaining / $rate}")

    # Format as HH:MM:SS
    printf "%02d:%02d:%02d" $((eta_seconds/3600)) $((eta_seconds%3600/60)) $((eta_seconds%60))
}

get_cpu_usage() {
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d% -f1
}

get_memory_usage() {
    free -h | awk '/^Mem:/ {printf "%s / %s (%.1f%%)", $3, $2, ($3/$2)*100}'
}

# ============================================================================
# Display Functions
# ============================================================================

display_system_stats() {
    local cpu_usage=$(get_cpu_usage)
    local mem_usage=$(get_memory_usage)

    echo -e "${YELLOW}System Resources:${NC}"
    echo -e "  CPU Usage: ${cpu_usage}%"
    echo -e "  Memory: ${mem_usage}"
    echo
}

display_project_status() {
    local project=$1
    local session_name="frame_extraction_$project"

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Project: ${project}${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Session status
    local status=$(get_session_status "$session_name")
    local status_color=$RED
    case $status in
        RUNNING) status_color=$GREEN ;;
        IDLE) status_color=$YELLOW ;;
        NOT_FOUND) status_color=$RED ;;
    esac
    echo -e "  Tmux Session: ${status_color}${status}${NC} (${session_name})"

    # Find latest log file
    local latest_log=$(ls -t "$LOG_DIR/${project}"_*.log 2>/dev/null | head -1)
    if [[ -n "$latest_log" ]]; then
        echo -e "  Log File: $(basename "$latest_log")"

        # Elapsed time
        local elapsed=$(get_elapsed_time "$latest_log")
        echo -e "  Elapsed Time: ${elapsed}"

        # Check extraction results
        local results=$(get_extraction_results "$project")
        IFS='|' read -r total_frames successful_episodes total_episodes result_status <<< "$results"

        if [[ "$result_status" == "COMPLETE" ]]; then
            echo -e "  ${GREEN}Status: COMPLETE${NC}"
            echo -e "  Total Frames: ${total_frames}"
            echo -e "  Episodes: ${successful_episodes}/${total_episodes}"
        else
            # Parse log for progress (if still running)
            echo -e "  ${YELLOW}Status: IN PROGRESS${NC}"

            # Show last few log lines
            echo -e "\n  ${BLUE}Recent Log:${NC}"
            get_log_tail "$latest_log" 3 | sed 's/^/    /'
        fi
    else
        echo -e "  ${RED}No log file found${NC}"
    fi

    echo
}

display_summary_table() {
    local projects=("$@")

    echo -e "${YELLOW}╔══════════════╦═══════════╦══════════════╦═══════════════╗${NC}"
    echo -e "${YELLOW}║   Project    ║  Status   ║ Total Frames ║  Elapsed Time ║${NC}"
    echo -e "${YELLOW}╠══════════════╬═══════════╬══════════════╬═══════════════╣${NC}"

    for project in "${projects[@]}"; do
        local session_name="frame_extraction_$project"
        local status=$(get_session_status "$session_name")
        local latest_log=$(ls -t "$LOG_DIR/${project}"_*.log 2>/dev/null | head -1)

        local total_frames="N/A"
        local elapsed="N/A"

        if [[ -n "$latest_log" ]]; then
            elapsed=$(get_elapsed_time "$latest_log")
            local results=$(get_extraction_results "$project")
            IFS='|' read -r frames _ _ result_status <<< "$results"
            if [[ "$result_status" == "COMPLETE" ]]; then
                total_frames="$frames"
                status="COMPLETE"
            fi
        fi

        # Format project name (pad to 12 chars)
        local project_padded=$(printf "%-12s" "$project")

        # Format status with color
        local status_display=""
        case $status in
            RUNNING)
                status_display="${GREEN}RUNNING  ${NC}"
                ;;
            COMPLETE)
                status_display="${GREEN}COMPLETE ${NC}"
                ;;
            IDLE)
                status_display="${YELLOW}IDLE     ${NC}"
                ;;
            NOT_FOUND)
                status_display="${RED}NOT FOUND${NC}"
                ;;
        esac

        # Format frames (pad to 12 chars)
        local frames_padded=$(printf "%12s" "$total_frames")

        # Format elapsed (pad to 13 chars)
        local elapsed_padded=$(printf "%13s" "$elapsed")

        echo -e "${YELLOW}║${NC} $project_padded ${YELLOW}║${NC} $status_display ${YELLOW}║${NC} $frames_padded ${YELLOW}║${NC} $elapsed_padded ${YELLOW}║${NC}"
    done

    echo -e "${YELLOW}╚══════════════╩═══════════╩══════════════╩═══════════════╝${NC}"
    echo
}

# ============================================================================
# Notification Functions
# ============================================================================

send_completion_notification() {
    local project=$1
    local total_frames=$2

    # Try to send desktop notification (if available)
    if command -v notify-send &> /dev/null; then
        notify-send "Frame Extraction Complete" "Project: $project\nTotal Frames: $total_frames"
    fi

    # Log completion
    echo "[$(date)] Completed: $project - $total_frames frames" >> "$LOG_DIR/completion_notifications.log"
}

# ============================================================================
# Main Monitoring Loop
# ============================================================================

monitor_projects() {
    local projects=("$@")
    local all_complete=false

    # Track completion state
    declare -A completion_notified

    while [[ "$all_complete" == "false" ]]; do
        print_header
        display_system_stats

        # Check each project
        local running_count=0
        for project in "${projects[@]}"; do
            local session_name="frame_extraction_$project"
            local status=$(get_session_status "$session_name")

            if [[ "$status" == "RUNNING" ]]; then
                ((running_count++))
            fi

            # Check for completion
            local results=$(get_extraction_results "$project")
            IFS='|' read -r total_frames _ _ result_status <<< "$results"

            if [[ "$result_status" == "COMPLETE" ]] && [[ -z "${completion_notified[$project]}" ]]; then
                send_completion_notification "$project" "$total_frames"
                completion_notified[$project]="1"
            fi
        done

        # Display summary table
        display_summary_table "${projects[@]}"

        # Display detailed status for each project
        for project in "${projects[@]}"; do
            display_project_status "$project"
        done

        # Check if all complete
        if [[ $running_count -eq 0 ]]; then
            all_complete=true
            echo -e "${GREEN}✓ All projects completed!${NC}"
            echo
            break
        fi

        echo -e "${BLUE}Refreshing in ${REFRESH_INTERVAL} seconds... (Press Ctrl+C to exit)${NC}"
        sleep $REFRESH_INTERVAL
    done
}

# ============================================================================
# Interactive Mode
# ============================================================================

interactive_mode() {
    # Find all active frame extraction sessions
    local sessions=$(tmux ls 2>/dev/null | grep "frame_extraction_" | cut -d: -f1 || echo "")

    if [[ -z "$sessions" ]]; then
        echo "No active frame extraction sessions found."
        return
    fi

    # Extract project names
    local projects=()
    while IFS= read -r session; do
        local project=$(echo "$session" | sed 's/frame_extraction_//')
        projects+=("$project")
    done <<< "$sessions"

    echo "Found ${#projects[@]} active extraction session(s)"
    echo

    monitor_projects "${projects[@]}"
}

# ============================================================================
# Argument Parsing
# ============================================================================

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Monitor active frame extraction tmux sessions and display progress.

OPTIONS:
    --projects PROJECTS     Comma-separated list of project names to monitor
    --interval N            Refresh interval in seconds (default: $REFRESH_INTERVAL)
    --once                  Display status once and exit (no continuous monitoring)
    -h, --help             Show this help message

EXAMPLES:
    # Monitor specific projects
    $0 --projects "onward,orion"

    # Auto-detect and monitor all active sessions
    $0

    # Display status once
    $0 --projects "onward,orion" --once

EOF
}

PROJECTS=""
ONCE_MODE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --projects)
            PROJECTS="$2"
            shift 2
            ;;
        --interval)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        --once)
            ONCE_MODE="true"
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

main() {
    # Create log directory if needed
    mkdir -p "$LOG_DIR"

    if [[ -n "$PROJECTS" ]]; then
        # Monitor specified projects
        IFS=',' read -ra PROJECT_ARRAY <<< "$PROJECTS"

        if [[ "$ONCE_MODE" == "true" ]]; then
            print_header
            display_system_stats
            display_summary_table "${PROJECT_ARRAY[@]}"
            for project in "${PROJECT_ARRAY[@]}"; do
                display_project_status "$project"
            done
        else
            monitor_projects "${PROJECT_ARRAY[@]}"
        fi
    else
        # Auto-detect active sessions
        interactive_mode
    fi
}

main
