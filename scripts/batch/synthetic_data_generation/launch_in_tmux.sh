#!/usr/bin/bash
"""
Launch Batch Synthetic Data Generation in tmux
===============================================

Launches the batch synthetic data generation pipeline in a detached tmux session.

Usage:
  bash launch_in_tmux.sh <config_file> [session_name]

Examples:
  bash launch_in_tmux.sh ../../configs/batch/synthetic_data_generation_example.yaml
  bash launch_in_tmux.sh myconfig.yaml my_custom_session

Author: LLMProvider Tooling
Date: 2025-11-30
"""

# Parse arguments
CONFIG_FILE="$1"
SESSION_NAME="${2:-synthetic_data_gen}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="$SCRIPT_DIR/batch_synthetic_data_pipeline.sh"

# Validate config file
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Config file required"
    echo ""
    echo "Usage:"
    echo "  bash $(basename "$0") <config_file> [session_name]"
    echo ""
    echo "Example:"
    echo "  bash $(basename "$0") ../../configs/batch/synthetic_data_generation_example.yaml"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Get absolute path
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"

# Extract workspace dir from config
WORKSPACE_DIR=$(python3 << EOF
import yaml
with open("$CONFIG_FILE", 'r') as f:
    config = yaml.safe_load(f)
print(config.get('workspace_dir', '/tmp/synthetic_data_output'))
EOF
)

LOG_FILE="$WORKSPACE_DIR/logs/main_pipeline_$(date +%Y%m%d_%H%M%S).log"

# Ensure workspace exists
mkdir -p "$WORKSPACE_DIR/logs"

# Kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Existing session found: $SESSION_NAME"
    echo ""
    read -p "Kill existing session and start fresh? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo "✅ Old session killed"
    else
        echo "❌ Aborted - attach to existing session with: tmux attach -t $SESSION_NAME"
        exit 1
    fi
fi

echo "========================================================================="
echo "🚀 Launching Batch Synthetic Data Generation Pipeline"
echo "========================================================================="
echo ""
echo "Session name:  $SESSION_NAME"
echo "Config file:   $CONFIG_FILE"
echo "Pipeline:      $PIPELINE_SCRIPT"
echo "Workspace:     $WORKSPACE_DIR"
echo "Log file:      $LOG_FILE"
echo ""
echo "Features:"
echo "  ✓ Checkpoint/Resume capability"
echo "  ✓ Automatic retry on failures"
echo "  ✓ GPU health monitoring"
echo "  ✓ Comprehensive error logging"
echo ""
echo "========================================================================="
echo ""

# Make pipeline script executable
chmod +x "$PIPELINE_SCRIPT"

# Create tmux session
tmux new-session -d -s "$SESSION_NAME" bash -c "\
    clear; \
    echo '========================================================================'; \
    echo 'BATCH SYNTHETIC DATA GENERATION PIPELINE'; \
    echo '========================================================================'; \
    echo ''; \
    echo 'Start time: \$(date)'; \
    echo 'Session: $SESSION_NAME'; \
    echo 'Config: $CONFIG_FILE'; \
    echo 'Log: $LOG_FILE'; \
    echo ''; \
    echo 'Press Ctrl+B, then D to detach'; \
    echo '========================================================================'; \
    echo ''; \
    sleep 3; \
    bash '$PIPELINE_SCRIPT' --config '$CONFIG_FILE' 2>&1 | tee '$LOG_FILE'; \
    EXIT_CODE=\$?; \
    echo ''; \
    echo '========================================================================'; \
    if [ \$EXIT_CODE -eq 0 ]; then \
        echo '✅ Pipeline completed successfully!'; \
    else \
        echo '❌ Pipeline exited with errors (code: '\$EXIT_CODE')'; \
    fi; \
    echo 'End time: \$(date)'; \
    echo '========================================================================'; \
    echo ''; \
    echo 'Press any key to close this tmux session...'; \
    read; \
    exit \
"

sleep 2

# Verify session started
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "✅ Pipeline launched successfully in tmux session: $SESSION_NAME"
    echo ""
    echo "📊 Next Steps:"
    echo ""
    echo "  1. Monitor progress:"
    echo "     bash $SCRIPT_DIR/monitor_progress.sh $WORKSPACE_DIR"
    echo ""
    echo "  2. Attach to view live output:"
    echo "     tmux attach -t $SESSION_NAME"
    echo ""
    echo "  3. Detach without stopping (from within tmux):"
    echo "     Press: Ctrl+B, then D"
    echo ""
    echo "  4. View logs:"
    echo "     tail -f $LOG_FILE"
    echo ""
    echo "🔄 Resume capability:"
    echo "  If interrupted, re-run with same config to resume."
    echo ""
    echo "========================================================================="
    echo ""

    # Show active sessions
    echo "Active tmux sessions:"
    tmux list-sessions 2>/dev/null || echo "  (none)"
    echo ""
else
    echo "❌ Failed to create tmux session"
    exit 1
fi
