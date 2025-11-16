#!/bin/bash

###############################################################################
# Standardized Tmux Training Launcher
# Purpose: Start long-running LoRA training in protected tmux session
# Usage: bash start_training_in_tmux.sh <session_name> <training_command>
###############################################################################

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "❌ Error: Insufficient arguments"
    echo ""
    echo "Usage:"
    echo "  bash start_training_in_tmux.sh <session_name> <training_command>"
    echo ""
    echo "Example:"
    echo '  bash start_training_in_tmux.sh trial35 "conda run -n kohya_ss python train_network.py --config_file config.toml"'
    exit 1
fi

SESSION_NAME="$1"
shift
TRAINING_COMMAND="$@"

# Validate session name (alphanumeric only, no special chars)
if ! [[ "$SESSION_NAME" =~ ^[a-zA-Z0-9]+$ ]]; then
    echo "❌ Error: Session name must be alphanumeric only (no underscores, dashes, or special characters)"
    echo "   Invalid: $SESSION_NAME"
    echo "   Valid examples: trial35, optv21, lucav6"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Warning: Tmux session '$SESSION_NAME' already exists!"
    echo ""
    read -p "Do you want to kill existing session and restart? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔪 Killing existing session..."
        tmux kill-session -t "$SESSION_NAME"
    else
        echo "❌ Aborted. Please choose a different session name or kill existing session manually."
        exit 1
    fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Tmux Training Launcher"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Session Name: $SESSION_NAME"
echo "Command: $TRAINING_COMMAND"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create new detached tmux session
echo "🚀 Creating tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME"

# Verify session creation
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "❌ Error: Failed to create tmux session"
    exit 1
fi

echo "✅ Tmux session '$SESSION_NAME' created successfully"

# Send command to tmux session
echo "📤 Sending training command to tmux..."
tmux send-keys -t "$SESSION_NAME" "$TRAINING_COMMAND" C-m

# Wait a moment for command to start
sleep 2

# Verify process started
echo "🔍 Verifying training process..."
PANE_CONTENT=$(tmux capture-pane -t "$SESSION_NAME" -p | tail -5)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Latest output from tmux session:"
echo "$PANE_CONTENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Success message
echo "✅ Training started successfully in tmux session '$SESSION_NAME'"
echo ""
echo "📋 Useful commands:"
echo "  • Attach to session:    tmux attach -t $SESSION_NAME"
echo "  • View session:         tmux capture-pane -t $SESSION_NAME -p | tail -50"
echo "  • List all sessions:    tmux list-sessions"
echo "  • Kill this session:    tmux kill-session -t $SESSION_NAME"
echo ""
echo "⚠️  Detach from session: Press Ctrl+B then D (do NOT close terminal directly)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 All done! Training is now protected from SSH disconnects."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
