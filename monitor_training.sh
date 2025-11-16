#!/bin/bash
# Monitor training progress

SESSION_NAME="luca_lora_training"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Training session is running!"
    echo "Attaching to session..."
    tmux attach -t "$SESSION_NAME"
else
    echo "Training session not found!"
    echo "Available sessions:"
    tmux ls
fi
