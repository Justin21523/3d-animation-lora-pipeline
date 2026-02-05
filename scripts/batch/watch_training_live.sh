#!/bin/bash
#
# Real-time Training Output Monitor
# Shows live progress from kohya_ss training
#

echo "========================================="
echo "🔴 LIVE Training Output Monitor"
echo "========================================="
echo ""
echo "Attempting to capture live training output..."
echo "Press Ctrl+C to exit"
echo ""
echo "========================================="
echo ""

# Method 1: Try to tail the tmux pane continuously
tmux pipe-pane -t sdxl_training -o 'cat >> /tmp/sdxl_training_live.log'
echo "✅ Started piping tmux output to /tmp/sdxl_training_live.log"
echo ""
echo "Now showing live output:"
echo "----------------------------------------"
echo ""

# Tail the live log
tail -f /tmp/sdxl_training_live.log
