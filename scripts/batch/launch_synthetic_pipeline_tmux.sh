#!/usr/bin/bash

# ============================================================================
# Launch Synthetic Data Pipeline in tmux
# ============================================================================
#
# This script launches the synthetic data generation pipeline in a detached
# tmux session for long-running background execution.
#
# Usage:
#   bash scripts/batch/launch_synthetic_pipeline_tmux.sh [options]
#
# Options are passed through to run_synthetic_data_pipeline.sh
#
# Examples:
#   # Launch full pipeline with default config
#   bash scripts/batch/launch_synthetic_pipeline_tmux.sh
#
#   # Launch with custom config
#   bash scripts/batch/launch_synthetic_pipeline_tmux.sh \
#     --config configs/batch/my_config.yaml
#
#   # Launch only generation phase for specific characters
#   bash scripts/batch/launch_synthetic_pipeline_tmux.sh \
#     --phase 2 --characters alberto,bryce
#
# Author: LLMProvider Tooling
# Date: 2025-11-30
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMUX_SESSION="synthetic_data_pipeline"

# Kill existing session if it exists
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

# Build command
PIPELINE_CMD="bash $SCRIPT_DIR/run_synthetic_data_pipeline.sh $@"

# Create new tmux session and run pipeline
tmux new-session -d -s "$TMUX_SESSION" "$PIPELINE_CMD"

echo ""
echo "========================================================================"
echo "✅ SYNTHETIC DATA PIPELINE LAUNCHED IN TMUX"
echo "========================================================================"
echo "Session name: $TMUX_SESSION"
echo ""
echo "Monitor with:"
echo "  tmux attach -t $TMUX_SESSION"
echo "  (Press Ctrl+B, then D to detach)"
echo ""
echo "Or view logs:"
echo "  tail -f /mnt/c/ai_projects/24h_lora_pipeline/logs/*.log"
echo ""
echo "Check status:"
echo "  tmux ls"
echo ""
echo "========================================================================"
