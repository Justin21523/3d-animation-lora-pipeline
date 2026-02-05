#!/usr/bin/env bash
#
# Launch P3D multi-character SDXL LoRA training in a detached tmux session.
#
# This uses the Kohya_ss SDXL training script with a TOML config, and is meant
# to run for many hours.
#
# Usage:
#   bash scripts/batch/launch_p3d_multichar_training_tmux.sh \
#     --config configs/training/p3d_multichar_sdxl_checkpoint_trial1.toml
#
# Env overrides:
#   TMUX_SESSION   (default: p3d_multichar_training)
#   CONDA_ENV      (default: kohya_ss)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TMUX_SESSION="${TMUX_SESSION:-p3d_multichar_training}"
CONDA_ENV="${CONDA_ENV:-kohya_ss}"
KOHYA_DIR="${KOHYA_DIR:-/mnt/c/ai_tools/kohya_ss}"

CONFIG_FILE="${1:-}"
if [[ -z "${CONFIG_FILE}" ]]; then
  echo "ERROR: Missing config argument." >&2
  echo "Usage: bash scripts/batch/launch_p3d_multichar_training_tmux.sh <config.toml>" >&2
  exit 2
fi

cd "$PROJECT_ROOT"

if [[ ! -d "$KOHYA_DIR" ]]; then
  echo "ERROR: Kohya_ss not found: $KOHYA_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Training config not found: $CONFIG_FILE" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found. Install tmux or run foreground:" >&2
  echo "  conda run -n \"$CONDA_ENV\" accelerate launch --num_cpu_threads_per_process=2 \\" >&2
  echo "    \"$KOHYA_DIR/sd-scripts/sdxl_train_network.py\" --config_file \"$CONFIG_FILE\"" >&2
  exit 1
fi

# Kill existing session if it exists
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

mkdir -p "$PROJECT_ROOT/logs"
LOG_FILE="${P3D_TRAIN_LOG_FILE:-$PROJECT_ROOT/logs/p3d_multichar_training_$(date +%Y%m%d_%H%M%S).log}"

RUN_CMD="cd \"$KOHYA_DIR\" && conda run -n \"$CONDA_ENV\" accelerate launch --num_cpu_threads_per_process=2 ./sd-scripts/sdxl_train_network.py --config_file \"$PROJECT_ROOT/$CONFIG_FILE\" 2>&1 | tee -a \"$LOG_FILE\""
tmux new-session -d -s "$TMUX_SESSION" "$RUN_CMD"

echo "✅ Launched training in tmux: $TMUX_SESSION"
echo "Log file: $LOG_FILE"
echo "Attach: tmux attach -t $TMUX_SESSION"
echo "Stop:   tmux kill-session -t $TMUX_SESSION"
