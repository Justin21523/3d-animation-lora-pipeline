#!/usr/bin/env bash
set -euo pipefail

# Launch SDXL full fine-tune (sdxl_train.py) in tmux.
#
# Usage:
#   bash scripts/batch/launch_p3d_sdxl_full_finetune_tmux.sh <config.toml>
#
# Env:
#   TMUX_SESSION (default: p3d_full_finetune)
#   KOHYA_DIR    (default: /mnt/c/ai_tools/kohya_ss)
#   CONDA_ENV    (default: kohya_ss)

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/batch/launch_p3d_sdxl_full_finetune_tmux.sh <config.toml>" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="$1"

TMUX_SESSION="${TMUX_SESSION:-p3d_full_finetune}"
KOHYA_DIR="${KOHYA_DIR:-/mnt/c/ai_tools/kohya_ss}"
CONDA_ENV="${CONDA_ENV:-kohya_ss}"

mkdir -p "$PROJECT_ROOT/logs"
LOG_FILE="${P3D_TRAIN_LOG_FILE:-$PROJECT_ROOT/logs/p3d_full_finetune_$(date +%Y%m%d_%H%M%S).log}"

if command -v tmux >/dev/null 2>&1; then
  :
else
  echo "tmux not found" >&2
  exit 1
fi

if [[ ! -f "$CFG" ]]; then
  echo "Config not found: $CFG" >&2
  exit 1
fi

if tmux ls 2>/dev/null | rg -q "^${TMUX_SESSION}:"; then
  echo "ERROR: tmux session already exists: $TMUX_SESSION" >&2
  echo "Stop with: tmux kill-session -t $TMUX_SESSION" >&2
  exit 1
fi

CMD="cd \"$KOHYA_DIR\" && conda run -n \"$CONDA_ENV\" accelerate launch --num_cpu_threads_per_process=2 ./sd-scripts/sdxl_train.py --config_file \"$PROJECT_ROOT/$CFG\""

tmux new-session -d -s "$TMUX_SESSION" "$CMD 2>&1 | tee -a \"$LOG_FILE\""

echo "✅ Launched full fine-tune in tmux: $TMUX_SESSION"
echo "Log file: $LOG_FILE"
echo "Attach: tmux attach -t $TMUX_SESSION"
echo "Stop:   tmux kill-session -t $TMUX_SESSION"

