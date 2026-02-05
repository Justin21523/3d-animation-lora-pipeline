#!/usr/bin/env bash
#
# ============================================================================
# Launch P3D Multi-Character Full Generation in tmux (Detached)
# ============================================================================
#
# This launches `run_p3d_multichar_full_generation.sh` in a detached tmux
# session so it keeps running after you disconnect.
#
# Usage:
#   bash scripts/batch/launch_p3d_multichar_full_generation_tmux.sh [options...]
#
# Options are passed through to run_p3d_multichar_full_generation.sh
#
# Examples:
#   # Run everything (interactions + AE + pose)
#   bash scripts/batch/launch_p3d_multichar_full_generation_tmux.sh
#
#   # Skip interactions (if already generated)
#   bash scripts/batch/launch_p3d_multichar_full_generation_tmux.sh --skip-interactions
#
# Author: Codex CLI
# Date: 2026-01-26
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TMUX_SESSION="${TMUX_SESSION:-p3d_full_generation}"
ARGS_ESCAPED="$(printf '%q ' "$@")"
RUNNER="bash $SCRIPT_DIR/run_p3d_multichar_full_generation.sh ${ARGS_ESCAPED}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found. Install tmux or run the runner directly with nohup:" >&2
  echo "  nohup $RUNNER > p3d_full_generation.log 2>&1 &" >&2
  exit 1
fi

# Kill existing session if it exists (non-interactive behavior)
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

tmux new-session -d -s "$TMUX_SESSION" "cd \"$PROJECT_ROOT\" && $RUNNER"

sleep 1
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  LOG_DIR="$PROJECT_ROOT/logs"
  LATEST_LOG="$(ls -1t "$LOG_DIR"/p3d_multichar_full_generation_*.log 2>/dev/null | head -n 1 || true)"
  LAST_LINES=""
  if [[ -n "$LATEST_LOG" ]]; then
    LAST_LINES="$(tail -n 80 "$LATEST_LOG" 2>/dev/null || true)"
  fi

  echo ""
  echo "========================================================================"
  echo "⚠️  tmux session exited immediately"
  echo "========================================================================"
  echo "Most likely causes:"
  echo "  - GPU/CUDA not available (nvidia-smi / torch.cuda.is_available())"
  echo "  - conda not found in PATH inside tmux"
  echo ""
  echo "Next steps:"
  echo "  - Check logs: ls -lt \"$LOG_DIR\" | head"
  echo "  - Run foreground to see error:"
  echo "      bash scripts/batch/run_p3d_multichar_full_generation.sh $ARGS_ESCAPED"
  if [[ -n "$LATEST_LOG" ]]; then
    echo ""
    echo "Latest log:"
    echo "  $LATEST_LOG"
    echo ""
    echo "Last lines:"
    echo "$LAST_LINES"
    if echo "$LAST_LINES" | rg -q "All requested steps completed\\."; then
      echo ""
      echo "✅ Looks like it finished successfully (exited quickly)."
      echo "========================================================================"
      exit 0
    fi
  fi
  echo "========================================================================"
  exit 1
fi

echo ""
echo "========================================================================"
echo "✅ P3D FULL GENERATION LAUNCHED IN TMUX"
echo "========================================================================"
echo "Session name: $TMUX_SESSION"
echo ""
echo "Attach:"
echo "  tmux attach -t $TMUX_SESSION"
echo ""
echo "Detach:"
echo "  Ctrl+B then D"
echo ""
echo "Stop (kills session):"
echo "  tmux kill-session -t $TMUX_SESSION"
echo "========================================================================"
