#!/usr/bin/env bash
#
# ============================================================================
# Launch P3D Multi-Character Full Generation via nohup (Background)
# ============================================================================
#
# Launches `run_p3d_multichar_full_generation.sh` in the background using nohup.
#
# Usage:
#   bash scripts/batch/launch_p3d_multichar_full_generation_nohup.sh [options...]
#
# Options are passed through to run_p3d_multichar_full_generation.sh
#
# Notes:
# - The runner already writes a timestamped log under `logs/`.
# - We redirect nohup stdout/stderr to /dev/null to avoid duplicate log lines
#   (the runner uses `tee` to write to its own log file).
#
# Author: Codex CLI
# Date: 2026-01-26
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/logs"

LOG_FILE="${P3D_LOG_FILE:-$PROJECT_ROOT/logs/p3d_multichar_full_generation_$(date +%Y%m%d_%H%M%S).log}"
PID_FILE="${P3D_PID_FILE:-$PROJECT_ROOT/logs/p3d_multichar_full_generation.nohup.pid}"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${existing_pid:-}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "ERROR: Already running (PID $existing_pid)."
    echo "Stop with: kill $existing_pid"
    echo "Or remove PID file if stale: rm -f \"$PID_FILE\""
    exit 1
  fi
fi

export P3D_LOG_FILE="$LOG_FILE"

nohup bash "$SCRIPT_DIR/run_p3d_multichar_full_generation.sh" "$@" </dev/null >/dev/null 2>&1 &
pid="$!"
echo "$pid" > "$PID_FILE"

echo ""
echo "========================================================================"
echo "✅ P3D FULL GENERATION LAUNCHED (nohup)"
echo "========================================================================"
echo "PID: $pid"
echo "PID file: $PID_FILE"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor:"
echo "  tail -f \"$LOG_FILE\""
echo ""
echo "Stop:"
echo "  kill $pid"
echo "========================================================================"

