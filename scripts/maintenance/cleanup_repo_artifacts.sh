#!/usr/bin/env bash
set -euo pipefail

# Cleanup large, generated artifacts inside this repo (safe defaults).
#
# This script intentionally focuses on repo-local artifacts such as ./outputs and ./logs.
# It will NOT delete external datasets/models under /mnt/data or /mnt/c/ai_models.
#
# Usage:
#   DRY_RUN=1 bash scripts/maintenance/cleanup_repo_artifacts.sh
#   bash scripts/maintenance/cleanup_repo_artifacts.sh
#
# Notes:
# - If Wan2.1 training is currently running (tracked by logs/wan21_train_from_cache_p3d_pairs_from_images.nohup.pid),
#   we keep the active PID file + the latest matching log to avoid breaking monitoring.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DRY_RUN="${DRY_RUN:-0}"

say() { echo "[cleanup] $*"; }
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ $*"
  else
    eval "$@"
  fi
}

keep_logs=()
active_pid_file="logs/wan21_train_from_cache_p3d_pairs_from_images.nohup.pid"
if [[ -f "$active_pid_file" ]]; then
  active_pid="$(cat "$active_pid_file" 2>/dev/null || true)"
  if [[ -n "${active_pid:-}" ]] && kill -0 "$active_pid" 2>/dev/null; then
    say "Detected active Wan2.1 training PID: $active_pid"
    keep_logs+=("$(basename "$active_pid_file")")
    latest_train_log="$(ls -1t logs/wan21_train_from_cache_p3d_pairs_from_images_*.log 2>/dev/null | head -n 1 || true)"
    if [[ -n "${latest_train_log:-}" ]]; then
      keep_logs+=("$(basename "$latest_train_log")")
      say "Keeping active log: $(basename "$latest_train_log")"
    fi
  fi
fi

say "Deleting ./outputs (repo-local artifacts)"
if [[ -d "outputs" ]]; then
  run "rm -rf outputs/*"
fi

say "Deleting ./logs except: ${keep_logs[*]:-(none)}"
if [[ -d "logs" ]]; then
  while IFS= read -r -d '' p; do
    b="$(basename "$p")"
    keep="0"
    for k in "${keep_logs[@]:-}"; do
      if [[ "$b" == "$k" ]]; then
        keep="1"
        break
      fi
    done
    if [[ "$keep" == "1" ]]; then
      continue
    fi
    run "rm -rf \"${p}\""
  done < <(find logs -mindepth 1 -maxdepth 1 -print0)
fi

say "Deleting common Python caches"
run "find . -type d -name __pycache__ -prune -print0 | xargs -0 -r rm -rf"
run "find . -type f \\( -name '*.pyc' -o -name '*.pyo' \\) -print0 | xargs -0 -r rm -f"

say "Done"

