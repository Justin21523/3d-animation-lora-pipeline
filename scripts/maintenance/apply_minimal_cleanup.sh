#!/usr/bin/env bash
set -euo pipefail

# Apply minimalization cleanup according to outputs/maintenance/minimal_keep_set_report.json
#
# Safety:
# - Refuses to modify scripts/ while Wan2.1 training is running (PID tracked by logs/wan21_train_from_cache_p3d_pairs_from_images.nohup.pid).
#
# Usage:
#   DRY_RUN=1 bash scripts/maintenance/apply_minimal_cleanup.sh
#   bash scripts/maintenance/apply_minimal_cleanup.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DRY_RUN="${DRY_RUN:-0}"
REPORT="${REPORT:-outputs/maintenance/minimal_keep_set_report.json}"

say() { echo "[apply-cleanup] $*"; }
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ $*"
  else
    eval "$@"
  fi
}

if [[ ! -f "$REPORT" ]]; then
  echo "Report not found: $REPORT" >&2
  echo "Generate it first with: python scripts/maintenance/derive_minimal_keep_set.py" >&2
  exit 1
fi

pid_file="logs/wan21_train_from_cache_p3d_pairs_from_images.nohup.pid"
if [[ -f "$pid_file" ]]; then
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: Wan2.1 training is still running (PID $pid). Refusing to delete/move scripts/." >&2
    echo "Wait for training to finish, then retry." >&2
    exit 1
  fi
fi

say "Loading delete candidates from $REPORT"
python - "$REPORT" <<'PY' | while IFS= read -r p; do
import json,sys
from pathlib import Path
report_path = Path(sys.argv[1])
rep = json.loads(report_path.read_text(encoding="utf-8"))
for p in rep.get("delete_candidates", []):
    print(p)
PY
  # Double-safety: never delete .git or requirements/configs roots.
  case "$p" in
    .git/*|.git|.gitignore) continue;;
    requirements/*|requirements) continue;;
    configs/*|configs) continue;;
    logs/*|logs) continue;;
    outputs/*|outputs) continue;;
  esac

  if [[ -e "$p" ]]; then
    run "rm -rf \"$p\""
  fi
done

say "Pruning empty directories under scripts/docs/prompts"
run "find scripts docs prompts -type d -empty -print0 2>/dev/null | xargs -0 -r rmdir"

say "Done"

