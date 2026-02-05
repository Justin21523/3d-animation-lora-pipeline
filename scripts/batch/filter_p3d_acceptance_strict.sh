#!/usr/bin/env bash
set -euo pipefail

# Strict filter: "few but high-quality"
#
# Usage:
#   bash scripts/batch/filter_p3d_acceptance_strict.sh outputs/<acceptance_run_dir>
#
# Expects:
#   <run_dir>/qc/acceptance_report.csv
#
# Outputs:
#   <run_dir>/qc/strict/accepted_prompts.txt
#   <run_dir>/qc/strict/rejected_prompts.txt

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV_GEN="${CONDA_ENV_GEN:-ai_env}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <acceptance_run_dir>" >&2
  exit 1
fi

RUN_DIR="$1"
REPORT="${RUN_DIR}/qc/acceptance_report.csv"
OUT_DIR="${RUN_DIR}/qc/strict"

if [[ ! -f "$REPORT" ]]; then
  echo "Report not found (run QC first): $REPORT" >&2
  exit 1
fi

conda run -n "$CONDA_ENV_GEN" python scripts/evaluation/p3d_filter_acceptance_results.py \
  --report-csv "$REPORT" \
  --out-dir "$OUT_DIR" \
  --min-sharpness 140 \
  --min-luma-std 0.11 \
  --max-luma-mean 0.88 \
  --max-overexposed-frac 0.02 \
  --max-underexposed-frac 0.02 \
  --require-no-flags 1

echo "✅ strict outputs:"
echo "  $OUT_DIR/accepted_prompts.txt"
echo "  $OUT_DIR/rejected_prompts.txt"

