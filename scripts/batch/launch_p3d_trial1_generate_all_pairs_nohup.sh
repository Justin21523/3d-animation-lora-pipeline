#!/usr/bin/env bash
set -euo pipefail

# Generate a large all-pairs dataset for manual curation.
#
# Default target:
# - 14 characters -> 91 unique pairs
# - 50 images per pair -> 4550 images total
#
# Usage:
#   bash scripts/batch/launch_p3d_trial1_generate_all_pairs_nohup.sh
#
# Monitor:
#   tail -f logs/p3d_trial1_all_pairs_gen_*.log
#
# Stop:
#   kill $(cat logs/p3d_trial1_all_pairs_gen.nohup.pid)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV_GEN="${CONDA_ENV_GEN:-ai_env}"

CHECKPOINT="${CHECKPOINT:-/mnt/data/ai_data/models/checkpoints/p3d_multichar/p3d_multichar_sdxl_trial1_merged_20260129_210411.safetensors}"
OUT_DIR="${OUT_DIR:-/mnt/data/ai_data/synthetic_lora_data/pairs_generated_trial1_$(date +%Y%m%d_%H%M%S)}"

IMAGES_PER_PAIR="${IMAGES_PER_PAIR:-50}"
STEPS="${STEPS:-45}"
CFG="${CFG:-5.5}"

# 0 = random seed
SEED="${SEED:-0}"

# 0/1
OFFLOAD="${OFFLOAD:-1}"

mkdir -p logs
LOG_FILE="logs/p3d_trial1_all_pairs_gen_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="logs/p3d_trial1_all_pairs_gen.nohup.pid"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${existing_pid:-}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "ERROR: Already running (PID $existing_pid)."
    echo "Stop with: kill $existing_pid"
    exit 1
  fi
fi

offload_args=()
if [[ "$OFFLOAD" == "1" ]]; then
  offload_args+=(--offload)
fi

nohup bash -lc "
  set -euo pipefail
  echo '[gen] start: ' \$(date -Is)
  echo '[gen] CHECKPOINT=$CHECKPOINT'
  echo '[gen] OUT_DIR=$OUT_DIR'
  echo '[gen] IMAGES_PER_PAIR=$IMAGES_PER_PAIR STEPS=$STEPS CFG=$CFG SEED=$SEED OFFLOAD=$OFFLOAD'
  conda run -n \"$CONDA_ENV_GEN\" python scripts/batch/generate_p3d_all_pairs_dataset.py \\
    --checkpoint \"$CHECKPOINT\" \\
    --out-dir \"$OUT_DIR\" \\
    --images-per-pair \"$IMAGES_PER_PAIR\" \\
    --steps \"$STEPS\" \\
    --cfg \"$CFG\" \\
    --seed \"$SEED\" \\
    \${offload_args[@]}
  echo '[gen] done: ' \$(date -Is)
" >"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "✅ all-pairs generation launched (nohup)"
echo "PID: $pid"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"
echo "Out: $OUT_DIR"

