#!/usr/bin/env bash
set -euo pipefail

# Image batch acceptance for P3D final SDXL checkpoint over:
# - 91 all-pairs coverage
# - 50 motion-ready interaction templates
# Total prompts: 4550
#
# Generates:
#   outputs/p3d_final_image_acceptance_allpairs50_<ts>/
#     images/img_*.png + img_*.txt
#     records.jsonl
#     meta.json
#     qc/acceptance_report.csv + acceptance_summary.json + flagged/
#
# Monitor:
#   tail -f logs/p3d_final_image_acceptance_allpairs50_*.log
#   ls outputs/p3d_final_image_acceptance_allpairs50_*/images | wc -l
#
# Stop:
#   kill $(cat logs/p3d_final_image_acceptance_allpairs50.nohup.pid)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV_GEN="${CONDA_ENV_GEN:-ai_env}"

CKPT="${CKPT:-/mnt/c/ai_models/stable-diffusion/checkpoints/p3d_multichar_sdxl_final_latest.safetensors}"
PROMPTS="${PROMPTS:-prompts/p3d/p3d_multichar_all_pairs_prompts_50actions_v1_video.txt}"
NEG_FILE="${NEG_FILE:-prompts/p3d/pair_negative.txt}"

STEPS="${STEPS:-45}"
CFG="${CFG:-5.5}"
SEED="${SEED:--1}"        # -1 = random per prompt, >=0 deterministic
OFFLOAD="${OFFLOAD:-1}"   # 0/1 (recommended 1 for 16GB safety)
LIMIT="${LIMIT:-0}"       # 0=all, for smoke set e.g. 50
SKIP_EXISTING="${SKIP_EXISTING:-1}"

BLUR_T="${BLUR_T:-35}"
LOWC_T="${LOWC_T:-0.08}"
OVER_T="${OVER_T:-0.08}"
UNDER_T="${UNDER_T:-0.08}"
COPY_FLAGGED="${COPY_FLAGGED:-200}"

mkdir -p logs outputs
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/p3d_final_image_acceptance_allpairs50_${TS}"
LOG_FILE="logs/p3d_final_image_acceptance_allpairs50_${TS}.log"
PID_FILE="logs/p3d_final_image_acceptance_allpairs50.nohup.pid"

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

skip_args=()
if [[ "$SKIP_EXISTING" == "1" ]]; then
  skip_args+=(--skip-existing)
fi

nohup bash -lc "
  set -euo pipefail
  echo '[acceptance] start: ' \$(date -Is)
  echo '[acceptance] CKPT=$CKPT'
  echo '[acceptance] PROMPTS=$PROMPTS'
  echo '[acceptance] NEG_FILE=$NEG_FILE'
  echo '[acceptance] OUT_DIR=$OUT_DIR'
  echo '[acceptance] STEPS=$STEPS CFG=$CFG SEED=$SEED OFFLOAD=$OFFLOAD LIMIT=$LIMIT SKIP_EXISTING=$SKIP_EXISTING'

  mkdir -p \"$OUT_DIR\"

  conda run -n \"$CONDA_ENV_GEN\" python scripts/evaluation/p3d_batch_generate_images_sdxl.py \\
    --checkpoint \"$CKPT\" \\
    --prompts \"$PROMPTS\" \\
    --out-dir \"$OUT_DIR\" \\
    --negative-file \"$NEG_FILE\" \\
    --steps \"$STEPS\" \\
    --cfg \"$CFG\" \\
    --seed \"$SEED\" \\
    --limit \"$LIMIT\" \\
    ${offload_args[@]} \\
    ${skip_args[@]}

  conda run -n \"$CONDA_ENV_GEN\" python scripts/evaluation/p3d_pair_acceptance_report.py \\
    --images-dir \"$OUT_DIR/images\" \\
    --out \"$OUT_DIR/qc\" \\
    --blur-threshold \"$BLUR_T\" \\
    --low-contrast \"$LOWC_T\" \\
    --overexposed \"$OVER_T\" \\
    --underexposed \"$UNDER_T\" \\
    --copy-flagged \"$COPY_FLAGGED\"

  echo '[acceptance] done: ' \$(date -Is)
" >"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "✅ acceptance launched (nohup)"
echo "PID: $pid"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"
echo "Out: $OUT_DIR"

