#!/usr/bin/env bash
set -euo pipefail

# Build a Wan2.1 LoRA dataset from the 4550 image-acceptance outputs using a *mild* QC filter.
#
# Source (images + per-image captions):
#   outputs/p3d_final_image_acceptance_allpairs50_20260202_213530/images/img_*.png
#   outputs/p3d_final_image_acceptance_allpairs50_20260202_213530/images/img_*.txt
# QC report:
#   outputs/p3d_final_image_acceptance_allpairs50_20260202_213530/qc/acceptance_report.csv
#
# Output dataset:
#   /mnt/data/datasets/general/wan2.1/lora_datasets/p3d_pairs_from_images_qc_mild_832x480_16fps_16f/
#
# Monitor:
#   tail -f logs/prepare_wan21_p3d_images_qc_mild_*.log
#
# Stop:
#   kill $(cat logs/prepare_wan21_p3d_images_qc_mild.nohup.pid)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

ACCEPT_RUN_DIR="${ACCEPT_RUN_DIR:-outputs/p3d_final_image_acceptance_allpairs50_20260202_213530}"
REPORT_CSV="${REPORT_CSV:-${ACCEPT_RUN_DIR}/qc/acceptance_report.csv}"

SIZE="${SIZE:-832x480}"
FPS="${FPS:-16}"
FRAMES="${FRAMES:-16}"

WAN21_DATASETS_ROOT="${WAN21_DATASETS_ROOT:-/mnt/data/datasets/general/wan2.1/lora_datasets}"
OUT_DIR="${OUT_DIR:-${WAN21_DATASETS_ROOT}/p3d_pairs_from_images_qc_mild_${SIZE}_${FPS}fps_${FRAMES}f}"

# Mild QC knobs (not too strict)
MIN_SHARPNESS="${MIN_SHARPNESS:-25}"
MIN_LUMA_STD="${MIN_LUMA_STD:-0.05}"
MIN_LUMA_MEAN="${MIN_LUMA_MEAN:-0.12}"
MAX_LUMA_MEAN="${MAX_LUMA_MEAN:-0.93}"
MAX_OVER_FRAC="${MAX_OVER_FRAC:-0.12}"
MAX_UNDER_FRAC="${MAX_UNDER_FRAC:-0.12}"
REQUIRE_NO_FLAGS="${REQUIRE_NO_FLAGS:-0}"

MAX_ITEMS="${MAX_ITEMS:-0}"  # 0=all 4550

mkdir -p logs "$(dirname "$OUT_DIR")"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/prepare_wan21_p3d_images_qc_mild_${TS}.log"
PID_FILE="logs/prepare_wan21_p3d_images_qc_mild.nohup.pid"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${existing_pid:-}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "ERROR: Already running (PID $existing_pid)."
    echo "Stop with: kill $existing_pid"
    exit 1
  fi
fi

if [[ ! -f "$REPORT_CSV" ]]; then
  echo "Report not found: $REPORT_CSV" >&2
  exit 1
fi

nohup bash -lc "
  set -euo pipefail
  echo '[prepare] start:' \$(date -Is)
  echo '[prepare] REPORT_CSV=$REPORT_CSV'
  echo '[prepare] OUT_DIR=$OUT_DIR'
  echo '[prepare] SIZE=$SIZE FPS=$FPS FRAMES=$FRAMES'
  echo '[prepare] QC min_sharp=$MIN_SHARPNESS min_std=$MIN_LUMA_STD mean=[$MIN_LUMA_MEAN,$MAX_LUMA_MEAN] over<=$MAX_OVER_FRAC under<=$MAX_UNDER_FRAC require_no_flags=$REQUIRE_NO_FLAGS'

  python scripts/batch/prepare_wan21_dataset_from_p3d_image_acceptance.py \\
    --report-csv \"$REPORT_CSV\" \\
    --out-dir \"$OUT_DIR\" \\
    --size \"$SIZE\" \\
    --fps \"$FPS\" \\
    --frames \"$FRAMES\" \\
    --min-sharpness \"$MIN_SHARPNESS\" \\
    --min-luma-std \"$MIN_LUMA_STD\" \\
    --min-luma-mean \"$MIN_LUMA_MEAN\" \\
    --max-luma-mean \"$MAX_LUMA_MEAN\" \\
    --max-overexposed-frac \"$MAX_OVER_FRAC\" \\
    --max-underexposed-frac \"$MAX_UNDER_FRAC\" \\
    --require-no-flags \"$REQUIRE_NO_FLAGS\" \\
    $( [[ \"$MAX_ITEMS\" != \"0\" ]] && echo \"--max-items $MAX_ITEMS\" )

  echo '[prepare] done:' \$(date -Is)
" >"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "✅ dataset build launched (nohup)"
echo "PID: $pid"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"
echo "Out: $OUT_DIR"

