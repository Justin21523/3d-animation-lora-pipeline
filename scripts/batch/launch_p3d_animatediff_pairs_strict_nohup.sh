#!/usr/bin/env bash
set -euo pipefail

# Batch-generate AnimateDiff videos from STRICT accepted pair prompts (188).
#
# Inputs (from the image acceptance run):
#   outputs/p3d_final_image_acceptance_allpairs50_20260202_213530/qc/strict/captions/*.txt
#   outputs/p3d_final_image_acceptance_allpairs50_20260202_213530/qc/strict/video_negative.txt
#
# Uses:
#   configs/animatediff/p3d_multichar_final_pairs_strict_sdxl.yaml
#   /mnt/c/ai_projects/ai-gen-hub/scripts/batch_generate_animatediff.sh
#
# Monitor:
#   tail -f logs/p3d_animatediff_pairs_strict_*.log
#   ls outputs/p3d_animatediff_pairs_strict_*/videos | wc -l
#
# Stop:
#   kill $(cat logs/p3d_animatediff_pairs_strict.nohup.pid)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

AI_GEN_HUB_ROOT="${AI_GEN_HUB_ROOT:-/mnt/c/ai_projects/ai-gen-hub}"
ANIMATEDIFF_BATCH="${ANIMATEDIFF_BATCH:-${AI_GEN_HUB_ROOT}/scripts/batch_generate_animatediff.sh}"

ACCEPT_RUN_DIR="${ACCEPT_RUN_DIR:-outputs/p3d_final_image_acceptance_allpairs50_20260202_213530}"
CAPTIONS_DIR="${CAPTIONS_DIR:-${ACCEPT_RUN_DIR}/qc/strict/captions}"
NEG_FILE="${NEG_FILE:-${ACCEPT_RUN_DIR}/qc/strict/video_negative.txt}"

BASE_CONFIG="${BASE_CONFIG:-${PROJECT_ROOT}/configs/animatediff/p3d_multichar_final_pairs_strict_sdxl.yaml}"

OUT_BASE="${OUT_BASE:-outputs}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${OUT_BASE}/p3d_animatediff_pairs_strict_${TS}}"

CONDA_ENV="${ANIMATEDIFF_CONDA_ENV:-animate_diff}"
LIMIT="${LIMIT:-0}"          # 0=all (188)
LENGTH="${LENGTH:-16}"
STEPS="${STEPS:-30}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1024}"
SEED="${SEED:--1}"           # -1 = random

MIN_FREE_MIB="${MIN_FREE_MIB:-2000}"

mkdir -p logs "$OUT_DIR"
LOG_FILE="logs/p3d_animatediff_pairs_strict_${TS}.log"
PID_FILE="logs/p3d_animatediff_pairs_strict.nohup.pid"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${existing_pid:-}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "ERROR: Already running (PID $existing_pid)."
    echo "Stop with: kill $existing_pid"
    exit 1
  fi
fi

if [[ ! -x "$ANIMATEDIFF_BATCH" ]]; then
  echo "AnimateDiff batch script not found/executable: $ANIMATEDIFF_BATCH" >&2
  exit 1
fi
if [[ ! -d "$CAPTIONS_DIR" ]]; then
  echo "Captions dir not found: $CAPTIONS_DIR" >&2
  exit 1
fi
if [[ ! -f "$NEG_FILE" ]]; then
  echo "Negative file not found: $NEG_FILE" >&2
  exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Base config not found: $BASE_CONFIG" >&2
  exit 1
fi

NEGATIVE_PROMPT="$(cat "$NEG_FILE")"

nohup bash -lc "
  set -euo pipefail
  # ai-gen-hub env defaults to Miguel; explicitly clear any prompt prefix/suffix.
  export ANIMATEDIFF_PROMPT_PREFIX=''
  export ANIMATEDIFF_PROMPT_SUFFIX=''
  echo '[animatediff] start: ' \$(date -Is)
  echo '[animatediff] ACCEPT_RUN_DIR=$ACCEPT_RUN_DIR'
  echo '[animatediff] CAPTIONS_DIR=$CAPTIONS_DIR'
  echo '[animatediff] OUT_DIR=$OUT_DIR'
  echo '[animatediff] BASE_CONFIG=$BASE_CONFIG'
  echo '[animatediff] CONDA_ENV=$CONDA_ENV'
  echo '[animatediff] LIMIT=$LIMIT LENGTH=$LENGTH STEPS=$STEPS HxW=${HEIGHT}x${WIDTH} SEED=$SEED'

  bash \"$ANIMATEDIFF_BATCH\" \\
    --captions-dir \"$CAPTIONS_DIR\" \\
    --out-dir \"$OUT_DIR/videos\" \\
    --base-config \"$BASE_CONFIG\" \\
    --prompt-prefix \"\" \\
    --prompt-suffix \"\" \\
    --length \"$LENGTH\" \\
    --steps \"$STEPS\" \\
    --height \"$HEIGHT\" \\
    --width \"$WIDTH\" \\
    --seed \"$SEED\" \\
    --negative-prompt \"$NEGATIVE_PROMPT\" \\
    --min-free-mib \"$MIN_FREE_MIB\" \\
    $( [[ \"$LIMIT\" != \"0\" ]] && echo \"--limit $LIMIT\" )

  echo '[animatediff] done: ' \$(date -Is)
" >"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "✅ animatediff batch launched (nohup)"
echo "PID: $pid"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"
echo "Out: $OUT_DIR/videos"
