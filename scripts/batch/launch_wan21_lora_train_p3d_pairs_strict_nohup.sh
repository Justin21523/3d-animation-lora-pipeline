#!/usr/bin/env bash
set -euo pipefail

# Train a Wan2.1 T2V 1.3B LoRA using the curated P3D pair videos dataset.
#
# Prereq:
#   - ai-gen-hub repo exists at /mnt/c/ai_projects/ai-gen-hub
#   - DiffSynth-Studio exists at /mnt/c/ai_tools/DiffSynth-Studio
#
# Dataset (auto-prepared under the latest AnimateDiff run folder):
#   outputs/p3d_animatediff_pairs_strict_*/dataset_pairs_strict_clean_wan21_832x480_16f_16fps
#
# Monitor:
#   tail -f logs/wan21_train_p3d_pairs_strict_*.log
#
# Stop:
#   kill $(cat logs/wan21_train_p3d_pairs_strict.nohup.pid)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

AI_GEN_HUB_ROOT="${AI_GEN_HUB_ROOT:-/mnt/c/ai_projects/ai-gen-hub}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${AI_GEN_HUB_ROOT}/scripts/train/train_wan21_lora_t2v_1_3b.sh}"

RUN_DIR="${RUN_DIR:-$(ls -1dt outputs/p3d_animatediff_pairs_strict_* | head -n1)}"
DATASET_DIR="${DATASET_DIR:-${RUN_DIR}/dataset_pairs_strict_clean_wan21_832x480_16f_16fps}"

MODEL_ROOT="${MODEL_ROOT:-/mnt/c/ai_models/wan2.1/Wan2.1-T2V-1.3B}"
MODEL_NAME="${MODEL_NAME:-Wan2.1-T2V-1.3B}"

OUT_DIR="${OUT_DIR:-/mnt/c/ai_models/wan2.1/lora/${MODEL_NAME}/p3d_pairs_strict_v1}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-16}"

EPOCHS="${EPOCHS:-5}"
DATASET_REPEAT="${DATASET_REPEAT:-3}"
LR="${LR:-1e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LORA_RANK="${LORA_RANK:-32}"
DATASET_WORKERS="${DATASET_WORKERS:-8}"
CPU_THREADS="${CPU_THREADS:-8}"
SAVE_STEPS="${SAVE_STEPS:-0}"

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/wan21_train_p3d_pairs_strict_${TS}.log"
PID_FILE="logs/wan21_train_p3d_pairs_strict.nohup.pid"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${existing_pid:-}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "ERROR: Already running (PID $existing_pid)."
    echo "Stop with: kill $existing_pid"
    exit 1
  fi
fi

if [[ ! -x "$TRAIN_SCRIPT" ]]; then
  echo "Train script not found/executable: $TRAIN_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset dir not found: $DATASET_DIR" >&2
  exit 1
fi

nohup bash -lc "
  set -euo pipefail
  echo '[wan21] start: ' \$(date -Is)
  echo '[wan21] RUN_DIR=$RUN_DIR'
  echo '[wan21] DATASET_DIR=$DATASET_DIR'
  echo '[wan21] OUT_DIR=$OUT_DIR'
  echo '[wan21] HxW=${HEIGHT}x${WIDTH} FRAMES=$NUM_FRAMES'
  echo '[wan21] EPOCHS=$EPOCHS REPEAT=$DATASET_REPEAT LR=$LR ACCUM=$GRAD_ACCUM RANK=$LORA_RANK'

  bash \"$TRAIN_SCRIPT\" \\
    --dataset-dir \"$DATASET_DIR\" \\
    --output-dir \"$OUT_DIR\" \\
    --model-root \"$MODEL_ROOT\" \\
    --model-name \"$MODEL_NAME\" \\
    --height \"$HEIGHT\" \\
    --width \"$WIDTH\" \\
    --num-frames \"$NUM_FRAMES\" \\
    --epochs \"$EPOCHS\" \\
    --dataset-repeat \"$DATASET_REPEAT\" \\
    --lr \"$LR\" \\
    --grad-accum \"$GRAD_ACCUM\" \\
    --lora-rank \"$LORA_RANK\" \\
    --dataset-workers \"$DATASET_WORKERS\" \\
    --cpu-threads \"$CPU_THREADS\" \\
    --save-steps \"$SAVE_STEPS\"

  echo '[wan21] done: ' \$(date -Is)
" >"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "✅ wan2.1 lora training launched (nohup)"
echo "PID: $pid"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"
echo "Out: $OUT_DIR"

