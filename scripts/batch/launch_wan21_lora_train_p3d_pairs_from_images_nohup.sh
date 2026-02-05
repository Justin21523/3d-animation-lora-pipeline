#!/usr/bin/env bash
set -euo pipefail

# Train a Wan2.1 T2V 1.3B LoRA using the dataset built from the 4550 image-acceptance images (after mild QC).
#
# Note:
#   If you hit OOM during prompt encoding (text encoder), use the two-stage flow instead:
#     1) CPU cache: scripts/batch/launch_wan21_sft_cache_p3d_pairs_from_images_cpu_nohup.sh
#     2) GPU train: scripts/batch/launch_wan21_sft_train_from_cache_p3d_pairs_from_images_nohup.sh
#
# Prereq:
#   - First run: scripts/batch/launch_prepare_wan21_dataset_from_p3d_images_qc_mild_nohup.sh
#     which produces a dataset dir with videos/ + metadata.jsonl
#
# Monitor:
#   tail -f logs/wan21_train_p3d_pairs_from_images_*.log
#
# Stop:
#   kill $(cat logs/wan21_train_p3d_pairs_from_images.nohup.pid)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

AI_GEN_HUB_ROOT="${AI_GEN_HUB_ROOT:-/mnt/c/ai_projects/ai-gen-hub}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${AI_GEN_HUB_ROOT}/scripts/train/train_wan21_lora_t2v_1_3b.sh}"

WAN21_DATASETS_ROOT="${WAN21_DATASETS_ROOT:-/mnt/data/datasets/general/wan2.1/lora_datasets}"
DATASET_DIR="${DATASET_DIR:-${WAN21_DATASETS_ROOT}/p3d_pairs_from_images_qc_mild_832x480_16fps_16f}"

# Use ai_env for Wan2.1 training (peft/accelerate compatibility) while keeping animate_diff intact for AnimateDiff.
WAN21_TRAIN_PY="${WAN21_TRAIN_PY:-/home/justin/miniconda3/envs/ai_env/bin/python}"

MODEL_ROOT="${MODEL_ROOT:-/mnt/c/ai_models/wan2.1/Wan2.1-T2V-1.3B}"
MODEL_NAME="${MODEL_NAME:-Wan2.1-T2V-1.3B}"

OUT_DIR="${OUT_DIR:-/mnt/c/ai_models/wan2.1/lora/${MODEL_NAME}/p3d_pairs_from_images_qc_mild_v1}"

# Reduce training footprint (more stable on 16GB VRAM):
# - Use a smaller spatial size (must remain divisible by 16)
# - Use fewer frames (must satisfy time_division_factor=4, remainder=1)
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-448}"
NUM_FRAMES="${NUM_FRAMES:-9}"

EPOCHS="${EPOCHS:-5}"
DATASET_REPEAT="${DATASET_REPEAT:-3}"
LR="${LR:-1e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LORA_RANK="${LORA_RANK:-8}"
DATASET_WORKERS="${DATASET_WORKERS:-8}"
CPU_THREADS="${CPU_THREADS:-8}"
SAVE_STEPS="${SAVE_STEPS:-0}"
# Default to offloading both the text encoder + VAE (most stable on 16GB VRAM).
OFFLOAD_MODELS="${OFFLOAD_MODELS:-/mnt/c/ai_models/wan2.1/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth,/mnt/c/ai_models/wan2.1/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth}"
WAN21_TOKENIZER_SEQ_LEN="${WAN21_TOKENIZER_SEQ_LEN:-32}"
USE_GC_OFFLOAD="${USE_GC_OFFLOAD:-1}"

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/wan21_train_p3d_pairs_from_images_${TS}.log"
PID_FILE="logs/wan21_train_p3d_pairs_from_images.nohup.pid"

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
if [[ ! -f "$DATASET_DIR/metadata.jsonl" ]]; then
  echo "Dataset not ready (missing metadata.jsonl): $DATASET_DIR" >&2
  echo "Build it first with: bash scripts/batch/launch_prepare_wan21_dataset_from_p3d_images_qc_mild_nohup.sh" >&2
  exit 1
fi

nohup bash -lc "
  set -euo pipefail
  echo '[wan21] start: ' \$(date -Is)
  echo '[wan21] DATASET_DIR=$DATASET_DIR'
  echo '[wan21] OUT_DIR=$OUT_DIR'
  echo '[wan21] HxW=${HEIGHT}x${WIDTH} FRAMES=$NUM_FRAMES'
  echo '[wan21] EPOCHS=$EPOCHS REPEAT=$DATASET_REPEAT LR=$LR ACCUM=$GRAD_ACCUM RANK=$LORA_RANK'
  echo '[wan21] WAN21_TRAIN_PY=$WAN21_TRAIN_PY'
  echo '[wan21] OFFLOAD_MODELS=$OFFLOAD_MODELS'
  echo '[wan21] WAN21_TOKENIZER_SEQ_LEN=$WAN21_TOKENIZER_SEQ_LEN'
  echo '[wan21] USE_GC_OFFLOAD=$USE_GC_OFFLOAD'

  export WAN21_TRAIN_PY=\"$WAN21_TRAIN_PY\"
  export WAN21_TOKENIZER_SEQ_LEN=\"$WAN21_TOKENIZER_SEQ_LEN\"
  export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}\"
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
    --save-steps \"$SAVE_STEPS\" \\
    --offload-models \"$OFFLOAD_MODELS\" \\
    $( [[ \"$USE_GC_OFFLOAD\" == \"1\" ]] && echo \"--use-gc-offload\" )

  echo '[wan21] done: ' \$(date -Is)
" >"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "✅ wan2.1 lora training launched (nohup)"
echo "PID: $pid"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"
echo "Out: $OUT_DIR"
