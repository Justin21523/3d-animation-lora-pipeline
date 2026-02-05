#!/usr/bin/env bash
set -euo pipefail

# Qualitative evaluation for Wan2.1 LoRA checkpoints (base vs epoch-*.safetensors).
#
# Usage:
#   bash scripts/batch/launch_wan21_eval_p3d_pairs_from_images_nohup.sh
#   tail -f logs/wan21_eval_p3d_pairs_from_images_*.log
#
# Stop:
#   kill $(cat logs/wan21_eval_p3d_pairs_from_images.nohup.pid)

WAN21_DATASETS_ROOT="${WAN21_DATASETS_ROOT:-/mnt/data/datasets/general/wan2.1/lora_datasets}"
DATASET_NAME="${DATASET_NAME:-p3d_pairs_from_images_qc_mild_832x480_16fps_16f}"
DATASET_DIR="${DATASET_DIR:-${WAN21_DATASETS_ROOT}/${DATASET_NAME}}"
METADATA_JSONL="${METADATA_JSONL:-${DATASET_DIR}/metadata.jsonl}"

MODEL_ROOT="${MODEL_ROOT:-/mnt/c/ai_models/wan2.1/Wan2.1-T2V-1.3B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ROOT}/google/umt5-xxl}"
LORA_DIR="${LORA_DIR:-/mnt/c/ai_models/wan2.1/lora/Wan2.1-T2V-1.3B/p3d_pairs_from_images_qc_mild_v1}"

OUT_DIR="${OUT_DIR:-outputs/evaluation/wan21/${DATASET_NAME}/$(date +%Y%m%d_%H%M%S)}"

NUM_SAMPLES="${NUM_SAMPLES:-8}"
BASE_SEED="${BASE_SEED:-0}"
STEPS="${STEPS:-30}"
LORA_ALPHA="${LORA_ALPHA:-0.3}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"

# Keep inference settings aligned with training cache
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-448}"
NUM_FRAMES="${NUM_FRAMES:-9}"
WAN21_TOKENIZER_SEQ_LEN="${WAN21_TOKENIZER_SEQ_LEN:-32}"

DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-/mnt/c/ai_tools/DiffSynth-Studio}"
PYTHON_BIN="${PYTHON_BIN:-/home/justin/miniconda3/envs/ai_env/bin/python}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/wan21_eval_p3d_pairs_from_images_${TS}.log"
PID_FILE="logs/wan21_eval_p3d_pairs_from_images.nohup.pid"

INCLUDE_BASE_FLAG=""
if [[ "${INCLUDE_BASE}" == "1" ]]; then
  INCLUDE_BASE_FLAG="--include-base"
fi

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$OUT_DIR")"

if [[ ! -f "${DIFFSYNTH_ROOT}/diffsynth/__init__.py" ]]; then
  echo "DiffSynth root not found: ${DIFFSYNTH_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${METADATA_JSONL}" ]]; then
  echo "metadata.jsonl not found: ${METADATA_JSONL}" >&2
  exit 1
fi

if [[ ! -d "${LORA_DIR}" ]]; then
  echo "LoRA dir not found: ${LORA_DIR}" >&2
  exit 1
fi

nohup bash -lc "
  set -euo pipefail
  echo '[wan21-eval] start:' \$(date -Is)
  echo '[wan21-eval] MODEL_ROOT=${MODEL_ROOT}'
  echo '[wan21-eval] TOKENIZER_PATH=${TOKENIZER_PATH}'
  echo '[wan21-eval] LORA_DIR=${LORA_DIR}'
  echo '[wan21-eval] METADATA_JSONL=${METADATA_JSONL}'
  echo '[wan21-eval] OUT_DIR=${OUT_DIR}'
  echo '[wan21-eval] HxW=${HEIGHT}x${WIDTH} FRAMES=${NUM_FRAMES} SEQ_LEN=${WAN21_TOKENIZER_SEQ_LEN}'
  echo '[wan21-eval] NUM_SAMPLES=${NUM_SAMPLES} BASE_SEED=${BASE_SEED} STEPS=${STEPS}'
  echo '[wan21-eval] LORA_ALPHA=${LORA_ALPHA} INCLUDE_BASE=${INCLUDE_BASE}'

  export TOKENIZERS_PARALLELISM=false
  export WAN21_TOKENIZER_SEQ_LEN='${WAN21_TOKENIZER_SEQ_LEN}'
  export PYTHONPATH='${DIFFSYNTH_ROOT}:'\"\${PYTHONPATH:-}\"

  \"${PYTHON_BIN}\" scripts/evaluation/evaluate_wan21_lora_checkpoints.py \\
    --model-root \"${MODEL_ROOT}\" \\
    --tokenizer-path \"${TOKENIZER_PATH}\" \\
    --lora-dir \"${LORA_DIR}\" \\
    --metadata-jsonl \"${METADATA_JSONL}\" \\
    --out-dir \"${OUT_DIR}\" \\
    ${INCLUDE_BASE_FLAG} \\
    --num-samples \"${NUM_SAMPLES}\" \\
    --base-seed \"${BASE_SEED}\" \\
    --alpha \"${LORA_ALPHA}\" \\
    --height \"${HEIGHT}\" \\
    --width \"${WIDTH}\" \\
    --num-frames \"${NUM_FRAMES}\" \\
    --steps \"${STEPS}\" \\
    --tiled

  echo '[wan21-eval] done:' \$(date -Is)
" >"$LOG_FILE" 2>&1 &

echo $! >"$PID_FILE"
echo "✅ wan2.1 evaluation launched (nohup)"
echo "  - log: $LOG_FILE"
echo "  - pid: $(cat "$PID_FILE")"
echo "  - out: $OUT_DIR"
