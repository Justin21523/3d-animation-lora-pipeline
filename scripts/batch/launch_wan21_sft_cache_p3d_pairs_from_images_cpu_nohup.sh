#!/usr/bin/env bash
set -euo pipefail

# Two-stage Wan2.1 LoRA training (stage 1/2): CPU cache (SFT data_process).
#
# This generates .pth caches (prompt embeddings + preprocessed inputs) on CPU, so the GPU training stage
# can skip the text encoder and avoid prompt-encoding OOM.
#
# Monitor:
#   tail -f logs/wan21_cache_p3d_pairs_from_images_*.log
#
# Stop:
#   kill $(cat logs/wan21_cache_p3d_pairs_from_images.nohup.pid)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

WAN21_DATASETS_ROOT="${WAN21_DATASETS_ROOT:-/mnt/data/datasets/general/wan2.1/lora_datasets}"
DATASET_DIR="${DATASET_DIR:-${WAN21_DATASETS_ROOT}/p3d_pairs_from_images_qc_mild_832x480_16fps_16f}"
DATASET_META="${DATASET_META:-${DATASET_DIR}/metadata.jsonl}"

# Training footprint knobs (must match the later train-from-cache stage).
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-448}"
NUM_FRAMES="${NUM_FRAMES:-9}"
WAN21_TOKENIZER_SEQ_LEN="${WAN21_TOKENIZER_SEQ_LEN:-32}"

# Where to write .pth cache shards.
CACHE_DIR="${CACHE_DIR:-${DATASET_DIR}/cache_sft_h${HEIGHT}x${WIDTH}_f${NUM_FRAMES}_seq${WAN21_TOKENIZER_SEQ_LEN}}"

# Use ai_env for Wan2.1 training (peft/accelerate compatibility).
WAN21_TRAIN_PY="${WAN21_TRAIN_PY:-/home/justin/miniconda3/envs/ai_env/bin/python}"
WAN21_ACCELERATE_BIN="${WAN21_ACCELERATE_BIN:-$(dirname "$WAN21_TRAIN_PY")/accelerate}"

DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-/mnt/c/ai_tools/DiffSynth-Studio}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/c/ai_models/wan2.1/Wan2.1-T2V-1.3B}"
MODEL_PATHS_JSON="${MODEL_PATHS_JSON:-[\"${MODEL_ROOT}/diffusion_pytorch_model.safetensors\",\"${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth\",\"${MODEL_ROOT}/Wan2.1_VAE.pth\"]}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ROOT}/google/umt5-xxl}"

DATASET_WORKERS="${DATASET_WORKERS:-4}"
CPU_THREADS="${CPU_THREADS:-8}"

# Only used for splitting units; LoRA is not applied in :data_process tasks.
LORA_RANK="${LORA_RANK:-8}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q,k,v,o,ffn.0,ffn.2}"

mkdir -p logs "$CACHE_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/wan21_cache_p3d_pairs_from_images_${TS}.log"
PID_FILE="logs/wan21_cache_p3d_pairs_from_images.nohup.pid"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${existing_pid:-}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "ERROR: Already running (PID $existing_pid)."
    echo "Stop with: kill $existing_pid"
    exit 1
  fi
fi

if [[ ! -f "$DATASET_META" ]]; then
  echo "Dataset not ready (missing metadata.jsonl): $DATASET_META" >&2
  exit 1
fi

if [[ ! -x "$WAN21_TRAIN_PY" ]]; then
  echo "Python not found/executable: $WAN21_TRAIN_PY" >&2
  exit 1
fi
if [[ ! -x "$WAN21_ACCELERATE_BIN" ]]; then
  echo "accelerate not found/executable: $WAN21_ACCELERATE_BIN" >&2
  exit 1
fi
if [[ ! -f "${DIFFSYNTH_ROOT}/examples/wanvideo/model_training/train.py" ]]; then
  echo "DiffSynth train.py not found: ${DIFFSYNTH_ROOT}/examples/wanvideo/model_training/train.py" >&2
  exit 1
fi

nohup bash -lc "
  set -euo pipefail
  echo '[wan21-cache] start:' \$(date -Is)
  echo '[wan21-cache] DATASET_DIR=$DATASET_DIR'
  echo '[wan21-cache] DATASET_META=$DATASET_META'
  echo '[wan21-cache] CACHE_DIR=$CACHE_DIR'
  echo '[wan21-cache] HxW=${HEIGHT}x${WIDTH} FRAMES=$NUM_FRAMES SEQ_LEN=$WAN21_TOKENIZER_SEQ_LEN'
  echo '[wan21-cache] WAN21_TRAIN_PY=$WAN21_TRAIN_PY'
  echo '[wan21-cache] WAN21_ACCELERATE_BIN=$WAN21_ACCELERATE_BIN'
  echo '[wan21-cache] DIFFSYNTH_ROOT=$DIFFSYNTH_ROOT'
  echo '[wan21-cache] MODEL_ROOT=$MODEL_ROOT'
  echo '[wan21-cache] DATASET_WORKERS=$DATASET_WORKERS CPU_THREADS=$CPU_THREADS'

  # Force CPU-only run (slow but stable).
  export CUDA_VISIBLE_DEVICES=\"\"
  export ACCELERATE_USE_CPU=1
  export TOKENIZERS_PARALLELISM=false
  export WAN21_TOKENIZER_SEQ_LEN=\"$WAN21_TOKENIZER_SEQ_LEN\"
  export PYTHONPATH=\"${DIFFSYNTH_ROOT}:\${PYTHONPATH:-}\"

  \"$WAN21_ACCELERATE_BIN\" launch --cpu --num_cpu_threads_per_process \"$CPU_THREADS\" \\
    \"${DIFFSYNTH_ROOT}/examples/wanvideo/model_training/train.py\" \\
    --task sft:data_process \\
    --initialize_model_on_cpu \\
    --dataset_base_path \"$DATASET_DIR\" \\
    --dataset_metadata_path \"$DATASET_META\" \\
    --dataset_repeat 1 \\
    --dataset_num_workers \"$DATASET_WORKERS\" \\
    --data_file_keys video \\
    --height \"$HEIGHT\" \\
    --width \"$WIDTH\" \\
    --num_frames \"$NUM_FRAMES\" \\
    --model_paths '$MODEL_PATHS_JSON' \\
    --tokenizer_path \"$TOKENIZER_PATH\" \\
    --output_path \"$CACHE_DIR\" \\
    --lora_base_model dit \\
    --lora_target_modules \"$LORA_TARGET_MODULES\" \\
    --lora_rank \"$LORA_RANK\" \\
    --use_gradient_checkpointing \\
    --gradient_accumulation_steps 1

  echo '[wan21-cache] done:' \$(date -Is)
" >"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "✅ wan2.1 sft:data_process cache launched (nohup)"
echo "PID: $pid"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"
echo "Cache: $CACHE_DIR"

