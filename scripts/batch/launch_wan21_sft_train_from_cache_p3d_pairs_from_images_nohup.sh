#!/usr/bin/env bash
set -euo pipefail

# Two-stage Wan2.1 LoRA training (stage 2/2): GPU train (SFT :train) from .pth cache.
#
# Prereq:
#   - Run: scripts/batch/launch_wan21_sft_cache_p3d_pairs_from_images_cpu_nohup.sh
#
# Monitor:
#   tail -f logs/wan21_train_from_cache_p3d_pairs_from_images_*.log
#
# Stop:
#   kill $(cat logs/wan21_train_from_cache_p3d_pairs_from_images.nohup.pid)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

WAN21_DATASETS_ROOT="${WAN21_DATASETS_ROOT:-/mnt/data/datasets/general/wan2.1/lora_datasets}"
DATASET_DIR="${DATASET_DIR:-${WAN21_DATASETS_ROOT}/p3d_pairs_from_images_qc_mild_832x480_16fps_16f}"

# Must match the cache stage.
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-448}"
NUM_FRAMES="${NUM_FRAMES:-9}"
WAN21_TOKENIZER_SEQ_LEN="${WAN21_TOKENIZER_SEQ_LEN:-32}"

CACHE_DIR="${CACHE_DIR:-${DATASET_DIR}/cache_sft_h${HEIGHT}x${WIDTH}_f${NUM_FRAMES}_seq${WAN21_TOKENIZER_SEQ_LEN}}"

WAIT_FOR_CACHE="${WAIT_FOR_CACHE:-1}"
WAIT_FOR_CACHE_DONE="${WAIT_FOR_CACHE_DONE:-1}"
CACHE_PID_FILE="${CACHE_PID_FILE:-logs/wan21_cache_p3d_pairs_from_images.nohup.pid}"
POLL_SECS="${POLL_SECS:-60}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-0}" # 0 = wait forever

WAN21_TRAIN_PY="${WAN21_TRAIN_PY:-/home/justin/miniconda3/envs/ai_env/bin/python}"
WAN21_ACCELERATE_BIN="${WAN21_ACCELERATE_BIN:-$(dirname "$WAN21_TRAIN_PY")/accelerate}"

DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-/mnt/c/ai_tools/DiffSynth-Studio}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/c/ai_models/wan2.1/Wan2.1-T2V-1.3B}"
MODEL_NAME="${MODEL_NAME:-Wan2.1-T2V-1.3B}"
MODEL_PATHS_JSON="${MODEL_PATHS_JSON:-[\"${MODEL_ROOT}/diffusion_pytorch_model.safetensors\",\"${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth\",\"${MODEL_ROOT}/Wan2.1_VAE.pth\"]}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ROOT}/google/umt5-xxl}"

OUT_DIR="${OUT_DIR:-/mnt/c/ai_models/wan2.1/lora/${MODEL_NAME}/p3d_pairs_from_images_qc_mild_v1}"

EPOCHS="${EPOCHS:-5}"
DATASET_REPEAT="${DATASET_REPEAT:-3}"
LR="${LR:-1e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LORA_RANK="${LORA_RANK:-8}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q,k,v,o,ffn.0,ffn.2}"

DATASET_WORKERS="${DATASET_WORKERS:-2}"
CPU_THREADS="${CPU_THREADS:-8}"
SAVE_STEPS="${SAVE_STEPS:-0}"

# Keep non-DiT models off GPU during the train split (text encoder + VAE are not used in :train units).
OFFLOAD_MODELS="${OFFLOAD_MODELS:-${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth,${MODEL_ROOT}/Wan2.1_VAE.pth}"
USE_GC_OFFLOAD="${USE_GC_OFFLOAD:-1}"

mkdir -p logs "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/wan21_train_from_cache_p3d_pairs_from_images_${TS}.log"
PID_FILE="logs/wan21_train_from_cache_p3d_pairs_from_images.nohup.pid"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${existing_pid:-}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "ERROR: Already running (PID $existing_pid)."
    echo "Stop with: kill $existing_pid"
    exit 1
  fi
fi

if [[ ! -d "$CACHE_DIR" ]]; then
  echo "Cache dir not found: $CACHE_DIR" >&2
  echo "Run first: bash scripts/batch/launch_wan21_sft_cache_p3d_pairs_from_images_cpu_nohup.sh" >&2
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
  echo '[wan21-train] start:' \$(date -Is)
  echo '[wan21-train] CACHE_DIR=$CACHE_DIR'
  echo '[wan21-train] OUT_DIR=$OUT_DIR'
  echo '[wan21-train] HxW=${HEIGHT}x${WIDTH} FRAMES=$NUM_FRAMES SEQ_LEN=$WAN21_TOKENIZER_SEQ_LEN'
  echo '[wan21-train] EPOCHS=$EPOCHS REPEAT=$DATASET_REPEAT LR=$LR ACCUM=$GRAD_ACCUM RANK=$LORA_RANK'
  echo '[wan21-train] OFFLOAD_MODELS=$OFFLOAD_MODELS'
  echo '[wan21-train] USE_GC_OFFLOAD=$USE_GC_OFFLOAD'
  echo '[wan21-train] WAIT_FOR_CACHE=$WAIT_FOR_CACHE WAIT_FOR_CACHE_DONE=$WAIT_FOR_CACHE_DONE POLL_SECS=$POLL_SECS MAX_WAIT_SECS=$MAX_WAIT_SECS'

  export TOKENIZERS_PARALLELISM=false
  export WAN21_TOKENIZER_SEQ_LEN=\"$WAN21_TOKENIZER_SEQ_LEN\"
  export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}\"
  export PYTHONPATH=\"${DIFFSYNTH_ROOT}:\${PYTHONPATH:-}\"

  wait_for_cache=\"$WAIT_FOR_CACHE\"
  wait_for_cache_done=\"$WAIT_FOR_CACHE_DONE\"
  cache_pid_file=\"$CACHE_PID_FILE\"
  poll_secs=\"$POLL_SECS\"
  max_wait_secs=\"$MAX_WAIT_SECS\"
  start_ts=\$(date +%s)
  while true; do
    has_pth=\"0\"
    if find \"$CACHE_DIR\" -type f -name '*.pth' -print -quit | grep -q .; then
      has_pth=\"1\"
    fi

    cache_running=\"0\"
    if [[ -f \"\$cache_pid_file\" ]]; then
      cache_pid=\"\$(cat \"\$cache_pid_file\" 2>/dev/null || true)\"
      if [[ -n \"\${cache_pid:-}\" ]] && kill -0 \"\$cache_pid\" 2>/dev/null; then
        cache_cmd=\"\$(ps -p \"\$cache_pid\" -o cmd= 2>/dev/null || true)\"
        if echo \"\$cache_cmd\" | grep -q -- \"sft:data_process\"; then
          cache_running=\"1\"
        else
          cache_running=\"0\"
        fi
      fi
    fi

    if [[ \"\$wait_for_cache\" != \"1\" ]]; then
      break
    fi

    if [[ \"\$wait_for_cache_done\" == \"1\" ]]; then
      if [[ \"\$cache_running\" == \"0\" && \"\$has_pth\" == \"1\" ]]; then
        break
      fi
    else
      if [[ \"\$has_pth\" == \"1\" ]]; then
        break
      fi
    fi

    if [[ \"\$max_wait_secs\" != \"0\" ]]; then
      now_ts=\$(date +%s)
      elapsed=\$((now_ts - start_ts))
      if (( elapsed > max_wait_secs )); then
        echo \"[wan21-train] ERROR: timed out waiting for cache (MAX_WAIT_SECS=\$max_wait_secs)\" >&2
        exit 1
      fi
    fi

    echo \"[wan21-train] waiting for cache... has_pth=\$has_pth cache_running=\$cache_running cache_dir=$CACHE_DIR\" >&2
    sleep \"\$poll_secs\"
  done

  if ! find \"$CACHE_DIR\" -type f -name '*.pth' -print -quit | grep -q .; then
    echo \"[wan21-train] ERROR: cache dir has no .pth shards: $CACHE_DIR\" >&2
    exit 1
  fi

  \"$WAN21_ACCELERATE_BIN\" launch --mixed_precision bf16 --num_cpu_threads_per_process \"$CPU_THREADS\" \\
    \"${DIFFSYNTH_ROOT}/examples/wanvideo/model_training/train.py\" \\
    --task sft:train \\
    --dataset_base_path \"$CACHE_DIR\" \\
    --dataset_repeat \"$DATASET_REPEAT\" \\
    --dataset_num_workers \"$DATASET_WORKERS\" \\
    --data_file_keys video \\
    --height \"$HEIGHT\" \\
    --width \"$WIDTH\" \\
    --num_frames \"$NUM_FRAMES\" \\
    --model_paths '$MODEL_PATHS_JSON' \\
    --tokenizer_path \"$TOKENIZER_PATH\" \\
    --learning_rate \"$LR\" \\
    --num_epochs \"$EPOCHS\" \\
    --remove_prefix_in_ckpt pipe.dit. \\
    --output_path \"$OUT_DIR\" \\
    --lora_base_model dit \\
    --lora_target_modules \"$LORA_TARGET_MODULES\" \\
    --lora_rank \"$LORA_RANK\" \\
    --use_gradient_checkpointing \\
    --gradient_accumulation_steps \"$GRAD_ACCUM\" \\
    --offload_models \"$OFFLOAD_MODELS\" \\
    $( [[ \"$SAVE_STEPS\" != \"0\" ]] && echo \"--save_steps $SAVE_STEPS\" ) \\
    $( [[ \"$USE_GC_OFFLOAD\" == \"1\" ]] && echo \"--use_gradient_checkpointing_offload\" )

  echo '[wan21-train] done:' \$(date -Is)
" >"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "✅ wan2.1 sft:train launched (nohup, from cache)"
echo "PID: $pid"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"
echo "Cache: $CACHE_DIR"
echo "Out: $OUT_DIR"
