#!/usr/bin/env bash
set -euo pipefail

# Wait for trial3 training output, then:
#  1) Merge trial3 LoRA -> single SDXL checkpoint
#  2) Generate style generalization dataset (6000 images by default)
#
# This is designed to be launched immediately (while training is running).
#
# Usage:
#   bash scripts/batch/launch_p3d_trial3_merge_and_stylegen_nohup.sh
#
# Monitor:
#   tail -f logs/p3d_trial3_postprocess_*.log
#
# Stop:
#   kill $(cat logs/p3d_trial3_postprocess.nohup.pid)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV_GEN="${CONDA_ENV_GEN:-ai_env}"

TRIAL3_OUTPUT_DIR="${TRIAL3_OUTPUT_DIR:-/mnt/data/ai_data/models/lora/p3d_multichar/sdxl_checkpoint_trial3_stylecalib}"
CHECKPOINT_OUT_DIR="${CHECKPOINT_OUT_DIR:-/mnt/data/ai_data/models/checkpoints/p3d_multichar}"
DATA_OUT_ROOT="${DATA_OUT_ROOT:-/mnt/data/ai_data/synthetic_lora_data}"

NUM_PROMPTS="${NUM_PROMPTS:-1200}"
IMAGES_PER_PROMPT="${IMAGES_PER_PROMPT:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
STEPS="${STEPS:-35}"
CFG="${CFG:-6.0}"
SEED="${SEED:-42}"
OFFLOAD="${OFFLOAD:-0}"

POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p logs
LOG_FILE="logs/p3d_trial3_postprocess_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="logs/p3d_trial3_postprocess.nohup.pid"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${existing_pid:-}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "ERROR: Already running (PID $existing_pid)."
    echo "Stop with: kill $existing_pid"
    exit 1
  fi
fi

nohup bash -lc "
  set -euo pipefail
  echo '[post] start: ' \$(date -Is)
  echo '[post] TRIAL3_OUTPUT_DIR=$TRIAL3_OUTPUT_DIR'
  echo '[post] CHECKPOINT_OUT_DIR=$CHECKPOINT_OUT_DIR'
  echo '[post] DATA_OUT_ROOT=$DATA_OUT_ROOT'
  echo '[post] NUM_PROMPTS=$NUM_PROMPTS IMAGES_PER_PROMPT=$IMAGES_PER_PROMPT BATCH_SIZE=$BATCH_SIZE STEPS=$STEPS CFG=$CFG SEED=$SEED OFFLOAD=$OFFLOAD'
  mkdir -p \"$CHECKPOINT_OUT_DIR\"

  echo '[post] waiting for trial3 LoRA output (.safetensors)...'
  lora=''
  while true; do
    # pick the newest safetensors if any exists
    lora=\$(ls -1t \"$TRIAL3_OUTPUT_DIR\"/*.safetensors 2>/dev/null | head -n 1 || true)
    if [[ -n \"\$lora\" ]]; then
      echo \"[post] found LoRA: \$lora\"
      break
    fi
    echo \"[post] not found yet; sleeping ${POLL_SECONDS}s\"
    sleep \"$POLL_SECONDS\"
  done

  merged=\"$CHECKPOINT_OUT_DIR/p3d_multichar_sdxl_trial3_merged_\$(date +%Y%m%d_%H%M%S).safetensors\"
  echo \"[post] merging -> \$merged\"
  bash scripts/batch/merge_p3d_multichar_lora_to_checkpoint.sh --lora \"\$lora\" --out \"\$merged\"

  style_out=\"$DATA_OUT_ROOT/style_generalization_trial3_\$(date +%Y%m%d_%H%M%S)\"
  echo \"[post] generating style dataset -> \$style_out\"
  offload_args=''
  if [[ \"$OFFLOAD\" == '1' ]]; then
    offload_args='--offload'
  fi
  conda run -n \"$CONDA_ENV_GEN\" python scripts/batch/generate_p3d_style_generalization_dataset.py \\
    --checkpoint \"\$merged\" \\
    --out-dir \"\$style_out\" \\
    --num-prompts \"$NUM_PROMPTS\" \\
    --images-per-prompt \"$IMAGES_PER_PROMPT\" \\
    --batch-size \"$BATCH_SIZE\" \\
    --steps \"$STEPS\" \\
    --cfg \"$CFG\" \\
    --seed \"$SEED\" \\
    \$offload_args

  echo \"[post] done: \$(date -Is)\"
  echo \"[post] merged_checkpoint=\$merged\"
  echo \"[post] style_dataset=\$style_out\"
" >"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "✅ trial3 postprocess launched (nohup)"
echo "PID: $pid"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"
