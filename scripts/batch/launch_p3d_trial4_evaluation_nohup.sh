#!/usr/bin/env bash
set -euo pipefail

# Trial4 detailed evaluation:
#  1) Merge trial4 LoRA into trial1 merged checkpoint -> new merged checkpoint
#  2) Generate baseline vs merged images for:
#     - v3 pair validation prompts (20 prompts) for 2 seeds
#     - all-pairs prompts (91 pairs) for 1 seed (optional)
#
# Usage:
#   bash scripts/batch/launch_p3d_trial4_evaluation_nohup.sh
#
# Monitor:
#   tail -f logs/p3d_trial4_eval_*.log
#
# Stop:
#   kill $(cat logs/p3d_trial4_eval.nohup.pid)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV_GEN="${CONDA_ENV_GEN:-ai_env}"

BASE_CKPT="${BASE_CKPT:-/mnt/data/ai_data/models/checkpoints/p3d_multichar/p3d_multichar_sdxl_trial1_merged_20260129_210411.safetensors}"
TRIAL4_LORA="${TRIAL4_LORA:-/mnt/data/ai_data/models/lora/p3d_multichar/sdxl_checkpoint_trial4_pairconsistency/p3d_multichar_sdxl_trial4_pairconsistency.safetensors}"
CKPT_OUT_DIR="${CKPT_OUT_DIR:-/mnt/data/ai_data/models/checkpoints/p3d_multichar}"

PROMPTS_V3="${PROMPTS_V3:-prompts/p3d/p3d_multichar_pair_validation_prompts_v3_inference.txt}"
PROMPTS_ALL_PAIRS="${PROMPTS_ALL_PAIRS:-prompts/p3d/p3d_multichar_all_pairs_prompts.txt}"

STEPS="${STEPS:-45}"
CFG="${CFG:-5.5}"

SEEDS_V3="${SEEDS_V3:-42 123}"
SEED_ALL_PAIRS="${SEED_ALL_PAIRS:-42}"
RUN_ALL_PAIRS="${RUN_ALL_PAIRS:-1}" # 0/1

OFFLOAD="${OFFLOAD:-1}" # 0/1 (generation safety)

NEGATIVE="${NEGATIVE:-collage, cutout, pasted, sticker, composited, inconsistent lighting, mismatched shadows, different color temperature, harsh specular, blown highlights, overexposed, extra person, three people, crowd, group, extra limbs, extra fingers, blurry, low quality, worst quality, watermark, text}"

mkdir -p logs outputs
LOG_FILE="logs/p3d_trial4_eval_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="logs/p3d_trial4_eval.nohup.pid"

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
  echo '[eval] start: ' \$(date -Is)
  echo '[eval] BASE_CKPT=$BASE_CKPT'
  echo '[eval] TRIAL4_LORA=$TRIAL4_LORA'
  echo '[eval] PROMPTS_V3=$PROMPTS_V3'
  echo '[eval] PROMPTS_ALL_PAIRS=$PROMPTS_ALL_PAIRS'
  echo '[eval] STEPS=$STEPS CFG=$CFG OFFLOAD=$OFFLOAD'
  echo '[eval] SEEDS_V3=$SEEDS_V3'
  echo '[eval] RUN_ALL_PAIRS=$RUN_ALL_PAIRS SEED_ALL_PAIRS=$SEED_ALL_PAIRS'

  merged=\"$CKPT_OUT_DIR/p3d_multichar_sdxl_trial4_pairconsistency_merged_\$(date +%Y%m%d_%H%M%S).safetensors\"
  echo \"[eval] merging trial4 LoRA -> \$merged\"
  bash scripts/batch/merge_p3d_multichar_lora_to_checkpoint.sh --lora \"$TRIAL4_LORA\" --base \"$BASE_CKPT\" --out \"\$merged\"

  echo \"[eval] merged_checkpoint=\$merged\"

  for seed in $SEEDS_V3; do
    out_base=\"outputs/p3d_eval_trial4_baseline_v3_seed\${seed}_\$(date +%Y%m%d_%H%M%S)\"
    out_merged=\"outputs/p3d_eval_trial4_merged_v3_seed\${seed}_\$(date +%Y%m%d_%H%M%S)\"

    echo \"[eval] baseline v3 seed=\$seed -> \$out_base\"
    conda run -n \"$CONDA_ENV_GEN\" python scripts/evaluation/p3d_multichar_checkpoint_smoke_test.py \\
      --checkpoint \"$BASE_CKPT\" \\
      --prompts \"$PROMPTS_V3\" \\
      --out-dir \"\$out_base\" \\
      --steps \"$STEPS\" \\
      --cfg \"$CFG\" \\
      --seed \"\$seed\" \\
      --limit 0 \\
      --negative \"$NEGATIVE\" \\
      \${offload_args[@]} \\
      --log-level INFO

    echo \"[eval] merged v3 seed=\$seed -> \$out_merged\"
    conda run -n \"$CONDA_ENV_GEN\" python scripts/evaluation/p3d_multichar_checkpoint_smoke_test.py \\
      --checkpoint \"\$merged\" \\
      --prompts \"$PROMPTS_V3\" \\
      --out-dir \"\$out_merged\" \\
      --steps \"$STEPS\" \\
      --cfg \"$CFG\" \\
      --seed \"\$seed\" \\
      --limit 0 \\
      --negative \"$NEGATIVE\" \\
      \${offload_args[@]} \\
      --log-level INFO
  done

  if [[ \"$RUN_ALL_PAIRS\" == '1' ]]; then
    out_base=\"outputs/p3d_eval_trial4_baseline_allpairs_seed${SEED_ALL_PAIRS}_\$(date +%Y%m%d_%H%M%S)\"
    out_merged=\"outputs/p3d_eval_trial4_merged_allpairs_seed${SEED_ALL_PAIRS}_\$(date +%Y%m%d_%H%M%S)\"

    echo \"[eval] baseline all-pairs seed=${SEED_ALL_PAIRS} -> \$out_base\"
    conda run -n \"$CONDA_ENV_GEN\" python scripts/evaluation/p3d_multichar_checkpoint_smoke_test.py \\
      --checkpoint \"$BASE_CKPT\" \\
      --prompts \"$PROMPTS_ALL_PAIRS\" \\
      --out-dir \"\$out_base\" \\
      --steps \"$STEPS\" \\
      --cfg \"$CFG\" \\
      --seed \"${SEED_ALL_PAIRS}\" \\
      --limit 0 \\
      --negative \"$NEGATIVE\" \\
      \${offload_args[@]} \\
      --log-level INFO

    echo \"[eval] merged all-pairs seed=${SEED_ALL_PAIRS} -> \$out_merged\"
    conda run -n \"$CONDA_ENV_GEN\" python scripts/evaluation/p3d_multichar_checkpoint_smoke_test.py \\
      --checkpoint \"\$merged\" \\
      --prompts \"$PROMPTS_ALL_PAIRS\" \\
      --out-dir \"\$out_merged\" \\
      --steps \"$STEPS\" \\
      --cfg \"$CFG\" \\
      --seed \"${SEED_ALL_PAIRS}\" \\
      --limit 0 \\
      --negative \"$NEGATIVE\" \\
      \${offload_args[@]} \\
      --log-level INFO
  fi

  echo \"[eval] done: \$(date -Is)\"
" >"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "✅ trial4 detailed evaluation launched (nohup)"
echo "PID: $pid"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"

