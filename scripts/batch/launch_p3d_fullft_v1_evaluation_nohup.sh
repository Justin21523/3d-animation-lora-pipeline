#!/usr/bin/env bash
set -euo pipefail

# Detailed evaluation for the full fine-tuned checkpoint vs baseline.
#
# Compares:
# - Baseline: trial1 merged checkpoint
# - Candidate: fullft v1 block fine-tune checkpoint
#
# Generates:
# - Solo validation prompts (identity sanity)
# - Pair v3 prompts (interaction cohesion)
# - All-pairs (91) prompts (coverage)
#
# Usage:
#   bash scripts/batch/launch_p3d_fullft_v1_evaluation_nohup.sh
#
# Monitor:
#   tail -f logs/p3d_fullft_eval_*.log
#
# Stop:
#   kill $(cat logs/p3d_fullft_eval.nohup.pid)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV_GEN="${CONDA_ENV_GEN:-ai_env}"

BASELINE_CKPT="${BASELINE_CKPT:-/mnt/data/ai_data/models/checkpoints/p3d_multichar/p3d_multichar_sdxl_trial1_merged_20260129_210411.safetensors}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-/mnt/data/ai_data/models/checkpoints/p3d_multichar/p3d_multichar_sdxl_fullft_v1_blockft.safetensors}"

PROMPTS_SOLO="${PROMPTS_SOLO:-prompts/p3d/p3d_multichar_validation_prompts.txt}"
PROMPTS_PAIR_V3="${PROMPTS_PAIR_V3:-prompts/p3d/p3d_multichar_pair_validation_prompts_v3_inference.txt}"
PROMPTS_ALL_PAIRS="${PROMPTS_ALL_PAIRS:-prompts/p3d/p3d_multichar_all_pairs_prompts.txt}"

STEPS="${STEPS:-45}"
CFG="${CFG:-5.5}"
SEEDS="${SEEDS:-42 123}"

RUN_SOLO="${RUN_SOLO:-1}"
RUN_PAIR_V3="${RUN_PAIR_V3:-1}"
RUN_ALL_PAIRS="${RUN_ALL_PAIRS:-1}"

OFFLOAD="${OFFLOAD:-1}" # 0/1

# Solo negative should strongly avoid duplicates.
NEG_SOLO="${NEG_SOLO:-two people, multiple people, crowd, group, duplicate character, collage, cutout, pasted, sticker, composited, inconsistent lighting, mismatched shadows, harsh specular, blown highlights, overexposed, extra limbs, extra fingers, blurry, low quality, worst quality, watermark, text}"

# Pair/all-pairs negative must NOT include 'two people'/'multiple people'.
NEG_PAIR="${NEG_PAIR:-collage, cutout, pasted, sticker, composited, inconsistent lighting, mismatched shadows, different color temperature, harsh specular, blown highlights, overexposed, extra person, three people, crowd, group, extra limbs, extra fingers, blurry, low quality, worst quality, watermark, text}"

mkdir -p logs outputs
LOG_FILE="logs/p3d_fullft_eval_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="logs/p3d_fullft_eval.nohup.pid"

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
  echo '[eval] BASELINE_CKPT=$BASELINE_CKPT'
  echo '[eval] CANDIDATE_CKPT=$CANDIDATE_CKPT'
  echo '[eval] STEPS=$STEPS CFG=$CFG SEEDS=$SEEDS OFFLOAD=$OFFLOAD'
  echo '[eval] RUN_SOLO=$RUN_SOLO RUN_PAIR_V3=$RUN_PAIR_V3 RUN_ALL_PAIRS=$RUN_ALL_PAIRS'

  for seed in $SEEDS; do
    ts=\$(date +%Y%m%d_%H%M%S)
    if [[ \"$RUN_SOLO\" == '1' ]]; then
      out_b=\"outputs/p3d_eval_fullft_baseline_solo_seed\${seed}_\$ts\"
      out_c=\"outputs/p3d_eval_fullft_candidate_solo_seed\${seed}_\$ts\"
      echo \"[eval] baseline solo seed=\$seed -> \$out_b\"
      conda run -n \"$CONDA_ENV_GEN\" python scripts/evaluation/p3d_multichar_checkpoint_smoke_test.py \\
        --checkpoint \"$BASELINE_CKPT\" \\
        --prompts \"$PROMPTS_SOLO\" \\
        --out-dir \"\$out_b\" \\
        --steps \"$STEPS\" \\
        --cfg \"$CFG\" \\
        --seed \"\$seed\" \\
        --limit 0 \\
        --negative \"$NEG_SOLO\" \\
        \${offload_args[@]} \\
        --log-level INFO
      echo \"[eval] candidate solo seed=\$seed -> \$out_c\"
      conda run -n \"$CONDA_ENV_GEN\" python scripts/evaluation/p3d_multichar_checkpoint_smoke_test.py \\
        --checkpoint \"$CANDIDATE_CKPT\" \\
        --prompts \"$PROMPTS_SOLO\" \\
        --out-dir \"\$out_c\" \\
        --steps \"$STEPS\" \\
        --cfg \"$CFG\" \\
        --seed \"\$seed\" \\
        --limit 0 \\
        --negative \"$NEG_SOLO\" \\
        \${offload_args[@]} \\
        --log-level INFO
    fi

    if [[ \"$RUN_PAIR_V3\" == '1' ]]; then
      out_b=\"outputs/p3d_eval_fullft_baseline_pairv3_seed\${seed}_\$ts\"
      out_c=\"outputs/p3d_eval_fullft_candidate_pairv3_seed\${seed}_\$ts\"
      echo \"[eval] baseline pairv3 seed=\$seed -> \$out_b\"
      conda run -n \"$CONDA_ENV_GEN\" python scripts/evaluation/p3d_multichar_checkpoint_smoke_test.py \\
        --checkpoint \"$BASELINE_CKPT\" \\
        --prompts \"$PROMPTS_PAIR_V3\" \\
        --out-dir \"\$out_b\" \\
        --steps \"$STEPS\" \\
        --cfg \"$CFG\" \\
        --seed \"\$seed\" \\
        --limit 0 \\
        --negative \"$NEG_PAIR\" \\
        \${offload_args[@]} \\
        --log-level INFO
      echo \"[eval] candidate pairv3 seed=\$seed -> \$out_c\"
      conda run -n \"$CONDA_ENV_GEN\" python scripts/evaluation/p3d_multichar_checkpoint_smoke_test.py \\
        --checkpoint \"$CANDIDATE_CKPT\" \\
        --prompts \"$PROMPTS_PAIR_V3\" \\
        --out-dir \"\$out_c\" \\
        --steps \"$STEPS\" \\
        --cfg \"$CFG\" \\
        --seed \"\$seed\" \\
        --limit 0 \\
        --negative \"$NEG_PAIR\" \\
        \${offload_args[@]} \\
        --log-level INFO
    fi

    if [[ \"$RUN_ALL_PAIRS\" == '1' ]]; then
      out_b=\"outputs/p3d_eval_fullft_baseline_allpairs_seed\${seed}_\$ts\"
      out_c=\"outputs/p3d_eval_fullft_candidate_allpairs_seed\${seed}_\$ts\"
      echo \"[eval] baseline allpairs seed=\$seed -> \$out_b\"
      conda run -n \"$CONDA_ENV_GEN\" python scripts/evaluation/p3d_multichar_checkpoint_smoke_test.py \\
        --checkpoint \"$BASELINE_CKPT\" \\
        --prompts \"$PROMPTS_ALL_PAIRS\" \\
        --out-dir \"\$out_b\" \\
        --steps \"$STEPS\" \\
        --cfg \"$CFG\" \\
        --seed \"\$seed\" \\
        --limit 0 \\
        --negative \"$NEG_PAIR\" \\
        \${offload_args[@]} \\
        --log-level INFO
      echo \"[eval] candidate allpairs seed=\$seed -> \$out_c\"
      conda run -n \"$CONDA_ENV_GEN\" python scripts/evaluation/p3d_multichar_checkpoint_smoke_test.py \\
        --checkpoint \"$CANDIDATE_CKPT\" \\
        --prompts \"$PROMPTS_ALL_PAIRS\" \\
        --out-dir \"\$out_c\" \\
        --steps \"$STEPS\" \\
        --cfg \"$CFG\" \\
        --seed \"\$seed\" \\
        --limit 0 \\
        --negative \"$NEG_PAIR\" \\
        \${offload_args[@]} \\
        --log-level INFO
    fi
  done

  echo \"[eval] done: \$(date -Is)\"
" >"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "✅ fullft detailed evaluation launched (nohup)"
echo "PID: $pid"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"

