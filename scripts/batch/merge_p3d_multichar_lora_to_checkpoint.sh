#!/usr/bin/env bash
set -euo pipefail

# Merge a trained P3D multi-character SDXL LoRA into the SDXL base model,
# producing a single .safetensors checkpoint.
#
# Usage:
#   bash scripts/batch/merge_p3d_multichar_lora_to_checkpoint.sh \
#     --lora /path/to/lora.safetensors \
#     --out /path/to/p3d_multichar_merged.safetensors
#
# Optional:
#   --base /path/to/sd_xl_base_1.0.safetensors
#   --ratio 1.0
#
# Requires:
#   - Kohya_ss at /mnt/c/ai_tools/kohya_ss
#   - conda env: kohya_ss

KOHYA_DIR="${KOHYA_DIR:-/mnt/c/ai_tools/kohya_ss}"
CONDA_ENV="${CONDA_ENV:-kohya_ss}"

BASE_MODEL="${BASE_MODEL:-/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors}"
RATIO="${RATIO:-1.0}"

LORA=""
OUT=""

usage() {
  cat <<EOF
Usage: bash scripts/batch/merge_p3d_multichar_lora_to_checkpoint.sh --lora FILE --out FILE [--base FILE] [--ratio FLOAT]

Env:
  KOHYA_DIR   (default: /mnt/c/ai_tools/kohya_ss)
  CONDA_ENV   (default: kohya_ss)
  BASE_MODEL  (default: /mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors)
  RATIO       (default: 1.0)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lora) LORA="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --base) BASE_MODEL="${2:-}"; shift 2 ;;
    --ratio) RATIO="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$LORA" || -z "$OUT" ]]; then
  echo "ERROR: --lora and --out are required." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$BASE_MODEL" ]]; then
  echo "ERROR: base model not found: $BASE_MODEL" >&2
  exit 1
fi
if [[ ! -f "$LORA" ]]; then
  echo "ERROR: LoRA not found: $LORA" >&2
  exit 1
fi
if [[ ! -d "$KOHYA_DIR" ]]; then
  echo "ERROR: Kohya_ss not found: $KOHYA_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

echo "[p3d-merge] base=$BASE_MODEL"
echo "[p3d-merge] lora=$LORA"
echo "[p3d-merge] out=$OUT"
echo "[p3d-merge] ratio=$RATIO"

# Avoid `cd $KOHYA_DIR` because pyenv can shadow `python` via local `.python-version`,
# causing conda-run to pick the wrong interpreter. Use explicit PYTHONPATH and path.
export PYTHONPATH="$KOHYA_DIR/sd-scripts${PYTHONPATH:+:$PYTHONPATH}"

conda run -n "$CONDA_ENV" python "$KOHYA_DIR/sd-scripts/networks/sdxl_merge_lora.py" \
  --sd_model "$BASE_MODEL" \
  --save_to "$OUT" \
  --models "$LORA" \
  --ratios "$RATIO" \
  --precision "bf16" \
  --save_precision "bf16"

echo "[p3d-merge] done"
