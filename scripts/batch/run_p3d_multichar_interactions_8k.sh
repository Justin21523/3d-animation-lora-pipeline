#!/usr/bin/env bash
set -euo pipefail

# Build a token-aware pair-interaction dataset (8k) for multi-character SDXL checkpoint training.
#
# Pipeline:
#  1) Generate green-screen single-character images (uses existing BEST identity LoRAs)
#  2) Chroma-key cutout bank (RGBA + bbox json)
#  3) Compose 2-character interaction dataset with *pair* captions using p3d tokens
#
# Output (default):
#   /mnt/data/ai_data/synthetic_lora_data/interaction_pairs_dataset_p3d_8k_20260126/
#     images/*.png
#     captions/*.txt
#
# Config/Docs:
#   - Strategy: docs/3d-training/SDXL_MULTI_CHARACTER_CHECKPOINT.md
#   - Token map: configs/training/p3d_token_map_14chars.json
#   - Single-image bank config: configs/batch/interaction_single_generation.yaml

ROOT_DIR="${ROOT_DIR:-/mnt/c/ai_projects/3d-animation-lora-pipeline}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/ai_data/synthetic_lora_data}"
CONDA_ENV="${CONDA_ENV:-ai_env}"

NUM_IMAGES="${NUM_IMAGES:-8000}"
SEED="${SEED:-20260126}"
TRIGGER="${TRIGGER:-p3d_style}"
TOLERANCE="${TOLERANCE:-42}"

OUT_DIR="${OUT_DIR:-$DATA_ROOT/interaction_pairs_dataset_p3d_8k_20260126}"
TOKEN_MAP="${TOKEN_MAP:-$ROOT_DIR/configs/training/p3d_token_map_14chars.json}"

SKIP_SINGLE="${SKIP_SINGLE:-0}"     # 1 = skip step1 (single-character bank)
SKIP_CUTOUTS="${SKIP_CUTOUTS:-0}"   # 1 = skip step2 (cutout bank)
SKIP_COMPOSE="${SKIP_COMPOSE:-0}"   # 1 = skip step3 (pair compose)

cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1

echo "[p3d-pairs] start: $(date -Is)"
echo "[p3d-pairs] ROOT_DIR=$ROOT_DIR"
echo "[p3d-pairs] DATA_ROOT=$DATA_ROOT"
echo "[p3d-pairs] CONDA_ENV=$CONDA_ENV"
echo "[p3d-pairs] OUT_DIR=$OUT_DIR"
echo "[p3d-pairs] NUM_IMAGES=$NUM_IMAGES SEED=$SEED TRIGGER=$TRIGGER"
echo "[p3d-pairs] TOKEN_MAP=$TOKEN_MAP"

if [[ ! -f "$TOKEN_MAP" ]]; then
  echo "[p3d-pairs] ERROR: token map not found: $TOKEN_MAP"
  exit 1
fi

chars=(alberto alberto_seamonster barley_lightfoot bryce caleb elio giulia ian_lightfoot luca luca_seamonster miguel orion russell tyler)

if [[ "$SKIP_SINGLE" != "1" ]]; then
  echo "[p3d-pairs] step1: interaction_single_orchestrator (green-screen single character bank)"
  conda run -n "$CONDA_ENV" python -u scripts/batch/interaction_single_orchestrator.py \
    --config configs/batch/interaction_single_generation.yaml \
    --log-level INFO
else
  echo "[p3d-pairs] step1: skipped (SKIP_SINGLE=1)"
fi

if [[ "$SKIP_CUTOUTS" != "1" ]]; then
  echo "[p3d-pairs] step2: chroma key cutout bank"
  for char in "${chars[@]}"; do
    in_dir="$DATA_ROOT/interaction_single/$char/images"
    out_dir="$DATA_ROOT/interaction_cutouts/$char"
    if [[ ! -d "$in_dir" ]]; then
      echo "[p3d-pairs] WARN: missing input dir, skipping cutouts: $in_dir"
      continue
    fi
    echo "[p3d-pairs] cutout: $char in=$in_dir out=$out_dir tol=$TOLERANCE"
    conda run -n "$CONDA_ENV" python -u scripts/generic/training/interaction/chroma_key_cutout_bank.py \
      --in-dir "$in_dir" \
      --out-dir "$out_dir" \
      --tolerance "$TOLERANCE"
  done
else
  echo "[p3d-pairs] step2: skipped (SKIP_CUTOUTS=1)"
fi

if [[ "$SKIP_COMPOSE" != "1" ]]; then
  echo "[p3d-pairs] step3: compose pair dataset with p3d tokens"
  conda run -n "$CONDA_ENV" python -u scripts/generic/training/interaction/compose_interaction_dataset.py \
    --cutout-root "$DATA_ROOT/interaction_cutouts" \
    --out-dir "$OUT_DIR" \
    --num-images "$NUM_IMAGES" \
    --seed "$SEED" \
    --trigger "$TRIGGER" \
    --caption-mode pair \
    --char-token-map "$TOKEN_MAP"

  echo "[p3d-pairs] step3b: rewrite captions for multi-character checkpoint training"
  conda run -n "$CONDA_ENV" python -u scripts/generic/training/interaction/convert_pair_captions_to_multichar.py \
    --captions-dir "$OUT_DIR/captions"
else
  echo "[p3d-pairs] step3: skipped (SKIP_COMPOSE=1)"
fi

echo "[p3d-pairs] done: $(date -Is)"
