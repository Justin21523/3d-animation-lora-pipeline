#!/usr/bin/env bash
set -euo pipefail

# Universal Two-Character Interaction LoRA - Data Prep Pipeline
#
# Steps:
#  0) (optional) wait for yokai-watch generator to finish
#  1) generate green-screen single-character images for 14 characters
#  2) chroma-key cutout bank (RGBA)
#  3) compose 2-character interaction dataset (identity-agnostic captions)
#
# This script is designed to be run in background (nohup) and is restart-friendly:
#  - interaction_single_orchestrator has its own checkpoint
#  - per-character image generation uses batch_image_generator checkpointing
#  - cutout step is idempotent (overwrites same filenames)

ROOT_DIR="/mnt/c/ai_projects/3d-animation-lora-pipeline"
DATA_ROOT="/mnt/data/ai_data/synthetic_lora_data"
CONDA_ENV="ai_env"
PYTHON="/home/justin/miniconda3/envs/${CONDA_ENV}/bin/python"

WAIT_FOR_YOKAI="${WAIT_FOR_YOKAI:-1}"   # 1 = wait, 0 = start immediately
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"

PAIR_IMAGES="${PAIR_IMAGES:-320}"
PAIR_SEED="${PAIR_SEED:-1234}"
PAIR_TRIGGER="${PAIR_TRIGGER:-pair_interaction}"

echo "[universal-interaction] start: $(date -Is)"
echo "[universal-interaction] ROOT_DIR=$ROOT_DIR"
echo "[universal-interaction] DATA_ROOT=$DATA_ROOT"
echo "[universal-interaction] CONDA_ENV=$CONDA_ENV"
echo "[universal-interaction] PYTHON=$PYTHON"

cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

if [[ "$WAIT_FOR_YOKAI" == "1" ]]; then
  echo "[universal-interaction] waiting for yokai-watch generator to finish..."
  while pgrep -f "yokai_watch_round_robin_generator.py" >/dev/null 2>&1; do
    echo "[universal-interaction] yokai-watch still running; sleeping ${SLEEP_SECONDS}s..."
    sleep "$SLEEP_SECONDS"
  done
  echo "[universal-interaction] yokai-watch generator not detected; continuing."
fi

echo "[universal-interaction] step1: interaction_single_orchestrator"
if [[ ! -x "$PYTHON" ]]; then
  echo "[universal-interaction] ERROR: python not found: $PYTHON"
  exit 1
fi

"$PYTHON" -u scripts/batch/interaction_single_orchestrator.py \
  --config configs/batch/interaction_single_generation.yaml \
  --log-level INFO

echo "[universal-interaction] step2: chroma key cutout bank"
python - <<'PY'
from pathlib import Path
chars = [
  "alberto","alberto_seamonster","barley_lightfoot","bryce","caleb","elio",
  "giulia","ian_lightfoot","luca","luca_seamonster","miguel","orion","russell","tyler"
]
data_root = Path("/mnt/data/ai_data/synthetic_lora_data")
for c in chars:
    in_dir = data_root / "interaction_single" / c / "images"
    out_dir = data_root / "interaction_cutouts" / c
    print(f"[cutout] {c} in={in_dir} out={out_dir}")
PY

for char in alberto alberto_seamonster barley_lightfoot bryce caleb elio giulia ian_lightfoot luca luca_seamonster miguel orion russell tyler; do
  "$PYTHON" -u scripts/generic/training/interaction/chroma_key_cutout_bank.py \
    --in-dir "$DATA_ROOT/interaction_single/$char/images" \
    --out-dir "$DATA_ROOT/interaction_cutouts/$char" \
    --tolerance 42
done

echo "[universal-interaction] step3: compose interaction dataset"
"$PYTHON" -u scripts/generic/training/interaction/compose_interaction_dataset.py \
  --cutout-root "$DATA_ROOT/interaction_cutouts" \
  --out-dir "$DATA_ROOT/interaction_pairs_dataset" \
  --num-images "$PAIR_IMAGES" \
  --seed "$PAIR_SEED" \
  --trigger "$PAIR_TRIGGER"

echo "[universal-interaction] done: $(date -Is)"
