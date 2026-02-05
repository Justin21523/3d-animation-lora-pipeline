#!/usr/bin/env bash
#
# ============================================================================
# P3D Multi-Character Full Generation (Sequential)
# ============================================================================
#
# Runs, in order:
#   1) Multi-character interactions dataset generation (8k) (optional)
#   2) Single-character action+expression SDXL generation (round-robin)
#   3) Single-character pose SDXL generation (round-robin)
#
# This script is meant to be launched in the background (tmux/nohup), and is
# non-interactive by default.
#
# Defaults match the 2026-01-26 configs referenced in progress notes.
#
# Usage:
#   bash scripts/batch/run_p3d_multichar_full_generation.sh
#
# Common options:
#   --skip-interactions
#   --skip-action-expression
#   --skip-pose
#   --no-gpu-check
#   --fill-missing            (round-robin only; generate missing files only)
#   --no-resume               (round-robin only; ignore checkpoints)
#
# Author: Codex CLI
# Date: 2026-01-26
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-ai_env}"

INTERACTIONS_SCRIPT_DEFAULT="scripts/batch/run_p3d_multichar_interactions_8k.sh"
AE_CONFIG_DEFAULT="configs/batch/non_yokai_action_expression_100x5_20260126.yaml"
POSE_CONFIG_DEFAULT="configs/batch/non_yokai_pose_30x5_20260126.yaml"

RUN_INTERACTIONS=1
RUN_ACTION_EXPRESSION=1
RUN_POSE=1
GPU_CHECK=1

ROUND_ROBIN_ARGS=()

usage() {
  cat <<EOF
Usage: bash scripts/batch/run_p3d_multichar_full_generation.sh [options]

Options:
  --skip-interactions        Skip interactions dataset generation
  --skip-action-expression   Skip single-character action+expression generation
  --skip-pose                Skip single-character pose generation
  --no-gpu-check             Skip CUDA/GPU availability preflight checks
  --fill-missing             Round-robin: generate only missing files
  --no-resume                Round-robin: ignore checkpoints and start fresh
  -h, --help                 Show help

Environment:
  CONDA_ENV_NAME             Conda env to use (default: ai_env)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-interactions) RUN_INTERACTIONS=0; shift ;;
    --skip-action-expression) RUN_ACTION_EXPRESSION=0; shift ;;
    --skip-pose) RUN_POSE=0; shift ;;
    --no-gpu-check) GPU_CHECK=0; shift ;;
    --fill-missing) ROUND_ROBIN_ARGS+=("--fill-missing"); shift ;;
    --no-resume) ROUND_ROBIN_ARGS+=("--no-resume"); shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$PROJECT_ROOT"

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  # Try common conda install locations (non-interactive shells / tmux sessions).
  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "/usr/local/miniconda3/etc/profile.d/conda.sh"
  )
  local conda_sh=""
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      conda_sh="$c"
      break
    fi
  done

  if [[ -n "$conda_sh" ]]; then
    # shellcheck disable=SC1090
    source "$conda_sh"
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found in PATH (needed for 'conda run -n $CONDA_ENV_NAME ...')." >&2
    echo "       Fix by ensuring conda is installed and available in non-interactive shells," >&2
    echo "       or run this script from a shell where 'conda' is in PATH." >&2
    exit 1
  fi
}

DEFAULT_LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$DEFAULT_LOG_DIR"
LOG_FILE="${P3D_LOG_FILE:-$DEFAULT_LOG_DIR/p3d_multichar_full_generation_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

preflight_gpu_check() {
  if [[ "$GPU_CHECK" -ne 1 ]]; then
    log "GPU preflight check: SKIPPED (--no-gpu-check)"
    return 0
  fi

  log "GPU preflight check: nvidia-smi"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Are NVIDIA drivers installed?" >&2
    exit 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed. Fix NVIDIA driver/CUDA setup first." >&2
    nvidia-smi || true
    exit 1
  fi

  log "GPU preflight check: torch.cuda.is_available() in conda env '$CONDA_ENV_NAME'"
  if ! conda run -n "$CONDA_ENV_NAME" python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: torch.cuda.is_available() == False in conda env '$CONDA_ENV_NAME'." >&2
    echo "       Ensure the correct CUDA-enabled PyTorch build and a visible GPU." >&2
    conda run -n "$CONDA_ENV_NAME" python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('cuda_version', torch.version.cuda)" || true
    exit 1
  fi
}

run_step() {
  local name="$1"
  shift
  log "==== START: $name ===="
  "$@"
  log "==== DONE:  $name ===="
}

INTERACTIONS_SCRIPT="$INTERACTIONS_SCRIPT_DEFAULT"
AE_CONFIG="$AE_CONFIG_DEFAULT"
POSE_CONFIG="$POSE_CONFIG_DEFAULT"

log "Project root: $PROJECT_ROOT"
log "Log file: $LOG_FILE"
log "Conda env: $CONDA_ENV_NAME"
log "Interactions script: $INTERACTIONS_SCRIPT"
log "AE config: $AE_CONFIG"
log "Pose config: $POSE_CONFIG"
log "Round-robin args: ${ROUND_ROBIN_ARGS[*]:-(none)}"

ensure_conda
preflight_gpu_check

if [[ "$RUN_INTERACTIONS" -eq 1 ]]; then
  run_step "P3D interactions (8k)" bash "$INTERACTIONS_SCRIPT"
else
  log "Skipping interactions (--skip-interactions)"
fi

if [[ "$RUN_ACTION_EXPRESSION" -eq 1 ]]; then
  run_step "Single-char action+expression (round-robin)" \
    conda run -n "$CONDA_ENV_NAME" python scripts/batch/round_robin_image_generator.py \
      --config "$AE_CONFIG" "${ROUND_ROBIN_ARGS[@]}"
else
  log "Skipping action+expression (--skip-action-expression)"
fi

if [[ "$RUN_POSE" -eq 1 ]]; then
  run_step "Single-char pose (round-robin)" \
    conda run -n "$CONDA_ENV_NAME" python scripts/batch/round_robin_image_generator.py \
      --config "$POSE_CONFIG" "${ROUND_ROBIN_ARGS[@]}"
else
  log "Skipping pose (--skip-pose)"
fi

log "All requested steps completed."
