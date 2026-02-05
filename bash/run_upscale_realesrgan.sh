#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/upscale_realesrgan.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/upscale_frames_realesrgan.py" --config "${CONFIG_PATH}" "$@"
