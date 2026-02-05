#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/build_embeddings.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_embeddings.py" --config "${CONFIG_PATH}" "$@"
