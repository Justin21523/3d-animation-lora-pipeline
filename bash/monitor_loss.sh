#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

LOG_DIR="${1:-${PROJECT_ROOT}/logs}"
PATTERN="${2:-loss=}"
COUNT="${3:-5}"
TAIL="${4:-200}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/monitor_loss.py" --log-dir "${LOG_DIR}" --pattern "${PATTERN}" --count "${COUNT}" --tail "${TAIL}"
