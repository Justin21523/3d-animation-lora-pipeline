#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

LOG_DIR="${1:-${PROJECT_ROOT}/logs}"
COUNT="${2:-5}"
LINES="${3:-20}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/monitor_logs.py" --log-dir "${LOG_DIR}" --count "${COUNT}" --lines "${LINES}"
