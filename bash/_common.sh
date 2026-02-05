#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
