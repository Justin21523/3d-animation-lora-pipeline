#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <metadata1> [metadata2 ...]"
  exit 1
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/qc_samples.py" --metadata "$@"
