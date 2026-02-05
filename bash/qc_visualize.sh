#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DETECTIONS="${1:-${PROJECT_ROOT}/metadata/detections.parquet}"
POSES="${2:-${PROJECT_ROOT}/metadata/poses.parquet}"
FG="${3:-${PROJECT_ROOT}/metadata/fg.parquet}"
OUTPUT="${4:-${PROJECT_ROOT}/outputs/qc_contact_sheet.png}"
SAMPLES="${5:-4}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/qc_visualize.py" \
  --detections "${DETECTIONS}" \
  --poses "${POSES}" \
  --fg "${FG}" \
  --output "${OUTPUT}" \
  --samples "${SAMPLES}"
