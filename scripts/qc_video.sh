#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

VIDEO_PATH="${1:-${PROJECT_ROOT}/outputs/animation/animation.mp4}"

if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "[WARN] Video not found: ${VIDEO_PATH}"
  exit 0
fi

ffprobe -v error -show_entries stream=width,height,nb_frames,duration,codec_name -of default=noprint_wrappers=1 "${VIDEO_PATH}"
