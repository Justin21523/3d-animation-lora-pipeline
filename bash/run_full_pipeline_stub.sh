#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Stub end-to-end pipeline: extract -> dedupe -> detect -> segment -> pose -> embeddings -> datasets -> inference -> animation -> upscale -> interpolate

echo "[1/10] extract_frames (stub)"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/extract_frames.py" --config "${PROJECT_ROOT}/configs/extract_frames.yaml" --use-stub

echo "[2/10] dedupe_frames"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/dedupe_frames.py" --config "${PROJECT_ROOT}/configs/dedupe_frames.yaml"

echo "[3/10] run_yolo_tracking (stub)"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_yolo_tracking.py" --config "${PROJECT_ROOT}/configs/run_yolo_tracking.yaml" --use-stub

echo "[4/10] segment_fg_bg (stub)"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/segment_fg_bg.py" --config "${PROJECT_ROOT}/configs/segment_fg_bg.yaml" --use-stub

echo "[5/10] extract_pose (stub)"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/extract_pose.py" --config "${PROJECT_ROOT}/configs/extract_pose.yaml" --use-stub

echo "[6/10] build_embeddings (stub)"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_embeddings.py" --config "${PROJECT_ROOT}/configs/build_embeddings.yaml" --use-stub

echo "[7/10] build_lora_dataset_characters (stub)"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_lora_dataset_characters.py" --config "${PROJECT_ROOT}/configs/build_lora_dataset_characters.yaml" --use-stub

echo "[8/10] build_controlnet_pose_dataset (stub)"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_controlnet_pose_dataset.py" --config "${PROJECT_ROOT}/configs/build_controlnet_pose_dataset.yaml" --use-stub

echo "[9/10] infer_lora_controlnet_pose (stub)"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/infer_lora_controlnet_pose.py" --config "${PROJECT_ROOT}/configs/infer_lora_controlnet_pose.yaml" --use-stub

echo "[10/10] generate_animation (stub) -> upscale -> interpolate"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/generate_animation_lora_controlnet_pose.py" --config "${PROJECT_ROOT}/configs/generate_animation_lora_controlnet_pose.yaml" --use-stub
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/upscale_frames_realesrgan.py" --config "${PROJECT_ROOT}/configs/upscale_realesrgan.yaml" --use-stub
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/interpolate_frames_rife.py" --config "${PROJECT_ROOT}/configs/interpolate_rife.yaml" --use-stub

echo "Stub pipeline completed."
