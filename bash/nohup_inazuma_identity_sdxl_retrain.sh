#!/bin/bash
set -euo pipefail

# Full reset retrain for Inazuma Eleven identity LoRAs (SDXL).
#
# Outputs: /mnt/data/training/lora/inazuma_eleven/
# Logs:    /mnt/data/training/lora/inazuma_eleven/logs_master/

REPO_ROOT="/mnt/c/ai_projects/3d-animation-lora-pipeline"
OUT_ROOT="/mnt/data/training/lora/inazuma_eleven"
MASTER_LOG_DIR="/mnt/data/training/lora/_logs_master/inazuma_eleven"

mkdir -p "$MASTER_LOG_DIR"

LOG_FILE="$MASTER_LOG_DIR/retrain_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Inazuma Eleven SDXL identity LoRA retrain..."
echo "Repo: $REPO_ROOT"
echo "Output root: $OUT_ROOT"
echo "Log file: $LOG_FILE"

cd "$REPO_ROOT"

nohup python -m scripts.training.inazuma_identity_sdxl_retrain --clean-output-root \
  2>&1 | tee "$LOG_FILE" &

echo "PID: $!"
echo "Tail: tail -f $LOG_FILE"
