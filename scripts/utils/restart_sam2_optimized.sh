#!/bin/bash
# Optimized SAM2 Restart Script
# Kills old session and starts with optimized parameters

echo "🔄 Restarting SAM2 with optimized settings..."

# Kill old session
tmux kill-session -t luca_sam2 2>/dev/null && echo "✓ Old session terminated"

# Wait for GPU to clear
sleep 3

# Start new session with optimized parameters
tmux new-session -d -s luca_sam2 "
conda run -n ai_env python scripts/generic/segmentation/instance_segmentation.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/frames_sampled \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled \
  --model sam2_hiera_large \
  --device cuda \
  --visualize \
  2>&1 | tee logs/sam2_optimized_$(date +%Y%m%d_%H%M%S).log
"

echo "✓ New SAM2 session started with:"
echo "  - points_per_side: 16 (was 32)"
echo "  - crop_n_layers: 0 (was 1)"
echo "  - GPU cache clear: every 10 frames (was 50)"
echo "  - Error handling: 3 retries + auto-skip"
echo ""
echo "Monitor with: bash scripts/utils/monitor_sam2.sh"
