#!/bin/bash
# Fast SAM2 Restart - No Visualization
# Maximum speed with stability

echo "🚀 Restarting SAM2 in FAST MODE (no visualization)..."

# Kill old session
tmux kill-session -t luca_sam2 2>/dev/null && echo "✓ Old session terminated"

# Wait for GPU to clear
sleep 3

# Start new session WITHOUT --visualize for maximum speed
tmux new-session -d -s luca_sam2 "
conda run -n ai_env python scripts/generic/segmentation/instance_segmentation.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/frames_sampled \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled \
  --model sam2_hiera_large \
  --device cuda \
  2>&1 | tee logs/sam2_fast_$(date +%Y%m%d_%H%M%S).log
"

echo "✓ SAM2 FAST MODE started:"
echo "  ⚡ NO visualization (2-3x faster)"
echo "  🎯 points_per_side: 16"
echo "  💾 GPU cache clear: every 10 frames"
echo "  🛡️ Error handling: 3 retries + auto-skip"
echo "  📊 Expected speed: ~30 scenes/min (was ~13)"
echo ""
echo "Monitor: bash scripts/utils/monitor_sam2.sh"
