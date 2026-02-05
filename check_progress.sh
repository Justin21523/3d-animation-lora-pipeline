#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Round-Robin Synthetic Data Generation Progress        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if checkpoint file exists
CHECKPOINT="/mnt/data/ai_data/synthetic_lora_data/checkpoints/round_robin_checkpoint.json"
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint file not found!"
    exit 1
fi

# Parse checkpoint data
echo "📊 Progress Summary:"
echo "─────────────────────────────────────────────────────────────"
python3 << 'EOF'
import json
import sys
from datetime import datetime

try:
    with open('/mnt/data/ai_data/synthetic_lora_data/checkpoints/round_robin_checkpoint.json', 'r') as f:
        data = json.load(f)

    current_round = data['current_round']
    total_generated = data['total_generated']
    target_rounds = 800

    # Calculate percentages
    round_pct = (current_round / target_rounds) * 100

    # Estimate remaining
    remaining_rounds = target_rounds - current_round
    avg_per_round = total_generated / max(current_round, 1)
    estimated_remaining = remaining_rounds * avg_per_round
    estimated_total = total_generated + estimated_remaining

    print(f"  Current Round: {current_round} / {target_rounds} ({round_pct:.1f}%)")
    print(f"  Total Images: {total_generated:,}")
    print(f"  Avg per Round: {avg_per_round:.1f} images")
    print(f"  Estimated Total: {estimated_total:,.0f} images")
    print(f"  Remaining: ~{estimated_remaining:,.0f} images")

except Exception as e:
    print(f"Error reading checkpoint: {e}")
    sys.exit(1)
EOF

echo ""
echo "⚙️  Process Status:"
echo "─────────────────────────────────────────────────────────────"
if pgrep -f "round_robin_generator.py" > /dev/null; then
    ps aux | grep "round_robin_generator.py" | grep -v grep | awk '{printf "  ✅ Running (PID: %s)\n  CPU: %s%% | Memory: %s%% | Runtime: %s\n", $2, $3, $4, $10}'
else
    echo "  ❌ Not running"
fi

echo ""
echo "🖼️  Latest Generated Images (most recent 5):"
echo "─────────────────────────────────────────────────────────────"
ls -lt /mnt/data/ai_data/synthetic_lora_data/generated_data/*/*/generated/*.png 2>/dev/null | head -5 | while read line; do
    file=$(echo "$line" | awk '{print $9}')
    time=$(echo "$line" | awk '{print $6, $7, $8}')
    basename=$(basename "$file")
    char=$(echo "$file" | cut -d'/' -f8)
    type=$(echo "$file" | cut -d'/' -f9)
    echo "  📸 $char/$type: $basename"
done

echo ""
echo "📈 Per-Character Statistics:"
echo "─────────────────────────────────────────────────────────────"
for char in barley_lightfoot bryce caleb elio giulia ian_lightfoot luca luca_seamonster miguel orion russell tyler; do
    pose_count=$(find /mnt/data/ai_data/synthetic_lora_data/generated_data/$char/pose/generated -name "*.png" 2>/dev/null | wc -l)
    expr_count=$(find /mnt/data/ai_data/synthetic_lora_data/generated_data/$char/expression/generated -name "*.png" 2>/dev/null | wc -l)
    action_count=$(find /mnt/data/ai_data/synthetic_lora_data/generated_data/$char/action/generated -name "*.png" 2>/dev/null | wc -l)
    total=$((pose_count + expr_count + action_count))

    if [ $total -gt 0 ]; then
        printf "  %-18s P:%-4d E:%-4d A:%-4d Total:%-5d\n" "$char" $pose_count $expr_count $action_count $total
    fi
done

echo ""
echo "⏰ Time Information:"
echo "─────────────────────────────────────────────────────────────"
echo "  Current Time: $(date '+%Y-%m-%d %H:%M:%S')"

# Get first generated image timestamp
first_img=$(ls -t /mnt/data/ai_data/synthetic_lora_data/generated_data/*/*/generated/*.png 2>/dev/null | tail -1)
if [ -n "$first_img" ]; then
    first_time=$(stat -c %y "$first_img" | cut -d'.' -f1)
    echo "  Start Time: $first_time"
fi

echo ""
echo "💡 Quick Commands:"
echo "─────────────────────────────────────────────────────────────"
echo "  Monitor live:  watch -n 30 bash check_progress.sh"
echo "  View log:      tail -f /mnt/data/ai_data/synthetic_lora_data/logs/round_robin_generation.log"
echo "  Stop process:  pkill -f round_robin_generator.py"
echo "  GPU status:    nvidia-smi"
echo ""
