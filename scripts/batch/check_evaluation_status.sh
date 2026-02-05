#!/bin/bash
#
# Quick Evaluation Status Checker
# Shows current evaluation progress and GPU status
#

echo "================================================================================"
echo "🔍 SDXL LoRA Evaluation Status"
echo "================================================================================"
echo ""

# Check if evaluation process is running
echo "📊 Evaluation Process:"
if ps aux | grep "sdxl_lora_evaluator.py" | grep -v grep > /dev/null; then
    echo "  ✅ RUNNING"
    ps aux | grep "sdxl_lora_evaluator.py" | grep -v grep | awk '{print "  PID: " $2 " | CPU: " $3"% | MEM: " $4"%"}'
else
    echo "  ❌ NOT RUNNING"
fi
echo ""

# Check GPU status
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
    awk -F',' '{printf "  GPU: %s | VRAM: %s / %s | Temp: %s\n", $1, $2, $3, $4}'
echo ""

# Check tmux sessions
echo "📺 Tmux Sessions:"
echo "  Training:   $(tmux has-session -t sdxl_training 2>/dev/null && echo '✅ ACTIVE' || echo '❌ INACTIVE')"
echo "  Evaluation: $(tmux has-session -t checkpoint_eval 2>/dev/null && echo '✅ ACTIVE' || echo '❌ INACTIVE')"
echo ""

# Check latest checkpoint
echo "📦 Latest Checkpoint:"
ls -lht /mnt/data/ai_data/models/lora_sdxl/coco/miguel_identity/*.safetensors 2>/dev/null | head -1 | \
    awk '{print "  " $9 " (" $5 ") - " $6 " " $7 " " $8}'
echo ""

# Check evaluation outputs
echo "🎨 Evaluation Results:"
eval_dirs=$(find /mnt/data/ai_data/models/lora_sdxl/coco/miguel_identity -maxdepth 1 -type d -name "eval_*" 2>/dev/null | wc -l)
if [ $eval_dirs -gt 0 ]; then
    echo "  ✅ $eval_dirs checkpoint(s) evaluated"
    for eval_dir in $(find /mnt/data/ai_data/models/lora_sdxl/coco/miguel_identity -maxdepth 1 -type d -name "eval_*" 2>/dev/null | sort); do
        eval_name=$(basename "$eval_dir")
        num_images=$(find "$eval_dir" -name "*.png" 2>/dev/null | wc -l)
        has_results=$([ -f "$eval_dir/evaluation.json" ] && echo "✓" || echo "⏳")
        echo "     $has_results $eval_name ($num_images images)"
    done
else
    echo "  ⏳ No completed evaluations yet"
fi
echo ""

# Check monitor log
echo "📝 Latest Monitor Log:"
latest_log=$(ls -t /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/logs/checkpoint_monitor_miguel_*.log 2>/dev/null | head -1)
if [ -n "$latest_log" ]; then
    echo "  $(basename "$latest_log")"
    tail -5 "$latest_log" | sed 's/^/    /'
else
    echo "  No log file found"
fi
echo ""

echo "================================================================================"
echo "💡 Commands:"
echo "  View evaluation tmux: tmux attach -t checkpoint_eval"
echo "  View training tmux:   tmux attach -t sdxl_training"
echo "  Full monitor:         bash scripts/batch/monitor_sdxl_training.sh"
echo "================================================================================"
