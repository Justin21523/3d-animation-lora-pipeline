#!/bin/bash
# Check training progress and status
# 檢查訓練進度和狀態

echo "========================================================================"
echo "TRAINING PROGRESS REPORT"
echo "========================================================================"
echo

# Check if training is running
TRAINING_PID=$(ps aux | grep "[l]aunch_iterative_training.py" | awk '{print $2}')
if [ -z "$TRAINING_PID" ]; then
    echo "Status: ❌ NOT RUNNING"
else
    echo "Status: ✅ RUNNING (PID: $TRAINING_PID)"

    # Show process uptime
    START_TIME=$(ps -p $TRAINING_PID -o lstart= 2>/dev/null)
    echo "Started: $START_TIME"
fi
echo

# Check checkpoint
CHECKPOINT_FILE="/mnt/data/ai_data/models/lora/luca/iterative_overnight_v5/checkpoint.json"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "========================================================================"
    echo "CHECKPOINT INFORMATION"
    echo "========================================================================"
    cat "$CHECKPOINT_FILE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Last Iteration:  {data.get('last_iteration', 'N/A')}\")
print(f\"Total Results:   {data.get('total_results', 'N/A')}\")
print(f\"Timestamp:       {data.get('timestamp', 'N/A')}\")
print(f\"Characters:      {', '.join(data.get('characters', []))}\")
" 2>/dev/null || cat "$CHECKPOINT_FILE"
    echo
fi

# Count completed iterations
OUTPUT_DIR="/mnt/data/ai_data/models/lora/luca/iterative_overnight_v5"
if [ -d "$OUTPUT_DIR" ]; then
    echo "========================================================================"
    echo "COMPLETED ITERATIONS"
    echo "========================================================================"
    ITERATION_COUNT=$(ls -d ${OUTPUT_DIR}/iteration_* 2>/dev/null | wc -l)
    echo "Total iterations: $ITERATION_COUNT"
    echo

    if [ $ITERATION_COUNT -gt 0 ]; then
        echo "Iteration directories:"
        ls -lh ${OUTPUT_DIR}/iteration_* 2>/dev/null | tail -5
        echo
    fi
fi

# Check GPU status
echo "========================================================================"
echo "GPU STATUS"
echo "========================================================================"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo

# Show recent log entries
LOG_FILE="${OUTPUT_DIR}/training.log"
if [ -f "$LOG_FILE" ]; then
    echo "========================================================================"
    echo "RECENT LOG ENTRIES (Last 15 lines)"
    echo "========================================================================"
    tail -15 "$LOG_FILE" | grep -E "ITERATION|Score|Time:|Training|Evaluating|remaining" || tail -15 "$LOG_FILE"
    echo
fi

# Show last evaluation results
echo "========================================================================"
echo "LATEST EVALUATION SCORES"
echo "========================================================================"

# Find most recent evaluation files
LATEST_EVALS=$(find ${OUTPUT_DIR} -name "evaluation_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -4)

if [ ! -z "$LATEST_EVALS" ]; then
    echo "$LATEST_EVALS" | while read timestamp filepath; do
        CHARACTER=$(echo $filepath | grep -oP '(luca_human|alberto_human)')
        ITERATION=$(echo $filepath | grep -oP 'iteration_\d+' | grep -oP '\d+')

        SCORE=$(cat "$filepath" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    score = data.get('composite_score', data.get('clip_score', 'N/A'))
    print(f'{score:.4f}' if isinstance(score, (int, float)) else score)
except:
    print('N/A')
" 2>/dev/null)

        echo "  Iteration $ITERATION - $CHARACTER: $SCORE"
    done
else
    echo "  No evaluation results found yet."
fi
echo

echo "========================================================================"
echo "QUICK ACTIONS"
echo "========================================================================"
if [ -z "$TRAINING_PID" ]; then
    echo "  Start:   bash scripts/training/restart_training.sh"
else
    echo "  Stop:    bash scripts/training/stop_training.sh"
    echo "  Monitor: tail -f $LOG_FILE"
    echo "  Watch:   bash scripts/monitoring/monitor_lora_training.sh"
fi
echo "========================================================================"
