#!/bin/bash
# Hyperparameter Search with tmux
# Runs 20 trials to find optimal LoRA configuration

set -e

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
BASE_CONFIG="$PROJECT_ROOT/configs/training/luca_final_v1_pure.toml"
OUTPUT_DIR="/mnt/data/ai_data/models/lora/luca/hyperparameter_search"
N_TRIALS=20

echo "=========================================="
echo "Luca LoRA Hyperparameter Search"
echo "=========================================="
echo ""
echo "Base config: $BASE_CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo "Trials: $N_TRIALS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create tmux session for hyperparameter search
SESSION_NAME="luca_hyperparam_search"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "WARNING: tmux session '$SESSION_NAME' already exists!"
    read -p "Kill existing session and start new? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
    else
        echo "Attaching to existing session..."
        tmux attach -t "$SESSION_NAME"
        exit 0
    fi
fi

# Create new tmux session
tmux new-session -d -s "$SESSION_NAME" -n "hyperparam_search"

# Send commands to tmux session
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_ROOT" C-m
tmux send-keys -t "$SESSION_NAME" "conda activate ai_env" C-m
tmux send-keys -t "$SESSION_NAME" "echo '=========================================='" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Luca LoRA Hyperparameter Search'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Session: $SESSION_NAME'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Trials: $N_TRIALS'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Estimated time: 120-160 hours (5-7 days)'" C-m
tmux send-keys -t "$SESSION_NAME" "echo '=========================================='" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m

# Start hyperparameter search
tmux send-keys -t "$SESSION_NAME" "python scripts/training/lora_hyperparameter_search.py --base-config $BASE_CONFIG --output-dir $OUTPUT_DIR --n-trials $N_TRIALS --method random 2>&1 | tee $OUTPUT_DIR/search_log_\$(date +%Y%m%d_%H%M%S).log" C-m

echo "✓ Hyperparameter search started in tmux session: $SESSION_NAME"
echo ""
echo "To monitor progress:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach from session: Ctrl+B, then D"
echo "To list sessions: tmux ls"
echo ""
echo "Results will be saved to: $OUTPUT_DIR/"
echo ""
echo "Estimated completion time: 5-7 days"
echo ""

# Create monitoring script
cat > "$PROJECT_ROOT/monitor_hyperparam_search.sh" << 'MONITOR_EOF'
#!/bin/bash
# Monitor hyperparameter search progress

SESSION_NAME="luca_hyperparam_search"
OUTPUT_DIR="/mnt/data/ai_data/models/lora/luca/hyperparameter_search"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Hyperparameter search session is running!"
    echo ""
    
    # Show progress
    if [ -f "$OUTPUT_DIR/trial_results.json" ]; then
        echo "Current progress:"
        python3 -c "
import json
with open('$OUTPUT_DIR/trial_results.json', 'r') as f:
    results = json.load(f)
total = len(results)
successful = sum(1 for r in results if r.get('status') == 'success')
failed = sum(1 for r in results if r.get('status') == 'failed')
print(f'  Completed: {total}/20 trials')
print(f'  Successful: {successful}')
print(f'  Failed: {failed}')
"
        echo ""
    fi
    
    echo "Attaching to session..."
    tmux attach -t "$SESSION_NAME"
else
    echo "Hyperparameter search session not found!"
    echo ""
    
    # Check if search completed
    if [ -f "$OUTPUT_DIR/trial_results.json" ]; then
        echo "Search appears to be complete. Results:"
        python3 -c "
import json
with open('$OUTPUT_DIR/trial_results.json', 'r') as f:
    results = json.load(f)
total = len(results)
successful = sum(1 for r in results if r.get('status') == 'success')
print(f'Total trials: {total}')
print(f'Successful: {successful}')
print(f'Failed: {total - successful}')
"
    fi
fi
MONITOR_EOF

chmod +x "$PROJECT_ROOT/monitor_hyperparam_search.sh"

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Use './monitor_hyperparam_search.sh' to check progress."
echo ""

