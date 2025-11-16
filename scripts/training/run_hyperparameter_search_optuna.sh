#!/bin/bash
# Hyperparameter Search with Optuna TPE + tmux
# Runs 20 trials using Tree-structured Parzen Estimator for intelligent parameter search

set -e

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
BASE_CONFIG="$PROJECT_ROOT/configs/training/luca_final_v1_pure.toml"
OUTPUT_DIR="/mnt/data/ai_data/models/lora/luca/hyperparameter_search_optuna"
N_TRIALS=20

echo "=========================================="
echo "Luca LoRA Hyperparameter Search (Optuna TPE)"
echo "=========================================="
echo ""
echo "Method: Optuna TPE (Tree-structured Parzen Estimator)"
echo "Base config: $BASE_CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo "Trials: $N_TRIALS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create tmux session for hyperparameter search
SESSION_NAME="luca_hyperparam_optuna"

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
tmux new-session -d -s "$SESSION_NAME" -n "optuna_search"

# Send commands to tmux session
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_ROOT" C-m
tmux send-keys -t "$SESSION_NAME" "conda activate kohya_ss" C-m
tmux send-keys -t "$SESSION_NAME" "echo '=========================================='" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Luca LoRA Hyperparameter Search (Optuna TPE)'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Session: $SESSION_NAME'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Trials: $N_TRIALS'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Method: Tree-structured Parzen Estimator'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Estimated time: 120-160 hours (5-7 days)'" C-m
tmux send-keys -t "$SESSION_NAME" "echo '=========================================='" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m

# Start hyperparameter search with Optuna
tmux send-keys -t "$SESSION_NAME" "python scripts/training/lora_hyperparameter_search_optuna.py --base-config $BASE_CONFIG --output-dir $OUTPUT_DIR --n-trials $N_TRIALS --search-strategy aggressive 2>&1 | tee $OUTPUT_DIR/search_log_\$(date +%Y%m%d_%H%M%S).log" C-m

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
cat > "$PROJECT_ROOT/monitor_hyperparam_optuna.sh" << 'MONITOR_EOF'
#!/bin/bash
# Monitor Optuna hyperparameter search progress

SESSION_NAME="luca_hyperparam_optuna"
OUTPUT_DIR="/mnt/data/ai_data/models/lora/luca/hyperparameter_search_optuna"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Hyperparameter search session is running!"
    echo ""

    # Show progress from Optuna database
    if [ -f "$OUTPUT_DIR/optuna_study.db" ]; then
        echo "Current progress:"
        python3 -c "
import optuna

try:
    study = optuna.load_study(
        study_name='lora_hyperparameter_search',
        storage='sqlite:///$OUTPUT_DIR/optuna_study.db'
    )

    total = len(study.trials)
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    print(f'  Total trials: {total}')
    print(f'  Completed: {completed}')
    print(f'  Pruned (safety constraints): {pruned}')
    print(f'  Failed: {failed}')

    if completed > 0:
        print(f'  Best score so far: {study.best_value:.4f}')
        print(f'  Best trial number: {study.best_trial.number}')
except Exception as e:
    print(f'Error reading Optuna database: {e}')
"
        echo ""
    fi

    echo "Attaching to session..."
    tmux attach -t "$SESSION_NAME"
else
    echo "Hyperparameter search session not found!"
    echo ""

    # Check if search completed
    if [ -f "$OUTPUT_DIR/best_parameters.json" ]; then
        echo "Search appears to be complete. Best parameters:"
        cat "$OUTPUT_DIR/best_parameters.json"
    fi
fi
MONITOR_EOF

chmod +x "$PROJECT_ROOT/monitor_hyperparam_optuna.sh"

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Use './monitor_hyperparam_optuna.sh' to check progress."
echo ""
