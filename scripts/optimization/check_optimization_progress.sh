#!/usr/bin/bash
# Quick Optimization Progress Checker
# Shows current status of hyperparameter optimization

OUTPUT_DIR="/mnt/data/ai_data/models/lora/luca/optimization_overnight"
DB_FILE="$OUTPUT_DIR/optuna_study.db"
LOG_FILE="$OUTPUT_DIR/optimization.log"

echo "=========================================="
echo "📊 HYPERPARAMETER OPTIMIZATION STATUS"
echo "=========================================="
echo ""

# Check if process is running
PID=$(ps aux | grep "optuna_hyperparameter_search.py" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "✅ Status: RUNNING (PID: $PID)"
    CPU_MEM=$(ps aux | grep "$PID" | grep -v grep | awk '{print "CPU: "$3"%, Memory: "$4"%"}')
    echo "   $CPU_MEM"
else
    echo "❌ Status: NOT RUNNING"
fi
echo ""

# Check database if it exists
if [ -f "$DB_FILE" ]; then
    echo "📈 Trial Progress:"
    echo "----------------------------------------"

    # Count trials by status
    TOTAL=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM trials;" 2>/dev/null || echo "0")
    COMPLETE=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';" 2>/dev/null || echo "0")
    RUNNING=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM trials WHERE state='RUNNING';" 2>/dev/null || echo "0")
    FAILED=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM trials WHERE state='FAIL';" 2>/dev/null || echo "0")

    echo "  Total Trials: $TOTAL"
    echo "  ✅ Completed: $COMPLETE"
    echo "  🔄 Running: $RUNNING"
    echo "  ❌ Failed: $FAILED"
    echo ""

    # Show best trials
    if [ "$COMPLETE" -gt 0 ]; then
        echo "🏆 Top 5 Best Trials (Lower is better):"
        echo "----------------------------------------"
        sqlite3 "$DB_FILE" "SELECT
            'Trial #' || number || ': Score = ' || ROUND(value, 4)
        FROM trials
        WHERE state='COMPLETE'
        ORDER BY value
        LIMIT 5;" 2>/dev/null || echo "Error reading trials"
        echo ""

        # Show parameters of best trial
        echo "🎯 Best Trial Parameters:"
        echo "----------------------------------------"
        sqlite3 "$DB_FILE" "
        SELECT
            tp.param_name || ' = ' || tv.value
        FROM trial_params tp
        JOIN trial_values tv ON tp.trial_id = tv.trial_id
        WHERE tp.trial_id = (
            SELECT trial_id
            FROM trials
            WHERE state='COMPLETE'
            ORDER BY value
            LIMIT 1
        )
        ORDER BY tp.param_name;
        " 2>/dev/null || echo "Error reading parameters"
        echo ""
    fi
else
    echo "⚠️  Database not found yet (optimization just started)"
    echo ""
fi

# Show recent log entries
if [ -f "$LOG_FILE" ]; then
    echo "📝 Recent Log (last 10 lines):"
    echo "----------------------------------------"
    tail -10 "$LOG_FILE"
    echo ""
fi

# Check for generated images
PROGRESS_DIR="$OUTPUT_DIR/progress_checks"
if [ -d "$PROGRESS_DIR" ]; then
    IMG_COUNT=$(find "$PROGRESS_DIR" -name "*.png" 2>/dev/null | wc -l)
    echo "🖼️  Generated Images: $IMG_COUNT samples"
    echo "   Location: $PROGRESS_DIR"
else
    echo "⚠️  Progress images directory not created yet"
fi

echo ""
echo "=========================================="
echo "📋 Monitoring Commands:"
echo "  Live log: tail -f $LOG_FILE"
echo "  Kill process: kill $PID"
echo "=========================================="
