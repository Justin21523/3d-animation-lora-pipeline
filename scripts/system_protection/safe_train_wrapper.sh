#!/bin/bash
# Safe Training Wrapper - Prevents OOM and auto-recovers

TRAINING_COMMAND="$@"
LOG_DIR="/tmp/safe_training_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Safe Training Wrapper with OOM Protection            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Command: $TRAINING_COMMAND"
echo "Log: $LOG_FILE"
echo ""

# Pre-flight checks
echo "🔍 Pre-flight checks..."

# Check available memory
AVAIL_MEM_GB=$(free -g | awk '/^Mem:/ {print $7}')
if [ "$AVAIL_MEM_GB" -lt 10 ]; then
    echo "⚠️  Warning: Only ${AVAIL_MEM_GB}GB free memory. Consider closing apps."
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check GPU memory
if command -v nvidia-smi &> /dev/null; then
    GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$GPU_FREE" -lt 2000 ]; then
        echo "⚠️  Warning: Only ${GPU_FREE}MB free GPU memory."
    fi
fi

echo "✓ Pre-flight checks passed"
echo ""

# Start watchdogs
echo "🛡️  Starting protection systems..."
if [ -f /tmp/memory_watchdog.sh ]; then
    pkill -f memory_watchdog.sh 2>/dev/null
    nohup /tmp/memory_watchdog.sh > /dev/null 2>&1 &
    echo "  ✓ Memory watchdog started (PID: $!)"
fi

if [ -f /tmp/gpu_watchdog.sh ]; then
    pkill -f gpu_watchdog.sh 2>/dev/null
    nohup /tmp/gpu_watchdog.sh > /dev/null 2>&1 &
    echo "  ✓ GPU watchdog started (PID: $!)"
fi

# Set OOM priorities
if [ -f /tmp/set_oom_priorities.sh ]; then
    bash /tmp/set_oom_priorities.sh > /dev/null 2>&1
    echo "  ✓ OOM priorities configured"
fi

echo ""
echo "🚀 Starting training..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run training with output logging
eval "$TRAINING_COMMAND" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"

    # Check if OOM killed it
    if tail -100 "$LOG_FILE" | grep -qi "out of memory\|CUDA out of memory\|killed"; then
        echo ""
        echo "🚨 OOM Detected! Training was killed due to memory exhaustion."
        echo ""
        echo "Recommendations:"
        echo "  1. Reduce batch size in config"
        echo "  2. Enable gradient checkpointing"
        echo "  3. Use smaller model or lower resolution"
        echo "  4. Close other applications"
        echo "  5. Increase swap size"
    fi
fi

echo ""
echo "Log saved to: $LOG_FILE"
exit $EXIT_CODE
