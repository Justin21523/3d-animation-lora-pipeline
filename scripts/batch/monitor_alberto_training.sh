#!/bin/bash
# Monitor Alberto Anti-Blur Training Progress

OUTPUT_DIR="/mnt/data/ai_data/models/lora_sdxl/luca/alberto_identity"

echo "🔍 Alberto Training Monitor"
echo "==========================="
echo ""

# Check if training is running
if pgrep -f "sdxl_train_network.*alberto" > /dev/null; then
    echo "✅ Training is RUNNING"
    echo ""

    # Show GPU usage
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
      awk -F',' '{printf "   GPU: %s%% | VRAM: %s/%s MB | Temp: %s°C\n", $1, $2, $3, $4}'
    echo ""
else
    echo "⚠️  Training is NOT running"
    echo ""
fi

# Show checkpoints
echo "📦 Checkpoints:"
if ls "$OUTPUT_DIR"/*.safetensors 1> /dev/null 2>&1; then
    ls -lh "$OUTPUT_DIR"/*.safetensors | tail -5 | awk '{printf "   %s %s %s\n", $5, $9, $6" "$7}'
else
    echo "   No checkpoints yet"
fi
echo ""

# Show latest samples
echo "🖼️  Latest Sample Images:"
if ls "$OUTPUT_DIR"/sample/*.png 1> /dev/null 2>&1; then
    ls -lt "$OUTPUT_DIR"/sample/*.png | head -6 | awk '{printf "   %s %s %s\n", $9, $6, $7}'
    echo ""
    echo "   👉 Open sample folder:"
    echo "      explorer.exe $(wslpath -w "$OUTPUT_DIR/sample")"
else
    echo "   No samples yet"
fi
echo ""

# Show training log tail
echo "📝 Recent Log (last 10 lines):"
if ls "$OUTPUT_DIR"/logs/*.log 1> /dev/null 2>&1; then
    tail -10 "$OUTPUT_DIR"/logs/*.log 2>/dev/null | grep -E "(epoch|step|loss|saving)" | tail -5 | sed 's/^/   /'
else
    echo "   No logs yet"
fi
echo ""

# Show epoch progress
echo "📊 Training Progress:"
if ls "$OUTPUT_DIR"/*.safetensors 1> /dev/null 2>&1; then
    CHECKPOINT_COUNT=$(ls "$OUTPUT_DIR"/*-0000*.safetensors 2>/dev/null | wc -l)
    echo "   Completed epochs: $CHECKPOINT_COUNT / 6"

    # Estimate time remaining
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        FIRST_CHECKPOINT=$(ls -t "$OUTPUT_DIR"/*-0000*.safetensors 2>/dev/null | tail -1)
        LATEST_CHECKPOINT=$(ls -t "$OUTPUT_DIR"/*-0000*.safetensors 2>/dev/null | head -1)

        if [ -f "$FIRST_CHECKPOINT" ] && [ -f "$LATEST_CHECKPOINT" ]; then
            START_TIME=$(stat -c %Y "$FIRST_CHECKPOINT")
            CURRENT_TIME=$(stat -c %Y "$LATEST_CHECKPOINT")
            ELAPSED=$((CURRENT_TIME - START_TIME))

            if [ $CHECKPOINT_COUNT -gt 1 ]; then
                SECONDS_PER_EPOCH=$((ELAPSED / (CHECKPOINT_COUNT - 1)))
                REMAINING_EPOCHS=$((6 - CHECKPOINT_COUNT))
                ESTIMATED_REMAINING=$((SECONDS_PER_EPOCH * REMAINING_EPOCHS))

                REMAINING_HOURS=$((ESTIMATED_REMAINING / 3600))
                REMAINING_MINS=$(((ESTIMATED_REMAINING % 3600) / 60))

                echo "   Estimated time remaining: ${REMAINING_HOURS}h ${REMAINING_MINS}m"
            fi
        fi
    fi
else
    echo "   Not started yet"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 Commands:"
echo "   • Watch live: watch -n 10 $0"
echo "   • View logs: tail -f $OUTPUT_DIR/logs/*.log"
echo "   • GPU monitor: watch -n 2 nvidia-smi"
echo "   • Stop training: pkill -f sdxl_train_network"
echo ""
