#!/bin/bash
#
# Check Image Preprocessing Progress
#

echo "========================================="
echo "Image Preprocessing Progress"
echo "========================================="
echo ""

# Check if preprocessing is running
if pgrep -f "preprocess_images_for_sdxl.py" > /dev/null; then
    echo "✅ Preprocessing is RUNNING"
    echo ""

    # Show process info
    ps aux | grep "preprocess_images_for_sdxl.py" | grep -v grep | awk '{printf "  PID: %s, CPU: %s%%, MEM: %s%%\n", $2, $3, $4}'

    # Show GPU usage
    echo ""
    echo "📊 GPU Status:"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F, '{printf "  VRAM: %dMB / %dMB, GPU: %d%%\n", $1, $2, $3}'

    echo ""
    echo "⏱️  Estimated completion: 30-60 minutes from start"

else
    echo "⚠️  Preprocessing is NOT running"
    echo ""

    # Check if it completed successfully
    if [ -f "logs/preprocessing_report.json" ]; then
        echo "✅ Found preprocessing report - may have completed"
        echo ""
        echo "📊 Report summary:"
        cat logs/preprocessing_report.json | python3 -m json.tool | grep -A 5 "summary"
    else
        echo "❌ No report found - preprocessing may have failed"
    fi
fi

echo ""
echo "========================================="
