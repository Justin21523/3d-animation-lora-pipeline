#!/bin/bash
#
# Quick Pipeline Status Checker
#
# Usage: bash scripts/monitoring/check_pipeline_status.sh

echo "════════════════════════════════════════════════════════════"
echo "  Luca Pipeline Status Monitor"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check SAM2 process
if ps aux | grep -q "[i]nstance_segmentation.py"; then
    echo "🔄 SAM2 Segmentation: RUNNING"

    PROCESSED=$(find /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/instances -name "scene*_inst0.png" 2>/dev/null | wc -l)
    TOTAL=5143
    PERCENTAGE=$(echo "scale=1; $PROCESSED * 100 / $TOTAL" | bc 2>/dev/null || echo "?")

    echo "   Progress: $PROCESSED / $TOTAL frames ($PERCENTAGE%)"

    # Estimate completion time
    LATEST_FILE=$(ls -t /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/instances/*.png 2>/dev/null | head -1)
    if [ ! -z "$LATEST_FILE" ]; then
        AGE_SECONDS=$(( $(date +%s) - $(stat -c %Y "$LATEST_FILE" 2>/dev/null || echo 0) ))
        if [ $AGE_SECONDS -lt 300 ]; then
            echo "   Status: Active (last file $AGE_SECONDS seconds ago)"
        else
            echo "   Status: Idle? (last file $((AGE_SECONDS / 60)) minutes ago)"
        fi
    fi

    REMAINING=$((TOTAL - PROCESSED))
    if [ $PROCESSED -gt 0 ]; then
        # Rough estimate based on average time
        EST_MINUTES=$(echo "scale=0; $REMAINING * 0.8" | bc 2>/dev/null || echo "?")
        echo "   Estimated remaining: ~$EST_MINUTES minutes"
    fi
else
    echo "✅ SAM2 Segmentation: COMPLETED"

    INSTANCES_COUNT=$(find /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/instances -name "*.png" 2>/dev/null | wc -l)
    CONTEXT_COUNT=$(find /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/instances_context -name "*.png" 2>/dev/null | wc -l)

    echo "   Total instances: $INSTANCES_COUNT"
    echo "   Context instances: $CONTEXT_COUNT"
fi

echo ""

# Check post-SAM2 pipeline
if ps aux | grep -q "[p]ost_sam2_auto_pipeline.sh"; then
    echo "🔄 Post-SAM2 Pipeline: RUNNING"

    # Check which phase
    if ps aux | grep -q "[i]ntelligent_frame_processor.py"; then
        echo "   Current Phase: Intelligent Processing"
    elif ps aux | grep -q "[i]ntelligent_dataset_curator.py"; then
        echo "   Current Phase: Quality Curation"
    else
        echo "   Current Phase: Waiting for SAM2 or between phases"
    fi

    # Show recent log
    if [ -f /tmp/post_sam2_monitor.log ]; then
        echo ""
        echo "   Recent activity:"
        tail -3 /tmp/post_sam2_monitor.log | sed 's/^/   | /'
    fi
else
    echo "⏸️  Post-SAM2 Pipeline: NOT RUNNING"

    # Check if outputs exist
    if [ -d /mnt/data/ai_data/datasets/3d-anime/luca/luca_curated_400 ]; then
        FINAL_COUNT=$(find /mnt/data/ai_data/datasets/3d-anime/luca/luca_curated_400 -name "*.png" 2>/dev/null | wc -l)
        echo "   ✅ Final dataset exists: $FINAL_COUNT images"
    fi
fi

echo ""

# Show output directories status
echo "────────────────────────────────────────────────────────────"
echo "Output Directories:"
echo "────────────────────────────────────────────────────────────"

dirs=(
    "/mnt/data/ai_data/datasets/3d-anime/luca/luca_frames_filtered:Face Filtered Frames"
    "/mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/instances_context:SAM2 Context Instances"
    "/mnt/data/ai_data/datasets/3d-anime/luca/luca_intelligent_candidates:Intelligent Candidates"
    "/mnt/data/ai_data/datasets/3d-anime/luca/luca_curated_400:Final Curated Dataset"
)

for entry in "${dirs[@]}"; do
    IFS=':' read -r dir label <<< "$entry"
    if [ -d "$dir" ]; then
        count=$(find "$dir" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        printf "  %-30s %8d files (%s)\n" "$label:" "$count" "$size"
    else
        printf "  %-30s %s\n" "$label:" "Not created yet"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo ""
echo "💡 Commands:"
echo "   Watch progress:  watch -n 60 bash scripts/monitoring/check_pipeline_status.sh"
echo "   View SAM2 log:   tail -f /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/*.log"
echo "   View auto log:   tail -f /tmp/post_sam2_monitor.log"
echo ""
