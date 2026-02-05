#!/usr/bin/bash
"""
Monitor Batch Synthetic Data Generation Progress
=================================================

Displays real-time progress of the batch synthetic data generation pipeline.

Usage:
  bash monitor_progress.sh [workspace_dir]

Author: LLMProvider Tooling
Date: 2025-11-30
"""

WORKSPACE="${1:-/tmp/synthetic_data_output}"

if [ ! -d "$WORKSPACE" ]; then
    echo "Error: Workspace directory not found: $WORKSPACE"
    echo ""
    echo "Usage:"
    echo "  bash $(basename "$0") [workspace_dir]"
    exit 1
fi

clear
echo "========================================================================"
echo "📊 Batch Synthetic Data Generation - Progress Monitor"
echo "========================================================================"
echo "Workspace: $WORKSPACE"
echo "Updated: $(date)"
echo ""

# Check if pipeline is running
PIPELINE_RUNNING=false
if pgrep -f "batch_synthetic_data_pipeline.sh" > /dev/null; then
    PIPELINE_RUNNING=true
    echo "✅ Pipeline Status: RUNNING"
else
    echo "⏸️  Pipeline Status: NOT RUNNING"
fi
echo ""

# Phase 1: Vocabulary Generation
echo "──────────────────────────────────────────────────────────────────"
echo "PHASE 1: Vocabulary Generation"
echo "──────────────────────────────────────────────────────────────────"
echo ""

if [ -d "$WORKSPACE/generated_data" ]; then
    for char_dir in "$WORKSPACE/generated_data"/*; do
        if [ -d "$char_dir" ]; then
            char=$(basename "$char_dir")

            # Count vocab files
            vocab_count=0
            for type_dir in "$char_dir"/*; do
                if [ -f "$type_dir/prompts.json" ]; then
                    vocab_count=$((vocab_count + 1))
                fi
            done

            if [ $vocab_count -eq 3 ]; then
                echo "  ✅ $char: All vocabularies generated"
            elif [ $vocab_count -gt 0 ]; then
                echo "  ⏳ $char: $vocab_count/3 vocabularies"
            else
                echo "  ⏸️  $char: Not started"
            fi
        fi
    done
else
    echo "  No vocabularies generated yet"
fi

echo ""

# Phase 2: Image Generation
echo "──────────────────────────────────────────────────────────────────"
echo "PHASE 2: Image Generation"
echo "──────────────────────────────────────────────────────────────────"
echo ""

TOTAL_GENERATED=0

if [ -d "$WORKSPACE/generated_data" ]; then
    for char_dir in "$WORKSPACE/generated_data"/*; do
        if [ -d "$char_dir" ]; then
            char=$(basename "$char_dir")

            char_total=0
            for type_dir in "$char_dir"/*/generated; do
                if [ -d "$type_dir" ]; then
                    count=$(find "$type_dir" -name "*.png" 2>/dev/null | wc -l)
                    char_total=$((char_total + count))
                fi
            done

            TOTAL_GENERATED=$((TOTAL_GENERATED + char_total))

            if [ $char_total -gt 0 ]; then
                echo "  $char: $char_total images generated"
            fi
        fi
    done

    echo ""
    echo "  Total generated: $TOTAL_GENERATED images"
else
    echo "  No images generated yet"
fi

echo ""

# Phase 3: Quality Filtering
echo "──────────────────────────────────────────────────────────────────"
echo "PHASE 3: Quality Filtering"
echo "──────────────────────────────────────────────────────────────────"
echo ""

TOTAL_FILTERED=0

if [ -d "$WORKSPACE/generated_data" ]; then
    for char_dir in "$WORKSPACE/generated_data"/*; do
        if [ -d "$char_dir" ]; then
            char=$(basename "$char_dir")

            char_filtered=0
            for type_dir in "$char_dir"/*/filtered; do
                if [ -d "$type_dir" ]; then
                    count=$(find "$type_dir" -name "*.png" 2>/dev/null | wc -l)
                    char_filtered=$((char_filtered + count))
                fi
            done

            TOTAL_FILTERED=$((TOTAL_FILTERED + char_filtered))

            if [ $char_filtered -gt 0 ]; then
                echo "  $char: $char_filtered images filtered"
            fi
        fi
    done

    echo ""
    echo "  Total filtered: $TOTAL_FILTERED images"

    if [ $TOTAL_GENERATED -gt 0 ]; then
        retention=$((TOTAL_FILTERED * 100 / TOTAL_GENERATED))
        echo "  Retention rate: ${retention}%"
    fi
else
    echo "  No images filtered yet"
fi

echo ""

# Phase 4: Dataset Organization
echo "──────────────────────────────────────────────────────────────────"
echo "PHASE 4: Dataset Organization"
echo "──────────────────────────────────────────────────────────────────"
echo ""

if [ -d "$WORKSPACE/datasets" ]; then
    dataset_count=$(find "$WORKSPACE/datasets" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)

    if [ $dataset_count -gt 0 ]; then
        echo "  Datasets organized: $dataset_count"

        for dataset_dir in "$WORKSPACE/datasets"/*; do
            if [ -d "$dataset_dir" ]; then
                dataset=$(basename "$dataset_dir")
                image_count=$(find "$dataset_dir" -name "*.png" 2>/dev/null | wc -l)
                caption_count=$(find "$dataset_dir" -name "*.txt" 2>/dev/null | wc -l)

                if [ $image_count -gt 0 ]; then
                    echo "    $dataset: $image_count images, $caption_count captions"
                fi
            fi
        done
    else
        echo "  No datasets organized yet"
    fi
else
    echo "  No datasets organized yet"
fi

echo ""

# Checkpoint Status
echo "──────────────────────────────────────────────────────────────────"
echo "Checkpoint Status"
echo "──────────────────────────────────────────────────────────────────"
echo ""

CHECKPOINT_FILE="$WORKSPACE/checkpoints/pipeline_progress.json"

if [ -f "$CHECKPOINT_FILE" ]; then
    completed_count=$(grep -c "\"completed\"" "$CHECKPOINT_FILE" 2>/dev/null || echo 0)
    echo "  Completed tasks: $completed_count"
    echo "  Checkpoint file: $CHECKPOINT_FILE"
else
    echo "  No checkpoint file found"
fi

echo ""

# Recent Activity
echo "──────────────────────────────────────────────────────────────────"
echo "Recent Activity (last 10 log entries)"
echo "──────────────────────────────────────────────────────────────────"
echo ""

STATUS_LOG="$WORKSPACE/logs/status.log"

if [ -f "$STATUS_LOG" ]; then
    tail -10 "$STATUS_LOG" | sed 's/^/  /'
else
    echo "  No status log found"
fi

echo ""

# Useful Commands
echo "──────────────────────────────────────────────────────────────────"
echo "Useful Commands"
echo "──────────────────────────────────────────────────────────────────"
echo ""
echo "  View main log:"
echo "    tail -f $WORKSPACE/logs/main_pipeline_*.log"
echo ""
echo "  View status log:"
echo "    tail -f $WORKSPACE/logs/status.log"
echo ""
echo "  View error log:"
echo "    tail -f $WORKSPACE/logs/errors.log"
echo ""
echo "  Check GPU usage:"
echo "    nvidia-smi"
echo ""
echo "  Re-run this monitor:"
echo "    bash $(realpath "$0") $WORKSPACE"
echo ""
echo "========================================================================"
