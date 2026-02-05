#!/bin/bash
# Manual Tasks After Batch Processing
# Run these after the main batch processor completes

PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
DATA_ROOT="/mnt/data/ai_data/datasets/3d-anime"

cd "$PROJECT_ROOT"

echo "========================================================================"
echo "🔧 Manual Processing Tasks"
echo "========================================================================"
echo ""

# Task 1: Resume Orion SAM2 (from 383/2803)
echo "Task 1: Resume Orion SAM2 Segmentation"
echo "  Current progress: 383/2803 (13.6%)"
echo "  Command:"
echo "  conda run -n ai_env python scripts/generic/segmentation/instance_segmentation.py \\"
echo "    $DATA_ROOT/orion/frames_final \\"
echo "    --output-dir $DATA_ROOT/orion/orion_instances_sam2_v2 \\"
echo "    --model sam2_hiera_large \\"
echo "    --device cuda \\"
echo "    --min-size 4096 \\"
echo "    --save-masks \\"
echo "    --save-backgrounds \\"
echo "    --context-mode transparent"
echo ""
echo "  (Will auto-skip already processed frames)"
echo ""

# Task 2: Luca LaMa Inpainting
echo "Task 2: Luca LaMa Inpainting"
echo "  Total frames: 14410"
echo "  Command:"
echo "  conda run -n ai_env python scripts/generic/inpainting/sam2_background_inpainting.py \\"
echo "    --sam2-dir $DATA_ROOT/luca/luca_instances_sam2_v2 \\"
echo "    --output-dir $DATA_ROOT/luca/backgrounds_lama_v2 \\"
echo "    --method lama \\"
echo "    --batch-size 8 \\"
echo "    --device cuda \\"
echo "    --mask-dilate 20"
echo ""

# Task 3: Orion LaMa Inpainting (after SAM2 completes)
echo "Task 3: Orion LaMa Inpainting (run after Orion SAM2 completes)"
echo "  Total frames: 2803"
echo "  Command:"
echo "  conda run -n ai_env python scripts/generic/inpainting/sam2_background_inpainting.py \\"
echo "    --sam2-dir $DATA_ROOT/orion/orion_instances_sam2_v2 \\"
echo "    --output-dir $DATA_ROOT/orion/backgrounds_lama_v2 \\"
echo "    --method lama \\"
echo "    --batch-size 8 \\"
echo "    --device cuda \\"
echo "    --mask-dilate 20"
echo ""

echo "========================================================================"
echo "💡 Usage:"
echo "  1. Wait for main batch processor to complete (or stop it)"
echo "  2. Run tasks one by one manually"
echo "  3. Or execute all in sequence (WARNING: may take many hours):"
echo "     bash $0 --execute-all"
echo "========================================================================"

# Execute mode
if [ "$1" == "--execute-all" ]; then
    echo ""
    echo "⚠️  Starting sequential execution of all manual tasks..."
    echo ""

    # Task 1: Orion SAM2
    echo "🎬 Starting Task 1: Orion SAM2..."
    conda run -n ai_env python scripts/generic/segmentation/instance_segmentation.py \
        "$DATA_ROOT/orion/frames_final" \
        --output-dir "$DATA_ROOT/orion/orion_instances_sam2_v2" \
        --model sam2_hiera_large \
        --device cuda \
        --min-size 4096 \
        --save-masks \
        --save-backgrounds \
        --context-mode transparent

    if [ $? -eq 0 ]; then
        echo "✅ Task 1 completed successfully"
    else
        echo "❌ Task 1 failed, stopping"
        exit 1
    fi

    # Task 2: Luca LaMa
    echo ""
    echo "🎬 Starting Task 2: Luca LaMa Inpainting..."
    conda run -n ai_env python scripts/generic/inpainting/sam2_background_inpainting.py \
        --sam2-dir "$DATA_ROOT/luca/luca_instances_sam2_v2" \
        --output-dir "$DATA_ROOT/luca/backgrounds_lama_v2" \
        --method lama \
        --batch-size 8 \
        --device cuda \
        --mask-dilate 20

    if [ $? -eq 0 ]; then
        echo "✅ Task 2 completed successfully"
    else
        echo "❌ Task 2 failed, stopping"
        exit 1
    fi

    # Task 3: Orion LaMa
    echo ""
    echo "🎬 Starting Task 3: Orion LaMa Inpainting..."
    conda run -n ai_env python scripts/generic/inpainting/sam2_background_inpainting.py \
        --sam2-dir "$DATA_ROOT/orion/orion_instances_sam2_v2" \
        --output-dir "$DATA_ROOT/orion/backgrounds_lama_v2" \
        --method lama \
        --batch-size 8 \
        --device cuda \
        --mask-dilate 20

    if [ $? -eq 0 ]; then
        echo "✅ Task 3 completed successfully"
    else
        echo "❌ Task 3 failed"
        exit 1
    fi

    echo ""
    echo "🎉 All manual tasks completed!"
fi
