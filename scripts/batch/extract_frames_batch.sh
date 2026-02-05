#!/bin/bash
# Batch Frame Extraction for 3D Animation Projects
# Processes videos across 9 projects sequentially with 32 threads
# Date: 2025-12-07

# Don't exit on error - continue to next project
set +e

# Configuration
GENERAL_DIR="/mnt/data/datasets/general"
WORKERS=32
MODE="scene"
SCENE_THRESHOLD=27.0
FRAMES_PER_SCENE=3
JPEG_QUALITY=95

# Frame extractor script
EXTRACTOR="/mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/generic/video/universal_frame_extractor.py"

# Log file
LOG_FILE="${GENERAL_DIR}/frame_extraction_log_$(date +%Y%m%d_%H%M%S).txt"

# Projects to process (skip astro-boy - already completed)
PROJECTS=(
    "astro-kid"
    "cars"
    "inside-out"
    "lorax"
    "mitch"
    "monster-house"
    "monster-inc"
    "rone"
)

# Function to process a project
process_project() {
    local project="$1"
    local video_dir="${GENERAL_DIR}/${project}/videos"
    local output_dir="${GENERAL_DIR}/${project}/frames"

    echo "========================================"
    echo "$(date): Processing ${project}"
    echo "Video dir: ${video_dir}"
    echo "Output dir: ${output_dir}"
    echo "========================================"

    # Check if video directory exists
    if [ ! -d "${video_dir}" ]; then
        echo "Video directory not found: ${video_dir}"
        return 1
    fi

    # Check if already has frames
    local existing_frames=$(find "${output_dir}" -name "*.jpg" 2>/dev/null | wc -l)
    if [ "${existing_frames}" -gt 1000 ]; then
        echo "Already has ${existing_frames} frames, skipping..."
        return 0
    fi

    # Create output directory
    mkdir -p "${output_dir}"

    # Run frame extraction
    echo "Starting frame extraction with ${WORKERS} workers..."
    conda run -n ai_env python "${EXTRACTOR}" \
        "${video_dir}" \
        --output-dir "${output_dir}" \
        --mode "${MODE}" \
        --scene-threshold "${SCENE_THRESHOLD}" \
        --frames-per-scene "${FRAMES_PER_SCENE}" \
        --jpeg-quality "${JPEG_QUALITY}" \
        --workers "${WORKERS}" \
        --episode-pattern "none"

    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        local frame_count=$(find "${output_dir}" -name "*.jpg" 2>/dev/null | wc -l)
        echo "Completed ${project}: ${frame_count} frames extracted"
    else
        echo "Failed ${project} with exit code ${exit_code}"
    fi

    echo ""
    return ${exit_code}
}

# Start processing
echo "========================================"
echo "Batch Frame Extraction Started"
echo "Date: $(date)"
echo "Total projects: ${#PROJECTS[@]}"
echo "Workers: ${WORKERS}"
echo "Mode: ${MODE}"
echo "========================================"

# Process each project sequentially
completed=0
failed=0

for project in "${PROJECTS[@]}"; do
    if process_project "${project}"; then
        ((completed++))
    else
        ((failed++))
    fi
done

# Summary
echo "========================================"
echo "Batch Frame Extraction Completed"
echo "Date: $(date)"
echo "Completed: ${completed}/${#PROJECTS[@]}"
echo "Failed: ${failed}/${#PROJECTS[@]}"
echo "========================================"

# Show final status
echo ""
echo "Final frame counts:"
for project in astro-boy astro-kid cars inside-out lorax mitch monster-house monster-inc rone; do
    count=$(find "${GENERAL_DIR}/${project}/frames" -name "*.jpg" 2>/dev/null | wc -l)
    echo "  ${project}: ${count} frames"
done
