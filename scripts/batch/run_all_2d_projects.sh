#!/bin/bash
#
# Run 2D Animation Pipeline for Multiple Projects
# ================================================
#
# This script runs the complete pipeline for all specified 2D animation projects.
#
# Usage:
#   ./run_all_2d_projects.sh
#
# Or specify projects:
#   ./run_all_2d_projects.sh gumbell wylde-pak
#

set -e

# Default projects
PROJECTS="${@:-gumbell wylde-pak}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_2d_animation_pipeline.py"
CONDA_ENV="ai_env"

echo "============================================================"
echo "2D ANIMATION PIPELINE - BATCH RUNNER"
echo "============================================================"
echo "Projects: ${PROJECTS}"
echo "Script: ${PIPELINE_SCRIPT}"
echo "============================================================"

# Function to run pipeline for a single project
run_project() {
    local project=$1
    echo ""
    echo "============================================================"
    echo "Starting: ${project}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    conda run -n ${CONDA_ENV} python "${PIPELINE_SCRIPT}" \
        --project "${project}" \
        --all-stages \
        --frame-interval 10 \
        --yolo-confidence 0.5 \
        --device cuda

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ ${project} completed successfully"
    else
        echo "✗ ${project} failed with exit code: ${exit_code}"
    fi

    return $exit_code
}

# Track results
SUCCESSFUL=()
FAILED=()

# Process each project
for project in ${PROJECTS}; do
    if run_project "${project}"; then
        SUCCESSFUL+=("${project}")
    else
        FAILED+=("${project}")
    fi

    echo ""
    echo "Waiting 10 seconds before next project..."
    sleep 10
done

# Print summary
echo ""
echo "============================================================"
echo "BATCH PROCESSING COMPLETE"
echo "============================================================"
echo "Successful (${#SUCCESSFUL[@]}):"
for p in "${SUCCESSFUL[@]}"; do
    echo "  ✓ ${p}"
done
echo ""
echo "Failed (${#FAILED[@]}):"
for p in "${FAILED[@]}"; do
    echo "  ✗ ${p}"
done
echo "============================================================"
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Exit with error if any project failed
[ ${#FAILED[@]} -eq 0 ]
