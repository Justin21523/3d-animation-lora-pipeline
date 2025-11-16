#!/bin/bash
#
# Instance Enhancement for 3D Character Dataset - Project-Agnostic
# Apply brightness, contrast, and sharpness enhancements to extracted character instances
#
# Usage:
#   ./run_instance_enhancement.sh [project]     # project defaults to "luca" if not specified
#
# Examples:
#   ./run_instance_enhancement.sh              # Use luca project
#   ./run_instance_enhancement.sh alberto      # Use alberto project
#
# Parameters:
#   - sharpen: 1.2 (moderate sharpening)
#   - denoise: 5 (light denoising)
#   - clahe_clip: 2.0 (adaptive contrast)
#

set -e  # Exit on error

cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline || exit 1

# Project configuration
PROJECT="${1:-luca}"  # Default to luca if no argument provided
PROJECT_CONFIG="configs/projects/${PROJECT}.yaml"

# Verify project config exists
if [ ! -f "$PROJECT_CONFIG" ]; then
    echo "❌ Error: Project config not found: $PROJECT_CONFIG"
    echo "Available projects in configs/projects/:"
    ls -1 configs/projects/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//' || echo "  (none found)"
    exit 1
fi

# Read project configuration from YAML
PROJECT_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$PROJECT_CONFIG'))['project']['name'])")
BASE_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$PROJECT_CONFIG'))['paths']['base_dir'])")

# Define input/output directories
INPUT_DIR="${BASE_DIR}/clustered_filtered"
OUTPUT_DIR="${BASE_DIR}/clustered_enhanced"

echo "========================================"
echo "${PROJECT_NAME^^} INSTANCE ENHANCEMENT"
echo "========================================"
echo "Project: ${PROJECT_NAME}"
echo "Working directory: $(pwd)"
echo ""
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Parameters:"
echo "  - sharpen: 1.2"
echo "  - denoise: 5"
echo "  - clahe_clip: 2.0"
echo "  - clahe_grid: 8"
echo "========================================"
echo ""

# Run instance enhancement with ai_env
conda run -n ai_env python scripts/generic/enhancement/instance_enhancement.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --sharpen 1.2 \
  --denoise 5 \
  --clahe-clip 2.0 \
  --clahe-grid 8 \
  --skip-existing

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Instance enhancement completed successfully for ${PROJECT_NAME}"
    echo ""
    echo "Check results:"
    echo "  ls ${OUTPUT_DIR}/"
    echo "  cat ${OUTPUT_DIR}/enhancement_report.json"
else
    echo "❌ Instance enhancement failed with exit code: $EXIT_CODE"
    echo "Check logs above for error details"
fi
echo "========================================"

exit $EXIT_CODE
