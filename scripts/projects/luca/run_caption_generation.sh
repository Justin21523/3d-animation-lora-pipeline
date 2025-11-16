#!/bin/bash
#
# Caption Generation for 3D Character Dataset (Using Qwen2-VL) - Project-Agnostic
# Generates detailed captions for character clusters
#
# Usage:
#   ./run_caption_generation.sh [project]     # project defaults to "luca" if not specified
#
# Examples:
#   ./run_caption_generation.sh              # Use luca project
#   ./run_caption_generation.sh alberto      # Use alberto project
#

set -e

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
INPUT_DIR="${BASE_DIR}/clustered_enhanced"
OUTPUT_DIR="${BASE_DIR}/training_data"

echo "========================================"
echo "${PROJECT_NAME^^} CAPTION GENERATION (Qwen2-VL)"
echo "========================================"
echo "Project: ${PROJECT_NAME}"
echo "Working directory: $(pwd)"
echo ""
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Model: Qwen2-VL-2B (optimized for 3D animation)"
echo "========================================"
echo ""

# Run caption generation with Qwen2-VL-7B (reduced batch size for larger model)
/home/b0979/.conda/envs/ai_env/bin/python scripts/generic/training/qwen_caption_generator.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --batch-size 2

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Caption generation completed successfully for ${PROJECT_NAME}"
    echo ""
    echo "Check results:"
    echo "  ls ${OUTPUT_DIR}/"
    echo "  cat ${OUTPUT_DIR}/caption_generation_report.json"
else
    echo "❌ Caption generation failed with exit code: $EXIT_CODE"
    echo "Check logs above for error details"
fi
echo "========================================"

exit $EXIT_CODE
