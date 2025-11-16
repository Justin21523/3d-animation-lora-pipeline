#!/bin/bash
#
# Quality Filtering for 3D Character Dataset - Project-Agnostic
# Filters organized character instances for high-quality training data
#
# Usage:
#   ./run_quality_filter.sh [project]     # project defaults to "luca" if not specified
#
# Examples:
#   ./run_quality_filter.sh              # Use luca project
#   ./run_quality_filter.sh alberto      # Use alberto project
#
# Parameters:
#   - min_sharpness: 50 (relaxed from 100 for inpainted images)
#   - min_completeness: 0.85 (85% alpha coverage)
#   - target_per_cluster: 200 (balanced sampling)
#   - diversity_method: clip (CLIP-based visual diversity clustering)
#   - diversity_clusters: 5 (K-Means subclusters for stratified sampling)
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
INPUT_DIR="${BASE_DIR}/clustered_v2_inpainted"
OUTPUT_DIR="${BASE_DIR}/clustered_filtered"

echo "========================================"
echo "${PROJECT_NAME^^} QUALITY FILTERING"
echo "========================================"
echo "Project: ${PROJECT_NAME}"
echo "Working directory: $(pwd)"
echo ""
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Parameters:"
echo "  - min_sharpness: 50 (relaxed for inpainted images)"
echo "  - min_completeness: 0.85"
echo "  - target_per_cluster: 10000 (保留所有通過質量檢查的圖片)"
echo "  - diversity_method: clip (full CLIP embedding analysis)"
echo "  - diversity_clusters: 5"
echo "  - device: cuda"
echo "========================================"
echo ""

# Run quality filter with ai_env
conda run -n ai_env python scripts/generic/training/quality_filter.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --target-per-cluster 10000 \
  --min-sharpness 50 \
  --min-completeness 0.85 \
  --diversity-method clip \
  --diversity-clusters 5 \
  --device cuda

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Quality filtering completed successfully for ${PROJECT_NAME}"
    echo ""
    echo "Check results:"
    echo "  ls ${OUTPUT_DIR}/"
    echo "  cat ${OUTPUT_DIR}/quality_filter_report.json"
else
    echo "❌ Quality filtering failed with exit code: $EXIT_CODE"
    echo "Check logs above for error details"
fi
echo "========================================"

exit $EXIT_CODE
