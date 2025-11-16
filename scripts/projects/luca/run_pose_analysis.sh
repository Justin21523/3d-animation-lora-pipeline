#!/bin/bash
#
# Pose/View Analysis for 3D Character Dataset - Project-Agnostic
# Analyzes pose and view distribution across all character clusters
#
# Usage:
#   ./run_pose_analysis.sh [project]     # project defaults to "luca" if not specified
#
# Examples:
#   ./run_pose_analysis.sh              # Use luca project
#   ./run_pose_analysis.sh alberto      # Use alberto project
#
# This will:
# 1. Estimate pose keypoints using RTM-Pose
# 2. Classify view angles (front/three-quarter/profile/back)
# 3. Subcluster by pose+view features
# 4. Generate pose/angle distribution reports
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
OUTPUT_DIR="${BASE_DIR}/pose_analysis"

echo "========================================"
echo "${PROJECT_NAME^^} POSE/VIEW ANALYSIS"
echo "========================================"
echo "Project: ${PROJECT_NAME}"
echo "Working directory: $(pwd)"
echo ""
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Parameters:"
echo "  - pose_model: rtmpose-m"
echo "  - method: umap_hdbscan"
echo "  - min_cluster_size: 5"
echo "  - device: cuda"
echo "========================================"
echo ""

# Run pose subclustering with ai_env
conda run -n ai_env python scripts/generic/clustering/pose_subclustering.py \
  "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --pose-model rtmpose-m \
  --method umap_hdbscan \
  --min-cluster-size 5 \
  --device cuda \
  --visualize

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Pose analysis completed successfully for ${PROJECT_NAME}"
    echo ""
    echo "Check results:"
    echo "  ls ${OUTPUT_DIR}/"
    echo "  cat ${OUTPUT_DIR}/pose_subclustering.json"
else
    echo "❌ Pose analysis failed with exit code: $EXIT_CODE"
    echo "Check logs above for error details"
fi
echo "========================================"

exit $EXIT_CODE
