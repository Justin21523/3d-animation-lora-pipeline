#!/bin/bash
#
# SAM2 Instance Processing Pipeline
#
# Processes SAM2 segmentation results through the complete pipeline:
# 1. Auto-categorization (CLIP)
# 2. Manual review (Web UI)
# 3. Context-aware inpainting
# 4. Face identity clustering
# 5. Optional pose subclustering
#
# Usage: bash process_sam2_instances.sh <project_name>
#

set -e

PROJECT=${1:-luca}
CONDA_ENV="ai_env"

# Paths
BASE_DIR="/mnt/data/ai_data/datasets/3d-anime/${PROJECT}"
INSTANCES_DIR="${BASE_DIR}/instances_sampled/instances"
CATEGORIZED_DIR="${BASE_DIR}/instances_categorized"
FILTERED_DIR="${BASE_DIR}/instances_filtered"
INPAINTED_DIR="${BASE_DIR}/instances_inpainted"
CLUSTERS_DIR="${BASE_DIR}/face_clusters"
FRAMES_DIR="${BASE_DIR}/frames"

# Scripts
SCRIPT_DIR="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts"
CATEGORIZER="${SCRIPT_DIR}/generic/review/instance_categorizer.py"
FILTER_UI="${SCRIPT_DIR}/generic/review/instance_filter_ui.py"
INPAINTER="${SCRIPT_DIR}/generic/enhancement/inpaint_context_aware.py"
FACE_CLUSTERER="${SCRIPT_DIR}/generic/clustering/face_identity_clustering.py"
POSE_SUBCLUSTER="${SCRIPT_DIR}/generic/clustering/pose_subclustering.py"

echo "=========================================="
echo "SAM2 Instance Processing Pipeline"
echo "Project: ${PROJECT}"
echo "=========================================="
echo ""

# Check if instances exist
if [ ! -d "${INSTANCES_DIR}" ]; then
    echo "❌ Error: Instances directory not found: ${INSTANCES_DIR}"
    exit 1
fi

INSTANCE_COUNT=$(ls "${INSTANCES_DIR}" | wc -l)
echo "📊 Found ${INSTANCE_COUNT} instances to process"
echo ""

# ==========================================
# Step 1: Auto-categorization with CLIP
# ==========================================
echo "🔍 Step 1: Auto-categorizing instances with CLIP..."
echo "   Input: ${INSTANCES_DIR}"
echo "   Output: ${CATEGORIZED_DIR}"
echo ""

if [ ! -f "${CATEGORIZED_DIR}/categorization_results.json" ]; then
    conda run -n ${CONDA_ENV} python ${CATEGORIZER} \
        --instances-dir "${INSTANCES_DIR}" \
        --output-dir "${CATEGORIZED_DIR}" \
        --confidence-threshold 0.3 \
        --project ${PROJECT}

    echo "✅ Auto-categorization complete"
else
    echo "⏭️  Categorization already exists, skipping"
fi

echo ""
echo "📊 Categorization Summary:"
cat "${CATEGORIZED_DIR}/categorization_results.json" | grep -E "total_instances|category_counts" || echo "See ${CATEGORIZED_DIR}/categorization_results.json"
echo ""

# ==========================================
# Step 2: Manual Review (Interactive)
# ==========================================
echo "=========================================="
echo "👀 Step 2: Manual Review Required"
echo "=========================================="
echo ""
echo "Starting interactive web UI for instance filtering..."
echo "URL: http://localhost:5555"
echo ""
echo "Instructions:"
echo "  - Press 'K' to keep an instance"
echo "  - Press 'D' to discard an instance"
echo "  - Use arrow keys to navigate"
echo "  - Changes auto-save to progress.json"
echo ""
echo "When finished reviewing, press Ctrl+C to continue pipeline"
echo ""

read -p "Press Enter to launch web UI (or Ctrl+C to skip)..."

conda run -n ${CONDA_ENV} python ${FILTER_UI} \
    --instances-dir "${INSTANCES_DIR}" \
    --categories-json "${CATEGORIZED_DIR}/categorization_results.json" \
    --output-dir "${FILTERED_DIR}" \
    --port 5555 \
    --project ${PROJECT} || echo "⚠️  UI closed or skipped"

echo ""

# Check if filtering was done
if [ ! -d "${FILTERED_DIR}/keep" ]; then
    echo "⚠️  Warning: No filtered instances found. Skipping remaining steps."
    echo "   Please run the manual review UI first."
    exit 0
fi

KEPT_COUNT=$(ls "${FILTERED_DIR}/keep" 2>/dev/null | wc -l)
echo "✅ Manual review complete: ${KEPT_COUNT} instances kept"
echo ""

# ==========================================
# Step 3: Context-Aware Inpainting
# ==========================================
echo "=========================================="
echo "🎨 Step 3: Context-Aware Inpainting"
echo "=========================================="
echo ""

if [ ! -f "${INPAINTED_DIR}/inpainting_results.json" ]; then
    echo "Running inpainting on ${KEPT_COUNT} instances..."
    echo "   Method: LaMa with temporal context"
    echo "   Output: ${INPAINTED_DIR}"
    echo ""

    conda run -n ${CONDA_ENV} python ${INPAINTER} \
        --instances-dir "${FILTERED_DIR}/keep" \
        --frames-dir "${FRAMES_DIR}" \
        --output-dir "${INPAINTED_DIR}" \
        --method lama \
        --use-temporal-context \
        --project ${PROJECT}

    echo "✅ Inpainting complete"
else
    echo "⏭️  Inpainting already exists, skipping"
fi

echo ""

# ==========================================
# Step 4: Face Identity Clustering
# ==========================================
echo "=========================================="
echo "👤 Step 4: Face Identity Clustering"
echo "=========================================="
echo ""

if [ ! -f "${CLUSTERS_DIR}/clustering_results.json" ]; then
    echo "Clustering inpainted instances by character identity..."
    echo "   Input: ${INPAINTED_DIR}"
    echo "   Output: ${CLUSTERS_DIR}"
    echo ""

    conda run -n ${CONDA_ENV} python ${FACE_CLUSTERER} \
        --instances-dir "${INPAINTED_DIR}" \
        --output-dir "${CLUSTERS_DIR}" \
        --method arcface \
        --min-cluster-size 5 \
        --project ${PROJECT}

    echo "✅ Face clustering complete"
else
    echo "⏭️  Clustering already exists, skipping"
fi

echo ""
echo "📊 Clustering Summary:"
cat "${CLUSTERS_DIR}/clustering_results.json" | grep -E "total_clusters|cluster_sizes" || echo "See ${CLUSTERS_DIR}/clustering_results.json"
echo ""

# ==========================================
# Step 5: Optional Pose Subclustering
# ==========================================
echo "=========================================="
echo "🤸 Step 5: Pose Subclustering (Optional)"
echo "=========================================="
echo ""

read -p "Run pose subclustering for training balance? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    POSE_DIR="${BASE_DIR}/pose_subclusters"

    echo "Running pose subclustering..."
    echo "   Input: ${CLUSTERS_DIR}"
    echo "   Output: ${POSE_DIR}"
    echo ""

    conda run -n ${CONDA_ENV} python ${POSE_SUBCLUSTER} \
        --clusters-dir "${CLUSTERS_DIR}" \
        --output-dir "${POSE_DIR}" \
        --method umap_hdbscan \
        --min-cluster-size 3 \
        --visualize \
        --project ${PROJECT}

    echo "✅ Pose subclustering complete"

    FINAL_DIR="${POSE_DIR}"
else
    echo "⏭️  Skipping pose subclustering"
    FINAL_DIR="${CLUSTERS_DIR}"
fi

echo ""

# ==========================================
# Pipeline Complete
# ==========================================
echo "=========================================="
echo "✅ Pipeline Complete!"
echo "=========================================="
echo ""
echo "Final Results:"
echo "   Character Clusters: ${FINAL_DIR}"
echo "   Next Steps:"
echo "     1. Review clusters and rename character folders"
echo "     2. Generate captions with prepare_training_data.py"
echo "     3. Build dataset with dataset_builder.py"
echo "     4. Train LoRA"
echo ""
echo "Quick Commands:"
echo "   # View clusters"
echo "   ls -lh ${FINAL_DIR}"
echo ""
echo "   # Generate captions for a character"
echo "   conda run -n ${CONDA_ENV} python ${SCRIPT_DIR}/generic/training/prepare_training_data.py \\"
echo "       --character-dirs ${FINAL_DIR}/character_0 \\"
echo "       --output-dir /path/to/training_data \\"
echo "       --character-name \"character_name\" \\"
echo "       --generate-captions"
echo ""
echo "   # Build dataset"
echo "   conda run -n ${CONDA_ENV} python ${SCRIPT_DIR}/generic/analysis/dataset_builder.py \\"
echo "       --input-dir ${FINAL_DIR} \\"
echo "       --output-dir /path/to/dataset \\"
echo "       --dataset-type character \\"
echo "       --format pytorch"
echo ""
