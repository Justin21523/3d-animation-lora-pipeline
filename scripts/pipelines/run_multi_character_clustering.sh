#!/bin/bash
#
# Multi-Character Identity Clustering Pipeline
# Complete pipeline for extracting and clustering multiple characters from 3D animation
#
# Usage:
#   bash scripts/pipelines/run_multi_character_clustering.sh <film_name>
#
# Example:
#   bash scripts/pipelines/run_multi_character_clustering.sh luca
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FILM_NAME=${1:-luca}
DATA_DIR="/mnt/data/ai_data/datasets/3d-anime/${FILM_NAME}"
FRAMES_DIR="${DATA_DIR}/frames"  # Frames directly in frames/ directory
INSTANCES_DIR="${DATA_DIR}/instances"
IDENTITY_DIR="${DATA_DIR}/identity_clusters"
POSE_SUBCLUSTER_DIR="${DATA_DIR}/pose_subclusters"
FINAL_DIR="${DATA_DIR}/final_clusters"

# Model settings
SAM2_MODEL="sam2_hiera_large"
MIN_INSTANCE_SIZE=16384  # 128x128 pixels
MIN_CLUSTER_SIZE=10
POSE_MODEL="rtmpose-m"
MIN_POSE_CLUSTER_SIZE=5
DEVICE="cuda"

# Resource optimization settings (32-core CPU + RTX 5080 GPU)
# CPU-intensive tasks: Use all 32 threads
# GPU-intensive tasks: Single GPU process, no conflicts
CPU_WORKERS=32              # For CPU-intensive clustering
GPU_BATCH_SIZE=8            # For SAM2 segmentation
FACE_BATCH_SIZE=32          # For face detection/embedding
POSE_BATCH_SIZE=16          # For pose estimation
DATALOADER_WORKERS=8        # For I/O operations

# Memory management
EMPTY_CACHE_INTERVAL=50     # Clear GPU cache every N batches

# Conda environment
CONDA_ENV="ai_env"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Multi-Character Identity Clustering${NC}"
echo -e "${BLUE}Film: ${FILM_NAME}${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Display resource optimization configuration
echo -e "${YELLOW}Resource Optimization Configuration:${NC}"
echo -e "  CPU Workers: ${CPU_WORKERS} threads (for CPU-intensive clustering)"
echo -e "  GPU Batch Size: ${GPU_BATCH_SIZE} (SAM2 segmentation)"
echo -e "  Face Batch Size: ${FACE_BATCH_SIZE} (face detection/embedding)"
echo -e "  Pose Batch Size: ${POSE_BATCH_SIZE} (pose estimation)"
echo -e "  DataLoader Workers: ${DATALOADER_WORKERS} (I/O operations)"
echo -e "  GPU Cache Clear Interval: Every ${EMPTY_CACHE_INTERVAL} batches"
echo -e "${YELLOW}Strategy:${NC} Single GPU process (no conflicts) + Full CPU parallelization${NC}\n"

# Check if frames exist
if [ ! -d "$FRAMES_DIR" ]; then
    echo -e "${RED}Error: Frames directory not found: $FRAMES_DIR${NC}"
    echo -e "${YELLOW}Please run frame extraction first.${NC}"
    exit 1
fi

# Count frames
FRAME_COUNT=$(find "$FRAMES_DIR" -name "*.jpg" -o -name "*.png" | wc -l)
echo -e "${GREEN}✓ Found $FRAME_COUNT frames${NC}\n"

# ============================================
# STAGE 1: Instance-level Segmentation (SAM2)
# ============================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}STAGE 1: Instance-level Segmentation${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Using SAM2 to extract each character instance..."
echo -e "Model: ${SAM2_MODEL}"
echo -e "Min instance size: ${MIN_INSTANCE_SIZE} pixels\n"

if [ -d "$INSTANCES_DIR" ]; then
    echo -e "${YELLOW}⚠ Instances directory already exists.${NC}"
    read -p "Overwrite? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Skipping instance segmentation...${NC}\n"
    else
        rm -rf "$INSTANCES_DIR"
        conda run -n "$CONDA_ENV" python scripts/generic/segmentation/instance_segmentation.py \
            "$FRAMES_DIR" \
            --output-dir "$INSTANCES_DIR" \
            --model "$SAM2_MODEL" \
            --device "$DEVICE" \
            --min-size "$MIN_INSTANCE_SIZE" \
            --visualize
    fi
else
    conda run -n "$CONDA_ENV" python scripts/generic/segmentation/instance_segmentation.py \
        "$FRAMES_DIR" \
        --output-dir "$INSTANCES_DIR" \
        --model "$SAM2_MODEL" \
        --device "$DEVICE" \
        --min-size "$MIN_INSTANCE_SIZE" \
        --visualize
fi

# Check instance segmentation results
if [ ! -d "${INSTANCES_DIR}/instances" ]; then
    echo -e "${RED}Error: Instance segmentation failed!${NC}"
    exit 1
fi

INSTANCE_COUNT=$(find "${INSTANCES_DIR}/instances" -name "*.png" | wc -l)
echo -e "${GREEN}✓ Extracted $INSTANCE_COUNT character instances${NC}"
echo -e "${GREEN}✓ Visualizations saved to: ${INSTANCES_DIR}/visualization/${NC}\n"

# ============================================
# STAGE 2: Face-centric Identity Clustering
# ============================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}STAGE 2: Face-centric Identity Clustering${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Using face recognition to group by character identity..."
echo -e "Min cluster size: ${MIN_CLUSTER_SIZE}\n"

if [ -d "$IDENTITY_DIR" ]; then
    echo -e "${YELLOW}⚠ Identity clusters directory already exists.${NC}"
    read -p "Overwrite? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Skipping identity clustering...${NC}\n"
    else
        rm -rf "$IDENTITY_DIR"
        conda run -n "$CONDA_ENV" python scripts/generic/clustering/face_identity_clustering.py \
            "${INSTANCES_DIR}/instances" \
            --output-dir "$IDENTITY_DIR" \
            --min-cluster-size "$MIN_CLUSTER_SIZE" \
            --device "$DEVICE" \
            --save-faces
    fi
else
    conda run -n "$CONDA_ENV" python scripts/generic/clustering/face_identity_clustering.py \
        "${INSTANCES_DIR}/instances" \
        --output-dir "$IDENTITY_DIR" \
        --min-cluster-size "$MIN_CLUSTER_SIZE" \
        --device "$DEVICE" \
        --save-faces
fi

# Check identity clustering results
if [ ! -f "${IDENTITY_DIR}/identity_clustering.json" ]; then
    echo -e "${RED}Error: Identity clustering failed!${NC}"
    exit 1
fi

# Parse results
N_IDENTITIES=$(cat "${IDENTITY_DIR}/identity_clustering.json" | grep -o '"n_identities": [0-9]*' | grep -o '[0-9]*')
echo -e "${GREEN}✓ Found $N_IDENTITIES character identities${NC}"
echo -e "${GREEN}✓ Face crops saved to: identity_*/faces/${NC}\n"

# ============================================
# STAGE 3: Interactive Review & Naming
# ============================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}STAGE 3: Interactive Review & Naming${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Launching web interface for manual review...\n"

echo -e "${YELLOW}Instructions:${NC}"
echo -e "1. Review each identity cluster"
echo -e "2. Rename clusters: identity_000 → character_name"
echo -e "   Examples:"
echo -e "   - identity_000 → luca_human_form"
echo -e "   - identity_001 → alberto_sea_monster"
echo -e "   - identity_002 → giulia"
echo -e "3. Merge incorrectly split identities"
echo -e "4. Move misclassified instances"
echo -e "5. Save changes when done\n"

read -p "Launch interactive review? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Starting web server on http://localhost:8000${NC}\n"

    conda run -n "$CONDA_ENV" python scripts/generic/clustering/launch_interactive_review.py \
        "$IDENTITY_DIR" \
        --port 8000

    echo -e "\n${GREEN}✓ Review complete!${NC}\n"
fi

# ============================================
# STAGE 4: Pose/View Subclustering (Optional)
# ============================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}STAGE 4: Pose/View Subclustering${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Subdivide each identity into pose/view buckets for better training control...\n"

echo -e "${YELLOW}Purpose:${NC}"
echo -e "  • Separate same character into pose/view groups (front/three-quarter/profile/back)"
echo -e "  • Enable balanced angle/pose sampling for LoRA training"
echo -e "  • Improve caption consistency and generalization\n"

echo -e "${YELLOW}Process:${NC}"
echo -e "  • Pose estimation: RTM-Pose keypoint detection"
echo -e "  • View classification: Front/Three-quarter/Profile/Back"
echo -e "  • Subclustering: UMAP + HDBSCAN by pose+view features\n"

echo -e "${YELLOW}Output:${NC}"
echo -e "  identity_XXX/pose_YYY/ folders with pose-specific instances\n"

read -p "Run pose subclustering? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Starting pose/view subclustering...${NC}\n"

    conda run -n "$CONDA_ENV" python scripts/generic/clustering/pose_subclustering.py \
        "$IDENTITY_DIR" \
        --output-dir "$POSE_SUBCLUSTER_DIR" \
        --pose-model "$POSE_MODEL" \
        --device "$DEVICE" \
        --method umap_hdbscan \
        --min-cluster-size "$MIN_POSE_CLUSTER_SIZE" \
        --visualize

    echo -e "\n${GREEN}✓ Pose subclustering complete!${NC}"
    echo -e "${GREEN}✓ Results saved to: ${POSE_SUBCLUSTER_DIR}/${NC}\n"

    # Update FINAL_DIR to use pose subclustered results
    FINAL_DIR="$POSE_SUBCLUSTER_DIR"
else
    echo -e "${YELLOW}Skipping pose subclustering...${NC}\n"
    FINAL_DIR="$IDENTITY_DIR"
fi

# ============================================
# STAGE 5: Summary & Next Steps
# ============================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Pipeline Complete!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${GREEN}Results:${NC}"
echo -e "  Frames extracted: ${FRAME_COUNT}"
echo -e "  Character instances: ${INSTANCE_COUNT}"
echo -e "  Identities found: ${N_IDENTITIES}"
echo -e ""
echo -e "${GREEN}Output directories:${NC}"
echo -e "  Instances: ${INSTANCES_DIR}/instances/"
echo -e "  Visualizations: ${INSTANCES_DIR}/visualization/"
echo -e "  Identity clusters: ${IDENTITY_DIR}/"
if [ "$FINAL_DIR" != "$IDENTITY_DIR" ]; then
    echo -e "  Pose subclusters: ${POSE_SUBCLUSTER_DIR}/"
fi
echo -e "  Final clusters: ${FINAL_DIR}/"
echo -e ""
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "  1. Generate captions for each character:"
echo -e "     ${BLUE}conda run -n ai_env python scripts/generic/training/prepare_training_data.py \\${NC}"
echo -e "     ${BLUE}  --character-dirs ${FINAL_DIR}/character_name* \\${NC}"
echo -e "     ${BLUE}  --character-name \"Character Name\" \\${NC}"
echo -e "     ${BLUE}  --generate-captions \\${NC}"
echo -e "     ${BLUE}  --vlm-model qwen2_vl${NC}"
echo -e ""
echo -e "  2. Train character LoRA"
echo -e ""
echo -e "  3. (Optional) For temporal consistency:"
echo -e "     ${BLUE}bash scripts/pipelines/run_temporal_consistency.sh ${FILM_NAME}${NC}"
echo -e ""
echo -e "${GREEN}Documentation:${NC}"
echo -e "  Architecture: docs/guides/MULTI_CHARACTER_CLUSTERING.md"
echo -e "  Interactive UI: scripts/generic/clustering/interactive_ui/README.md"
echo -e ""
echo -e "${BLUE}========================================${NC}\n"
