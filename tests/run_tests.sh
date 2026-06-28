#!/usr/bin/env bash
#
# Demo-safe test runner for 3D Animation LoRA Pipeline.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "3D Animation LoRA Pipeline Test Suite"
echo "========================================"
echo ""

# Get project root (parent of tests directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}Running demo-safe smoke suite${NC}"

python -m pytest \
  tests/demo \
  tests/test_config.py \
  tests/test_frames.py \
  tests/test_detection.py \
  tests/test_segmentation.py \
  tests/test_pose.py \
  tests/test_embeddings_and_datasets.py \
  tests/test_training.py \
  tests/test_controlnet_training.py \
  tests/test_inference_animation.py \
  tests/test_upscale_and_interpolate.py \
  -q

echo ""
echo -e "${GREEN}Demo-safe tests passed.${NC}"
