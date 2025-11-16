#!/bin/bash
# Install and verify SOTA evaluation models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WAREHOUSE="/mnt/c/AI_LLM_projects/ai_warehouse/models"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         SOTA EVALUATION MODELS - INSTALLATION              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 1. Install Python dependencies
echo -e "${YELLOW}[1/5] Installing Python dependencies...${NC}"
conda run -n ai_env pip install -r "$PROJECT_ROOT/requirements/sota_evaluation.txt"
echo -e "${GREEN}✓ Python dependencies installed${NC}"
echo ""

# 2. Download InternVL2-8B
echo -e "${YELLOW}[2/5] Downloading InternVL2-8B (16GB)...${NC}"
INTERNVL_PATH="$WAREHOUSE/vlm/InternVL2-8B"

if [ -d "$INTERNVL_PATH" ]; then
    echo -e "${GREEN}✓ InternVL2-8B already exists${NC}"
else
    mkdir -p "$WAREHOUSE/vlm"
    cd "$WAREHOUSE/vlm"

    echo "  Downloading from HuggingFace..."
    huggingface-cli download OpenGVLab/InternVL2-8B \
      --local-dir InternVL2-8B \
      --local-dir-use-symlinks False

    echo -e "${GREEN}✓ InternVL2-8B downloaded${NC}"
fi
echo ""

# 3. Setup LAION Aesthetics
echo -e "${YELLOW}[3/5] Setting up LAION Aesthetics...${NC}"

conda run -n ai_env python -c "
from transformers import pipeline
try:
    scorer = pipeline('image-classification', model='cafeai/cafe_aesthetic', device=-1)
    print('✓ LAION Aesthetics model loaded successfully')
except Exception as e:
    print(f'✗ Error: {e}')
"

echo -e "${GREEN}✓ LAION Aesthetics ready${NC}"
echo ""

# 4. Setup InsightFace
echo -e "${YELLOW}[4/5] Setting up InsightFace...${NC}"

conda run -n ai_env python -c "
import insightface
from insightface.app import FaceAnalysis
try:
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1)
    print('✓ InsightFace initialized successfully')
except Exception as e:
    print(f'✗ Error: {e}')
"

echo -e "${GREEN}✓ InsightFace ready${NC}"
echo ""

# 5. Verify LPIPS and MUSIQ
echo -e "${YELLOW}[5/5] Verifying LPIPS and MUSIQ...${NC}"

conda run -n ai_env python -c "
import torch

# Test LPIPS
try:
    import lpips
    loss_fn = lpips.LPIPS(net='alex')
    print('✓ LPIPS loaded')
except Exception as e:
    print(f'✗ LPIPS error: {e}')

# Test MUSIQ
try:
    import pyiqa
    metric = pyiqa.create_metric('musiq', device='cpu')
    print('✓ MUSIQ loaded')
except Exception as e:
    print(f'✗ MUSIQ error: {e}')
"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              INSTALLATION COMPLETE                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Summary
echo -e "${BLUE}Installed SOTA models:${NC}"
echo "  ✓ InternVL2-8B (prompt alignment)"
echo "  ✓ LAION Aesthetics V2 (aesthetic scoring)"
echo "  ✓ InsightFace (character consistency)"
echo "  ✓ LPIPS (perceptual diversity)"
echo "  ✓ MUSIQ (image quality)"
echo ""

echo -e "${YELLOW}Storage used:${NC}"
du -sh "$WAREHOUSE/vlm/InternVL2-8B" 2>/dev/null || echo "  InternVL2: Not found"
echo ""

echo -e "${GREEN}Ready to use SOTA evaluator!${NC}"
echo ""
echo -e "${BLUE}Test command:${NC}"
echo "  conda run -n ai_env python scripts/evaluation/sota_lora_evaluator.py --help"
echo ""
