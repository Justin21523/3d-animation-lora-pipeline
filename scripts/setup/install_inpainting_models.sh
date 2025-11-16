#!/bin/bash
# Install Inpainting Models and Dependencies
# Supports LaMa, Stable Diffusion Inpainting, and OpenCV

set -e  # Exit on error

echo "🎨 Installing Inpainting Models and Dependencies..."
echo ""

# Colors for output
GREEN='\033[0.32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${RED}❌ Error: No conda environment activated${NC}"
    echo "   Please activate ai_env first:"
    echo "   conda activate ai_env"
    exit 1
fi

echo -e "${GREEN}✓ Conda environment: $CONDA_DEFAULT_ENV${NC}"
echo ""

# Function to check if package is installed
check_package() {
    python -c "import $1" 2>/dev/null
    return $?
}

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install --upgrade pip

# Install LaMa dependencies
echo ""
echo "🔧 Installing LaMa (lama-cleaner)..."
if check_package "lama_cleaner"; then
    echo -e "${YELLOW}⚠️  lama-cleaner already installed${NC}"
else
    pip install lama-cleaner
    echo -e "${GREEN}✓ LaMa installed${NC}"
fi

# Install Stable Diffusion Inpainting dependencies
echo ""
echo "🔧 Installing Stable Diffusion Inpainting (diffusers)..."
if check_package "diffusers"; then
    echo -e "${YELLOW}⚠️  diffusers already installed${NC}"
else
    pip install diffusers transformers accelerate
    echo -e "${GREEN}✓ Diffusers installed${NC}"
fi

# Install supporting libraries
echo ""
echo "🔧 Installing supporting libraries..."
pip install opencv-python pillow numpy tqdm

# Download models
echo ""
echo "📥 Downloading inpainting models..."

# Create models directory
MODELS_DIR="/mnt/data/ai_data/models/inpainting"
mkdir -p "$MODELS_DIR"

echo ""
echo "🔧 Setting up LaMa model..."
python << 'EOF'
try:
    from lama_cleaner.model_manager import ModelManager
    from lama_cleaner.schema import Config
    import os

    # Initialize model manager (downloads model automatically)
    model_manager = ModelManager(
        name="lama",
        device="cpu"  # Just for download
    )
    print("✓ LaMa model downloaded successfully")
except Exception as e:
    print(f"⚠️  LaMa model setup: {e}")
EOF

echo ""
echo "🔧 Testing Stable Diffusion Inpainting model access..."
python << 'EOF'
try:
    from diffusers import StableDiffusionInpaintPipeline
    import torch

    # Check if model cache exists
    model_id = "runwayml/stable-diffusion-inpainting"
    print(f"✓ Stable Diffusion Inpainting model ID: {model_id}")
    print("  Note: Model will be downloaded on first use (~2GB)")
except Exception as e:
    print(f"⚠️  SD Inpainting setup: {e}")
EOF

# Test installations
echo ""
echo "🧪 Testing installations..."

python << 'EOF'
import sys

# Test imports
try:
    import cv2
    print("✓ OpenCV imported successfully")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print("✓ Pillow imported successfully")
except ImportError as e:
    print(f"❌ Pillow import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")
    sys.exit(1)

try:
    from lama_cleaner.model_manager import ModelManager
    print("✓ LaMa imported successfully")
except ImportError as e:
    print(f"⚠️  LaMa import failed: {e}")
    print("   (LaMa is optional but recommended)")

try:
    from diffusers import StableDiffusionInpaintPipeline
    print("✓ Diffusers imported successfully")
except ImportError as e:
    print(f"⚠️  Diffusers import failed: {e}")
    print("   (SD Inpainting is optional)")

print("")
print("✅ Core inpainting dependencies installed successfully!")
EOF

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Installation Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📋 Installed Components:"
echo "   ✓ OpenCV (Telea/NS inpainting)"
echo "   ✓ LaMa (lama-cleaner)"
echo "   ✓ Stable Diffusion Inpainting (diffusers)"
echo ""
echo "💡 Next Steps:"
echo "   1. Test inpainting on sample images:"
echo "      python scripts/generic/enhancement/inpaint_occlusions.py --help"
echo ""
echo "   2. Run on Luca instances (after SAM2 completes):"
echo "      conda run -n ai_env python scripts/generic/enhancement/inpaint_occlusions.py \\"
echo "        --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_sampled/instances \\"
echo "        --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/instances_inpainted \\"
echo "        --method lama \\"
echo "        --occlusion-threshold 0.15"
echo ""
echo "   3. Use character-specific prompts (for SD method):"
echo "      --config configs/inpainting/luca_prompts.json"
echo ""
echo "═══════════════════════════════════════════════════════════"
