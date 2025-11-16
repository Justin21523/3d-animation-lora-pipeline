#!/bin/bash
# Setup dedicated conda environment for Kohya LoRA training
# Compatible with RTX 5080 and PyTorch 2.7.1+cu128

set -e  # Exit on error

echo "========================================================================"
echo "KOHYA SS ENVIRONMENT SETUP"
echo "========================================================================"
echo "PyTorch: 2.7.1+cu128"
echo "Target GPU: RTX 5080 (CUDA 12.8)"
echo "========================================================================"
echo ""

ENV_NAME="kohya_ss"
PYTHON_VERSION="3.10"

# Step 1: Create conda environment
echo "Step 1/6: Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y || echo "Environment may already exist"

# Step 2: Install PyTorch with CUDA 12.8
echo ""
echo "Step 2/6: Installing PyTorch 2.7.1+cu128"
conda run -n $ENV_NAME pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Step 3: Install compatible triton (REQUIRED for bitsandbytes)
echo ""
echo "Step 3/6: Installing compatible triton"
# PyTorch 2.7.1 requires triton ~= 3.1
conda run -n $ENV_NAME pip install triton==3.1.0

# Step 4: Install bitsandbytes (compile from source or use compatible wheel)
echo ""
echo "Step 4/6: Installing bitsandbytes with CUDA 12.8 support"
# Try precompiled wheel first, fallback to source if needed
conda run -n $ENV_NAME pip install \
    bitsandbytes>=0.45.0 || \
    conda run -n $ENV_NAME pip install \
        --no-build-isolation \
        git+https://github.com/bitsandbytes-foundation/bitsandbytes.git

# Step 5: Install Kohya ss requirements
echo ""
echo "Step 5/6: Installing Kohya ss dependencies"
conda run -n $ENV_NAME pip install \
    accelerate==0.30.0 \
    transformers==4.44.0 \
    diffusers[torch]==0.25.0 \
    ftfy==6.1.1 \
    opencv-python==4.8.1.78 \
    einops==0.7.0 \
    safetensors==0.4.2 \
    huggingface-hub==0.24.5 \
    tensorboard \
    toml==0.10.2 \
    voluptuous==0.13.1 \
    imagesize==1.4.1 \
    altair==4.2.2 \
    easygui==0.98.3

# Optional optimizers
conda run -n $ENV_NAME pip install \
    prodigyopt==1.0 \
    lion-pytorch==0.0.6

# Step 6: Verify installation
echo ""
echo "Step 6/6: Verifying installation"
echo "========================================================================"

conda run -n $ENV_NAME python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print()

try:
    import bitsandbytes as bnb
    print(f'bitsandbytes: {bnb.__version__}')

    # Test AdamW8bit
    param = torch.nn.Parameter(torch.randn(10, 10).cuda())
    optimizer = bnb.optim.AdamW8bit([param], lr=1e-3)
    loss = (param ** 2).sum()
    loss.backward()
    optimizer.step()
    print('✓ AdamW8bit test: PASSED')
except Exception as e:
    print(f'✗ bitsandbytes test FAILED: {e}')
print()

import transformers
print(f'transformers: {transformers.__version__}')

import diffusers
print(f'diffusers: {diffusers.__version__}')

import accelerate
print(f'accelerate: {accelerate.__version__}')
"

echo ""
echo "========================================================================"
echo "✓ ENVIRONMENT SETUP COMPLETE"
echo "========================================================================"
echo ""
echo "To activate: conda activate $ENV_NAME"
echo "To test training: bash test_kohya_training.sh"
echo ""
