#!/bin/bash
#
# Pipeline Dependencies Installer
# Ensures all packages are compatible with PyTorch 2.7.1 + CUDA 12.8
# Run this BEFORE starting the 24h pipeline
#

set -e

echo "================================================================================"
echo "Installing Pipeline Dependencies (Compatible with PyTorch 2.7.1 + CUDA 12.8)"
echo "================================================================================"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ai_env

echo "=== Step 1: Verify Base PyTorch Installation ==="
python << 'EOF'
import torch
import sys

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.__version__.startswith("2.7"):
    print("ERROR: PyTorch version must be 2.7.x!")
    sys.exit(1)

if not torch.cuda.is_available():
    print("WARNING: CUDA not available!")

print("✓ Base PyTorch verified")
EOF

echo ""
echo "=== Step 2: Install Core CV/ML Libraries (Compatible Versions) ==="

# OpenCV - compatible with PyTorch 2.7.1
pip install opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84

# Scene detection
pip install scenedetect==0.6.4

# Image processing
pip install pillow==10.4.0 imageio==2.35.1

# Scientific computing (already installed, verify compatibility)
pip install numpy==1.26.4 scipy==1.14.1

# Visualization
pip install matplotlib==3.9.2

echo ""
echo "=== Step 3: Install Deep Learning Framework Extensions ==="

# Torchvision/Torchaudio already installed (2.7.1 compatible)
# Verify versions match
python << 'EOF'
import torch, torchvision, torchaudio
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"torchaudio: {torchaudio.__version__}")
EOF

# MMDetection/MMPose - Install compatible versions
pip install openmim==0.3.9
mim install mmengine==0.10.5
mim install mmcv==2.2.0  # Compatible with PyTorch 2.7
mim install mmdet==3.3.0
mim install mmpose==1.3.2

echo ""
echo "=== Step 4: Install Segmentation Models ==="

# Segmentation models
pip install timm==1.0.9  # Segformer dependency
pip install transformers==4.45.2  # HuggingFace models
pip install accelerate==0.34.2

echo ""
echo "=== Step 5: Install Optical Flow & Motion Analysis ==="

# RAFT/FlowNet dependencies
pip install tensorboard==2.17.1

echo ""
echo "=== Step 6: Install Frame Interpolation Models ==="

# RIFE/FILM dependencies already covered by torch/numpy

echo ""
echo "=== Step 7: Install OCR & Detection Tools ==="

# EasyOCR (lightweight, compatible)
pip install easyocr==1.7.2

# PaddleOCR
pip install paddlepaddle-gpu==2.6.2 -i https://mirror.baidu.com/pypi/simple
pip install paddleocr==2.8.1

echo ""
echo "=== Step 8: Install CLIP & Vision-Language Models ==="

pip install ftfy==6.2.3 regex==2024.9.11
pip install git+https://github.com/openai/CLIP.git

echo ""
echo "=== Step 9: Install Quality & Aesthetic Assessment ==="

# Image quality assessment
pip install scikit-image==0.24.0
pip install lpips==0.1.4

echo ""
echo "=== Step 10: Install Utilities ==="

pip install tqdm==4.66.5
pip install pyyaml==6.0.2
pip install pandas==2.2.3
pip install seaborn==0.13.2

echo ""
echo "=== Step 11: Verify All Installations ==="

python << 'EOF'
import sys

packages_to_verify = [
    ("torch", "2.7"),
    ("torchvision", "0.22"),
    ("cv2", None),
    ("numpy", None),
    ("PIL", None),
    ("mmdet", None),
    ("mmpose", None),
    ("transformers", None),
    ("easyocr", None),
    ("paddleocr", None),
    ("clip", None),
    ("lpips", None),
]

print("Verifying installed packages:")
print("-" * 60)

all_ok = True
for package, expected_version in packages_to_verify:
    try:
        module = __import__(package)
        version = getattr(module, "__version__", "unknown")

        if expected_version and not version.startswith(expected_version):
            print(f"✗ {package:20s} {version:15s} (expected {expected_version}.x)")
            all_ok = False
        else:
            print(f"✓ {package:20s} {version:15s}")
    except ImportError as e:
        print(f"✗ {package:20s} NOT INSTALLED")
        all_ok = False

print("-" * 60)

if all_ok:
    print("✓ All packages verified successfully!")
    sys.exit(0)
else:
    print("✗ Some packages missing or incompatible")
    sys.exit(1)
EOF

echo ""
echo "================================================================================"
echo "✓ Pipeline Dependencies Installation Complete!"
echo "================================================================================"
echo ""
echo "You can now run the 24h pipeline:"
echo "  ./start_24h_pipeline.sh"
echo ""
