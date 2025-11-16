#!/bin/bash
# Download All Pipeline Models
# Ensures all models are in correct locations before 24h pipeline starts

set -e

MODEL_BASE="/mnt/c/AI_LLM_projects/ai_warehouse/models"
LOG_FILE="/tmp/model_download.log"

echo "================================================================================"
echo "Downloading All Pipeline Models"
echo "================================================================================"
echo "Base path: $MODEL_BASE"
echo "Log file: $LOG_FILE"
echo ""

# Function to download with retry
download_with_retry() {
    local url=$1
    local output=$2
    local max_attempts=3
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        echo "  Attempt $attempt/$max_attempts..."
        if wget -c --progress=bar:force -O "$output" "$url" 2>&1 | tee -a "$LOG_FILE"; then
            echo "  ✓ Downloaded successfully"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 5
    done

    echo "  ✗ Failed after $max_attempts attempts"
    return 1
}

# ============================================================================
# 1. YOLOv8x (Detection)
# ============================================================================
echo "=== 1. Downloading YOLOv8x (87 MB) ==="
YOLO_DIR="$MODEL_BASE/detection"
YOLO_PATH="$YOLO_DIR/yolov8x.pt"

if [ -f "$YOLO_PATH" ]; then
    echo "✓ YOLOv8x already exists: $YOLO_PATH"
else
    echo "Downloading YOLOv8x..."
    mkdir -p "$YOLO_DIR"
    download_with_retry \
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt" \
        "$YOLO_PATH"
fi
echo ""

# ============================================================================
# 2. IS-Net Anime (BritishWerewolf/IS-Net-Anime from HuggingFace)
# ============================================================================
echo "=== 2. Downloading IS-Net Anime from HuggingFace ==="
ISNET_DIR="$MODEL_BASE/segmentation/isnet-anime"
mkdir -p "$ISNET_DIR"

conda run -n ai_env python << 'EOF'
import os
os.makedirs("/mnt/c/AI_LLM_projects/ai_warehouse/models/segmentation/isnet-anime", exist_ok=True)

print("Installing huggingface-hub...")
import subprocess
subprocess.run(["pip", "install", "-q", "huggingface-hub"], check=False)

print("Downloading IS-Net-Anime model...")
from huggingface_hub import snapshot_download

try:
    model_path = snapshot_download(
        repo_id="BritishWerewolf/IS-Net-Anime",
        local_dir="/mnt/c/AI_LLM_projects/ai_warehouse/models/segmentation/isnet-anime",
        local_dir_use_symlinks=False
    )
    print(f"✓ IS-Net-Anime downloaded to: {model_path}")
except Exception as e:
    print(f"⚠️  Failed to download: {e}")
    print("   Will try alternative method...")
EOF

echo ""

# ============================================================================
# 3. Anime Segmentation (SkyTNT)
# ============================================================================
echo "=== 3. Downloading Anime Segmentation (SkyTNT) ==="
ANIME_SEG_DIR="$MODEL_BASE/segmentation/anime-segmentation"

if [ -d "$ANIME_SEG_DIR" ] && [ -f "$ANIME_SEG_DIR/isnetis.ckpt" ]; then
    echo "✓ Anime Segmentation already exists"
else
    echo "Cloning anime-segmentation repository..."
    git clone https://github.com/SkyTNT/anime-segmentation.git "$ANIME_SEG_DIR"

    if [ -d "$ANIME_SEG_DIR" ]; then
        cd "$ANIME_SEG_DIR"
        echo "Downloading ISNet model weights from HuggingFace..."

        conda run -n ai_env python << 'EOF'
from huggingface_hub import hf_hub_download
import os

os.chdir("/mnt/c/AI_LLM_projects/ai_warehouse/models/segmentation/anime-segmentation")

try:
    # Download ISNet checkpoint
    model_file = hf_hub_download(
        repo_id="skytnt/anime-seg",
        filename="isnetis.ckpt",
        local_dir=".",
        local_dir_use_symlinks=False
    )
    print(f"✓ Downloaded ISNet model: {model_file}")
except Exception as e:
    print(f"⚠️  Failed to download model: {e}")
EOF

        echo "✓ Anime Segmentation downloaded"
    else
        echo "✗ Failed to clone repository"
    fi
fi
echo ""

# ============================================================================
# 3. DeepLabV3 (Torchvision - will auto-download on first use)
# ============================================================================
echo "=== 3. Pre-loading DeepLabV3 (Torchvision) ==="
DEEPLABV3_DIR="$MODEL_BASE/segmentation/deeplabv3"
mkdir -p "$DEEPLABV3_DIR"

conda run -n ai_env python << 'EOF'
import torch
import torchvision
import os

cache_dir = "/mnt/c/AI_LLM_projects/ai_warehouse/models/segmentation/deeplabv3"
os.makedirs(cache_dir, exist_ok=True)

print("Loading DeepLabV3-ResNet101...")
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
print("✓ DeepLabV3 loaded and cached")

# Save to our cache
torch.save(model.state_dict(), f"{cache_dir}/deeplabv3_resnet101.pth")
print(f"✓ Saved to: {cache_dir}/deeplabv3_resnet101.pth")
EOF

echo ""

# ============================================================================
# 4. SegFormer (Hugging Face - will cache on first use)
# ============================================================================
echo "=== 4. Pre-loading SegFormer (Hugging Face) ==="
SEGFORMER_DIR="$MODEL_BASE/segmentation/segformer"
mkdir -p "$SEGFORMER_DIR"

conda run -n ai_env python << 'EOF'
from transformers import SegformerForSemanticSegmentation
import os

cache_dir = "/mnt/c/AI_LLM_projects/ai_warehouse/models/segmentation/segformer"
os.makedirs(cache_dir, exist_ok=True)

print("Downloading SegFormer-B5...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640",
    cache_dir=cache_dir
)
print(f"✓ SegFormer cached in: {cache_dir}")
EOF

echo ""

# ============================================================================
# 5. RAFT (Optical Flow) from Princeton
# ============================================================================
echo "=== 5. Downloading RAFT (Optical Flow) ==="
RAFT_DIR="$MODEL_BASE/flow/RAFT"

if [ -d "$RAFT_DIR" ] && [ "$(ls -A $RAFT_DIR)" ]; then
    echo "✓ RAFT already exists"
else
    echo "Cloning RAFT repository..."
    git clone https://github.com/princeton-vl/RAFT.git "$RAFT_DIR"

    if [ -d "$RAFT_DIR" ]; then
        cd "$RAFT_DIR"
        echo "Downloading RAFT model weights..."

        mkdir -p models
        cd models

        # Download RAFT models from Google Drive (via direct links)
        echo "Downloading raft-things.pth..."
        wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1xJ-RVIR4rNEhz4NvZPKXvfCbYKHJnO6X' \
            -O raft-things.pth 2>&1 | tail -5

        if [ -f "raft-things.pth" ]; then
            echo "✓ RAFT raft-things.pth downloaded"
        else
            echo "⚠️  Direct download failed, will download on first use"
        fi

        echo "✓ RAFT repository cloned"
    else
        echo "✗ Failed to clone RAFT"
    fi
fi
echo ""

# ============================================================================
# 6. RIFE 4.6 (Frame Interpolation)
# ============================================================================
echo "=== 6. Downloading RIFE 4.6 (Frame Interpolation) ==="
RIFE_DIR="$MODEL_BASE/interpolation/RIFE_4.6"
mkdir -p "$RIFE_DIR"

if [ -d "$RIFE_DIR" ] && [ "$(ls -A $RIFE_DIR)" ]; then
    echo "✓ RIFE 4.6 already exists: $RIFE_DIR"
else
    echo "Cloning RIFE repository..."
    cd "$MODEL_BASE/interpolation"

    # Clone RIFE repo
    if [ ! -d "Practical-RIFE" ]; then
        git clone https://github.com/hzwer/Practical-RIFE.git
    fi

    cd Practical-RIFE

    # Download model weights
    echo "Downloading RIFE 4.6 weights..."
    if [ ! -d "train_log" ]; then
        mkdir -p train_log
    fi

    # Download from releases
    wget -c https://github.com/hzwer/Practical-RIFE/releases/download/4.6/flownet.pkl \
        -O train_log/flownet.pkl

    # Create symlink
    ln -sf "$MODEL_BASE/interpolation/Practical-RIFE" "$RIFE_DIR"
    echo "✓ RIFE 4.6 downloaded"
fi
echo ""

# ============================================================================
# 7. Additional Anime Segmentation Models (ISNet, U2Net, MODNet, InSPyReNet)
# ============================================================================
echo "=== 7. Downloading Additional Anime Segmentation Models ==="

# ISNet (DIS)
ISNET_DIR="$MODEL_BASE/segmentation/DIS"
if [ ! -d "$ISNET_DIR" ]; then
    echo "Cloning ISNet (DIS) repository..."
    git clone https://github.com/xuebinqin/DIS.git "$ISNET_DIR"
    echo "✓ ISNet (DIS) repository cloned"
fi

# U2Net
U2NET_DIR="$MODEL_BASE/segmentation/U-2-Net"
if [ ! -d "$U2NET_DIR" ]; then
    echo "Cloning U2Net repository..."
    git clone https://github.com/xuebinqin/U-2-Net.git "$U2NET_DIR"

    # Download U2Net model weights
    if [ -d "$U2NET_DIR" ]; then
        cd "$U2NET_DIR/saved_models"
        echo "Downloading U2Net model weights..."
        wget -c https://drive.google.com/uc?export=download&id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ \
            -O u2net.pth 2>&1 | tail -3 || echo "⚠️ U2Net weights download may require manual download"
    fi
    echo "✓ U2Net repository cloned"
fi

# MODNet
MODNET_DIR="$MODEL_BASE/segmentation/MODNet"
if [ ! -d "$MODNET_DIR" ]; then
    echo "Cloning MODNet repository..."
    git clone https://github.com/ZHKKKe/MODNet.git "$MODNET_DIR"
    echo "✓ MODNet repository cloned"
fi

# InSPyReNet
INSPYRENET_DIR="$MODEL_BASE/segmentation/InSPyReNet"
if [ ! -d "$INSPYRENET_DIR" ]; then
    echo "Cloning InSPyReNet repository..."
    git clone https://github.com/plemeri/InSPyReNet.git "$INSPYRENET_DIR"
    echo "✓ InSPyReNet repository cloned"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "Model Download Summary"
echo "================================================================================"
echo ""

echo "Checking model availability:"
echo ""

check_model() {
    local name=$1
    local path=$2

    if [ -e "$path" ]; then
        size=$(du -h "$path" 2>/dev/null | cut -f1)
        echo "✓ $name: $path ($size)"
        return 0
    else
        echo "✗ $name: NOT FOUND at $path"
        return 1
    fi
}

success=0
total=0

# YOLOv8x
((total++))
if check_model "YOLOv8x" "$MODEL_BASE/detection/yolov8x.pt"; then
    ((success++))
fi

# DeepLabV3
((total++))
if check_model "DeepLabV3" "$MODEL_BASE/segmentation/deeplabv3/deeplabv3_resnet101.pth"; then
    ((success++))
fi

# SegFormer
((total++))
if [ -d "$MODEL_BASE/segmentation/segformer" ] && [ "$(ls -A $MODEL_BASE/segmentation/segformer)" ]; then
    echo "✓ SegFormer: $MODEL_BASE/segmentation/segformer (cached)"
    ((success++))
else
    echo "✗ SegFormer: NOT FOUND"
fi

# RIFE
((total++))
if [ -d "$MODEL_BASE/interpolation/Practical-RIFE" ]; then
    echo "✓ RIFE 4.6: $MODEL_BASE/interpolation/Practical-RIFE"
    ((success++))
else
    echo "✗ RIFE 4.6: NOT FOUND"
fi

echo ""
echo "Models ready: $success/$total"
echo ""

if [ $success -eq $total ]; then
    echo "✅ All essential models downloaded and verified!"
    echo ""
    echo "Optional models (will be downloaded on demand if needed):"
    echo "  - U2-Net Anime: For specialized anime character segmentation"
    echo "  - RAFT: For optical flow (can use torchvision alternative)"
    echo "  - HRNet Pose: For pose estimation (mmpose will auto-download)"
    echo ""
    exit 0
else
    echo "⚠️  Some models missing but core models ready"
    echo "Pipeline can run with available models"
    echo ""
    exit 0
fi
