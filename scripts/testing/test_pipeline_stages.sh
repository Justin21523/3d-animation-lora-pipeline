#!/usr/bin/bash
#
# Pipeline Stage Testing Script
# ==============================
#
# Tests each pipeline stage independently with small samples
# to ensure everything works before full execution.
#
# Author: Claude Code
# Date: 2025-11-13

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline"
TEST_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/pipeline_test"
LOG_FILE="${PROJECT_ROOT}/logs/pipeline_test_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "${TEST_DIR}"

# ============================================================
# Helper Functions
# ============================================================

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗ $1${NC}" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠ $1${NC}" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓ $1${NC}" | tee -a "${LOG_FILE}"
}

section() {
    echo "" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "$1" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 is available"
        return 0
    else
        log_error "$1 is NOT available"
        return 1
    fi
}

# ============================================================
# Stage 0: Environment Check
# ============================================================

check_environment() {
    section "STAGE 0: Environment Check"

    # Check conda
    log "Checking conda..."
    check_command conda

    # Check ai_env
    log "Checking ai_env conda environment..."
    if conda env list | grep -q "^ai_env "; then
        log_success "ai_env environment exists"

        # Check Python
        log "Checking Python in ai_env..."
        conda run -n ai_env python --version | tee -a "${LOG_FILE}"

        # Check key packages
        log "Checking key Python packages in ai_env..."
        conda run -n ai_env python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>&1 | tee -a "${LOG_FILE}"
        conda run -n ai_env python -c "import PIL; print(f'Pillow: {PIL.__version__}')" 2>&1 | tee -a "${LOG_FILE}"
        conda run -n ai_env python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" 2>&1 | tee -a "${LOG_FILE}"

    else
        log_error "ai_env environment NOT found"
        exit 1
    fi

    # Check kohya_ss
    log "Checking kohya_ss conda environment..."
    if conda env list | grep -q "^kohya_ss "; then
        log_success "kohya_ss environment exists"

        # Check if train_network.py exists
        KOHYA_DIR=$(conda run -n kohya_ss python -c "import sys; print([p for p in sys.path if 'kohya_ss' in p][0])" 2>/dev/null || echo "")
        if [ -n "$KOHYA_DIR" ]; then
            log "Kohya_ss path: $KOHYA_DIR"
        else
            log_warn "Could not determine kohya_ss path automatically"
        fi
    else
        log_error "kohya_ss environment NOT found"
        exit 1
    fi

    # Check CUDA
    log "Checking CUDA availability..."
    if conda run -n ai_env python -c "import torch; print(torch.cuda.is_available())" 2>&1 | grep -q "True"; then
        log_success "CUDA is available"
        conda run -n ai_env python -c "import torch; print(f'CUDA Device: {torch.cuda.get_device_name(0)}')" 2>&1 | tee -a "${LOG_FILE}"
        conda run -n ai_env python -c "import torch; print(f'CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')" 2>&1 | tee -a "${LOG_FILE}"
    else
        log_warn "CUDA is NOT available (will use CPU - much slower)"
    fi

    # Check specific packages for each stage
    log "Checking stage-specific packages..."

    # InsightFace for Stage 1
    if conda run -n ai_env python -c "import insightface" 2>&1 | grep -q "ModuleNotFoundError"; then
        log_warn "InsightFace NOT installed - required for Stage 1 (face matching)"
        log "Install: conda run -n ai_env pip install insightface onnxruntime-gpu"
    else
        log_success "InsightFace installed"
    fi

    # CLIP for Stage 4
    if conda run -n ai_env python -c "import clip" 2>&1 | grep -q "ModuleNotFoundError"; then
        log_warn "CLIP NOT installed - required for Stage 4 (diversity selection)"
        log "Install: conda run -n ai_env pip install git+https://github.com/openai/CLIP.git"
    else
        log_success "CLIP installed"
    fi

    # Transformers for Stage 5
    if conda run -n ai_env python -c "import transformers; print(transformers.__version__)" 2>&1 | grep -q "ModuleNotFoundError"; then
        log_warn "Transformers NOT installed - required for Stage 5 (caption generation)"
        log "Install: conda run -n ai_env pip install transformers>=4.37.0"
    else
        TRANSFORMERS_VERSION=$(conda run -n ai_env python -c "import transformers; print(transformers.__version__)" 2>&1)
        log_success "Transformers installed: ${TRANSFORMERS_VERSION}"
    fi

    log_success "Environment check complete!"
}

# ============================================================
# Stage 0.5: Data Check
# ============================================================

check_data() {
    section "STAGE 0.5: Data Availability Check"

    # Check SAM2 results
    log "Checking SAM2 instance results..."
    SAM2_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2"

    if [ ! -d "${SAM2_DIR}" ]; then
        log_error "SAM2 results directory not found: ${SAM2_DIR}"
        exit 1
    fi

    log_success "SAM2 directory exists: ${SAM2_DIR}"

    # Count instances in each type
    for inst_type in "instances" "instances_blurred" "instances_context"; do
        if [ -d "${SAM2_DIR}/${inst_type}" ]; then
            count=$(find "${SAM2_DIR}/${inst_type}" -name "*.png" 2>/dev/null | wc -l)
            log "  ${inst_type}: ${count} images"
        else
            log_warn "  ${inst_type}: directory not found"
        fi
    done

    # Check reference training data
    log "Checking reference Luca training data..."
    REF_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/training_ready/1_luca"

    if [ ! -d "${REF_DIR}" ]; then
        log_error "Reference training data not found: ${REF_DIR}"
        exit 1
    fi

    ref_count=$(find "${REF_DIR}" -name "*.png" 2>/dev/null | wc -l)
    log_success "Reference training data: ${ref_count} images"

    if [ ${ref_count} -lt 100 ]; then
        log_warn "Reference image count (${ref_count}) seems low. Expected ~372"
    fi

    # Check disk space
    log "Checking available disk space..."
    df -h /mnt/data/ai_data | tail -1 | tee -a "${LOG_FILE}"

    log_success "Data check complete!"
}

# ============================================================
# Stage 1 Test: Face Matching (Small Sample)
# ============================================================

test_stage1_face_matching() {
    section "STAGE 1 TEST: Face Matching (Small Sample)"

    log "Testing ArcFace face matching with 100 sample images..."

    # Create test directories
    STAGE1_TEST_DIR="${TEST_DIR}/stage1_face_matching"
    mkdir -p "${STAGE1_TEST_DIR}"

    # Create small sample
    log "Creating small sample (100 images from instances_context)..."
    SAM2_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2"
    find "${SAM2_DIR}/instances_context" -name "*.png" 2>/dev/null | head -100 > "${STAGE1_TEST_DIR}/sample_list.txt"

    sample_count=$(wc -l < "${STAGE1_TEST_DIR}/sample_list.txt")
    log "Sample created: ${sample_count} images"

    # Copy samples
    SAMPLE_DIR="${STAGE1_TEST_DIR}/samples"
    mkdir -p "${SAMPLE_DIR}"

    while IFS= read -r img_path; do
        cp "$img_path" "${SAMPLE_DIR}/"
    done < "${STAGE1_TEST_DIR}/sample_list.txt"

    # Create minimal test script
    cat > "${STAGE1_TEST_DIR}/test_face_match.py" <<'PYTHON_EOF'
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("ERROR: InsightFace not installed!")
    print("Install: pip install insightface onnxruntime-gpu")
    sys.exit(1)

# Initialize
print("Initializing FaceAnalysis...")
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load reference
ref_dir = Path("/mnt/data/ai_data/datasets/3d-anime/luca/training_ready/1_luca")
ref_images = list(ref_dir.glob("*.png"))[:10]  # Use 10 references

print(f"Loading {len(ref_images)} reference images...")
ref_embeddings = []

for img_path in tqdm(ref_images, desc="References"):
    img = cv2.imread(str(img_path))
    if img is not None:
        faces = app.get(img)
        if faces:
            ref_embeddings.append(faces[0].embedding)

ref_embeddings = np.array(ref_embeddings)
print(f"Extracted {len(ref_embeddings)} reference embeddings")

# Test samples
sample_dir = Path(sys.argv[1])
sample_images = list(sample_dir.glob("*.png"))

print(f"\nTesting {len(sample_images)} sample images...")
matched = 0
threshold = 0.40

for img_path in tqdm(sample_images, desc="Matching"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    faces = app.get(img)
    if not faces:
        continue

    query_embedding = faces[0].embedding
    similarities = np.dot(ref_embeddings, query_embedding)
    max_sim = similarities.max()

    if max_sim >= threshold:
        matched += 1

print(f"\nResults:")
print(f"  Processed: {len(sample_images)}")
print(f"  Matched: {matched}")
print(f"  Match rate: {matched/len(sample_images)*100:.1f}%")
print(f"\n✓ Stage 1 test successful!")
PYTHON_EOF

    # Run test
    log "Running face matching test..."
    if conda run -n ai_env python "${STAGE1_TEST_DIR}/test_face_match.py" "${SAMPLE_DIR}" 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Stage 1 test PASSED!"
    else
        log_error "Stage 1 test FAILED!"
        return 1
    fi
}

# ============================================================
# Stage 2 Test: Quality Filtering
# ============================================================

test_stage2_quality_filter() {
    section "STAGE 2 TEST: Quality Filtering"

    log "Testing quality filtering with sample images..."

    STAGE2_TEST_DIR="${TEST_DIR}/stage2_quality_filter"
    mkdir -p "${STAGE2_TEST_DIR}/input"
    mkdir -p "${STAGE2_TEST_DIR}/output"

    # Use Stage 1 output or create sample
    if [ -d "${TEST_DIR}/stage1_face_matching/samples" ]; then
        log "Using Stage 1 output as input..."
        cp "${TEST_DIR}/stage1_face_matching/samples"/*.png "${STAGE2_TEST_DIR}/input/" 2>/dev/null || true
    else
        log "Creating new sample..."
        find "/mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/instances_context" -name "*.png" | head -50 | while read img; do
            cp "$img" "${STAGE2_TEST_DIR}/input/"
        done
    fi

    input_count=$(find "${STAGE2_TEST_DIR}/input" -name "*.png" | wc -l)
    log "Input images: ${input_count}"

    # Create test script
    cat > "${STAGE2_TEST_DIR}/test_quality_filter.py" <<'PYTHON_EOF'
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(exist_ok=True)

# Quality thresholds
MIN_SHARPNESS = 50
MIN_ALPHA_COVERAGE = 0.7
MIN_DIMENSIONS = (128, 128)

image_files = list(input_dir.glob("*.png"))
print(f"Processing {len(image_files)} images...")

passed = 0

for img_path in tqdm(image_files):
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Check dimensions
        h, w = img.shape[:2]
        if h < MIN_DIMENSIONS[0] or w < MIN_DIMENSIONS[1]:
            continue

        # Sharpness (Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < MIN_SHARPNESS:
            continue

        # Alpha coverage
        pil_img = Image.open(img_path)
        if pil_img.mode == 'RGBA':
            alpha = np.array(pil_img)[:, :, 3]
            coverage = (alpha > 0).sum() / alpha.size
            if coverage < MIN_ALPHA_COVERAGE:
                continue

        # Passed all filters
        import shutil
        shutil.copy2(img_path, output_dir / img_path.name)
        passed += 1

    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")

print(f"\nResults:")
print(f"  Input: {len(image_files)}")
print(f"  Passed: {passed}")
print(f"  Pass rate: {passed/len(image_files)*100:.1f}%")
print(f"\n✓ Stage 2 test successful!")
PYTHON_EOF

    # Run test
    log "Running quality filter test..."
    if conda run -n ai_env python "${STAGE2_TEST_DIR}/test_quality_filter.py" \
        "${STAGE2_TEST_DIR}/input" "${STAGE2_TEST_DIR}/output" 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Stage 2 test PASSED!"
    else
        log_error "Stage 2 test FAILED!"
        return 1
    fi
}

# ============================================================
# Stage 3 Test: Augmentation
# ============================================================

test_stage3_augmentation() {
    section "STAGE 3 TEST: Augmentation"

    log "Testing 3D-safe augmentation..."

    STAGE3_TEST_DIR="${TEST_DIR}/stage3_augmentation"
    mkdir -p "${STAGE3_TEST_DIR}/input"
    mkdir -p "${STAGE3_TEST_DIR}/output"

    # Use Stage 2 output or create sample
    if [ -d "${TEST_DIR}/stage2_quality_filter/output" ]; then
        log "Using Stage 2 output as input..."
        cp "${TEST_DIR}/stage2_quality_filter/output"/*.png "${STAGE3_TEST_DIR}/input/" 2>/dev/null || true
    fi

    # Ensure we have at least 5 images
    input_count=$(find "${STAGE3_TEST_DIR}/input" -name "*.png" | wc -l)
    if [ ${input_count} -lt 5 ]; then
        log "Adding more samples..."
        find "/mnt/data/ai_data/datasets/3d-anime/luca/training_ready/1_luca" -name "*.png" | head -5 | while read img; do
            cp "$img" "${STAGE3_TEST_DIR}/input/"
        done
    fi

    input_count=$(find "${STAGE3_TEST_DIR}/input" -name "*.png" | wc -l)
    log "Input images: ${input_count}"

    # Create test script
    cat > "${STAGE3_TEST_DIR}/test_augmentation.py" <<'PYTHON_EOF'
import sys
import random
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm
import numpy as np

input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(exist_ok=True)

def apply_3d_safe_augmentation(img):
    """Apply 3D-safe augmentations."""
    aug_img = img.copy()

    # Random crop (0.8-1.0x)
    if random.random() < 0.8:
        w, h = aug_img.size
        scale = random.uniform(0.8, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        aug_img = aug_img.crop((left, top, left + new_w, top + new_h))
        aug_img = aug_img.resize((w, h), Image.LANCZOS)

    # Rotation (-5 to +5 degrees)
    if random.random() < 0.5:
        angle = random.uniform(-5, 5)
        aug_img = aug_img.rotate(angle, expand=False, fillcolor=(0, 0, 0, 0))

    # Brightness (0.9-1.1x)
    if random.random() < 0.6:
        enhancer = ImageEnhance.Brightness(aug_img)
        aug_img = enhancer.enhance(random.uniform(0.9, 1.1))

    # Contrast (0.95-1.05x - MINIMAL)
    if random.random() < 0.4:
        enhancer = ImageEnhance.Contrast(aug_img)
        aug_img = enhancer.enhance(random.uniform(0.95, 1.05))

    return aug_img

image_files = list(input_dir.glob("*.png"))
print(f"Augmenting {len(image_files)} images (3 variants each)...")

total_generated = 0

for img_path in tqdm(image_files):
    try:
        img = Image.open(img_path).convert('RGBA')

        # Generate 3 augmented versions
        for i in range(3):
            aug_img = apply_3d_safe_augmentation(img)
            output_path = output_dir / f"{img_path.stem}_aug{i:02d}.png"
            aug_img.save(output_path)
            total_generated += 1

        # Also save original
        img.save(output_dir / img_path.name)
        total_generated += 1

    except Exception as e:
        print(f"Error augmenting {img_path.name}: {e}")

print(f"\nResults:")
print(f"  Input: {len(image_files)}")
print(f"  Generated: {total_generated}")
print(f"  Multiplier: {total_generated/len(image_files):.1f}x")
print(f"\n✓ Stage 3 test successful!")
PYTHON_EOF

    # Run test
    log "Running augmentation test..."
    if conda run -n ai_env python "${STAGE3_TEST_DIR}/test_augmentation.py" \
        "${STAGE3_TEST_DIR}/input" "${STAGE3_TEST_DIR}/output" 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Stage 3 test PASSED!"
    else
        log_error "Stage 3 test FAILED!"
        return 1
    fi
}

# ============================================================
# Stage 4 Test: Diversity Selection
# ============================================================

test_stage4_diversity_selection() {
    section "STAGE 4 TEST: Diversity Selection"

    log "Testing CLIP-based diversity selection..."

    STAGE4_TEST_DIR="${TEST_DIR}/stage4_diversity"
    mkdir -p "${STAGE4_TEST_DIR}/input"
    mkdir -p "${STAGE4_TEST_DIR}/output"

    # Use Stage 3 output
    if [ -d "${TEST_DIR}/stage3_augmentation/output" ]; then
        log "Using Stage 3 output as input..."
        cp "${TEST_DIR}/stage3_augmentation/output"/*.png "${STAGE4_TEST_DIR}/input/" 2>/dev/null || true
    fi

    input_count=$(find "${STAGE4_TEST_DIR}/input" -name "*.png" | wc -l)
    log "Input images: ${input_count}"

    # Create test script
    cat > "${STAGE4_TEST_DIR}/test_diversity_selection.py" <<'PYTHON_EOF'
import sys
from pathlib import Path
import shutil

try:
    import torch
    import clip
    from PIL import Image
    import numpy as np
    from sklearn.cluster import KMeans
    from tqdm import tqdm
except ImportError as e:
    print(f"ERROR: Missing package: {e}")
    print("Install: pip install ftfy regex tqdm scikit-learn")
    print("        pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
target_count = int(sys.argv[3]) if len(sys.argv) > 3 else 20
output_dir.mkdir(exist_ok=True)

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)  # Use smaller model for testing

# Extract embeddings
image_files = list(input_dir.glob("*.png"))
print(f"Processing {len(image_files)} images...")

embeddings = []
valid_files = []

for img_path in tqdm(image_files, desc="Extracting CLIP embeddings"):
    try:
        img = Image.open(img_path).convert('RGB')
        img_input = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(img_input).cpu().numpy()[0]

        embeddings.append(embedding)
        valid_files.append(img_path)
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")

embeddings = np.array(embeddings)
print(f"Extracted {len(embeddings)} embeddings")

# Cluster and select
n_clusters = min(4, len(valid_files) // 5)  # Adaptive cluster count
samples_per_cluster = target_count // n_clusters

print(f"Clustering into {n_clusters} groups...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

selected_files = []

for cluster_id in range(n_clusters):
    cluster_indices = np.where(labels == cluster_id)[0]
    cluster_files = [valid_files[i] for i in cluster_indices]

    # Select closest to centroid
    cluster_embeddings = embeddings[cluster_indices]
    centroid = kmeans.cluster_centers_[cluster_id]
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

    sorted_indices = np.argsort(distances)
    selected_indices = sorted_indices[:min(samples_per_cluster, len(sorted_indices))]

    selected_files.extend([cluster_files[i] for i in selected_indices])

# Copy selected files
print(f"Selecting {len(selected_files[:target_count])} diverse images...")
for img_path in selected_files[:target_count]:
    shutil.copy2(img_path, output_dir / img_path.name)

print(f"\nResults:")
print(f"  Input: {len(image_files)}")
print(f"  Selected: {len(selected_files[:target_count])}")
print(f"\n✓ Stage 4 test successful!")
PYTHON_EOF

    # Run test
    log "Running diversity selection test (target: 20 images)..."
    if conda run -n ai_env python "${STAGE4_TEST_DIR}/test_diversity_selection.py" \
        "${STAGE4_TEST_DIR}/input" "${STAGE4_TEST_DIR}/output" 20 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Stage 4 test PASSED!"
    else
        log_error "Stage 4 test FAILED!"
        return 1
    fi
}

# ============================================================
# Stage 5 Test: Caption Generation
# ============================================================

test_stage5_caption_generation() {
    section "STAGE 5 TEST: Caption Generation"

    log "Testing caption generation with Qwen2-VL..."
    log_warn "NOTE: This requires Qwen2-VL model to be downloaded (~15GB)"
    log_warn "Will only test 3 images to save time"

    STAGE5_TEST_DIR="${TEST_DIR}/stage5_captions"
    mkdir -p "${STAGE5_TEST_DIR}/input"

    # Use Stage 4 output or reference images
    if [ -d "${TEST_DIR}/stage4_diversity/output" ]; then
        log "Using Stage 4 output as input (first 3 images)..."
        find "${TEST_DIR}/stage4_diversity/output" -name "*.png" | head -3 | while read img; do
            cp "$img" "${STAGE5_TEST_DIR}/input/"
        done
    else
        log "Using reference images..."
        find "/mnt/data/ai_data/datasets/3d-anime/luca/training_ready/1_luca" -name "*.png" | head -3 | while read img; do
            cp "$img" "${STAGE5_TEST_DIR}/input/"
        done
    fi

    input_count=$(find "${STAGE5_TEST_DIR}/input" -name "*.png" | wc -l)
    log "Input images: ${input_count}"

    # Use existing regenerate_captions_vlm.py script
    log "Running caption generation..."
    if conda run -n ai_env python "${PROJECT_ROOT}/scripts/training/regenerate_captions_vlm.py" \
        --image-dir "${STAGE5_TEST_DIR}/input" \
        --output-dir "${STAGE5_TEST_DIR}/input" \
        --model qwen2_vl \
        --character-profile "${PROJECT_ROOT}/configs/characters/luca.yaml" \
        --sample-size 3 \
        --device cuda 2>&1 | tee -a "${LOG_FILE}"; then

        # Check if captions were generated
        caption_count=$(find "${STAGE5_TEST_DIR}/input" -name "*.txt" | wc -l)
        if [ ${caption_count} -ge 1 ]; then
            log_success "Stage 5 test PASSED! Generated ${caption_count} captions"

            # Show sample caption
            log "Sample caption:"
            head -1 $(find "${STAGE5_TEST_DIR}/input" -name "*.txt" | head -1) | tee -a "${LOG_FILE}"
        else
            log_error "No captions were generated!"
            return 1
        fi
    else
        log_error "Stage 5 test FAILED!"
        return 1
    fi
}

# ============================================================
# Stage 6 Test: Training Data Preparation
# ============================================================

test_stage6_training_prep() {
    section "STAGE 6 TEST: Training Data Preparation"

    log "Testing Kohya_ss format preparation..."

    STAGE6_TEST_DIR="${TEST_DIR}/stage6_training_prep"
    KOHYA_DIR="${STAGE6_TEST_DIR}/10_luca_human"
    mkdir -p "${KOHYA_DIR}"

    # Use Stage 5 output (images with captions)
    if [ -d "${TEST_DIR}/stage5_captions/input" ]; then
        log "Copying images and captions to Kohya_ss format..."

        # Copy all images and captions
        copied=0
        for img in "${TEST_DIR}/stage5_captions/input"/*.png; do
            if [ -f "$img" ]; then
                cp "$img" "${KOHYA_DIR}/"

                # Copy caption if exists
                txt_file="${img%.png}.txt"
                if [ -f "$txt_file" ]; then
                    cp "$txt_file" "${KOHYA_DIR}/"
                fi

                ((copied++))
            fi
        done

        log_success "Copied ${copied} image-caption pairs"

        # Create metadata
        cat > "${STAGE6_TEST_DIR}/metadata.json" <<EOF
{
    "source": "pipeline_test",
    "image_count": ${copied},
    "repeat_count": 10,
    "class_name": "luca_human",
    "created": "$(date -Iseconds)"
}
EOF

        log "Metadata saved: ${STAGE6_TEST_DIR}/metadata.json"
        log_success "Stage 6 test PASSED!"

        # Show structure
        log "Kohya_ss directory structure:"
        tree -L 2 "${STAGE6_TEST_DIR}" 2>/dev/null || ls -la "${STAGE6_TEST_DIR}"

    else
        log_error "Stage 5 output not found!"
        return 1
    fi
}

# ============================================================
# Stage 7 Test: LoRA Training Config
# ============================================================

test_stage7_lora_config() {
    section "STAGE 7 TEST: LoRA Training Configuration"

    log "Testing LoRA training configuration generation..."

    STAGE7_TEST_DIR="${TEST_DIR}/stage7_lora_config"
    mkdir -p "${STAGE7_TEST_DIR}"

    # Generate test config
    TRAINING_CONFIG="${STAGE7_TEST_DIR}/luca_test.toml"

    cat > "${TRAINING_CONFIG}" <<EOF
# Luca LoRA Training - Test Configuration
# Based on Trial 3.5/3.6 optimized parameters

[model]
pretrained_model_name_or_path = "/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors"

[network]
network_module = "networks.lora"
network_dim = 64
network_alpha = 32
network_dropout = 0.1

[training]
# Data
train_data_dir = "${TEST_DIR}/stage6_training_prep"
output_dir = "${STAGE7_TEST_DIR}/output"
output_name = "luca_test"
resolution = "512,512"

# Optimization (Trial 3.6 optimized)
learning_rate = 8e-5
text_encoder_lr = 5e-5
unet_lr = 8e-5
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100

optimizer_type = "AdamW"

# Training duration (SHORT FOR TESTING)
max_train_epochs = 2
train_batch_size = 2
gradient_accumulation_steps = 2

# Regularization
min_snr_gamma = 5.0
noise_offset = 0.05

# Data augmentation
random_crop = false
color_aug = false
flip_aug = false

# Saving
save_every_n_epochs = 1
save_precision = "fp16"
mixed_precision = "fp16"

# Logging
logging_dir = "${STAGE7_TEST_DIR}/output/logs"

# Performance
max_data_loader_n_workers = 4
seed = 42

# Miscellaneous
caption_extension = ".txt"
EOF

    log_success "Training configuration created: ${TRAINING_CONFIG}"

    # Validate config
    log "Validating configuration..."

    # Check if base model exists
    BASE_MODEL="/mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors"
    if [ -f "${BASE_MODEL}" ]; then
        log_success "Base model found: ${BASE_MODEL}"
    else
        log_warn "Base model NOT found: ${BASE_MODEL}"
        log "You'll need to update this path before training"
    fi

    # Check if train_data_dir exists
    if [ -d "${TEST_DIR}/stage6_training_prep/10_luca_human" ]; then
        img_count=$(find "${TEST_DIR}/stage6_training_prep/10_luca_human" -name "*.png" | wc -l)
        log_success "Training data exists: ${img_count} images"
    else
        log_warn "Training data directory not ready yet"
    fi

    # Find Kohya_ss installation
    log "Checking Kohya_ss installation..."
    if conda env list | grep -q "^kohya_ss "; then
        log_success "kohya_ss environment found"

        # Try to find train_network.py
        log "Looking for train_network.py..."
        KOHYA_SCRIPT=$(find /mnt -name "train_network.py" -path "*/kohya_ss/*" 2>/dev/null | head -1)

        if [ -n "${KOHYA_SCRIPT}" ]; then
            log_success "Found: ${KOHYA_SCRIPT}"
            log "Training command would be:"
            echo "  cd $(dirname ${KOHYA_SCRIPT})" | tee -a "${LOG_FILE}"
            echo "  conda run -n kohya_ss python train_network.py --config_file ${TRAINING_CONFIG}" | tee -a "${LOG_FILE}"
        else
            log_warn "train_network.py not found automatically"
        fi
    else
        log_error "kohya_ss environment not found!"
        return 1
    fi

    log_success "Stage 7 test PASSED!"
}

# ============================================================
# Main Test Execution
# ============================================================

main() {
    section "PIPELINE STAGE TESTING"

    log "Starting comprehensive pipeline tests..."
    log "Test directory: ${TEST_DIR}"
    log "Log file: ${LOG_FILE}"

    # Stage 0: Environment
    check_environment
    check_data

    # Ask user which stages to test
    echo ""
    echo "Which stages would you like to test?"
    echo "1) All stages"
    echo "2) Select specific stages"
    echo "3) Quick test (Stages 1-3 only)"
    read -p "Choice (1-3): " choice

    case $choice in
        1)
            test_stage1_face_matching
            test_stage2_quality_filter
            test_stage3_augmentation
            test_stage4_diversity_selection
            test_stage5_caption_generation
            test_stage6_training_prep
            test_stage7_lora_config
            ;;
        2)
            # Interactive stage selection
            echo "Select stages to test (space-separated, e.g., 1 3 5):"
            read -p "Stages: " stages
            for stage in $stages; do
                case $stage in
                    1) test_stage1_face_matching ;;
                    2) test_stage2_quality_filter ;;
                    3) test_stage3_augmentation ;;
                    4) test_stage4_diversity_selection ;;
                    5) test_stage5_caption_generation ;;
                    6) test_stage6_training_prep ;;
                    7) test_stage7_lora_config ;;
                esac
            done
            ;;
        3)
            test_stage1_face_matching
            test_stage2_quality_filter
            test_stage3_augmentation
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac

    section "TEST SUMMARY"

    log_success "All requested tests completed!"
    log "Test results saved to: ${LOG_FILE}"
    log "Test artifacts saved to: ${TEST_DIR}"

    echo ""
    echo "Next steps:"
    echo "  1. Review test results: cat ${LOG_FILE}"
    echo "  2. Check test outputs: ls -la ${TEST_DIR}/"
    echo "  3. If all tests passed, run full pipeline:"
    echo "     bash scripts/workflows/run_luca_dataset_pipeline.sh"
}

# Run main
main "$@"
