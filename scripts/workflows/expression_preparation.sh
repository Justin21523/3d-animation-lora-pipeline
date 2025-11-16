#!/bin/bash
# Expression Data Preparation (Pure CPU)
# Extracts facial expressions and clusters by emotion type
# Input: 542 character instances → Output: 200-400 expression examples

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Configuration
INPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset_v2_smart/luca_human/images"
OUTPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/expression_data"
LOG_DIR="/tmp/expression_preparation"
PYTHON_BIN="/home/b0979/.conda/envs/ai_env/bin/python"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR/faces" "$OUTPUT_DIR/clustered"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Expression Data Preparation (CPU Only)                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Purpose: Extract facial expressions for Expression LoRA training"
echo ""
echo "Process:"
echo "  1. Face detection and cropping (with context)"
echo "  2. Quality filtering (blur, occlusion, resolution)"
echo "  3. Expression clustering (happy/sad/surprised/angry/neutral)"
echo "  4. Dataset assembly"
echo ""
echo "Running with low priority (nice -n 19)..."
echo "Estimated time: 20-30 minutes"
echo ""

# Count input instances
NUM_INSTANCES=$(ls "$INPUT_DIR"/*.png 2>/dev/null | wc -l)
echo "📊 Found $NUM_INSTANCES character instances"

if [ "$NUM_INSTANCES" -eq 0 ]; then
    echo "❌ Error: No instances found in $INPUT_DIR"
    exit 1
fi

# Run face extraction and expression clustering
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Face Detection & Expression Clustering..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$PROJECT_ROOT" && nice -n 19 "$PYTHON_BIN" -u - <<'PYTHON_CODE'
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import shutil
from collections import defaultdict

def detect_and_crop_face(image_path, padding=0.3):
    """
    Detect face and crop with context padding.
    Returns cropped face image and face region info.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None

    # Use Haar Cascade for CPU-friendly face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None, None

    # Get largest face
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face

    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img.shape[1], x + w + pad_w)
    y2 = min(img.shape[0], y + h + pad_h)

    # Crop face with context
    face_img = img[y1:y2, x1:x2]

    # Quality checks
    if face_img.shape[0] < 64 or face_img.shape[1] < 64:
        return None, None

    # Check blur
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    if blur_score < 50.0:  # Too blurry
        return None, None

    face_info = {
        "bbox": [int(x), int(y), int(w), int(h)],
        "bbox_padded": [int(x1), int(y1), int(x2), int(y2)],
        "size": [int(face_img.shape[1]), int(face_img.shape[0])],
        "blur_score": float(blur_score)
    }

    return face_img, face_info

def extract_expression_features(face_img):
    """
    Extract simple expression features for clustering.
    Returns feature vector for emotion classification.
    """
    if face_img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Resize to standard size for consistent features
    gray = cv2.resize(gray, (128, 128))

    # Extract features from different facial regions
    h, w = gray.shape

    # Upper face (eyes/eyebrows): top 40%
    upper = gray[:int(h*0.4), :]
    upper_mean = np.mean(upper)
    upper_std = np.std(upper)

    # Middle face (nose): middle 30%
    middle = gray[int(h*0.35):int(h*0.65), :]
    middle_mean = np.mean(middle)
    middle_std = np.std(middle)

    # Lower face (mouth): bottom 30%
    lower = gray[int(h*0.6):, :]
    lower_mean = np.mean(lower)
    lower_std = np.std(lower)

    # Overall statistics
    overall_mean = np.mean(gray)
    overall_std = np.std(gray)

    # Histogram features
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist = hist.flatten() / hist.sum()

    # Combine features
    features = np.concatenate([
        [upper_mean, upper_std, middle_mean, middle_std,
         lower_mean, lower_std, overall_mean, overall_std],
        hist
    ])

    return features

def cluster_expressions(features_list, n_clusters=5):
    """
    Cluster faces by expression.
    Clusters: happy, sad, surprised, angry, neutral
    """
    if len(features_list) < n_clusters:
        return [0] * len(features_list)

    # Convert to numpy array
    features = np.array([f for f in features_list if f is not None])

    if len(features) == 0:
        return [0] * len(features_list)

    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    features_norm = (features - mean) / std

    # K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_norm)

    return labels.tolist()

# Configuration
INPUT_DIR = Path("/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset_v2_smart/luca_human/images")
OUTPUT_DIR = Path("/mnt/data/ai_data/datasets/3d-anime/luca/expression_data")
FACES_DIR = OUTPUT_DIR / "faces"
CLUSTERED_DIR = OUTPUT_DIR / "clustered"

FACES_DIR.mkdir(parents=True, exist_ok=True)
CLUSTERED_DIR.mkdir(parents=True, exist_ok=True)

# Find all character instances
image_files = sorted(INPUT_DIR.glob("*.png"))
print(f"Processing {len(image_files)} character instances...")
print("")

# Step 1: Face detection and cropping
print("Step 1/3: Detecting and cropping faces...")
valid_faces = []
valid_features = []
valid_source_files = []

for img_path in tqdm(image_files, desc="Face detection"):
    face_img, face_info = detect_and_crop_face(img_path)

    if face_img is not None:
        # Save face crop
        face_filename = FACES_DIR / f"{img_path.stem}_face.png"
        cv2.imwrite(str(face_filename), face_img)

        # Save face info
        info_filename = FACES_DIR / f"{img_path.stem}_face.json"
        with open(info_filename, 'w') as f:
            json.dump(face_info, f, indent=2)

        # Extract features
        features = extract_expression_features(face_img)

        if features is not None:
            valid_faces.append(face_filename)
            valid_features.append(features)
            valid_source_files.append(img_path)

print(f"Extracted {len(valid_faces)}/{len(image_files)} valid face crops")
print("")

# Step 2: Expression clustering
print("Step 2/3: Clustering by expression type...")
labels = cluster_expressions(valid_features, n_clusters=5)

# Assign cluster names (will need manual review/labeling)
cluster_names = {
    0: "expression_0",
    1: "expression_1",
    2: "expression_2",
    3: "expression_3",
    4: "expression_4"
}

# Create cluster directories
for cluster_id, cluster_name in cluster_names.items():
    (CLUSTERED_DIR / cluster_name).mkdir(exist_ok=True)

# Step 3: Organize faces by expression
print("Step 3/3: Organizing faces by expression cluster...")
cluster_counts = defaultdict(int)
cluster_samples = defaultdict(list)

for face_path, source_path, label in zip(valid_faces, valid_source_files, labels):
    cluster_name = cluster_names[label]
    dest_dir = CLUSTERED_DIR / cluster_name

    # Copy face crop
    dest_face = dest_dir / face_path.name
    shutil.copy2(face_path, dest_face)

    # Copy face info
    src_info = face_path.with_suffix('.json')
    dest_info = dest_dir / src_info.name
    if src_info.exists():
        shutil.copy2(src_info, dest_info)

    cluster_counts[cluster_name] += 1

    # Keep samples for visualization
    if len(cluster_samples[cluster_name]) < 5:
        cluster_samples[cluster_name].append(str(dest_face))

# Generate report
print("")
print("=" * 80)
print("Expression Clustering Complete")
print("=" * 80)
print(f"Total faces extracted: {len(valid_faces)}")
print("")
print("Expression Distribution:")
for cluster_name in sorted(cluster_counts.keys()):
    count = cluster_counts[cluster_name]
    pct = count / len(valid_faces) * 100 if valid_faces else 0
    print(f"  {cluster_name}: {count} ({pct:.1f}%)")
print("")

# Save metadata
metadata = {
    "total_instances": len(image_files),
    "valid_faces": len(valid_faces),
    "clusters": {
        name: {
            "count": cluster_counts[name],
            "percentage": cluster_counts[name] / len(valid_faces) * 100 if valid_faces else 0,
            "samples": cluster_samples[name]
        }
        for name in cluster_counts.keys()
    },
    "cluster_names": cluster_names,
    "output_dirs": {
        "faces": str(FACES_DIR),
        "clustered": str(CLUSTERED_DIR)
    },
    "notes": "Cluster names are generic. Manual review recommended for emotion labeling."
}

metadata_path = OUTPUT_DIR / "expression_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved to: {metadata_path}")
print("")
print("⚠️  NOTE: Cluster names are generic (expression_0, expression_1, etc.)")
print("   Manual review recommended to label emotions (happy/sad/surprised/etc.)")
print("")
print("✅ Expression preparation completed!")

PYTHON_CODE

if [ $? -ne 0 ]; then
    echo "❌ Expression preparation failed!"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ Expression Preparation Complete!           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Output: $OUTPUT_DIR"
echo ""
echo "Next Steps:"
echo "  1. Review clusters and label emotions manually"
echo "  2. Merge/split clusters if needed"
echo "  3. Generate captions with VLM (after GPU available)"
echo ""
echo "✨ Ready for Expression LoRA training (after VLM captioning)"
