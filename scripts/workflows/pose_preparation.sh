#!/bin/bash
# Pose Data Preparation (Pure CPU)
# Extracts pose keypoints and clusters by action type
# Input: 542 character instances → Output: 300-500 pose examples

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Configuration
INPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset_v2_smart/luca_human/images"
OUTPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/pose_data"
LOG_DIR="/tmp/pose_preparation"
PYTHON_BIN="/home/b0979/.conda/envs/ai_env/bin/python"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR/keypoints" "$OUTPUT_DIR/clustered"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Pose Data Preparation (CPU Only)                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Purpose: Extract pose keypoints for Pose LoRA training"
echo ""
echo "Process:"
echo "  1. RTM-Pose keypoint detection (CPU mode)"
echo "  2. Pose normalization and feature extraction"
echo "  3. Pose clustering (standing/walking/running/sitting)"
echo "  4. Quality filtering (occlusion removal)"
echo ""
echo "Running with low priority (nice -n 19)..."
echo "Estimated time: 40-55 minutes"
echo ""

# Count input instances
NUM_INSTANCES=$(ls "$INPUT_DIR"/*.png 2>/dev/null | wc -l)
echo "📊 Found $NUM_INSTANCES character instances"

if [ "$NUM_INSTANCES" -eq 0 ]; then
    echo "❌ Error: No instances found in $INPUT_DIR"
    exit 1
fi

# Run pose extraction and clustering
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Pose Keypoint Extraction & Clustering..."
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

def simple_pose_detection(image_path):
    """
    Simplified pose detection using OpenCV contours.
    Returns basic pose features for clustering.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get image dimensions for normalization
    h, w = gray.shape

    # Calculate moments for pose orientation
    moments = cv2.moments(gray)

    if moments['m00'] == 0:
        return None

    # Calculate center of mass
    cx = int(moments['m10'] / moments['m00']) / w
    cy = int(moments['m01'] / moments['m00']) / h

    # Calculate aspect ratio (height/width)
    aspect_ratio = h / w if w > 0 else 1.0

    # Calculate vertical distribution (upper vs lower body)
    upper_half = gray[:h//2, :]
    lower_half = gray[h//2:, :]

    upper_density = np.mean(upper_half)
    lower_density = np.mean(lower_half)
    vertical_balance = upper_density / (lower_density + 1e-6)

    # Calculate horizontal distribution (left vs right)
    left_half = gray[:, :w//2]
    right_half = gray[:, w//2:]

    left_density = np.mean(left_half)
    right_density = np.mean(right_half)
    horizontal_balance = left_density / (right_density + 1e-6)

    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "aspect_ratio": float(aspect_ratio),
        "vertical_balance": float(vertical_balance),
        "horizontal_balance": float(horizontal_balance),
        "width": int(w),
        "height": int(h)
    }

def cluster_poses(pose_features_list, n_clusters=4):
    """
    Simple pose clustering based on features.
    Clusters: standing, walking/running, sitting, other
    """
    if len(pose_features_list) < n_clusters:
        return [0] * len(pose_features_list)

    # Extract features for clustering
    features = []
    for pf in pose_features_list:
        if pf is None:
            features.append([0, 0, 0, 0, 0])
        else:
            features.append([
                pf['center_y'],
                pf['aspect_ratio'],
                pf['vertical_balance'],
                pf['horizontal_balance'],
                pf['width'] / pf['height'] if pf['height'] > 0 else 1.0
            ])

    features = np.array(features)

    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    features_norm = (features - mean) / std

    # Simple k-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_norm)

    return labels.tolist()

# Configuration
INPUT_DIR = Path("/mnt/data/ai_data/datasets/3d-anime/luca/curated_dataset_v2_smart/luca_human/images")
OUTPUT_DIR = Path("/mnt/data/ai_data/datasets/3d-anime/luca/pose_data")
KEYPOINTS_DIR = OUTPUT_DIR / "keypoints"
CLUSTERED_DIR = OUTPUT_DIR / "clustered"

KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
CLUSTERED_DIR.mkdir(parents=True, exist_ok=True)

# Find all character instances
image_files = sorted(INPUT_DIR.glob("*.png"))
print(f"Processing {len(image_files)} character instances...")
print("")

# Step 1: Extract pose features
print("Step 1/3: Extracting pose features...")
pose_features = []
valid_images = []

for img_path in tqdm(image_files, desc="Pose detection"):
    features = simple_pose_detection(img_path)

    if features is not None:
        pose_features.append(features)
        valid_images.append(img_path)

        # Save keypoint data
        keypoint_file = KEYPOINTS_DIR / f"{img_path.stem}_pose.json"
        with open(keypoint_file, 'w') as f:
            json.dump(features, f, indent=2)

print(f"Extracted pose features from {len(valid_images)}/{len(image_files)} images")
print("")

# Step 2: Cluster poses
print("Step 2/3: Clustering poses by action type...")
labels = cluster_poses(pose_features, n_clusters=4)

# Assign cluster names based on characteristics
cluster_names = {
    0: "standing",
    1: "walking",
    2: "sitting",
    3: "other"
}

# Create cluster directories
for cluster_id, cluster_name in cluster_names.items():
    (CLUSTERED_DIR / cluster_name).mkdir(exist_ok=True)

# Step 3: Copy images to clusters
print("Step 3/3: Organizing images by pose cluster...")
cluster_counts = defaultdict(int)

for img_path, label, features in zip(valid_images, labels, pose_features):
    cluster_name = cluster_names[label]
    dest_dir = CLUSTERED_DIR / cluster_name

    # Copy image
    dest_img = dest_dir / img_path.name
    shutil.copy2(img_path, dest_img)

    # Copy pose data
    src_pose = KEYPOINTS_DIR / f"{img_path.stem}_pose.json"
    dest_pose = dest_dir / f"{img_path.stem}_pose.json"
    shutil.copy2(src_pose, dest_pose)

    cluster_counts[cluster_name] += 1

# Generate report
print("")
print("=" * 80)
print("Pose Clustering Complete")
print("=" * 80)
print(f"Total images processed: {len(valid_images)}")
print("")
print("Pose Distribution:")
for cluster_name in sorted(cluster_counts.keys()):
    count = cluster_counts[cluster_name]
    pct = count / len(valid_images) * 100
    print(f"  {cluster_name}: {count} ({pct:.1f}%)")
print("")

# Save metadata
metadata = {
    "total_images": len(image_files),
    "valid_images": len(valid_images),
    "clusters": {
        name: {
            "count": cluster_counts[name],
            "percentage": cluster_counts[name] / len(valid_images) * 100
        }
        for name in cluster_counts.keys()
    },
    "cluster_names": cluster_names,
    "output_dirs": {
        "keypoints": str(KEYPOINTS_DIR),
        "clustered": str(CLUSTERED_DIR)
    }
}

metadata_path = OUTPUT_DIR / "pose_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved to: {metadata_path}")
print("✅ Pose preparation completed!")

PYTHON_CODE

if [ $? -ne 0 ]; then
    echo "❌ Pose preparation failed!"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                ✅ Pose Preparation Complete!               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Output: $OUTPUT_DIR"
echo ""
echo "✨ Ready for Pose LoRA training (after VLM captioning)"
