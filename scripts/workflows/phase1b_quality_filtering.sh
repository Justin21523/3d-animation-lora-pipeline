#!/bin/bash
# Phase 1b: Quality Filtering (Pure CPU)
# Filters low-quality frames based on blur, brightness, contrast
# Input: 3074 deduplicated frames → Output: ~2500 quality frames

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Configuration
INPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/frames_deduplicated"
OUTPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/frames_quality_filtered"
LOG_DIR="/tmp/quality_filtering_phase1b"
PYTHON_BIN="/home/b0979/.conda/envs/ai_env/bin/python"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Phase 1b: Quality Filtering (CPU Only)                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Purpose: Filter low-quality frames before GPU processing"
echo ""
echo "Quality Checks:"
echo "  1. Blur detection (Laplacian variance)"
echo "  2. Brightness check (histogram analysis)"
echo "  3. Contrast check (std deviation)"
echo "  4. Resolution verification"
echo ""
echo "Running with low priority (nice -n 19)..."
echo "Estimated time: 15-20 minutes"
echo ""

# Count input frames
NUM_FRAMES=$(ls "$INPUT_DIR"/*.jpg 2>/dev/null | wc -l)
echo "📊 Found $NUM_FRAMES deduplicated frames"

if [ "$NUM_FRAMES" -eq 0 ]; then
    echo "❌ Error: No frames found in $INPUT_DIR"
    exit 1
fi

# Run quality filtering
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Quality Filtering Processing..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$PROJECT_ROOT" && nice -n 19 "$PYTHON_BIN" -u - <<'PYTHON_CODE'
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import json
import sys

def calculate_blur(image):
    """Calculate blur using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_brightness(image):
    """Calculate average brightness."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2].mean()

def calculate_contrast(image):
    """Calculate contrast using standard deviation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()

def quality_check(img_path, thresholds):
    """
    Check image quality.

    Returns: (pass: bool, metrics: dict, reason: str)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False, {}, "Failed to load image"

    h, w = img.shape[:2]

    # Calculate metrics
    blur_score = calculate_blur(img)
    brightness = calculate_brightness(img)
    contrast = calculate_contrast(img)

    metrics = {
        "blur_score": float(blur_score),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "resolution": [w, h]
    }

    # Quality checks
    if blur_score < thresholds["min_blur"]:
        return False, metrics, f"Too blurry (score: {blur_score:.1f} < {thresholds['min_blur']})"

    if brightness < thresholds["min_brightness"] or brightness > thresholds["max_brightness"]:
        return False, metrics, f"Bad brightness ({brightness:.1f}, range: {thresholds['min_brightness']}-{thresholds['max_brightness']})"

    if contrast < thresholds["min_contrast"]:
        return False, metrics, f"Low contrast ({contrast:.1f} < {thresholds['min_contrast']})"

    if w < thresholds["min_width"] or h < thresholds["min_height"]:
        return False, metrics, f"Resolution too low ({w}x{h})"

    return True, metrics, "OK"

# Configuration
INPUT_DIR = Path("/mnt/data/ai_data/datasets/3d-anime/luca/frames_deduplicated")
OUTPUT_DIR = Path("/mnt/data/ai_data/datasets/3d-anime/luca/frames_quality_filtered")
LOG_DIR = Path("/tmp/quality_filtering_phase1b")

# Quality thresholds (tuned for 3D animation)
thresholds = {
    "min_blur": 80.0,        # Laplacian variance (lower = blurrier)
    "min_brightness": 25.0,  # Too dark
    "max_brightness": 240.0, # Too bright/overexposed
    "min_contrast": 25.0,    # Flat/washed out
    "min_width": 640,        # Minimum resolution
    "min_height": 480
}

print(f"Quality Thresholds:")
print(f"  Min blur score: {thresholds['min_blur']}")
print(f"  Brightness range: {thresholds['min_brightness']}-{thresholds['max_brightness']}")
print(f"  Min contrast: {thresholds['min_contrast']}")
print(f"  Min resolution: {thresholds['min_width']}x{thresholds['min_height']}")
print("")

# Find all images
image_files = list(INPUT_DIR.glob("*.jpg"))
print(f"Processing {len(image_files)} images...")
print("")

# Process images
passed = []
rejected = []
rejection_reasons = {}

for img_path in tqdm(image_files, desc="Quality filtering"):
    is_good, metrics, reason = quality_check(img_path, thresholds)

    if is_good:
        # Copy to output
        dest = OUTPUT_DIR / img_path.name
        shutil.copy2(img_path, dest)
        passed.append({
            "file": img_path.name,
            "metrics": metrics
        })
    else:
        rejected.append({
            "file": img_path.name,
            "metrics": metrics,
            "reason": reason
        })
        # Track rejection reasons
        rejection_reasons[reason.split("(")[0].strip()] = rejection_reasons.get(reason.split("(")[0].strip(), 0) + 1

# Generate report
report = {
    "input_count": len(image_files),
    "passed_count": len(passed),
    "rejected_count": len(rejected),
    "pass_rate": len(passed) / len(image_files) * 100 if image_files else 0,
    "thresholds": thresholds,
    "rejection_reasons": rejection_reasons,
    "passed_files": passed[:10],  # Sample
    "rejected_files": rejected[:10]  # Sample
}

# Save detailed metadata
metadata_path = OUTPUT_DIR / "quality_filtering_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(report, f, indent=2)

print("")
print("=" * 80)
print("Quality Filtering Complete")
print("=" * 80)
print(f"Total images: {report['input_count']}")
print(f"Passed: {report['passed_count']} ({report['pass_rate']:.1f}%)")
print(f"Rejected: {report['rejected_count']} ({100-report['pass_rate']:.1f}%)")
print("")
print("Rejection Breakdown:")
for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
    pct = count / len(rejected) * 100 if rejected else 0
    print(f"  {reason}: {count} ({pct:.1f}%)")
print("")
print(f"Metadata saved to: {metadata_path}")
print("✅ Quality filtering completed!")

PYTHON_CODE

if [ $? -ne 0 ]; then
    echo "❌ Quality filtering failed!"
    exit 1
fi

# Generate summary
FINAL_COUNT=$(ls "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l)
REJECTED_COUNT=$((NUM_FRAMES - FINAL_COUNT))
PASS_RATE=$(echo "scale=1; $FINAL_COUNT * 100 / $NUM_FRAMES" | bc)

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                ✅ Phase 1b Completed!                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary:"
echo "  Input frames: $NUM_FRAMES"
echo "  Quality frames: $FINAL_COUNT"
echo "  Rejected: $REJECTED_COUNT"
echo "  Pass rate: $PASS_RATE%"
echo ""
echo "Output: $OUTPUT_DIR"
echo ""
echo "✨ Ready for Phase 2 (GPU segmentation)"
