#!/bin/bash
# Lighting Classification (Pure CPU)
# Classifies frames by lighting conditions for Lighting LoRA training
# Input: 3074 deduplicated frames (or quality-filtered) → Output: 400-600 lighting examples

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Configuration - use quality-filtered frames if available, otherwise deduplicated
QUALITY_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/frames_quality_filtered"
DEDUP_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/frames_deduplicated"

if [ -d "$QUALITY_DIR" ] && [ "$(ls -A $QUALITY_DIR/*.jpg 2>/dev/null | wc -l)" -gt 0 ]; then
    INPUT_DIR="$QUALITY_DIR"
    echo "Using quality-filtered frames"
else
    INPUT_DIR="$DEDUP_DIR"
    echo "Using deduplicated frames (quality filtering not complete yet)"
fi

OUTPUT_DIR="/mnt/data/ai_data/datasets/3d-anime/luca/lighting_data"
LOG_DIR="/tmp/lighting_classification"
PYTHON_BIN="/home/b0979/.conda/envs/ai_env/bin/python"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR/classified"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Lighting Classification (CPU Only)                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Purpose: Classify lighting conditions for Lighting LoRA training"
echo ""
echo "Classification Categories:"
echo "  Time of Day: morning/noon/afternoon/evening/night"
echo "  Weather: sunny/cloudy/overcast"
echo "  Location: indoor/outdoor"
echo ""
echo "Running with low priority (nice -n 19)..."
echo "Estimated time: 30 minutes"
echo ""

# Count input frames
NUM_FRAMES=$(ls "$INPUT_DIR"/*.jpg 2>/dev/null | wc -l)
echo "📊 Found $NUM_FRAMES frames to classify"

if [ "$NUM_FRAMES" -eq 0 ]; then
    echo "❌ Error: No frames found in $INPUT_DIR"
    exit 1
fi

# Run lighting classification
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Lighting Condition Analysis..."
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

def analyze_lighting(image_path):
    """
    Analyze lighting conditions of an image.
    Returns classification and confidence scores.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Calculate brightness (V channel in HSV)
    brightness = hsv[:, :, 2].mean()

    # Calculate color temperature (approximate using B/R ratio)
    b_mean = img[:, :, 0].mean()
    r_mean = img[:, :, 2].mean()
    color_temp = b_mean / (r_mean + 1e-6)

    # Calculate contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()

    # Calculate saturation
    saturation = hsv[:, :, 1].mean()

    # Histogram analysis for lighting distribution
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    hist_v = hist_v.flatten() / hist_v.sum()

    # Check for highlights (overexposure)
    highlights = np.sum(hsv[:, :, 2] > 240) / hsv[:, :, 2].size

    # Check for shadows (underexposure)
    shadows = np.sum(hsv[:, :, 2] < 30) / hsv[:, :, 2].size

    # Lighting uniformity (std of brightness)
    uniformity = 1.0 / (hsv[:, :, 2].std() + 1e-6)

    lighting_features = {
        "brightness": float(brightness),
        "color_temperature": float(color_temp),
        "contrast": float(contrast),
        "saturation": float(saturation),
        "highlights_ratio": float(highlights),
        "shadows_ratio": float(shadows),
        "uniformity": float(uniformity)
    }

    return lighting_features

def classify_time_of_day(features):
    """
    Classify time of day based on lighting features.
    Returns: morning, noon, afternoon, evening, night
    """
    brightness = features["brightness"]
    color_temp = features["color_temperature"]
    saturation = features["saturation"]

    # Night: low brightness
    if brightness < 80:
        return "night"

    # Morning: cool color temperature, moderate brightness
    if color_temp > 1.05 and 100 < brightness < 160:
        return "morning"

    # Noon: high brightness, neutral color temp
    if brightness > 180 and 0.95 < color_temp < 1.05:
        return "noon"

    # Afternoon: warm color temperature, high brightness
    if color_temp < 0.95 and brightness > 140:
        return "afternoon"

    # Evening: warm color, moderate to low brightness
    if color_temp < 0.95 and 80 < brightness < 140:
        return "evening"

    # Default to afternoon for ambiguous cases
    return "afternoon"

def classify_weather(features):
    """
    Classify weather condition based on lighting features.
    Returns: sunny, cloudy, overcast
    """
    contrast = features["contrast"]
    highlights = features["highlights_ratio"]
    shadows = features["shadows_ratio"]
    uniformity = features["uniformity"]

    # Sunny: high contrast, strong highlights and shadows
    if contrast > 50 and (highlights > 0.05 or shadows > 0.15):
        return "sunny"

    # Overcast: low contrast, uniform lighting
    if contrast < 35 and uniformity > 0.02:
        return "overcast"

    # Cloudy: moderate contrast
    return "cloudy"

def classify_location(features):
    """
    Classify indoor vs outdoor based on lighting features.
    Returns: indoor, outdoor
    """
    brightness = features["brightness"]
    saturation = features["saturation"]
    uniformity = features["uniformity"]

    # Indoor: more uniform lighting, potentially lower saturation
    if uniformity > 0.025 and brightness < 160:
        return "indoor"

    # Outdoor: more varied lighting, higher saturation
    return "outdoor"

def balance_sampling(classifications, target_per_category=50):
    """
    Balance dataset by sampling frames from each category.
    Returns selected frame indices.
    """
    # Group by combined classification
    category_indices = defaultdict(list)

    for idx, (time, weather, location) in enumerate(classifications):
        category = f"{time}_{weather}_{location}"
        category_indices[category].append(idx)

    # Sample from each category
    selected_indices = []

    for category, indices in category_indices.items():
        # Sample up to target_per_category from this category
        n_samples = min(len(indices), target_per_category)
        sampled = np.random.choice(indices, n_samples, replace=False)
        selected_indices.extend(sampled)

    return sorted(selected_indices)

# Configuration
INPUT_DIR = Path("''' + INPUT_DIR + '''")
OUTPUT_DIR = Path("/mnt/data/ai_data/datasets/3d-anime/luca/lighting_data")
CLASSIFIED_DIR = OUTPUT_DIR / "classified"

CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

# Find all frames
image_files = sorted(INPUT_DIR.glob("*.jpg"))
print(f"Processing {len(image_files)} frames...")
print("")

# Step 1: Analyze all frames
print("Step 1/3: Analyzing lighting conditions...")
lighting_data = []

for img_path in tqdm(image_files, desc="Lighting analysis"):
    features = analyze_lighting(img_path)

    if features is not None:
        time_of_day = classify_time_of_day(features)
        weather = classify_weather(features)
        location = classify_location(features)

        lighting_data.append({
            "filename": img_path.name,
            "path": str(img_path),
            "features": features,
            "time_of_day": time_of_day,
            "weather": weather,
            "location": location
        })

print(f"Analyzed {len(lighting_data)}/{len(image_files)} frames")
print("")

# Step 2: Classification statistics
print("Step 2/3: Generating classification statistics...")

time_counts = defaultdict(int)
weather_counts = defaultdict(int)
location_counts = defaultdict(int)
combined_counts = defaultdict(int)

for data in lighting_data:
    time_counts[data["time_of_day"]] += 1
    weather_counts[data["weather"]] += 1
    location_counts[data["location"]] += 1

    combined_key = f"{data['time_of_day']}_{data['weather']}_{data['location']}"
    combined_counts[combined_key] += 1

print("Time of Day Distribution:")
for time, count in sorted(time_counts.items(), key=lambda x: -x[1]):
    pct = count / len(lighting_data) * 100
    print(f"  {time}: {count} ({pct:.1f}%)")
print("")

print("Weather Distribution:")
for weather, count in sorted(weather_counts.items(), key=lambda x: -x[1]):
    pct = count / len(lighting_data) * 100
    print(f"  {weather}: {count} ({pct:.1f}%)")
print("")

print("Location Distribution:")
for location, count in sorted(location_counts.items(), key=lambda x: -x[1]):
    pct = count / len(lighting_data) * 100
    print(f"  {location}: {count} ({pct:.1f}%)")
print("")

# Step 3: Balance and organize dataset
print("Step 3/3: Balancing and organizing dataset...")

# Create category directories
categories = set()
for data in lighting_data:
    category = f"{data['time_of_day']}_{data['weather']}_{data['location']}"
    categories.add(category)
    (CLASSIFIED_DIR / category).mkdir(exist_ok=True)

# Copy files to category folders
category_counts = defaultdict(int)

for data in lighting_data:
    category = f"{data['time_of_day']}_{data['weather']}_{data['location']}"
    src_path = Path(data["path"])
    dest_path = CLASSIFIED_DIR / category / src_path.name

    shutil.copy2(src_path, dest_path)

    # Save metadata
    metadata_path = dest_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            "time_of_day": data["time_of_day"],
            "weather": data["weather"],
            "location": data["location"],
            "features": data["features"]
        }, f, indent=2)

    category_counts[category] += 1

# Generate report
print("")
print("=" * 80)
print("Lighting Classification Complete")
print("=" * 80)
print(f"Total frames classified: {len(lighting_data)}")
print("")
print("Category Distribution:")
for category in sorted(category_counts.keys()):
    count = category_counts[category]
    pct = count / len(lighting_data) * 100
    print(f"  {category}: {count} ({pct:.1f}%)")
print("")

# Save summary metadata
summary_metadata = {
    "total_frames": len(image_files),
    "classified_frames": len(lighting_data),
    "time_of_day_distribution": dict(time_counts),
    "weather_distribution": dict(weather_counts),
    "location_distribution": dict(location_counts),
    "category_distribution": dict(category_counts),
    "categories": sorted(list(categories)),
    "output_dir": str(CLASSIFIED_DIR)
}

metadata_path = OUTPUT_DIR / "lighting_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(summary_metadata, f, indent=2)

print(f"Metadata saved to: {metadata_path}")
print("✅ Lighting classification completed!")

PYTHON_CODE

if [ $? -ne 0 ]; then
    echo "❌ Lighting classification failed!"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║            ✅ Lighting Classification Complete!            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Output: $OUTPUT_DIR"
echo ""
echo "Next Steps:"
echo "  1. Review classification results"
echo "  2. Sample diverse examples from each category"
echo "  3. Generate lighting-focused captions with VLM (after GPU available)"
echo ""
echo "✨ Ready for Lighting LoRA training (after VLM captioning)"
