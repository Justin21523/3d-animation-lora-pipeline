#!/usr/bin/env bash
# Generic Kohya Dataset Preparation Script
# Converts flat image+caption directories to Kohya format ({repeat}_{name}/)

set -e

# Usage information
usage() {
    cat << EOF
Usage: $0 --source-dir <dir> --output-dir <dir> --repeat <num> --name <name> [OPTIONS]

Required Arguments:
  --source-dir DIR     Source directory containing PNG/JPG + TXT files
  --output-dir DIR     Output directory for Kohya-format dataset
  --repeat NUM         Repeat multiplier (e.g., 10 for 10_charactername)
  --name NAME          Character/concept name (e.g., luca, alberto)

Optional Arguments:
  --validate          Validate caption completeness before copying (default: true)
  --force             Overwrite existing output directory
  --help              Show this help message

Example:
  $0 --source-dir /path/to/curated_images \\
     --output-dir /path/to/training_data \\
     --repeat 10 \\
     --name luca

Kohya Format:
  Creates directory structure: {output-dir}/{repeat}_{name}/
  Example: /path/to/training_data/10_luca/

EOF
    exit 1
}

# Parse arguments
SOURCE_DIR=""
OUTPUT_DIR=""
REPEAT=""
NAME=""
VALIDATE=true
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --repeat)
            REPEAT="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --validate)
            VALIDATE=true
            shift
            ;;
        --no-validate)
            VALIDATE=false
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "❌ Unknown argument: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$SOURCE_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$REPEAT" ] || [ -z "$NAME" ]; then
    echo "❌ Error: Missing required arguments"
    usage
fi

# Validate source directory
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Error: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Create Kohya directory structure
KOHYA_SUBDIR="${OUTPUT_DIR}/${REPEAT}_${NAME}"

echo "======================================================================"
echo "Kohya Dataset Preparation"
echo "======================================================================"
echo "Source:       $SOURCE_DIR"
echo "Output:       $KOHYA_SUBDIR"
echo "Repeat:       ${REPEAT}x"
echo "Name:         $NAME"
echo "Validate:     $VALIDATE"
echo "======================================================================"
echo ""

# Count source files
PNG_COUNT=$(find "$SOURCE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
TXT_COUNT=$(find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.txt" | wc -l)

echo "Source files:"
echo "  Images:   $PNG_COUNT"
echo "  Captions: $TXT_COUNT"
echo ""

# Validation
if [ "$VALIDATE" = true ]; then
    if [ "$PNG_COUNT" -ne "$TXT_COUNT" ]; then
        echo "❌ Error: Image count ($PNG_COUNT) ≠ Caption count ($TXT_COUNT)"
        echo "   Each image must have a matching caption file!"
        exit 1
    fi

    if [ "$PNG_COUNT" -eq 0 ]; then
        echo "❌ Error: No images found in source directory!"
        exit 1
    fi

    echo "✓ Validation passed"
    echo ""
fi

# Check if output directory exists
if [ -d "$KOHYA_SUBDIR" ]; then
    if [ "$FORCE" = false ]; then
        echo "❌ Error: Output directory already exists: $KOHYA_SUBDIR"
        echo "   Use --force to overwrite"
        exit 1
    else
        echo "⚠  Overwriting existing directory: $KOHYA_SUBDIR"
        rm -rf "$KOHYA_SUBDIR"
    fi
fi

# Create Kohya directory structure
echo "Creating Kohya directory structure..."
mkdir -p "$KOHYA_SUBDIR"

# Copy files
echo "Copying files..."
find "$SOURCE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) -exec cp {} "$KOHYA_SUBDIR/" \;
find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.txt" -exec cp {} "$KOHYA_SUBDIR/" \;

# Verify final counts
FINAL_PNG=$(find "$KOHYA_SUBDIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
FINAL_TXT=$(find "$KOHYA_SUBDIR" -type f -name "*.txt" | wc -l)

echo ""
echo "======================================================================"
echo "Dataset Preparation Complete!"
echo "======================================================================"
echo "Final dataset:"
echo "  Images:   $FINAL_PNG"
echo "  Captions: $FINAL_TXT"
echo "  Location: $KOHYA_SUBDIR"
echo ""
echo "Training configuration:"
echo "  Repeat multiplier: ${REPEAT}x"
echo "  Effective iterations: $((FINAL_PNG * REPEAT))"
echo ""

if [ "$FINAL_PNG" -ne "$FINAL_TXT" ]; then
    echo "⚠  Warning: Final image count ($FINAL_PNG) ≠ caption count ($FINAL_TXT)"
else
    echo "✓ All checks passed - Ready for training!"
fi

echo "======================================================================"
