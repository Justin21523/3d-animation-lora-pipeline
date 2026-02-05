#!/bin/bash
#
# Move Elio SDXL Checkpoints to Correct Location
# Usage: bash scripts/batch/move_elio_checkpoints.sh
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "🚀 Move Elio Checkpoints to ai_models"
echo "=========================================="
echo ""

# Paths
SOURCE_DIR="/mnt/data/training/lora/elio/elio_identity"
TARGET_DIR="/mnt/c/ai_models/lora_sdxl/elio/elio_identity"

# Check source directory
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}❌ Source directory not found: $SOURCE_DIR${NC}"
    exit 1
fi

# Count checkpoints
CHECKPOINT_COUNT=$(ls "$SOURCE_DIR"/*.safetensors 2>/dev/null | wc -l)

if [ "$CHECKPOINT_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No checkpoints found in source directory${NC}"
    exit 1
fi

echo -e "${BLUE}📁 Source:${NC} $SOURCE_DIR"
echo -e "${BLUE}📁 Target:${NC} $TARGET_DIR"
echo -e "${GREEN}📊 Found $CHECKPOINT_COUNT checkpoint(s)${NC}"
echo ""

# Create target directory
echo "Creating target directory..."
mkdir -p "$TARGET_DIR"

# List checkpoints to be moved
echo "Checkpoints to be moved:"
ls -lh "$SOURCE_DIR"/*.safetensors | awk '{print "  - " $9 " (" $5 ")"}'
echo ""

# Ask for confirmation
read -p "Proceed with moving files? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Operation cancelled${NC}"
    exit 0
fi

echo ""
echo "Moving files..."

# Move checkpoints
mv -v "$SOURCE_DIR"/*.safetensors "$TARGET_DIR/"

# Move logs
if [ -d "$SOURCE_DIR/logs" ]; then
    echo "Moving logs..."
    if [ ! -d "$TARGET_DIR/logs" ]; then
        mkdir -p "$TARGET_DIR/logs"
    fi
    mv -v "$SOURCE_DIR/logs"/* "$TARGET_DIR/logs/" 2>/dev/null || true
fi

# Create symlink for backward compatibility (optional)
echo ""
echo "Creating symlink for backward compatibility..."
if [ -L "$SOURCE_DIR" ]; then
    rm "$SOURCE_DIR"
fi
ln -s "$TARGET_DIR" "$SOURCE_DIR"

echo ""
echo -e "${GREEN}✅ Successfully moved all files!${NC}"
echo ""
echo "Final structure:"
ls -lh "$TARGET_DIR"/*.safetensors 2>/dev/null | head -10
echo ""
echo -e "${BLUE}Checkpoints location:${NC} $TARGET_DIR"
echo -e "${BLUE}Symlink created:${NC} $SOURCE_DIR -> $TARGET_DIR"
echo ""
echo "=========================================="
echo "✅ Done!"
echo "=========================================="
