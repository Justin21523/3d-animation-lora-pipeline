#!/bin/bash
# Generic Batch Processing Launcher
# Wrapper script to launch Python batch processor with proper environment
#
# Usage:
#   bash scripts/batch/run_batch_processing.sh [config_file] [options]
#
# Examples:
#   # Run SAM2 + LaMa batch processing (with resume)
#   bash scripts/batch/run_batch_processing.sh configs/batch/sam2_lama.yaml
#
#   # Run SAM2 only
#   bash scripts/batch/run_batch_processing.sh configs/batch/sam2_only.yaml
#
#   # Dry run (test configuration)
#   bash scripts/batch/run_batch_processing.sh configs/batch/sam2_lama.yaml --dry-run
#
#   # Fresh start (ignore progress file)
#   bash scripts/batch/run_batch_processing.sh configs/batch/sam2_lama.yaml --no-resume

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default config
CONFIG="${1:-configs/batch/sam2_lama.yaml}"
shift || true  # Remove first arg (config), keep remaining for passthrough

echo "======================================================================"
echo "🚀 Generic Batch Processing Pipeline"
echo "======================================================================"
echo ""
echo "Project:  $PROJECT_ROOT"
echo "Config:   $CONFIG"
echo "Args:     $@"
echo ""
echo "======================================================================"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    echo ""
    echo "Available configs:"
    ls -1 configs/batch/*.yaml 2>/dev/null || echo "  (none)"
    exit 1
fi

# Check if Python script exists
PROCESSOR="$SCRIPT_DIR/batch_processor.py"
if [ ! -f "$PROCESSOR" ]; then
    echo "❌ Error: Batch processor not found: $PROCESSOR"
    exit 1
fi

# Make processor executable
chmod +x "$PROCESSOR"

# Create log directory
mkdir -p logs/batch_processing

# Run batch processor with conda environment
echo "🔄 Starting batch processor..."
echo ""

conda run -n ai_env python "$PROCESSOR" \
    --config "$CONFIG" \
    "$@"

EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Batch processing complete"
else
    echo "❌ Batch processing failed (exit code: $EXIT_CODE)"
fi
echo "======================================================================"
echo ""
echo "📄 Logs: logs/batch_processing/"
echo "📊 Progress: logs/batch_processing/progress.json"
echo ""

exit $EXIT_CODE
