#!/usr/bin/bash

# ============================================================================
# Synthetic Data Generation Pipeline - Universal Configurable Version
# ============================================================================
#
# This script runs the complete synthetic data generation pipeline using
# a YAML configuration file for maximum flexibility and reusability.
#
# Usage:
#   bash scripts/batch/run_synthetic_data_pipeline.sh [options]
#
# Options:
#   --config PATH          Path to YAML config file
#                          (default: configs/batch/synthetic_data_generation.yaml)
#   --resume               Resume from checkpoint (default: true)
#   --no-resume            Start fresh, ignoring checkpoints
#   --phase PHASE          Run specific phase only (1=vocab, 2=generation, all=both)
#                          (default: all)
#   --characters CHARS     Comma-separated character list (overrides config)
#   --lora-types TYPES     Comma-separated lora types (overrides config)
#   --dry-run              Show what would be done without executing
#   --help                 Show this help message
#
# Examples:
#   # Run full pipeline with default config
#   bash scripts/batch/run_synthetic_data_pipeline.sh
#
#   # Run with custom config
#   bash scripts/batch/run_synthetic_data_pipeline.sh \
#     --config configs/batch/my_custom_config.yaml
#
#   # Run only vocabulary generation for specific characters
#   bash scripts/batch/run_synthetic_data_pipeline.sh \
#     --phase 1 --characters alberto,bryce,caleb
#
#   # Resume generation phase only
#   bash scripts/batch/run_synthetic_data_pipeline.sh \
#     --phase 2 --resume
#
# Author: LLMProvider Tooling
# Date: 2025-11-30
# ============================================================================

set -euo pipefail

# ============================================================================
# DEFAULT SETTINGS
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_CONFIG="$PROJECT_ROOT/configs/batch/synthetic_data_generation.yaml"

CONFIG_FILE="$DEFAULT_CONFIG"
RESUME=true
PHASE="all"
OVERRIDE_CHARACTERS=""
OVERRIDE_LORA_TYPES=""
DRY_RUN=false

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

show_help() {
    head -n 50 "$0" | grep "^#" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --no-resume)
            RESUME=false
            shift
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --characters)
            OVERRIDE_CHARACTERS="$2"
            shift 2
            ;;
        --lora-types)
            OVERRIDE_LORA_TYPES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# VALIDATE CONFIG
# ============================================================================

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "========================================================================"
echo "🚀 SYNTHETIC DATA GENERATION PIPELINE"
echo "========================================================================"
echo "Config file: $CONFIG_FILE"
echo "Project root: $PROJECT_ROOT"
echo "Resume: $RESUME"
echo "Phase: $PHASE"
[ -n "$OVERRIDE_CHARACTERS" ] && echo "Override characters: $OVERRIDE_CHARACTERS"
[ -n "$OVERRIDE_LORA_TYPES" ] && echo "Override LoRA types: $OVERRIDE_LORA_TYPES"
[ "$DRY_RUN" = true ] && echo "DRY RUN MODE (no execution)"
echo "========================================================================"
echo ""

# ============================================================================
# CHANGE TO PROJECT ROOT
# ============================================================================

cd "$PROJECT_ROOT" || {
    echo "FATAL: Failed to change to project root: $PROJECT_ROOT"
    exit 1
}

# ============================================================================
# RUN PYTHON ORCHESTRATOR
# ============================================================================

PYTHON_SCRIPT="$PROJECT_ROOT/scripts/batch/synthetic_data_orchestrator.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python orchestrator not found: $PYTHON_SCRIPT"
    echo "Please ensure the file exists in the project."
    exit 1
fi

# Build command
CMD="conda run -n ai_env python \"$PYTHON_SCRIPT\" --config \"$CONFIG_FILE\" --phase $PHASE"

if [ "$RESUME" = true ]; then
    CMD="$CMD --resume"
else
    CMD="$CMD --no-resume"
fi

if [ -n "$OVERRIDE_CHARACTERS" ]; then
    CMD="$CMD --characters \"$OVERRIDE_CHARACTERS\""
fi

if [ -n "$OVERRIDE_LORA_TYPES" ]; then
    CMD="$CMD --lora-types \"$OVERRIDE_LORA_TYPES\""
fi

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
fi

# Execute
echo "Executing:"
echo "  $CMD"
echo ""

if [ "$DRY_RUN" = false ]; then
    eval "$CMD"
else
    echo "[DRY RUN] Command prepared but not executed"
fi

echo ""
echo "========================================================================"
echo "✅ PIPELINE EXECUTION COMPLETE"
echo "========================================================================"
