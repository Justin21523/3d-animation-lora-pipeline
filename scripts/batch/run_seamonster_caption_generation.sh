#!/bin/bash
# Wrapper script to generate captions for sea monster images
# Usage: bash run_seamonster_caption_generation.sh [your-api-key]

# Check if API key provided as argument
if [ -n "$1" ]; then
    export LLM_VENDOR_API_KEY="$1"
    echo "✓ Using API key from command line argument"
elif [ -z "$LLM_VENDOR_API_KEY" ]; then
    echo "❌ ERROR: LLM_VENDOR_API_KEY not set"
    echo ""
    echo "Usage:"
    echo "  Option 1 - Set environment variable first:"
    echo "    export LLM_VENDOR_API_KEY='your-api-key-here'"
    echo "    bash $0"
    echo ""
    echo "  Option 2 - Pass as argument:"
    echo "    bash $0 your-api-key-here"
    echo ""
    exit 1
else
    echo "✓ Using API key from environment variable"
fi

# Show configuration
echo ""
echo "========================================================================"
echo "Sea Monster Caption Generation with LLMProvider Haiku"
echo "========================================================================"
echo "Alberto images: /mnt/data/datasets/general/luca/sdxl_seamonster_training/alberto_seamonster_sdxl"
echo "Luca images: /mnt/data/datasets/general/luca/sdxl_seamonster_training/luca_seamonster_sdxl"
echo "Model: llm_provider-3-5-haiku-20241022"
echo "========================================================================"
echo ""

# Activate conda environment and run
conda run -n ai_env python /mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/batch/generate_seamonster_captions.py \
  --image-dirs \
    /mnt/data/datasets/general/luca/sdxl_seamonster_training/alberto_seamonster_sdxl \
    /mnt/data/datasets/general/luca/sdxl_seamonster_training/luca_seamonster_sdxl \
  --character-names alberto_seamonster luca_seamonster \
  2>&1 | tee /tmp/seamonster_caption_generation.log

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Caption generation completed successfully!"
    echo "   Log saved to: /tmp/seamonster_caption_generation.log"
else
    echo ""
    echo "❌ Caption generation failed. Check log for details:"
    echo "   /tmp/seamonster_caption_generation.log"
    exit 1
fi
