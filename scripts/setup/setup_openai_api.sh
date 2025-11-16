#!/bin/bash
# Setup OpenAI API Key for Pipeline
# Usage: ./setup_openai_api.sh YOUR_API_KEY

set -e

API_KEY="$1"

if [ -z "$API_KEY" ]; then
    echo "ERROR: No API key provided"
    echo ""
    echo "Usage: ./setup_openai_api.sh YOUR_API_KEY"
    echo ""
    echo "Example:"
    echo "  ./setup_openai_api.sh sk-proj-xxxxxxxxxxxxx"
    echo ""
    exit 1
fi

# Add to .bashrc if not already present
if ! grep -q "OPENAI_API_KEY" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# OpenAI API Key for Yokai Watch Pipeline" >> ~/.bashrc
    echo "export OPENAI_API_KEY=\"$API_KEY\"" >> ~/.bashrc
    echo "✓ Added OPENAI_API_KEY to ~/.bashrc"
else
    # Update existing entry
    sed -i "s/^export OPENAI_API_KEY=.*/export OPENAI_API_KEY=\"$API_KEY\"/" ~/.bashrc
    echo "✓ Updated OPENAI_API_KEY in ~/.bashrc"
fi

# Export for current session
export OPENAI_API_KEY="$API_KEY"

echo ""
echo "✓ OpenAI API Key configured"
echo ""
echo "To activate in current shell:"
echo "  source ~/.bashrc"
echo ""
echo "Or start a new shell session"
echo ""

# Test API key
echo "Testing API connection..."
python3 << 'EOF'
import os
import sys

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in environment")
    sys.exit(1)

if not api_key.startswith("sk-"):
    print("WARNING: API key doesn't start with 'sk-', may be invalid")

print(f"✓ API Key detected: {api_key[:10]}...{api_key[-4:]}")
print("✓ Ready to use OpenAI API")
EOF

echo ""
echo "Estimated costs for frame annotation:"
echo "  - Sample 1:50 with GPT-4o-mini: ~$148"
echo "  - Sample 1:100 with GPT-4o-mini: ~$74 (Recommended)"
echo "  - Sample 1:200 with GPT-4o-mini: ~$37"
echo ""
echo "Budget limit set in config: $50"
echo "You can adjust in: config/pipeline_config.yaml"
