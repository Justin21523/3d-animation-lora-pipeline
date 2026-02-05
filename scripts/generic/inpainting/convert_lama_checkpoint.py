#!/usr/bin/env python3
"""
One-time converter: Extract LaMa generator weights from Lightning checkpoint
This creates a clean PyTorch state_dict without pytorch_lightning dependencies.

Usage:
    python convert_lama_checkpoint.py \
        --input /path/to/best.ckpt \
        --output /path/to/lama_generator.pth
"""

import argparse
import torch
from pathlib import Path
import sys

# Patch the version checking bug before any pytorch_lightning imports
def patch_version_check():
    """
    Patch packaging.version.Version to handle non-string inputs.
    This works around the lightning_utilities scipy version check bug.
    """
    try:
        from packaging import version
        original_init = version.Version.__init__

        def patched_init(self, version_str):
            # Convert to string if not already
            if not isinstance(version_str, str):
                version_str = str(version_str)
            original_init(self, version_str)

        version.Version.__init__ = patched_init
        print("✅ Applied version check patch")
    except Exception as e:
        print(f"⚠️  Could not apply patch: {e}")

def convert_checkpoint(input_path: Path, output_path: Path):
    """
    Extract generator weights from Lightning checkpoint and save as clean state_dict.
    """
    print(f"Loading checkpoint from: {input_path}")

    try:
        # Load checkpoint (this may trigger pytorch_lightning imports)
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

        # Extract generator state dict
        state_dict = checkpoint['state_dict']
        gen_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith('generator.'):
                # Remove 'generator.' prefix
                new_key = k.replace('generator.', '')
                gen_state_dict[new_key] = v

        print(f"✅ Extracted {len(gen_state_dict)} generator parameters")

        # Extract generator config (needed for model initialization)
        generator_config = {
            'input_nc': 4,
            'output_nc': 3,
            'ngf': 64,
            'n_downsampling': 3,
            'n_blocks': 9,
            'max_features': 512,
            'add_out_act': 'sigmoid',
            'init_conv_kwargs': {'ratio_gin': 0, 'ratio_gout': 0},
            'downsample_conv_kwargs': {'ratio_gin': 0, 'ratio_gout': 0},
            'resnet_conv_kwargs': {'ratio_gin': 0, 'ratio_gout': 0.75}
        }

        # Save clean checkpoint with config
        clean_checkpoint = {
            'state_dict': gen_state_dict,
            'config': generator_config
        }

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save clean checkpoint
        torch.save(clean_checkpoint, output_path)

        print(f"✅ Saved clean checkpoint to: {output_path}")
        print(f"📊 File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        return True

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Apply patch before any torch operations
    patch_version_check()

    parser = argparse.ArgumentParser(description="Convert LaMa Lightning checkpoint to clean PyTorch state_dict")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to Lightning checkpoint (best.ckpt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save clean state_dict (.pth)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Input checkpoint not found: {input_path}")
        sys.exit(1)

    success = convert_checkpoint(input_path, output_path)

    if success:
        print("\n✅ Conversion complete!")
        print(f"   You can now delete the original checkpoint if needed.")
        print(f"   The inpainting script will automatically use the clean checkpoint.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
