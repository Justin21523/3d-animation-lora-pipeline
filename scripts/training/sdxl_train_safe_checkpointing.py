#!/usr/bin/env python3
"""
SDXL Training Wrapper with Safe Gradient Checkpointing
=======================================================

This wrapper enables gradient checkpointing with use_reentrant=False to avoid CUDA errors.

Key Features:
- Uses PyTorch's recommended non-reentrant gradient checkpointing
- Prevents CUDA checkpoint errors and None gradient issues
- Compatible with Kohya sd-scripts SDXL training

Technical Background:
- PyTorch 2.x recommends use_reentrant=False for gradient checkpointing
- The old reentrant mode causes CUDA errors, gradient issues, and incompatibilities
- Diffusers 0.35.2+ supports custom gradient_checkpointing_func parameter

Usage:
    python scripts/training/sdxl_train_safe_checkpointing.py \
        --config configs/training/sdxl_16gb_optimized.toml
"""

import sys
import os
import functools
from typing import Any, Tuple
import torch

# Add sd-scripts to path
KOHYA_ROOT = "/mnt/c/AI_LLM_projects/kohya_ss"
sys.path.insert(0, os.path.join(KOHYA_ROOT, "sd-scripts"))

def safe_gradient_checkpointing_func(*args, **kwargs):
    """
    Gradient checkpointing function with use_reentrant=False.

    This is the PyTorch-recommended approach for gradient checkpointing
    that avoids CUDA errors and other issues with the reentrant variant.
    """
    return torch.utils.checkpoint.checkpoint(
        *args,
        use_reentrant=False,
        **kwargs
    )

def patch_gradient_checkpointing():
    """
    Patch diffusers models to use safe gradient checkpointing.

    This must be called before model initialization.
    """
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

    # Store original method
    original_unet_enable = UNet2DConditionModel.enable_gradient_checkpointing

    def safe_unet_enable_gradient_checkpointing(self, gradient_checkpointing_func=None):
        """Patched UNet gradient checkpointing enabler."""
        if gradient_checkpointing_func is None:
            gradient_checkpointing_func = safe_gradient_checkpointing_func
        return original_unet_enable(self, gradient_checkpointing_func)

    # Apply patch
    UNet2DConditionModel.enable_gradient_checkpointing = (
        safe_unet_enable_gradient_checkpointing
    )

    print("✅ Gradient checkpointing patched to use_reentrant=False (safe mode)")

def main():
    """Main entry point."""
    # Patch gradient checkpointing BEFORE importing training script
    patch_gradient_checkpointing()

    # Import training modules
    import sdxl_train_network
    import train_network

    # Call the actual training function
    parser = sdxl_train_network.setup_parser()
    args = parser.parse_args()

    import train_util
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train_network.train(args)

if __name__ == "__main__":
    main()
