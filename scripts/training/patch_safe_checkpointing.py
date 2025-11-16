"""
Monkey-patch to enable safe gradient checkpointing (use_reentrant=False).

This module patches PyTorch's gradient checkpointing before any models are loaded.
Import this at the very beginning of your training script.
"""

import torch
import functools

# Store original checkpoint function
_original_checkpoint = torch.utils.checkpoint.checkpoint

@functools.wraps(_original_checkpoint)
def safe_checkpoint(*args, use_reentrant=None, **kwargs):
    """
    Patched checkpoint function that defaults to use_reentrant=False.

    PyTorch 2.x recommends use_reentrant=False for better stability
    and to avoid CUDA errors.
    """
    if use_reentrant is None:
        use_reentrant = False
    return _original_checkpoint(*args, use_reentrant=use_reentrant, **kwargs)

# Apply the patch
torch.utils.checkpoint.checkpoint = safe_checkpoint

print("✅ Gradient checkpointing patched: use_reentrant=False by default")
