#!/usr/bin/env python3
"""
SAM2 Background Inpainting Script
Processes background images from SAM2 segmentation output by merging all instance masks
and applying LaMa or BrushNet inpainting.

Usage:
    python sam2_background_inpainting.py \
        --sam2-dir /path/to/sam2_output \
        --output-dir /path/to/clean_backgrounds \
        --method lama \
        --batch-size 16 \
        --device cuda
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from typing import List, Tuple, Dict
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 Background Inpainting")
    parser.add_argument("--sam2-dir", type=str, required=True,
                        help="Directory containing SAM2 outputs (backgrounds/ and masks/ subdirs)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for cleaned backgrounds")
    parser.add_argument("--method", type=str, choices=["lama", "brushnet"], default="lama",
                        help="Inpainting method: 'lama' (fast) or 'brushnet' (quality)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--quality-check", action="store_true",
                        help="Enable quality validation (PSNR/SSIM)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images to process (for testing)")
    parser.add_argument("--save-metadata", action="store_true", default=True,
                        help="Save processing metadata")
    parser.add_argument("--mask-dilate", type=int, default=15,
                        help="Dilate mask by N pixels to cover character edges (default: 15)")
    return parser.parse_args()

def find_matching_masks(background_path: Path, masks_dir: Path) -> List[Path]:
    """
    Find all instance masks corresponding to a background image.

    Example:
        Background: scene0024_pos1_frame000241_t157.66s_background.jpg
        Masks: scene0024_pos1_frame000241_t157.66s_inst0_mask.png
               scene0024_pos1_frame000241_t157.66s_inst1_mask.png
               ...
    """
    # Extract base name (remove _background.jpg)
    base_name = background_path.stem.replace("_background", "")

    # Find all matching masks
    mask_pattern = f"{base_name}_inst*_mask.png"
    matching_masks = sorted(masks_dir.glob(mask_pattern))

    return matching_masks

def merge_masks(mask_paths: List[Path], dilate_pixels: int = 0) -> np.ndarray:
    """
    Merge multiple instance masks into a single binary mask.
    White (255) = areas to inpaint (character regions)
    Black (0) = areas to keep (background)

    Args:
        mask_paths: List of paths to instance masks
        dilate_pixels: Dilate mask by N pixels to cover character edges
    """
    if not mask_paths:
        return None

    # Load first mask to get dimensions
    first_mask = cv2.imread(str(mask_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first_mask is None:
        return None

    # Initialize merged mask
    merged = np.zeros_like(first_mask)

    # Merge all masks
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Any non-zero pixel becomes white in merged mask
            merged = np.maximum(merged, mask)

    # Dilate mask to cover character edges
    if dilate_pixels > 0:
        kernel = np.ones((dilate_pixels, dilate_pixels), np.uint8)
        merged = cv2.dilate(merged, kernel, iterations=1)

    return merged

def inpaint_lama(image: np.ndarray, mask: np.ndarray, model) -> np.ndarray:
    """
    Apply LaMa inpainting with the generator model.
    """
    import torch

    # Ensure mask is binary (0 or 1)
    mask_binary = (mask > 127).astype(np.float32)

    # Ensure RGB image
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    image_norm = image_rgb.astype(np.float32) / 255.0

    # Convert to torch tensors [C, H, W]
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0)

    # Move to device
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    mask_tensor = mask_tensor.to(device)

    # Concatenate image and mask [B, 4, H, W]
    masked_image = image_tensor * (1 - mask_tensor)
    model_input = torch.cat([masked_image, mask_tensor], dim=1)

    # Inpaint
    with torch.no_grad():
        result_tensor = model(model_input)

    # Convert back to numpy [H, W, C]
    result_np = result_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)

    # RGB to BGR
    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

    return result_bgr

def process_single_background(
    background_path: Path,
    masks_dir: Path,
    output_dir: Path,
    model,
    method: str,
    stats: Dict,
    dilate_pixels: int = 0
) -> bool:
    """
    Process a single background image.

    Args:
        dilate_pixels: Dilate mask by N pixels to cover character edges
    """
    try:
        # Find matching masks
        mask_paths = find_matching_masks(background_path, masks_dir)

        if not mask_paths:
            stats["no_masks"] += 1
            print(f"⚠️  No masks found for {background_path.name}")
            return False

        # Load background image
        bg_image = cv2.imread(str(background_path))
        if bg_image is None:
            stats["load_error"] += 1
            return False

        # Merge all instance masks with optional dilation
        merged_mask = merge_masks(mask_paths, dilate_pixels=dilate_pixels)
        if merged_mask is None:
            stats["merge_error"] += 1
            return False

        # Check if mask is empty (no characters to remove)
        mask_coverage = (merged_mask > 127).sum() / merged_mask.size * 100
        if mask_coverage < 0.1:  # Less than 0.1% coverage
            # Just copy the original (no characters detected)
            output_path = output_dir / background_path.name
            cv2.imwrite(str(output_path), bg_image)
            stats["no_characters"] += 1
            return True

        # Apply inpainting
        if method == "lama":
            result = inpaint_lama(bg_image, merged_mask, model)
        else:
            # TODO: Implement BrushNet
            raise NotImplementedError("BrushNet not yet implemented")

        # Save result
        output_path = output_dir / background_path.name
        cv2.imwrite(str(output_path), result)

        stats["success"] += 1
        stats["total_masks"] += len(mask_paths)
        stats["mask_coverage_sum"] += mask_coverage

        return True

    except Exception as e:
        stats["error"] += 1
        print(f"❌ Error processing {background_path.name}: {e}")
        return False

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
    except Exception:
        pass  # Silently ignore if patch fails

def load_lama_model(device: str):
    """
    Load LaMa model for inference directly without trainers module.
    """
    try:
        import torch
        from omegaconf import OmegaConf

        # Apply version check patch first
        patch_version_check()

        # Add LaMa modules to path
        lama_code_path = Path(__file__).parent / "lama"
        if str(lama_code_path) not in sys.path:
            sys.path.insert(0, str(lama_code_path))

        # Try clean checkpoint first (no pytorch_lightning dependencies)
        clean_checkpoint_path = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/inpainting/lama/checkpoints/lama_generator_clean.pth")
        original_checkpoint_path = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/inpainting/lama/checkpoints/best.ckpt")

        # Determine which checkpoint to use
        if clean_checkpoint_path.exists():
            checkpoint_path = clean_checkpoint_path
            is_clean = True
            print(f"Using clean checkpoint: {checkpoint_path.name}")
        elif original_checkpoint_path.exists():
            checkpoint_path = original_checkpoint_path
            is_clean = False
            print(f"Using original checkpoint: {checkpoint_path.name}")
        else:
            raise FileNotFoundError(f"No checkpoint found. Tried:\n  - {clean_checkpoint_path}\n  - {original_checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Handle clean vs original checkpoint formats
        if is_clean:
            # Clean checkpoint: direct state_dict + config
            gen_state_dict = checkpoint['state_dict']
            generator_config = checkpoint['config']
        else:
            # Original checkpoint: nested Lightning format
            # Get generator config from OmegaConf
            from omegaconf import OmegaConf
            generator_config_raw = checkpoint['hyper_parameters']['generator']

            # Convert OmegaConf to dict, resolving all interpolations
            generator_config_full = OmegaConf.to_container(generator_config_raw, resolve=True)

            # Filter to only valid FFCResNetGenerator parameters
            valid_params = ['input_nc', 'output_nc', 'ngf', 'n_downsampling', 'n_blocks',
                           'max_features', 'add_out_act', 'init_conv_kwargs',
                           'downsample_conv_kwargs', 'resnet_conv_kwargs', 'spatial_transform_kwargs',
                           'norm_layer', 'activation_layer', 'padding_type']
            generator_config = {k: v for k, v in generator_config_full.items() if k in valid_params}

            # Extract generator weights from state_dict
            state_dict = checkpoint['state_dict']
            gen_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('generator.'):
                    gen_state_dict[k.replace('generator.', '')] = v

        # Import generator directly
        from saicinpainting.training.modules.ffc import FFCResNetGenerator
        generator = FFCResNetGenerator(**generator_config)

        # Load weights (strict=False to handle architecture variations)
        missing_keys, unexpected_keys = generator.load_state_dict(gen_state_dict, strict=False)

        if missing_keys or unexpected_keys:
            print(f"⚠️  Weight loading with some mismatches:")
            if missing_keys:
                print(f"   Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"   Unexpected keys: {len(unexpected_keys)}")
        generator.eval()
        generator.to(device)

        print("✅ LaMa model loaded successfully")
        return generator

    except Exception as e:
        print(f"❌ Failed to load LaMa model: {e}")
        import traceback
        traceback.print_exc()
        print("   Falling back to OpenCV inpainting...")
        return None

def opencv_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fallback: OpenCV inpainting (telea method).
    """
    mask_binary = (mask > 127).astype(np.uint8)
    result = cv2.inpaint(image, mask_binary, 3, cv2.INPAINT_TELEA)
    return result

def main():
    args = parse_args()

    print("=" * 70)
    print("SAM2 BACKGROUND INPAINTING")
    print("=" * 70)
    print(f"SAM2 directory: {args.sam2_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Method: {args.method}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)
    print()

    # Setup paths
    sam2_dir = Path(args.sam2_dir)
    backgrounds_dir = sam2_dir / "backgrounds"
    masks_dir = sam2_dir / "masks"
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not backgrounds_dir.exists():
        print(f"❌ Backgrounds directory not found: {backgrounds_dir}")
        sys.exit(1)
    if not masks_dir.exists():
        print(f"❌ Masks directory not found: {masks_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all background images
    background_files = sorted(backgrounds_dir.glob("*_background.jpg"))
    total_backgrounds = len(background_files)

    if total_backgrounds == 0:
        print("❌ No background images found!")
        sys.exit(1)

    print(f"📊 Found {total_backgrounds} background images")

    # Apply limit if specified
    if args.limit:
        background_files = background_files[:args.limit]
        print(f"⚙️  Limited to {args.limit} images for testing")

    print()

    # Load inpainting model
    print(f"Loading {args.method.upper()} model on {args.device}...")

    if args.method == "lama":
        model = load_lama_model(args.device)
        if model is None:
            print("⚠️  Using OpenCV fallback inpainting")
            use_opencv = True
        else:
            print("✅ LaMa model loaded")
            use_opencv = False
    else:
        print("❌ BrushNet not yet implemented")
        sys.exit(1)

    print()

    # Processing statistics
    stats = {
        "success": 0,
        "no_masks": 0,
        "no_characters": 0,
        "load_error": 0,
        "merge_error": 0,
        "error": 0,
        "total_masks": 0,
        "mask_coverage_sum": 0.0
    }

    # Process all backgrounds
    print("🎨 Processing backgrounds...")
    print()

    for bg_path in tqdm(background_files, desc="Inpainting"):
        if use_opencv:
            # Use OpenCV fallback
            try:
                bg_image = cv2.imread(str(bg_path))
                mask_paths = find_matching_masks(bg_path, masks_dir)
                if mask_paths:
                    merged_mask = merge_masks(mask_paths, dilate_pixels=args.mask_dilate)
                    result = opencv_inpaint(bg_image, merged_mask)
                    output_path = output_dir / bg_path.name
                    cv2.imwrite(str(output_path), result)
                    stats["success"] += 1
            except:
                stats["error"] += 1
        else:
            process_single_background(bg_path, masks_dir, output_dir, model, args.method, stats, dilate_pixels=args.mask_dilate)

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✅ Successfully processed: {stats['success']}")
    print(f"⚠️  No masks found: {stats['no_masks']}")
    print(f"ℹ️  No characters (copied): {stats['no_characters']}")
    print(f"❌ Errors: {stats['error'] + stats['load_error'] + stats['merge_error']}")
    print(f"📊 Total instance masks merged: {stats['total_masks']}")

    if stats['success'] > 0:
        avg_coverage = stats['mask_coverage_sum'] / stats['success']
        print(f"📊 Average mask coverage: {avg_coverage:.2f}%")

    print()
    print(f"📁 Output directory: {output_dir}")
    print("=" * 70)

    # Save metadata
    if args.save_metadata:
        metadata = {
            "total_backgrounds": total_backgrounds,
            "processed": len(background_files),
            "stats": stats,
            "method": args.method,
            "device": args.device
        }

        metadata_path = output_dir / "inpainting_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📄 Metadata saved: {metadata_path}")

if __name__ == "__main__":
    main()
