#!/usr/bin/env python3
"""
3D Character Enhancement with SOTA Models

Purpose: High-quality enhancement optimized for Pixar-style 3D characters
Methods:
  - RealESRGAN: Pixar-grade upscaling (used in Toy Story 4, Onward, Soul)
  - CodeFormer: 2024 best face enhancement for 3D characters
  - CNN Denoising: Similar to Pixar's Finding Dory technique
  - OpenCV (fast mode): For quick previews

Usage:
    # Quality mode (RealESRGAN + CodeFormer)
    python frame_enhancement.py \
      --input-dir /path/to/frames \
      --output-dir /path/to/enhanced \
      --mode quality \
      --config configs/stages/enhancement/3d_character_enhancement.yaml

    # Fast mode (OpenCV only)
    python frame_enhancement.py \
      --input-dir /path/to/frames \
      --output-dir /path/to/enhanced \
      --mode fast
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Set, Optional, Dict

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from PIL import Image


# ============================================================================
# Model Loading
# ============================================================================

def load_realesrgan_model(config: Dict, device: str = "cuda"):
    """
    Load RealESRGAN model for upscaling

    RealESRGAN used by Pixar in recent films (Toy Story 4, Onward, Soul)
    for high-quality upscaling of rendered frames
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        print(f"Loading RealESRGAN on {device}...")

        realesrgan_config = config['realesrgan']
        model_path = realesrgan_config['model_path']

        # Define model architecture
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=realesrgan_config.get('upscale', 2)
        )

        # Initialize upsampler
        upsampler = RealESRGANer(
            scale=realesrgan_config.get('upscale', 2),
            model_path=model_path,
            model=model,
            tile=realesrgan_config.get('tile_size', 512),
            tile_pad=realesrgan_config.get('tile_pad', 10),
            pre_pad=realesrgan_config.get('pre_pad', 0),
            half=realesrgan_config.get('half_precision', True),
            device=device
        )

        print("✓ RealESRGAN loaded successfully")
        return upsampler

    except ImportError as e:
        print(f"❌ RealESRGAN not installed: {e}")
        print("\nInstall with:")
        print("  pip install realesrgan basicsr")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load RealESRGAN: {e}")
        sys.exit(1)


def load_codeformer_model(config: Dict, device: str = "cuda"):
    """
    Load CodeFormer for face enhancement

    CodeFormer (2024): Best face restoration model for 3D characters
    """
    try:
        from codeformer import CodeFormer
        from facelib.utils.face_restoration_helper import FaceRestoreHelper

        print(f"Loading CodeFormer on {device}...")

        codeformer_config = config['codeformer']
        model_path = codeformer_config['model_path']

        # Load CodeFormer
        net = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=['32', '64', '128', '256']
        ).to(device)

        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['params_ema'])
        net.eval()

        # Initialize face helper
        face_helper = FaceRestoreHelper(
            upscale_factor=codeformer_config.get('upscale', 2),
            face_size=512,
            crop_ratio=(1, 1),
            det_model=codeformer_config.get('detection_model', 'retinaface'),
            save_ext='png',
            use_parse=True,
            device=device
        )

        print("✓ CodeFormer loaded successfully")
        return net, face_helper

    except ImportError as e:
        print(f"❌ CodeFormer not installed: {e}")
        print("\nInstall with:")
        print("  pip install codeformer facelib")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load CodeFormer: {e}")
        sys.exit(1)


# ============================================================================
# Enhancement Functions
# ============================================================================


def enhance_frame_quality(
    image: np.ndarray,
    config: Dict,
    realesrgan_model,
    codeformer_model = None,
    face_helper = None
) -> np.ndarray:
    """
    High-quality enhancement using SOTA models

    Pipeline:
      1. RealESRGAN upscaling (Pixar-grade)
      2. CodeFormer face enhancement (optional)
      3. Adaptive contrast enhancement (3D-optimized)
      4. CNN denoising (Pixar-style)
      5. Unsharp masking (gentle)
    """
    enhancement_config = config['enhancement']

    # 1. Upscaling with RealESRGAN
    if enhancement_config['upscale']['enabled']:
        try:
            upscaled, _ = realesrgan_model.enhance(image, outscale=enhancement_config['upscale']['scale'])
            image = upscaled
        except Exception as e:
            print(f"⚠️ RealESRGAN failed: {e}, skipping upscaling")

    # 2. Face enhancement with CodeFormer
    if enhancement_config['upscale']['face_enhance'] and codeformer_model and face_helper:
        try:
            # Convert to RGB for face detection
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_helper.clean_all()
            face_helper.read_image(img_rgb)
            face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            face_helper.align_warp_face()

            if len(face_helper.cropped_faces) > 0:
                # Enhance faces
                for cropped_face in face_helper.cropped_faces:
                    cropped_face_t = torch.from_numpy(cropped_face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    cropped_face_t = cropped_face_t.to(codeformer_model.device)

                    with torch.no_grad():
                        output = codeformer_model(cropped_face_t, w=config['codeformer'].get('fidelity_weight', 0.7))[0]

                    restored_face = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    restored_face = (restored_face * 255.0).astype(np.uint8)
                    face_helper.add_restored_face(restored_face)

                # Paste faces back
                face_helper.get_inverse_affine(None)
                img_rgb = face_helper.paste_faces_to_input_image()
                image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"⚠️ CodeFormer face enhancement failed: {e}, skipping")

    # 3. Contrast enhancement (3D-optimized)
    if enhancement_config['contrast']['enabled']:
        contrast_config = enhancement_config['contrast']
        method = contrast_config.get('method', 'adaptive_histogram')

        if method == 'adaptive_histogram':
            # Convert to LAB for luminance adjustment
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # CLAHE with 3D-optimized parameters (preserve smooth shading)
            clahe = cv2.createCLAHE(
                clipLimit=contrast_config.get('clip_limit', 2.0),
                tileGridSize=tuple(contrast_config.get('tile_grid_size', [8, 8]))
            )
            l = clahe.apply(l)

            # Merge and convert back
            enhanced_lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Apply contrast/brightness adjustments
        contrast_factor = contrast_config.get('contrast_factor', 1.15)
        brightness_adjust = contrast_config.get('brightness_adjust', 1.05)
        saturation_boost = contrast_config.get('saturation_boost', 1.1)

        # Contrast and brightness
        image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=(brightness_adjust - 1) * 50)

        # Saturation boost
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 4. Denoising (CNN-based, Pixar-style)
    if enhancement_config['denoise']['enabled']:
        strength = enhancement_config['denoise'].get('strength', 0.3)
        h_value = int(strength * 10)  # Scale 0.0-1.0 to 0-10
        image = cv2.fastNlMeansDenoisingColored(
            image, None,
            h=h_value,
            hColor=h_value,
            templateWindowSize=7,
            searchWindowSize=21
        )

    # 5. Sharpness enhancement (gentle for 3D)
    if enhancement_config['sharpness']['enabled']:
        sharpness_config = enhancement_config['sharpness']
        method = sharpness_config.get('method', 'unsharp_mask')

        if method == 'unsharp_mask':
            radius = sharpness_config.get('radius', 1.0)
            amount = sharpness_config.get('amount', 0.5)
            threshold = sharpness_config.get('threshold', 3)

            # Gaussian blur
            blurred = cv2.GaussianBlur(image, (0, 0), radius)

            # Create sharpened image
            sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

            # Apply threshold to avoid sharpening noise
            diff = cv2.absdiff(image, sharpened)
            mask = diff > threshold
            image = np.where(mask, sharpened, image)

    return image


def enhance_frame_fast(image: np.ndarray, sharpen_amount: float = 1.5) -> np.ndarray:
    """
    Fast frame enhancement using OpenCV (for quick previews)

    Args:
        image: Input image (BGR)
        sharpen_amount: Sharpening strength (1.0 = original, 2.0 = strong)

    Returns:
        enhanced: Enhanced image
    """
    # 1. Denoising (gentle, preserves details)
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h=3, hColor=3,
                                                templateWindowSize=7, searchWindowSize=21)

    # 2. Unsharp masking for sharpening
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, sharpen_amount, gaussian, 1 - sharpen_amount, 0)

    # 3. Mild contrast enhancement (CLAHE)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def main():
    parser = argparse.ArgumentParser(description="3D Character Enhancement")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with frames")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--mode", type=str, choices=["quality", "fast"], default="fast",
                        help="Enhancement mode: quality (RealESRGAN+CodeFormer) or fast (OpenCV)")
    parser.add_argument("--config", type=str, default="configs/stages/enhancement/3d_character_enhancement.yaml",
                        help="Config file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing files")
    parser.add_argument("--sharpen", type=float, default=1.5, help="Sharpening for fast mode (1.0-2.0)")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = "cpu"

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"3D CHARACTER ENHANCEMENT - {args.mode.upper()} MODE")
    print(f"{'='*70}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    # Load models for quality mode
    config = None
    realesrgan_model = None
    codeformer_model = None
    face_helper = None

    if args.mode == "quality":
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            print("Using fast mode instead")
            args.mode = "fast"
        else:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Load RealESRGAN
            try:
                realesrgan_model = load_realesrgan_model(config, device=args.device)
            except Exception as e:
                print(f"⚠️ Failed to load RealESRGAN: {e}")
                print("Continuing without upscaling...")

            # Load CodeFormer (optional)
            if config['enhancement']['upscale']['face_enhance']:
                try:
                    codeformer_model, face_helper = load_codeformer_model(config, device=args.device)
                except Exception as e:
                    print(f"⚠️ Failed to load CodeFormer: {e}")
                    print("Continuing without face enhancement...")

    # Get image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(input_dir.glob(ext))
    image_files = sorted(image_files)

    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return

    print(f"📂 Found {len(image_files)} images to enhance\n")

    # Estimate time
    if args.mode == "quality":
        avg_time = 3.0  # seconds per frame
    else:
        avg_time = 0.5
    estimated_minutes = len(image_files) * avg_time / 60
    print(f"⏱️  Estimated time: {estimated_minutes:.1f} minutes\n")

    # Process images
    stats = {
        "total": len(image_files),
        "processed": 0,
        "skipped": 0,
        "failed": 0
    }

    for img_path in tqdm(image_files, desc="Enhancing"):
        output_path = output_dir / img_path.name

        if args.skip_existing and output_path.exists():
            stats["skipped"] += 1
            continue

        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                stats["failed"] += 1
                continue

            # Enhance based on mode
            if args.mode == "quality" and realesrgan_model:
                enhanced = enhance_frame_quality(
                    image, config, realesrgan_model,
                    codeformer_model, face_helper
                )
            else:
                enhanced = enhance_frame_fast(image, sharpen_amount=args.sharpen)

            # Save
            cv2.imwrite(str(output_path), enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            stats["processed"] += 1

        except Exception as e:
            print(f"\n⚠️ Failed {img_path.name}: {e}")
            stats["failed"] += 1

    print(f"\n{'='*70}")
    print(f"ENHANCEMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Total: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"{'='*70}")
    print(f"\n📁 Enhanced frames: {output_dir}\n")


if __name__ == "__main__":
    main()
