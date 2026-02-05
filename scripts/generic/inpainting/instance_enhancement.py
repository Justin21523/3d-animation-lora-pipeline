#!/usr/bin/env python3
"""
Instance Enhancement - Improve quality of extracted character instances

Applies adaptive enhancements to character instances:
- Brightness/contrast correction (CLAHE)
- Sharpness enhancement
- Noise reduction
- Color balance

Usage:
    python scripts/generic/enhancement/instance_enhancement.py \\
        --input-dir /path/to/instances \\
        --output-dir /path/to/enhanced \\
        --sharpen 1.2 \\
        --denoise 5
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


class InstanceEnhancer:
    """Enhance extracted character instances for better training quality"""

    def __init__(
        self,
        sharpen_strength: float = 1.2,
        denoise_strength: int = 5,
        clahe_clip: float = 2.0,
        clahe_grid: int = 8,
    ):
        """
        Initialize enhancer

        Args:
            sharpen_strength: Sharpening intensity (1.0 = no change, >1.0 = sharper)
            denoise_strength: Denoising strength (0-20, 0 = off)
            clahe_clip: CLAHE clip limit for adaptive contrast
            clahe_grid: CLAHE grid size
        """
        self.sharpen_strength = sharpen_strength
        self.denoise_strength = denoise_strength
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid

        # Create CLAHE processor for adaptive contrast
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_grid, self.clahe_grid)
        )

    def enhance_instance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply enhancement pipeline to a single instance

        Args:
            image: Input image (BGR or BGRA)

        Returns:
            Enhanced image
        """
        # Extract alpha channel if present
        has_alpha = (image.shape[2] == 4)
        if has_alpha:
            alpha = image[:, :, 3]
            bgr = image[:, :, :3]
        else:
            bgr = image.copy()

        # 1. Denoise (preserve edges)
        if self.denoise_strength > 0:
            bgr = cv2.fastNlMeansDenoisingColored(
                bgr,
                None,
                self.denoise_strength,
                self.denoise_strength,
                7,
                21
            )

        # 2. Adaptive contrast enhancement (CLAHE)
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel only
        l_enhanced = self.clahe.apply(l)

        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 3. Sharpening (unsharp mask)
        if self.sharpen_strength > 1.0:
            gaussian = cv2.GaussianBlur(bgr, (0, 0), 2.0)
            bgr = cv2.addWeighted(
                bgr,
                self.sharpen_strength,
                gaussian,
                -(self.sharpen_strength - 1.0),
                0
            )

        # Recombine with alpha
        if has_alpha:
            enhanced = np.dstack([bgr, alpha])
        else:
            enhanced = bgr

        return enhanced

    def compute_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> dict:
        """Compute quality metrics for comparison"""
        # Extract BGR channels only
        if original.shape[2] == 4:
            orig_bgr = original[:, :, :3]
            enh_bgr = enhanced[:, :, :3]
        else:
            orig_bgr = original
            enh_bgr = enhanced

        # Convert to grayscale for metrics
        orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2GRAY)

        # Sharpness (Laplacian variance)
        orig_sharpness = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
        enh_sharpness = cv2.Laplacian(enh_gray, cv2.CV_64F).var()

        # Brightness (mean intensity)
        orig_brightness = orig_gray.mean()
        enh_brightness = enh_gray.mean()

        # Contrast (std deviation)
        orig_contrast = orig_gray.std()
        enh_contrast = enh_gray.std()

        return {
            "sharpness_before": float(orig_sharpness),
            "sharpness_after": float(enh_sharpness),
            "sharpness_gain": float(enh_sharpness / (orig_sharpness + 1e-6)),
            "brightness_before": float(orig_brightness),
            "brightness_after": float(enh_brightness),
            "contrast_before": float(orig_contrast),
            "contrast_after": float(enh_contrast),
        }


def process_cluster(
    cluster_dir: Path,
    output_dir: Path,
    enhancer: InstanceEnhancer,
    skip_existing: bool = True,
) -> dict:
    """
    Process all images in a cluster directory

    Args:
        cluster_dir: Input cluster directory
        output_dir: Output directory for enhanced images
        enhancer: InstanceEnhancer instance
        skip_existing: Skip if output already exists

    Returns:
        Processing statistics
    """
    # Find all images
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(cluster_dir.glob(ext))

    if len(image_paths) == 0:
        return {
            "cluster": cluster_dir.name,
            "total": 0,
            "processed": 0,
            "skipped": 0,
        }

    # Create output directory
    output_cluster_dir = output_dir / cluster_dir.name
    output_cluster_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    processed = 0
    skipped = 0
    metrics_list = []

    for img_path in tqdm(image_paths, desc=f"Processing {cluster_dir.name}"):
        output_path = output_cluster_dir / img_path.name

        # Skip if exists
        if skip_existing and output_path.exists():
            skipped += 1
            continue

        try:
            # Read image
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"⚠ Failed to read: {img_path.name}")
                continue

            # Enhance
            enhanced = enhancer.enhance_instance(image)

            # Compute metrics
            metrics = enhancer.compute_metrics(image, enhanced)
            metrics["filename"] = img_path.name
            metrics_list.append(metrics)

            # Save
            cv2.imwrite(str(output_path), enhanced)
            processed += 1

        except Exception as e:
            print(f"✗ Error processing {img_path.name}: {e}")
            continue

    # Compute aggregate metrics
    if metrics_list:
        avg_sharpness_gain = np.mean([m["sharpness_gain"] for m in metrics_list])
        avg_brightness_before = np.mean([m["brightness_before"] for m in metrics_list])
        avg_brightness_after = np.mean([m["brightness_after"] for m in metrics_list])
        avg_contrast_before = np.mean([m["contrast_before"] for m in metrics_list])
        avg_contrast_after = np.mean([m["contrast_after"] for m in metrics_list])
    else:
        avg_sharpness_gain = 0
        avg_brightness_before = 0
        avg_brightness_after = 0
        avg_contrast_before = 0
        avg_contrast_after = 0

    return {
        "cluster": cluster_dir.name,
        "total": len(image_paths),
        "processed": processed,
        "skipped": skipped,
        "avg_sharpness_gain": float(avg_sharpness_gain),
        "avg_brightness_before": float(avg_brightness_before),
        "avg_brightness_after": float(avg_brightness_after),
        "avg_contrast_before": float(avg_contrast_before),
        "avg_contrast_after": float(avg_contrast_after),
        "metrics": metrics_list[:10],  # Save first 10 for inspection
    }


def main():
    parser = argparse.ArgumentParser(
        description="Enhance extracted character instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing cluster subdirectories"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for enhanced instances"
    )

    parser.add_argument(
        "--sharpen",
        type=float,
        default=1.2,
        help="Sharpening strength (1.0 = no change, >1.0 = sharper)"
    )

    parser.add_argument(
        "--denoise",
        type=int,
        default=5,
        help="Denoising strength (0-20, 0 = off)"
    )

    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        help="CLAHE clip limit for adaptive contrast"
    )

    parser.add_argument(
        "--clahe-grid",
        type=int,
        default=8,
        help="CLAHE grid size"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already exist in output"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INSTANCE ENHANCEMENT")
    print("=" * 70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Sharpen: {args.sharpen}")
    print(f"Denoise: {args.denoise}")
    print(f"CLAHE clip: {args.clahe_clip}, grid: {args.clahe_grid}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 70)
    print()

    # Initialize enhancer
    enhancer = InstanceEnhancer(
        sharpen_strength=args.sharpen,
        denoise_strength=args.denoise,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
    )

    # Find all cluster directories
    SKIP_DIRS = {'noise', '__pycache__', '.git', '.DS_Store'}
    cluster_dirs = [
        d for d in args.input_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith('.')
    ]

    cluster_dirs = sorted(cluster_dirs, key=lambda x: x.name)

    print(f"Found {len(cluster_dirs)} clusters:")
    for d in cluster_dirs:
        count = len(list(d.glob("*.png")))
        print(f"   - {d.name}: {count} instances")
    print()

    # Process each cluster
    all_stats = []
    total_processed = 0
    total_skipped = 0

    for cluster_dir in cluster_dirs:
        print(f"\n{'='*60}")
        print(f"Processing {cluster_dir.name}")
        print(f"{'='*60}")

        stats = process_cluster(
            cluster_dir,
            args.output_dir,
            enhancer,
            skip_existing=args.skip_existing,
        )

        all_stats.append(stats)
        total_processed += stats["processed"]
        total_skipped += stats["skipped"]

        print(f"   Total: {stats['total']}, Processed: {stats['processed']}, Skipped: {stats['skipped']}")
        if stats['processed'] > 0:
            print(f"   Avg sharpness gain: {stats['avg_sharpness_gain']:.2f}x")
            print(f"   Brightness: {stats['avg_brightness_before']:.1f} → {stats['avg_brightness_after']:.1f}")
            print(f"   Contrast: {stats['avg_contrast_before']:.1f} → {stats['avg_contrast_after']:.1f}")

    # Save report
    report = {
        "parameters": {
            "sharpen": args.sharpen,
            "denoise": args.denoise,
            "clahe_clip": args.clahe_clip,
            "clahe_grid": args.clahe_grid,
        },
        "summary": {
            "total_clusters": len(cluster_dirs),
            "total_processed": total_processed,
            "total_skipped": total_skipped,
        },
        "clusters": all_stats,
    }

    report_path = args.output_dir / "enhancement_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("ENHANCEMENT COMPLETE")
    print("=" * 70)
    print(f"Total processed: {total_processed} images")
    print(f"Total skipped: {total_skipped} images")
    print(f"Report saved: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
