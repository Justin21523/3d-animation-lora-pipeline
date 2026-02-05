#!/usr/bin/env python3
"""
Score image quality without GPU models.

CPU-ONLY TOOL - No GPU required, uses 32-thread parallel processing.

This tool evaluates image quality using CPU-based methods:
- Blur detection (Laplacian variance)
- Resolution check
- Aspect ratio validation
- Duplicate detection (perceptual hash)
- Edge quality analysis
- Color distribution analysis

Usage:
    python scripts/generic/quality/image_quality_scorer.py --input-dir /path/to/images
    python scripts/generic/quality/image_quality_scorer.py --input-dir /path/to/images --threshold 50 --output report.json

Author: Justin Lu
Date: 2025-12-02
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import argparse
import json
import time
import hashlib
from collections import Counter, defaultdict
from datetime import datetime

# CPU-only imports
import numpy as np

try:
    from PIL import Image
    import imagehash
    PIL_AVAILABLE = True
    IMAGEHASH_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    IMAGEHASH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ImageQuality:
    """Quality assessment for a single image."""
    path: str
    filename: str
    width: int = 0
    height: int = 0
    format: str = ""
    size_kb: float = 0.0
    blur_score: float = 0.0  # Higher = sharper
    edge_density: float = 0.0
    aspect_ratio: float = 0.0
    color_variance: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    overall_score: float = 0.0  # 0-100
    phash: str = ""
    issues: List[str] = field(default_factory=list)
    passed: bool = True


@dataclass
class DirectoryStats:
    """Quality statistics for a directory."""
    path: str
    total_images: int = 0
    passed_images: int = 0
    failed_images: int = 0
    avg_score: float = 0.0
    avg_blur: float = 0.0
    avg_resolution: Tuple[int, int] = (0, 0)
    duplicate_count: int = 0
    issue_counts: Dict[str, int] = field(default_factory=dict)
    size_mb: float = 0.0


def compute_blur_score(image_array: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance.
    Higher values = sharper image.

    Args:
        image_array: Grayscale image as numpy array

    Returns:
        Blur score (variance of Laplacian)
    """
    if not CV2_AVAILABLE:
        return 0.0

    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    except Exception:
        return 0.0


def compute_edge_density(image_array: np.ndarray) -> float:
    """
    Compute edge density using Canny edge detection.

    Args:
        image_array: Image as numpy array

    Returns:
        Edge density (0-1)
    """
    if not CV2_AVAILABLE:
        return 0.0

    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        edges = cv2.Canny(gray, 100, 200)
        return float(np.count_nonzero(edges)) / edges.size
    except Exception:
        return 0.0


def compute_color_stats(image_array: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute color statistics.

    Returns:
        Tuple of (color_variance, brightness, contrast)
    """
    try:
        if len(image_array.shape) == 3:
            # Color variance across channels
            color_variance = float(np.std(image_array))
            # Brightness (mean of all pixels)
            brightness = float(np.mean(image_array))
            # Contrast (std of luminance)
            if image_array.shape[2] >= 3:
                luminance = 0.299 * image_array[:, :, 0] + 0.587 * image_array[:, :, 1] + 0.114 * image_array[:, :, 2]
            else:
                luminance = image_array[:, :, 0]
            contrast = float(np.std(luminance))
        else:
            color_variance = float(np.std(image_array))
            brightness = float(np.mean(image_array))
            contrast = color_variance

        return (color_variance, brightness, contrast)
    except Exception:
        return (0.0, 0.0, 0.0)


def compute_phash(image: Image.Image) -> str:
    """
    Compute perceptual hash for duplicate detection.

    Args:
        image: PIL Image

    Returns:
        Perceptual hash string
    """
    if not IMAGEHASH_AVAILABLE:
        return ""

    try:
        return str(imagehash.phash(image))
    except Exception:
        return ""


def assess_image_quality(
    image_path: Path,
    min_resolution: int = 256,
    max_resolution: int = 4096,
    blur_threshold: float = 100.0,
    min_edge_density: float = 0.01
) -> ImageQuality:
    """
    Assess quality of a single image.

    Args:
        image_path: Path to image file
        min_resolution: Minimum acceptable resolution (width or height)
        max_resolution: Maximum acceptable resolution
        blur_threshold: Minimum blur score (lower = more blur allowed)
        min_edge_density: Minimum edge density

    Returns:
        ImageQuality assessment
    """
    quality = ImageQuality(
        path=str(image_path),
        filename=image_path.name
    )

    if not PIL_AVAILABLE:
        quality.issues.append("PIL not available")
        quality.passed = False
        return quality

    try:
        # Load image
        with Image.open(image_path) as img:
            quality.width = img.width
            quality.height = img.height
            quality.format = img.format or image_path.suffix.lower()
            quality.size_kb = image_path.stat().st_size / 1024
            quality.aspect_ratio = img.width / img.height if img.height > 0 else 0

            # Compute perceptual hash
            quality.phash = compute_phash(img)

            # Convert to numpy for analysis
            if CV2_AVAILABLE:
                img_rgb = img.convert('RGB')
                img_array = np.array(img_rgb)

                # Blur detection
                quality.blur_score = compute_blur_score(img_array)

                # Edge density
                quality.edge_density = compute_edge_density(img_array)

                # Color stats
                color_var, brightness, contrast = compute_color_stats(img_array)
                quality.color_variance = color_var
                quality.brightness = brightness
                quality.contrast = contrast

        # Check for issues
        if quality.width < min_resolution or quality.height < min_resolution:
            quality.issues.append(f"resolution_too_low:{quality.width}x{quality.height}")

        if quality.width > max_resolution or quality.height > max_resolution:
            quality.issues.append(f"resolution_too_high:{quality.width}x{quality.height}")

        if quality.blur_score < blur_threshold and quality.blur_score > 0:
            quality.issues.append(f"too_blurry:{quality.blur_score:.1f}")

        if quality.edge_density < min_edge_density and quality.edge_density > 0:
            quality.issues.append(f"low_detail:{quality.edge_density:.3f}")

        if quality.aspect_ratio < 0.3 or quality.aspect_ratio > 3.0:
            quality.issues.append(f"extreme_aspect:{quality.aspect_ratio:.2f}")

        if quality.brightness < 20:
            quality.issues.append("too_dark")
        elif quality.brightness > 235:
            quality.issues.append("too_bright")

        if quality.contrast < 20:
            quality.issues.append("low_contrast")

        # Compute overall score (0-100)
        score = 100.0

        # Resolution score (prefer 512-1024)
        min_dim = min(quality.width, quality.height)
        if min_dim < 256:
            score -= 30
        elif min_dim < 512:
            score -= 10
        elif min_dim > 2048:
            score -= 5

        # Blur score (normalize to 0-30 points)
        if quality.blur_score > 0:
            blur_contrib = min(30, quality.blur_score / 100 * 30)
            score -= max(0, 30 - blur_contrib)

        # Edge density (0-10 points)
        if quality.edge_density > 0:
            edge_contrib = min(10, quality.edge_density * 100)
            score -= max(0, 10 - edge_contrib)

        # Contrast (0-10 points)
        if quality.contrast > 0:
            contrast_contrib = min(10, quality.contrast / 10)
            score -= max(0, 10 - contrast_contrib)

        # Issue penalties
        score -= len(quality.issues) * 10

        quality.overall_score = max(0, min(100, score))
        quality.passed = len(quality.issues) == 0 and quality.overall_score >= 50

    except Exception as e:
        quality.issues.append(f"read_error:{str(e)[:30]}")
        quality.passed = False

    return quality


def assess_directory(
    input_dir: Path,
    min_resolution: int = 256,
    max_resolution: int = 4096,
    blur_threshold: float = 100.0,
    num_threads: int = 32,
    verbose: bool = False
) -> Tuple[DirectoryStats, List[ImageQuality]]:
    """
    Assess quality of all images in a directory.

    Args:
        input_dir: Input directory
        min_resolution: Minimum resolution
        max_resolution: Maximum resolution
        blur_threshold: Blur threshold
        num_threads: Number of threads
        verbose: Print detailed output

    Returns:
        Tuple of (DirectoryStats, list of ImageQuality)
    """
    stats = DirectoryStats(path=str(input_dir))

    # Find images
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    image_files: List[Path] = []

    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    stats.total_images = len(image_files)

    if not image_files:
        return stats, []

    # Assess in parallel
    results: List[ImageQuality] = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                assess_image_quality,
                f,
                min_resolution,
                max_resolution,
                blur_threshold
            ): f
            for f in image_files
        }

        for future in as_completed(futures):
            try:
                quality = future.result()
                results.append(quality)
                if verbose:
                    status = "PASS" if quality.passed else "FAIL"
                    print(f"  {status}: {quality.filename} (score: {quality.overall_score:.0f})")
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {futures[future].name}: {e}")

    # Compile statistics
    if results:
        stats.passed_images = sum(1 for r in results if r.passed)
        stats.failed_images = stats.total_images - stats.passed_images
        stats.avg_score = sum(r.overall_score for r in results) / len(results)
        stats.avg_blur = sum(r.blur_score for r in results) / len(results)

        widths = [r.width for r in results if r.width > 0]
        heights = [r.height for r in results if r.height > 0]
        if widths and heights:
            stats.avg_resolution = (int(sum(widths) / len(widths)), int(sum(heights) / len(heights)))

        stats.size_mb = sum(r.size_kb for r in results) / 1024

        # Count issues
        issue_counter = Counter()
        for r in results:
            for issue in r.issues:
                # Group similar issues
                issue_type = issue.split(':')[0]
                issue_counter[issue_type] += 1
        stats.issue_counts = dict(issue_counter)

        # Detect duplicates
        phash_counts = Counter(r.phash for r in results if r.phash)
        stats.duplicate_count = sum(1 for h, c in phash_counts.items() if c > 1)

    return stats, results


def find_duplicates(results: List[ImageQuality], hamming_threshold: int = 8) -> List[List[str]]:
    """
    Find duplicate/similar images using perceptual hash.

    Args:
        results: List of ImageQuality assessments
        hamming_threshold: Max Hamming distance to consider duplicate

    Returns:
        List of duplicate groups (each group is list of file paths)
    """
    if not IMAGEHASH_AVAILABLE:
        return []

    # Group by exact hash first
    hash_groups: Dict[str, List[str]] = defaultdict(list)
    for r in results:
        if r.phash:
            hash_groups[r.phash].append(r.path)

    # Find exact duplicates
    duplicates = [paths for paths in hash_groups.values() if len(paths) > 1]

    # For near-duplicates, we'd need to compare hashes (expensive for large sets)
    # This is a simplified version that only finds exact matches

    return duplicates


def generate_report(
    input_dirs: List[Path],
    min_resolution: int = 256,
    max_resolution: int = 4096,
    blur_threshold: float = 100.0,
    num_threads: int = 32,
    verbose: bool = False
) -> Dict:
    """
    Generate quality report for multiple directories.

    Returns:
        Report dictionary
    """
    print(f"Assessing image quality in {len(input_dirs)} directories...")
    start_time = time.time()

    all_stats: List[DirectoryStats] = []
    all_results: List[ImageQuality] = []

    for input_dir in input_dirs:
        if verbose:
            print(f"\nProcessing: {input_dir}")

        stats, results = assess_directory(
            input_dir,
            min_resolution,
            max_resolution,
            blur_threshold,
            num_threads,
            verbose
        )
        all_stats.append(stats)
        all_results.extend(results)

    elapsed = time.time() - start_time

    # Global statistics
    total_images = sum(s.total_images for s in all_stats)
    total_passed = sum(s.passed_images for s in all_stats)
    total_failed = sum(s.failed_images for s in all_stats)

    # Find duplicates across all images
    duplicates = find_duplicates(all_results)

    # Aggregate issues
    global_issues = Counter()
    for s in all_stats:
        global_issues.update(s.issue_counts)

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "directories_scanned": len(input_dirs),
            "elapsed_seconds": round(elapsed, 2),
            "settings": {
                "min_resolution": min_resolution,
                "max_resolution": max_resolution,
                "blur_threshold": blur_threshold
            },
            "libraries": {
                "pil": PIL_AVAILABLE,
                "imagehash": IMAGEHASH_AVAILABLE,
                "cv2": CV2_AVAILABLE
            }
        },
        "summary": {
            "total_images": total_images,
            "passed": total_passed,
            "failed": total_failed,
            "pass_rate": round(total_passed / total_images * 100, 1) if total_images > 0 else 0,
            "avg_score": round(sum(r.overall_score for r in all_results) / len(all_results), 1) if all_results else 0,
            "duplicate_groups": len(duplicates),
            "duplicate_images": sum(len(g) for g in duplicates),
            "issue_counts": dict(global_issues)
        },
        "directories": {
            s.path: {
                "total": s.total_images,
                "passed": s.passed_images,
                "failed": s.failed_images,
                "avg_score": round(s.avg_score, 1),
                "avg_blur": round(s.avg_blur, 1),
                "avg_resolution": s.avg_resolution,
                "size_mb": round(s.size_mb, 2),
                "issues": s.issue_counts
            }
            for s in all_stats
        },
        "failed_images": [
            {
                "path": r.path,
                "score": round(r.overall_score, 1),
                "issues": r.issues
            }
            for r in sorted(all_results, key=lambda x: x.overall_score)[:100]
            if not r.passed
        ],
        "duplicates": [
            {"count": len(group), "files": group}
            for group in duplicates[:50]
        ]
    }

    return report


def print_summary(report: Dict) -> None:
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("Image Quality Assessment Summary")
    print("=" * 70)

    s = report['summary']
    print(f"\nTotal Images: {s['total_images']:,}")
    print(f"Passed: {s['passed']:,} ({s['pass_rate']:.1f}%)")
    print(f"Failed: {s['failed']:,}")
    print(f"Average Score: {s['avg_score']:.1f}/100")
    print(f"Duplicate Groups: {s['duplicate_groups']}")

    if s['issue_counts']:
        print("\n" + "-" * 70)
        print("Issues Found:")
        for issue, count in sorted(s['issue_counts'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {issue}: {count:,}")

    if report['failed_images']:
        print("\n" + "-" * 70)
        print("Worst Quality Images:")
        for img in report['failed_images'][:5]:
            print(f"  {Path(img['path']).name}: score {img['score']:.0f}")
            for issue in img['issues'][:2]:
                print(f"    - {issue}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Score image quality without GPU models"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process subdirectories recursively"
    )
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=256,
        help="Minimum acceptable resolution (default: 256)"
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=4096,
        help="Maximum acceptable resolution (default: 4096)"
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Minimum blur score (default: 100)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of threads (default: 32)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save JSON report to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        sys.exit(1)

    # Determine directories to process
    if args.recursive:
        input_dirs = [input_path] + [d for d in input_path.rglob("*") if d.is_dir()]
    else:
        input_dirs = [input_path]

    print(f"Input: {args.input_dir}")
    print(f"Directories to process: {len(input_dirs)}")
    print(f"Settings: min_res={args.min_resolution}, blur_thresh={args.blur_threshold}")

    # Check libraries
    if not CV2_AVAILABLE:
        print("Warning: OpenCV not available, blur detection disabled")
    if not IMAGEHASH_AVAILABLE:
        print("Warning: imagehash not available, duplicate detection disabled")

    # Generate report
    report = generate_report(
        input_dirs,
        args.min_resolution,
        args.max_resolution,
        args.blur_threshold,
        args.threads,
        args.verbose
    )

    # Print summary
    print_summary(report)

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
