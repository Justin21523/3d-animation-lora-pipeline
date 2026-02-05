#!/usr/bin/env python3
"""
Generate comprehensive statistics for all training datasets.

CPU-ONLY TOOL - No GPU required, uses 32-thread parallel processing.

This tool scans training data directories and generates detailed statistics
including image counts, caption analysis, resolution distribution, and more.

Usage:
    python scripts/utils/dataset_statistics_reporter.py
    python scripts/utils/dataset_statistics_reporter.py --dataset-root /mnt/data/training/lora
    python scripts/utils/dataset_statistics_reporter.py --output report.json --markdown report.md

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
from collections import Counter, defaultdict
from datetime import datetime

# CPU-only imports - use PIL for image dimensions without GPU
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ImageStats:
    """Statistics for a single image."""
    path: str
    width: int = 0
    height: int = 0
    format: str = ""
    size_kb: float = 0.0
    has_caption: bool = False
    caption_tokens: int = 0


@dataclass
class CharacterStats:
    """Statistics for a single character dataset."""
    name: str
    path: str
    image_count: int = 0
    caption_count: int = 0
    total_size_mb: float = 0.0
    avg_resolution: Tuple[int, int] = (0, 0)
    min_resolution: Tuple[int, int] = (0, 0)
    max_resolution: Tuple[int, int] = (0, 0)
    avg_caption_tokens: float = 0.0
    min_caption_tokens: int = 0
    max_caption_tokens: int = 0
    image_formats: Dict[str, int] = field(default_factory=dict)
    resolution_distribution: Dict[str, int] = field(default_factory=dict)
    caption_word_freq: Dict[str, int] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


@dataclass
class GlobalStats:
    """Global statistics across all datasets."""
    total_characters: int = 0
    total_images: int = 0
    total_captions: int = 0
    total_size_gb: float = 0.0
    avg_images_per_character: float = 0.0
    characters_by_size: List[Tuple[str, int]] = field(default_factory=list)
    global_resolution_dist: Dict[str, int] = field(default_factory=dict)
    global_format_dist: Dict[str, int] = field(default_factory=dict)
    global_word_freq: Dict[str, int] = field(default_factory=dict)
    characters_with_issues: List[str] = field(default_factory=list)


def get_image_dimensions(image_path: Path) -> Tuple[int, int, str]:
    """
    Get image dimensions without loading entire image.

    Returns:
        Tuple of (width, height, format)
    """
    if not PIL_AVAILABLE:
        return (0, 0, image_path.suffix.lower())

    try:
        with Image.open(image_path) as img:
            return (img.width, img.height, img.format or image_path.suffix.lower())
    except Exception:
        return (0, 0, image_path.suffix.lower())


def count_caption_tokens(caption_path: Path) -> int:
    """
    Count tokens in a caption file (simple whitespace tokenization).

    Returns:
        Number of tokens
    """
    try:
        text = caption_path.read_text(encoding='utf-8', errors='ignore').strip()
        # Simple tokenization by whitespace and punctuation
        tokens = text.replace(',', ' ').replace('.', ' ').split()
        return len(tokens)
    except Exception:
        return 0


def get_caption_words(caption_path: Path) -> List[str]:
    """
    Get words from a caption file for frequency analysis.

    Returns:
        List of words (lowercase)
    """
    try:
        text = caption_path.read_text(encoding='utf-8', errors='ignore').strip().lower()
        # Remove common punctuation
        for char in '.,!?;:()[]{}"\'-':
            text = text.replace(char, ' ')
        words = text.split()
        # Filter very short words and common stopwords
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'and', 'or'}
        return [w for w in words if len(w) > 1 and w not in stopwords]
    except Exception:
        return []


def resolution_bucket(width: int, height: int) -> str:
    """
    Categorize resolution into buckets.

    Returns:
        Resolution category string
    """
    if width == 0 or height == 0:
        return "unknown"

    pixels = width * height

    if pixels >= 2073600:  # >= 1920x1080
        return "hd+"
    elif pixels >= 921600:  # >= 1280x720
        return "hd"
    elif pixels >= 409600:  # >= 640x640
        return "medium"
    elif pixels >= 102400:  # >= 320x320
        return "small"
    else:
        return "tiny"


def analyze_character_directory(char_dir: Path) -> CharacterStats:
    """
    Analyze a single character directory.

    Args:
        char_dir: Path to character directory

    Returns:
        CharacterStats with analysis results
    """
    stats = CharacterStats(
        name=char_dir.name,
        path=str(char_dir)
    )

    if not char_dir.exists():
        stats.issues.append("Directory does not exist")
        return stats

    # Find all images
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    image_files: List[Path] = []

    for ext in image_extensions:
        image_files.extend(char_dir.glob(f"*{ext}"))
        image_files.extend(char_dir.glob(f"*{ext.upper()}"))

    stats.image_count = len(image_files)

    if stats.image_count == 0:
        stats.issues.append("No images found")
        return stats

    # Analyze images
    widths = []
    heights = []
    format_counter = Counter()
    resolution_counter = Counter()
    total_bytes = 0

    for img_path in image_files:
        try:
            w, h, fmt = get_image_dimensions(img_path)
            if w > 0 and h > 0:
                widths.append(w)
                heights.append(h)
            format_counter[fmt.lower().lstrip('.')] += 1
            resolution_counter[resolution_bucket(w, h)] += 1
            total_bytes += img_path.stat().st_size
        except Exception:
            pass

    stats.total_size_mb = total_bytes / (1024 * 1024)
    stats.image_formats = dict(format_counter)
    stats.resolution_distribution = dict(resolution_counter)

    if widths:
        stats.avg_resolution = (int(sum(widths) / len(widths)), int(sum(heights) / len(heights)))
        stats.min_resolution = (min(widths), min(heights))
        stats.max_resolution = (max(widths), max(heights))

    # Analyze captions
    caption_files = list(char_dir.glob("*.txt"))
    stats.caption_count = len(caption_files)

    if caption_files:
        token_counts = []
        word_counter = Counter()

        for cap_path in caption_files:
            tokens = count_caption_tokens(cap_path)
            token_counts.append(tokens)
            word_counter.update(get_caption_words(cap_path))

        if token_counts:
            stats.avg_caption_tokens = sum(token_counts) / len(token_counts)
            stats.min_caption_tokens = min(token_counts)
            stats.max_caption_tokens = max(token_counts)

        # Top 50 words
        stats.caption_word_freq = dict(word_counter.most_common(50))

    # Check for issues
    if stats.caption_count == 0:
        stats.issues.append("No caption files")
    elif stats.caption_count != stats.image_count:
        stats.issues.append(f"Caption mismatch: {stats.image_count} images, {stats.caption_count} captions")

    if stats.avg_caption_tokens < 10:
        stats.issues.append("Captions may be too short (avg < 10 tokens)")
    elif stats.avg_caption_tokens > 77:
        stats.issues.append("Captions may be too long (avg > 77 tokens)")

    if stats.image_count < 50:
        stats.issues.append(f"Small dataset ({stats.image_count} images)")

    return stats


def generate_report(
    dataset_root: Path,
    animation_type: str = "2d",
    num_threads: int = 32,
    verbose: bool = False
) -> Dict:
    """
    Generate comprehensive statistics report for all datasets.

    Args:
        dataset_root: Root directory for training data
        animation_type: "2d" or "3d"
        num_threads: Number of threads
        verbose: Print detailed output

    Returns:
        Report dictionary
    """
    print(f"Scanning datasets in: {dataset_root}")
    start_time = time.time()

    # Find character directories
    type_dir = dataset_root / f"{animation_type}_characters"
    if not type_dir.exists():
        # Try direct subdirectories
        type_dir = dataset_root

    char_dirs = [d for d in type_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    print(f"Found {len(char_dirs)} character directories")

    # Analyze in parallel
    all_stats: List[CharacterStats] = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(analyze_character_directory, d): d for d in char_dirs}

        for future in as_completed(futures):
            try:
                stats = future.result()
                all_stats.append(stats)
                if verbose:
                    print(f"  Analyzed: {stats.name} ({stats.image_count} images)")
            except Exception as e:
                if verbose:
                    print(f"  Error: {futures[future].name}: {e}")

    # Compile global statistics
    global_stats = GlobalStats()
    global_stats.total_characters = len(all_stats)
    global_stats.total_images = sum(s.image_count for s in all_stats)
    global_stats.total_captions = sum(s.caption_count for s in all_stats)
    global_stats.total_size_gb = sum(s.total_size_mb for s in all_stats) / 1024

    if global_stats.total_characters > 0:
        global_stats.avg_images_per_character = global_stats.total_images / global_stats.total_characters

    # Sort by image count
    global_stats.characters_by_size = sorted(
        [(s.name, s.image_count) for s in all_stats],
        key=lambda x: x[1],
        reverse=True
    )

    # Aggregate distributions
    resolution_agg = Counter()
    format_agg = Counter()
    word_agg = Counter()

    for stats in all_stats:
        resolution_agg.update(stats.resolution_distribution)
        format_agg.update(stats.image_formats)
        word_agg.update(stats.caption_word_freq)

        if stats.issues:
            global_stats.characters_with_issues.append(stats.name)

    global_stats.global_resolution_dist = dict(resolution_agg)
    global_stats.global_format_dist = dict(format_agg)
    global_stats.global_word_freq = dict(word_agg.most_common(100))

    elapsed = time.time() - start_time

    # Build report
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "dataset_root": str(dataset_root),
            "animation_type": animation_type,
            "elapsed_seconds": round(elapsed, 2),
            "pil_available": PIL_AVAILABLE
        },
        "global": {
            "total_characters": global_stats.total_characters,
            "total_images": global_stats.total_images,
            "total_captions": global_stats.total_captions,
            "total_size_gb": round(global_stats.total_size_gb, 2),
            "avg_images_per_character": round(global_stats.avg_images_per_character, 1),
            "characters_with_issues": len(global_stats.characters_with_issues),
            "resolution_distribution": global_stats.global_resolution_dist,
            "format_distribution": global_stats.global_format_dist,
            "top_caption_words": global_stats.global_word_freq
        },
        "characters": {
            s.name: {
                "path": s.path,
                "image_count": s.image_count,
                "caption_count": s.caption_count,
                "total_size_mb": round(s.total_size_mb, 2),
                "avg_resolution": s.avg_resolution,
                "avg_caption_tokens": round(s.avg_caption_tokens, 1),
                "image_formats": s.image_formats,
                "issues": s.issues
            }
            for s in sorted(all_stats, key=lambda x: x.image_count, reverse=True)
        },
        "rankings": {
            "by_image_count": global_stats.characters_by_size[:20],
            "with_issues": global_stats.characters_with_issues
        }
    }

    return report


def generate_markdown_report(report: Dict) -> str:
    """
    Generate Markdown summary from report.

    Returns:
        Markdown string
    """
    md = []
    md.append("# Training Dataset Statistics Report")
    md.append("")
    md.append(f"**Generated:** {report['metadata']['generated_at']}")
    md.append(f"**Dataset Root:** `{report['metadata']['dataset_root']}`")
    md.append(f"**Animation Type:** {report['metadata']['animation_type']}")
    md.append("")

    # Global stats
    md.append("## Global Statistics")
    md.append("")
    g = report['global']
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Total Characters | {g['total_characters']} |")
    md.append(f"| Total Images | {g['total_images']:,} |")
    md.append(f"| Total Captions | {g['total_captions']:,} |")
    md.append(f"| Total Size | {g['total_size_gb']:.2f} GB |")
    md.append(f"| Avg Images/Character | {g['avg_images_per_character']:.1f} |")
    md.append(f"| Characters with Issues | {g['characters_with_issues']} |")
    md.append("")

    # Resolution distribution
    md.append("## Resolution Distribution")
    md.append("")
    md.append("| Category | Count |")
    md.append("|----------|-------|")
    for cat, count in sorted(g['resolution_distribution'].items(), key=lambda x: x[1], reverse=True):
        md.append(f"| {cat} | {count:,} |")
    md.append("")

    # Format distribution
    md.append("## Image Format Distribution")
    md.append("")
    md.append("| Format | Count |")
    md.append("|--------|-------|")
    for fmt, count in sorted(g['format_distribution'].items(), key=lambda x: x[1], reverse=True):
        md.append(f"| {fmt} | {count:,} |")
    md.append("")

    # Top characters
    md.append("## Top Characters by Image Count")
    md.append("")
    md.append("| Rank | Character | Images | Captions | Size (MB) |")
    md.append("|------|-----------|--------|----------|-----------|")
    for i, (name, count) in enumerate(report['rankings']['by_image_count'][:20], 1):
        char = report['characters'].get(name, {})
        captions = char.get('caption_count', 0)
        size = char.get('total_size_mb', 0)
        md.append(f"| {i} | {name} | {count:,} | {captions:,} | {size:.1f} |")
    md.append("")

    # Issues
    if report['rankings']['with_issues']:
        md.append("## Characters with Issues")
        md.append("")
        for name in report['rankings']['with_issues'][:10]:
            char = report['characters'].get(name, {})
            issues = char.get('issues', [])
            md.append(f"### {name}")
            for issue in issues:
                md.append(f"- {issue}")
            md.append("")

    # Top caption words
    md.append("## Top Caption Words")
    md.append("")
    md.append("| Word | Frequency |")
    md.append("|------|-----------|")
    for word, freq in list(g['top_caption_words'].items())[:30]:
        md.append(f"| {word} | {freq:,} |")
    md.append("")

    md.append("---")
    md.append(f"*Report generated in {report['metadata']['elapsed_seconds']}s*")

    return "\n".join(md)


def print_summary(report: Dict) -> None:
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("Training Dataset Statistics Summary")
    print("=" * 70)

    g = report['global']
    print(f"\nTotal Characters: {g['total_characters']}")
    print(f"Total Images: {g['total_images']:,}")
    print(f"Total Captions: {g['total_captions']:,}")
    print(f"Total Size: {g['total_size_gb']:.2f} GB")
    print(f"Avg Images/Character: {g['avg_images_per_character']:.1f}")

    print("\n" + "-" * 70)
    print("Top 10 Characters by Image Count:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Character':<30} {'Images':>10} {'Size (MB)':>10}")
    print("-" * 56)

    for i, (name, count) in enumerate(report['rankings']['by_image_count'][:10], 1):
        char = report['characters'].get(name, {})
        size = char.get('total_size_mb', 0)
        print(f"{i:<6} {name:<30} {count:>10,} {size:>10.1f}")

    if g['characters_with_issues'] > 0:
        print(f"\nCharacters with issues: {g['characters_with_issues']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive statistics for training datasets"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/mnt/data/training/lora",
        help="Root directory for training data"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["2d", "3d"],
        default="2d",
        help="Animation type (default: 2d)"
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
        "--markdown",
        type=str,
        help="Save Markdown report to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Generate report
    report = generate_report(
        Path(args.dataset_root),
        args.type,
        args.threads,
        args.verbose
    )

    # Print summary
    print_summary(report)

    # Save JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report saved to: {output_path}")

    # Save Markdown if requested
    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = generate_markdown_report(report)
        with open(md_path, 'w') as f:
            f.write(md_content)
        print(f"Markdown report saved to: {md_path}")


if __name__ == "__main__":
    main()
