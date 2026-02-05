#!/usr/bin/env python3
"""
Organize training datasets into AI_WAREHOUSE 3.0 structure.

CPU-ONLY TOOL - No GPU required, uses 32-thread parallel processing.

This tool scans existing training data directories and reorganizes them
into the AI_WAREHOUSE 3.0 structure with proper character/project organization.

Usage:
    python scripts/utils/organize_training_datasets.py
    python scripts/utils/organize_training_datasets.py --source /path/to/data --target /mnt/data/training/lora
    python scripts/utils/organize_training_datasets.py --dry-run --verbose

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
import shutil
import time
import hashlib
from collections import defaultdict

# CPU-only imports
import yaml

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CharacterDataset:
    """Information about a character dataset."""
    name: str
    source_path: str
    image_count: int = 0
    caption_count: int = 0
    has_metadata: bool = False
    total_size_mb: float = 0.0
    image_formats: Set[str] = field(default_factory=set)
    issues: List[str] = field(default_factory=list)


@dataclass
class OrganizationResult:
    """Result of organizing a dataset."""
    character: str
    source: str
    target: str
    images_copied: int = 0
    captions_copied: int = 0
    errors: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def get_file_hash(filepath: Path, block_size: int = 65536) -> str:
    """Get MD5 hash of file for duplicate detection."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                hasher.update(block)
        return hasher.hexdigest()
    except Exception:
        return ""


def scan_character_directory(char_dir: Path) -> CharacterDataset:
    """
    Scan a single character directory for dataset information.

    Args:
        char_dir: Path to character directory

    Returns:
        CharacterDataset with scanned information
    """
    result = CharacterDataset(
        name=char_dir.name,
        source_path=str(char_dir)
    )

    if not char_dir.exists():
        result.issues.append("Directory does not exist")
        return result

    # Count images
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    for ext in image_extensions:
        images = list(char_dir.glob(f"*{ext}")) + list(char_dir.glob(f"*{ext.upper()}"))
        result.image_count += len(images)
        if images:
            result.image_formats.add(ext)

    # Count captions
    caption_files = list(char_dir.glob("*.txt"))
    result.caption_count = len(caption_files)

    # Check for metadata
    metadata_files = ['metadata.json', 'dataset.json', 'config.yaml', 'config.json']
    for mf in metadata_files:
        if (char_dir / mf).exists():
            result.has_metadata = True
            break

    # Calculate total size
    try:
        total_bytes = sum(f.stat().st_size for f in char_dir.rglob("*") if f.is_file())
        result.total_size_mb = total_bytes / (1024 * 1024)
    except Exception as e:
        result.issues.append(f"Could not calculate size: {str(e)[:50]}")

    # Validate structure
    if result.image_count == 0:
        result.issues.append("No images found")
    if result.caption_count == 0:
        result.issues.append("No caption files found")
    elif result.caption_count != result.image_count:
        result.issues.append(f"Image/caption mismatch: {result.image_count} images, {result.caption_count} captions")

    return result


def organize_character_dataset(
    source_dir: Path,
    target_dir: Path,
    character_name: str,
    dry_run: bool = False,
    overwrite: bool = False,
    verbose: bool = False
) -> OrganizationResult:
    """
    Organize a single character dataset into target structure.

    Args:
        source_dir: Source character directory
        target_dir: Target base directory
        character_name: Name to use for the character
        dry_run: If True, don't actually copy files
        overwrite: If True, overwrite existing files
        verbose: Print detailed information

    Returns:
        OrganizationResult with operation details
    """
    result = OrganizationResult(
        character=character_name,
        source=str(source_dir),
        target=str(target_dir / character_name)
    )

    if not source_dir.exists():
        result.skipped = True
        result.skip_reason = "Source directory does not exist"
        return result

    # Target directory
    char_target = target_dir / character_name

    if char_target.exists() and not overwrite:
        result.skipped = True
        result.skip_reason = "Target already exists (use --force to overwrite)"
        return result

    if dry_run:
        # Count files without copying
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        for ext in image_extensions:
            result.images_copied += len(list(source_dir.glob(f"*{ext}")))
            result.images_copied += len(list(source_dir.glob(f"*{ext.upper()}")))
        result.captions_copied = len(list(source_dir.glob("*.txt")))
        return result

    try:
        # Create target directory
        char_target.mkdir(parents=True, exist_ok=True)

        # Copy images
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        for ext in image_extensions:
            for img_file in source_dir.glob(f"*{ext}"):
                target_file = char_target / img_file.name
                if not target_file.exists() or overwrite:
                    shutil.copy2(img_file, target_file)
                    result.images_copied += 1
            for img_file in source_dir.glob(f"*{ext.upper()}"):
                target_file = char_target / img_file.name
                if not target_file.exists() or overwrite:
                    shutil.copy2(img_file, target_file)
                    result.images_copied += 1

        # Copy captions
        for caption_file in source_dir.glob("*.txt"):
            target_file = char_target / caption_file.name
            if not target_file.exists() or overwrite:
                shutil.copy2(caption_file, target_file)
                result.captions_copied += 1

        # Copy metadata files if they exist
        metadata_files = ['metadata.json', 'dataset.json', 'config.yaml', 'config.json']
        for mf in metadata_files:
            source_meta = source_dir / mf
            if source_meta.exists():
                shutil.copy2(source_meta, char_target / mf)

    except Exception as e:
        result.errors.append(f"Copy error: {str(e)[:100]}")

    return result


def discover_datasets(
    search_paths: List[Path],
    num_threads: int = 32,
    verbose: bool = False
) -> List[CharacterDataset]:
    """
    Discover all character datasets in search paths.

    Args:
        search_paths: List of paths to search
        num_threads: Number of threads for parallel scanning
        verbose: Print detailed information

    Returns:
        List of discovered CharacterDataset objects
    """
    all_char_dirs: List[Path] = []

    for search_path in search_paths:
        if not search_path.exists():
            if verbose:
                print(f"  Warning: Path does not exist: {search_path}")
            continue

        # Look for character directories (directories with images)
        for item in search_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it looks like a character directory
                has_images = any(
                    item.glob(f"*.{ext}")
                    for ext in ['png', 'jpg', 'jpeg', 'webp']
                )
                if has_images or any(item.iterdir()):
                    all_char_dirs.append(item)

    if verbose:
        print(f"  Found {len(all_char_dirs)} potential character directories")

    # Scan directories in parallel
    datasets: List[CharacterDataset] = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(scan_character_directory, d): d for d in all_char_dirs}

        for future in as_completed(futures):
            try:
                dataset = future.result()
                datasets.append(dataset)
                if verbose and dataset.image_count > 0:
                    print(f"    Scanned: {dataset.name} ({dataset.image_count} images)")
            except Exception as e:
                if verbose:
                    print(f"    Error scanning {futures[future]}: {e}")

    return datasets


def organize_all_datasets(
    source_paths: List[Path],
    target_base: Path,
    animation_type: str = "2d",
    num_threads: int = 32,
    dry_run: bool = False,
    overwrite: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Organize all discovered datasets into AI_WAREHOUSE 3.0 structure.

    Args:
        source_paths: Paths to search for datasets
        target_base: Base target directory (e.g., /mnt/data/training/lora)
        animation_type: "2d" or "3d"
        num_threads: Number of threads
        dry_run: Don't actually copy files
        overwrite: Overwrite existing files
        verbose: Print detailed information

    Returns:
        Summary dictionary
    """
    print(f"Discovering datasets from {len(source_paths)} source paths...")
    start_time = time.time()

    # Discover all datasets
    datasets = discover_datasets(source_paths, num_threads, verbose)

    if not datasets:
        return {"error": "No datasets found", "datasets_found": 0}

    print(f"Found {len(datasets)} character datasets")

    # Filter to only datasets with images
    valid_datasets = [d for d in datasets if d.image_count > 0]
    print(f"Datasets with images: {len(valid_datasets)}")

    # Target structure: /mnt/data/training/lora/{2d,3d}_characters/{character_name}
    type_dir = f"{animation_type}_characters"
    target_dir = target_base / type_dir

    if dry_run:
        print(f"\n[DRY RUN] Would organize to: {target_dir}")
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOrganizing to: {target_dir}")

    # Organize datasets
    results: List[OrganizationResult] = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for dataset in valid_datasets:
            source = Path(dataset.source_path)
            future = executor.submit(
                organize_character_dataset,
                source,
                target_dir,
                dataset.name,
                dry_run,
                overwrite,
                verbose
            )
            futures[future] = dataset

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if verbose:
                    status = "SKIPPED" if result.skipped else "OK"
                    print(f"  {status}: {result.character} ({result.images_copied} images)")
            except Exception as e:
                dataset = futures[future]
                if verbose:
                    print(f"  ERROR: {dataset.name}: {e}")

    elapsed = time.time() - start_time

    # Compile summary
    organized = [r for r in results if not r.skipped and not r.errors]
    skipped = [r for r in results if r.skipped]
    failed = [r for r in results if r.errors]

    summary = {
        "datasets_discovered": len(datasets),
        "datasets_valid": len(valid_datasets),
        "datasets_organized": len(organized),
        "datasets_skipped": len(skipped),
        "datasets_failed": len(failed),
        "total_images_copied": sum(r.images_copied for r in organized),
        "total_captions_copied": sum(r.captions_copied for r in organized),
        "target_directory": str(target_dir),
        "animation_type": animation_type,
        "dry_run": dry_run,
        "elapsed_seconds": round(elapsed, 2),
        "organized": [
            {
                "character": r.character,
                "images": r.images_copied,
                "captions": r.captions_copied
            }
            for r in organized
        ],
        "skipped": [
            {
                "character": r.character,
                "reason": r.skip_reason
            }
            for r in skipped
        ],
        "failed": [
            {
                "character": r.character,
                "errors": r.errors
            }
            for r in failed
        ]
    }

    return summary


def print_summary(summary: Dict) -> None:
    """Print organization summary."""
    if "error" in summary:
        print(f"\n{summary['error']}")
        return

    print("\n" + "=" * 70)
    print("Training Dataset Organization Summary")
    print("=" * 70)

    print(f"\nDatasets discovered: {summary['datasets_discovered']}")
    print(f"Datasets with images: {summary['datasets_valid']}")
    print(f"Datasets organized: {summary['datasets_organized']}")
    print(f"Datasets skipped: {summary['datasets_skipped']}")
    print(f"Datasets failed: {summary['datasets_failed']}")

    print(f"\nTotal images copied: {summary['total_images_copied']}")
    print(f"Total captions copied: {summary['total_captions_copied']}")
    print(f"Target directory: {summary['target_directory']}")
    print(f"Animation type: {summary['animation_type']}")
    print(f"Time elapsed: {summary['elapsed_seconds']}s")

    if summary['dry_run']:
        print("\n[DRY RUN - No files were actually copied]")

    # Show organized datasets (limited)
    if summary['organized']:
        print("\n" + "-" * 70)
        print("Organized datasets:")
        for i, d in enumerate(summary['organized'][:15]):
            print(f"  {d['character']}: {d['images']} images, {d['captions']} captions")
        if len(summary['organized']) > 15:
            print(f"  ... and {len(summary['organized']) - 15} more")

    # Show skipped
    if summary['skipped']:
        print("\n" + "-" * 70)
        print("Skipped datasets:")
        for d in summary['skipped'][:10]:
            print(f"  {d['character']}: {d['reason']}")

    # Show failures
    if summary['failed']:
        print("\n" + "-" * 70)
        print("Failed datasets:")
        for d in summary['failed']:
            print(f"  {d['character']}: {', '.join(d['errors'][:2])}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Organize training datasets into AI_WAREHOUSE 3.0 structure"
    )
    parser.add_argument(
        "--source",
        type=str,
        nargs="+",
        default=None,
        help="Source directories to scan (default: auto-detect from common locations)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="/mnt/data/training/lora",
        help="Target base directory (default: /mnt/data/training/lora)"
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
        help="Number of threads for parallel processing (default: 32)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually copy files, just show what would be done"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save summary to JSON file"
    )

    args = parser.parse_args()

    # Determine source paths
    if args.source:
        source_paths = [Path(p) for p in args.source]
    else:
        # Default search paths based on AI_WAREHOUSE 3.0 structure
        source_paths = [
            Path("/mnt/data/datasets/general"),
            Path("/mnt/data/ai_data/datasets/3d-anime"),  # Legacy path
            Path("/mnt/data/training/lora"),
        ]

        # Also check for training_data subdirectories in datasets
        datasets_dir = Path("/mnt/data/datasets/general")
        if datasets_dir.exists():
            for project_dir in datasets_dir.iterdir():
                if project_dir.is_dir():
                    training_data = project_dir / "lora_data" / "training_data"
                    if training_data.exists():
                        source_paths.append(training_data)

    print(f"Source paths: {len(source_paths)}")
    for sp in source_paths[:5]:
        print(f"  - {sp}")
    if len(source_paths) > 5:
        print(f"  ... and {len(source_paths) - 5} more")

    print(f"Target: {args.target}")
    print(f"Type: {args.type}")
    print(f"Threads: {args.threads}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be copied]")

    # Run organization
    summary = organize_all_datasets(
        source_paths,
        Path(args.target),
        args.type,
        args.threads,
        args.dry_run,
        args.force,
        args.verbose
    )

    # Print summary
    print_summary(summary)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
