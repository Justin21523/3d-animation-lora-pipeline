#!/usr/bin/env python3
"""
Validate dataset directory structures against expected AI_WAREHOUSE 3.0 layout.

CPU-ONLY TOOL - No GPU required, uses 32-thread parallel processing.

This tool validates that dataset directories follow the expected structure
for the 2D animation LoRA pipeline.

Usage:
    python scripts/utils/validate_dataset_structure.py
    python scripts/utils/validate_dataset_structure.py --dataset-root /mnt/data/datasets/general
    python scripts/utils/validate_dataset_structure.py --project simpsons

Author: Justin Lu
Date: 2025-12-02
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
import argparse
import json
import time

# CPU-only - no torch/cuda imports


@dataclass
class ProjectValidation:
    """Validation result for a single project."""
    project_name: str
    project_path: str
    has_frames: bool = False
    has_segmented: bool = False
    has_clustered: bool = False
    has_training_data: bool = False
    has_metadata: bool = False
    frame_count: int = 0
    segmented_count: int = 0
    clustered_characters: List[str] = field(default_factory=list)
    training_characters: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if project has minimum required structure."""
        return len(self.issues) == 0


# Expected subdirectories for a project
EXPECTED_SUBDIRS = {
    "frames": "Extracted video frames",
    "segmented": "Segmented character instances",
    "clustered": "Identity-clustered characters",
}

# Optional but recommended subdirectories
OPTIONAL_SUBDIRS = {
    "training_data": "Prepared training datasets",
    "captions": "Generated captions",
    "metadata": "JSON metadata files",
    "lora_data": "LoRA-specific organized data",
}

# Image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def count_images(directory: Path) -> int:
    """Count image files in directory (recursive)."""
    if not directory.exists():
        return 0

    count = 0
    for ext in IMAGE_EXTENSIONS:
        count += len(list(directory.rglob(f"*{ext}")))
    return count


def count_images_flat(directory: Path) -> int:
    """Count image files in directory (non-recursive)."""
    if not directory.exists():
        return 0

    count = 0
    for ext in IMAGE_EXTENSIONS:
        count += len(list(directory.glob(f"*{ext}")))
    return count


def list_subdirectories(directory: Path) -> List[str]:
    """List subdirectory names."""
    if not directory.exists():
        return []
    return [d.name for d in directory.iterdir() if d.is_dir()]


def validate_project(project_dir: Path) -> ProjectValidation:
    """
    Validate a single project directory structure.

    Args:
        project_dir: Project directory path

    Returns:
        ProjectValidation result
    """
    result = ProjectValidation(
        project_name=project_dir.name,
        project_path=str(project_dir)
    )

    # Check frames directory
    frames_dir = project_dir / "frames"
    result.has_frames = frames_dir.exists()
    if result.has_frames:
        result.frame_count = count_images(frames_dir)
        if result.frame_count == 0:
            result.warnings.append("frames/ directory exists but is empty")
    else:
        result.issues.append("Missing frames/ directory")

    # Check segmented directory
    segmented_dir = project_dir / "segmented"
    if not segmented_dir.exists():
        # Check alternative: characters directory
        segmented_dir = project_dir / "characters"

    result.has_segmented = segmented_dir.exists()
    if result.has_segmented:
        result.segmented_count = count_images(segmented_dir)
        if result.segmented_count == 0:
            result.warnings.append(f"{segmented_dir.name}/ directory exists but is empty")

    # Check clustered directory
    clustered_dir = project_dir / "clustered"
    if not clustered_dir.exists():
        # Check alternative: character_clusters
        clustered_dir = project_dir / "character_clusters"

    result.has_clustered = clustered_dir.exists()
    if result.has_clustered:
        # List character subdirectories
        result.clustered_characters = [
            d for d in list_subdirectories(clustered_dir)
            if not d.startswith(("noise", "unknown", "."))
        ]

    # Check training_data directory
    training_dir = project_dir / "training_data"
    if not training_dir.exists():
        training_dir = project_dir / "lora_data" / "training_data"

    result.has_training_data = training_dir.exists()
    if result.has_training_data:
        result.training_characters = list_subdirectories(training_dir)

    # Check metadata
    metadata_dir = project_dir / "metadata"
    result.has_metadata = metadata_dir.exists()

    # Additional validation
    if result.has_frames and result.frame_count < 100:
        result.warnings.append(f"Low frame count: {result.frame_count} (recommend 500+)")

    if result.has_clustered and len(result.clustered_characters) == 0:
        result.warnings.append("clustered/ exists but no character directories found")

    # Check for lora_data structure (newer format)
    lora_data_dir = project_dir / "lora_data"
    if lora_data_dir.exists():
        # Check for expected lora_data subdirectories
        lora_subdirs = list_subdirectories(lora_data_dir)
        if "characters_inpainted" in lora_subdirs:
            result.warnings.append("Using newer lora_data structure (good!)")

    return result


def validate_all_projects(
    dataset_root: Path,
    num_threads: int = 32,
    verbose: bool = False,
    project_filter: Optional[str] = None
) -> Dict:
    """
    Validate all project directories in dataset root.

    Args:
        dataset_root: Root directory containing projects
        num_threads: Number of threads for parallel processing
        verbose: Print detailed output
        project_filter: Only validate specific project

    Returns:
        Validation summary dictionary
    """
    # Find project directories
    if project_filter:
        project_dirs = [dataset_root / project_filter]
        if not project_dirs[0].exists():
            print(f"Error: Project not found: {project_dirs[0]}")
            sys.exit(1)
    else:
        project_dirs = [
            d for d in dataset_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    print(f"Validating {len(project_dirs)} project directories...")
    start_time = time.time()

    results: List[ProjectValidation] = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(validate_project, d): d for d in project_dirs}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if verbose:
                status = "✅" if result.is_valid else "❌"
                print(f"  {status} {result.project_name}")

    elapsed = time.time() - start_time

    # Compile summary
    total = len(results)
    valid = sum(1 for r in results if r.is_valid)
    with_frames = sum(1 for r in results if r.has_frames)
    with_segmented = sum(1 for r in results if r.has_segmented)
    with_clustered = sum(1 for r in results if r.has_clustered)
    with_training = sum(1 for r in results if r.has_training_data)
    total_frames = sum(r.frame_count for r in results)
    total_characters = sum(len(r.clustered_characters) for r in results)

    summary = {
        "total_projects": total,
        "valid_projects": valid,
        "projects_with_frames": with_frames,
        "projects_with_segmented": with_segmented,
        "projects_with_clustered": with_clustered,
        "projects_with_training_data": with_training,
        "total_frames": total_frames,
        "total_clustered_characters": total_characters,
        "elapsed_seconds": round(elapsed, 2),
        "projects": [
            {
                "name": r.project_name,
                "path": r.project_path,
                "valid": r.is_valid,
                "has_frames": r.has_frames,
                "has_segmented": r.has_segmented,
                "has_clustered": r.has_clustered,
                "has_training_data": r.has_training_data,
                "frame_count": r.frame_count,
                "clustered_characters": r.clustered_characters,
                "training_characters": r.training_characters,
                "issues": r.issues,
                "warnings": r.warnings
            }
            for r in sorted(results, key=lambda x: x.project_name)
        ]
    }

    return summary


def print_summary(summary: Dict, verbose: bool = False) -> None:
    """Print validation summary."""
    print("\n" + "=" * 80)
    print("Dataset Structure Validation Summary")
    print("=" * 80)

    # Summary table
    print(f"\n{'Metric':<35} {'Count':>10}")
    print("-" * 45)
    print(f"{'Total projects':<35} {summary['total_projects']:>10}")
    print(f"{'Valid projects':<35} {summary['valid_projects']:>10}")
    print(f"{'With frames/':<35} {summary['projects_with_frames']:>10}")
    print(f"{'With segmented/':<35} {summary['projects_with_segmented']:>10}")
    print(f"{'With clustered/':<35} {summary['projects_with_clustered']:>10}")
    print(f"{'With training_data/':<35} {summary['projects_with_training_data']:>10}")
    print("-" * 45)
    print(f"{'Total frames':<35} {summary['total_frames']:>10}")
    print(f"{'Total clustered characters':<35} {summary['total_clustered_characters']:>10}")
    print(f"{'Time elapsed':<35} {summary['elapsed_seconds']:>10.2f}s")

    # Per-project table
    print("\n" + "-" * 80)
    print(f"{'Project':<25} {'Frames':<10} {'Seg':<6} {'Clust':<6} {'Train':<6} {'Chars':<10}")
    print("-" * 80)

    for p in summary['projects']:
        frames_str = str(p['frame_count']) if p['has_frames'] else "-"
        seg = "✅" if p['has_segmented'] else "❌"
        clust = "✅" if p['has_clustered'] else "❌"
        train = "✅" if p['has_training_data'] else "❌"
        chars = len(p['clustered_characters']) if p['clustered_characters'] else 0

        print(f"{p['name']:<25} {frames_str:<10} {seg:<6} {clust:<6} {train:<6} {chars:<10}")

    # Issues and warnings
    has_issues = any(p['issues'] or p['warnings'] for p in summary['projects'])
    if has_issues or verbose:
        print("\n" + "-" * 80)
        print("Issues and Warnings:")
        print("-" * 80)

        for p in summary['projects']:
            if p['issues'] or p['warnings']:
                print(f"\n{p['name']}:")
                for issue in p['issues']:
                    print(f"  ❌ {issue}")
                for warning in p['warnings']:
                    print(f"  ⚠️  {warning}")

    print("\n" + "=" * 80)
    if summary['valid_projects'] == summary['total_projects']:
        print("✅ All projects have valid structure!")
    else:
        invalid = summary['total_projects'] - summary['valid_projects']
        print(f"⚠️  {invalid} project(s) have structural issues")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset directory structures"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/mnt/data/datasets/general",
        help="Root directory containing project datasets"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Validate only specific project"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of threads for parallel processing (default: 32)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save validation report to JSON file"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    if not dataset_root.exists():
        print(f"Error: Dataset root not found: {dataset_root}")
        sys.exit(1)

    print(f"Dataset root: {dataset_root}")
    print(f"Using {args.threads} threads")

    # Run validation
    summary = validate_all_projects(
        dataset_root,
        args.threads,
        args.verbose,
        args.project
    )

    # Print summary
    print_summary(summary, args.verbose)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
