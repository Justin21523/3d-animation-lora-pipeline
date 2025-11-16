#!/usr/bin/env python3
"""
Cluster Organizer - Interactive cluster review and renaming

Helps organize clusters by:
1. Showing cluster previews (first few images)
2. Allowing renaming (character_0 → luca_main, etc.)
3. Merging similar clusters
4. Deleting unwanted clusters
5. Generating cluster summary report

Usage:
    python cluster_organizer.py \
      --cluster-dir /path/to/clustered \
      --output-dir /path/to/organized
"""

import argparse
import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, List
import json


def show_cluster_preview(cluster_dir: Path, num_images: int = 9):
    """Show preview of cluster (first N images)"""
    images = list(cluster_dir.glob("*.png"))[:num_images]

    if not images:
        print(f"  No images found in {cluster_dir.name}")
        return

    # Load and resize images
    previews = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is not None:
            # Resize to 200x200 for preview
            img_resized = cv2.resize(img, (200, 200))
            previews.append(img_resized)

    if not previews:
        return

    # Create grid (3x3)
    rows = []
    for i in range(0, len(previews), 3):
        row = previews[i:i+3]
        # Pad row if needed
        while len(row) < 3:
            row.append(np.zeros((200, 200, 3), dtype=np.uint8))
        rows.append(np.hstack(row))

    # Pad rows if needed
    while len(rows) < 3:
        rows.append(np.zeros((200, 600, 3), dtype=np.uint8))

    grid = np.vstack(rows)

    # Save preview
    preview_path = cluster_dir / "cluster_preview.jpg"
    cv2.imwrite(str(preview_path), grid)

    print(f"  Preview saved: {preview_path}")


def generate_cluster_summary(cluster_dir: Path) -> Dict:
    """Generate summary for a cluster"""
    images = list(cluster_dir.glob("*.png"))

    return {
        "name": cluster_dir.name,
        "count": len(images),
        "preview_available": (cluster_dir / "cluster_preview.jpg").exists()
    }


def list_clusters_with_stats(cluster_dir: Path):
    """List all clusters with statistics"""
    clusters = sorted([d for d in cluster_dir.iterdir() if d.is_dir() and d.name.startswith("character_")])

    print(f"\n{'='*70}")
    print(f"CLUSTER SUMMARY")
    print(f"{'='*70}")
    print(f"Found {len(clusters)} clusters\n")

    print(f"{'Cluster Name':<30} {'Image Count':>15}")
    print(f"{'-'*30} {'-'*15}")

    for cluster in clusters:
        count = len(list(cluster.glob("*.png")))
        print(f"{cluster.name:<30} {count:>15}")

    print(f"{'='*70}\n")


def rename_cluster(old_path: Path, new_name: str, output_dir: Path) -> Path:
    """Rename a cluster"""
    new_path = output_dir / new_name

    if new_path.exists():
        print(f"⚠️  Warning: {new_name} already exists!")
        return old_path

    # Copy to new location
    shutil.copytree(old_path, new_path)
    print(f"✓ Renamed: {old_path.name} → {new_name}")

    return new_path


def merge_clusters(source_paths: List[Path], target_name: str, output_dir: Path) -> Path:
    """Merge multiple clusters into one"""
    target_path = output_dir / target_name
    target_path.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for source_path in source_paths:
        images = list(source_path.glob("*.png"))
        for img in images:
            # Create unique name if collision
            target_file = target_path / f"{source_path.name}_{img.name}"
            shutil.copy2(img, target_file)
            total_copied += 1

    print(f"✓ Merged {len(source_paths)} clusters → {target_name} ({total_copied} images)")

    return target_path


def generate_previews_for_all(cluster_dir: Path, num_images: int = 9):
    """Generate previews for all clusters"""
    clusters = [d for d in cluster_dir.iterdir() if d.is_dir() and d.name.startswith("character_")]

    print(f"\nGenerating previews for {len(clusters)} clusters...")

    for cluster in clusters:
        print(f"Processing {cluster.name}...")
        show_cluster_preview(cluster, num_images)

    print(f"\n✓ All previews generated!\n")


def main():
    parser = argparse.ArgumentParser(description="Cluster Organizer")
    parser.add_argument(
        "--cluster-dir",
        type=str,
        required=True,
        help="Directory with clusters to organize"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["list", "preview", "rename", "merge"],
        default="list",
        help="Action to perform (default: list)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for organized clusters (for rename/merge)"
    )
    parser.add_argument(
        "--old-name",
        type=str,
        help="Old cluster name (for rename)"
    )
    parser.add_argument(
        "--new-name",
        type=str,
        help="New cluster name (for rename/merge)"
    )
    parser.add_argument(
        "--merge-sources",
        type=str,
        nargs="+",
        help="Source cluster names to merge (for merge)"
    )

    args = parser.parse_args()

    cluster_dir = Path(args.cluster_dir)

    if not cluster_dir.exists():
        print(f"❌ Cluster directory not found: {cluster_dir}")
        return

    if args.action == "list":
        list_clusters_with_stats(cluster_dir)

    elif args.action == "preview":
        generate_previews_for_all(cluster_dir)

    elif args.action == "rename":
        if not args.output_dir or not args.old_name or not args.new_name:
            print("❌ --output-dir, --old-name, and --new-name required for rename")
            return

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        old_path = cluster_dir / args.old_name
        if not old_path.exists():
            print(f"❌ Cluster not found: {args.old_name}")
            return

        rename_cluster(old_path, args.new_name, output_dir)

    elif args.action == "merge":
        if not args.output_dir or not args.merge_sources or not args.new_name:
            print("❌ --output-dir, --merge-sources, and --new-name required for merge")
            return

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        source_paths = [cluster_dir / name for name in args.merge_sources]

        # Check all sources exist
        for path in source_paths:
            if not path.exists():
                print(f"❌ Cluster not found: {path.name}")
                return

        merge_clusters(source_paths, args.new_name, output_dir)


if __name__ == "__main__":
    main()
