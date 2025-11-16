#!/usr/bin/env python3
"""
Frame Deduplication Tool

Purpose: Remove near-duplicate frames to reduce dataset redundancy
Methods: pHash, SSIM, perceptual hashing, feature-based similarity
Use Cases: Preprocessing video frames before segmentation

Usage:
    python deduplicate_frames.py \
        --input-dir /path/to/frames \
        --output-dir /path/to/deduped \
        --method phash \
        --threshold 12
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import shutil


@dataclass
class DeduplicationConfig:
    """Configuration for frame deduplication"""
    method: str = "phash"  # phash, dhash, ahash, ssim, combined
    phash_threshold: int = 12  # Hamming distance threshold (lower = stricter)
    ssim_threshold: float = 0.92  # SSIM similarity threshold (higher = stricter)
    keep_mode: str = "first"  # first, best, random
    hash_size: int = 8  # Hash size for perceptual hashing
    create_hardlinks: bool = False  # Use hardlinks instead of copying
    save_duplicates_list: bool = True  # Save list of duplicates
    # NEW: Fast mode and temporal window
    mode: str = "balanced"  # fast, balanced, thorough
    temporal_window: Optional[int] = None  # Only compare frames within N positions (None = compare all)
    num_workers: int = 4  # Number of parallel workers for hash computation


class FrameDeduplicator:
    """Remove near-duplicate frames using various similarity metrics"""

    def __init__(self, config: DeduplicationConfig):
        """
        Initialize deduplicator

        Args:
            config: Deduplication configuration
        """
        self.config = config

    def compute_phash(self, image_path: Path) -> imagehash.ImageHash:
        """
        Compute perceptual hash (pHash)

        pHash is robust to minor variations like:
        - Slight color changes
        - Compression artifacts
        - Small geometric changes

        Args:
            image_path: Path to image

        Returns:
            Perceptual hash
        """
        with Image.open(image_path) as img:
            return imagehash.phash(img, hash_size=self.config.hash_size)

    def compute_dhash(self, image_path: Path) -> imagehash.ImageHash:
        """
        Compute difference hash (dHash)

        dHash is fast and good for detecting:
        - Exact duplicates
        - Near-exact duplicates

        Args:
            image_path: Path to image

        Returns:
            Difference hash
        """
        with Image.open(image_path) as img:
            return imagehash.dhash(img, hash_size=self.config.hash_size)

    def compute_ahash(self, image_path: Path) -> imagehash.ImageHash:
        """
        Compute average hash (aHash)

        aHash is very fast but less accurate

        Args:
            image_path: Path to image

        Returns:
            Average hash
        """
        with Image.open(image_path) as img:
            return imagehash.average_hash(img, hash_size=self.config.hash_size)

    def compute_ssim(self, img_path1: Path, img_path2: Path) -> float:
        """
        Compute structural similarity index (SSIM)

        SSIM is slower but more accurate for:
        - Detecting subtle differences
        - Comparing image quality

        Args:
            img_path1: First image path
            img_path2: Second image path

        Returns:
            SSIM score (0-1, higher = more similar)
        """
        import cv2

        # Load images as grayscale
        img1 = cv2.imread(str(img_path1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_path2), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return 0.0

        # Resize to same size if needed
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Compute SSIM
        score, _ = ssim(img1, img2, full=True)
        return float(score)

    def _compute_hash_worker(self, args: Tuple[Path, callable]) -> Tuple[Path, Optional[str]]:
        """
        Worker function for parallel hash computation

        Args:
            args: Tuple of (image_path, hash_function)

        Returns:
            Tuple of (image_path, hash_string or None)
        """
        img_path, hash_func = args
        try:
            img_hash = hash_func(img_path)
            return (img_path, str(img_hash))
        except Exception as e:
            print(f"⚠️ Error hashing {img_path.name}: {e}")
            return (img_path, None)

    def compute_hashes_parallel(
        self,
        image_files: List[Path],
        hash_func: callable
    ) -> Dict[Path, str]:
        """
        Compute hashes in parallel for speed

        Args:
            image_files: List of image paths
            hash_func: Hash function to use

        Returns:
            Dictionary mapping image path to hash string
        """
        from multiprocessing import Pool

        hash_map: Dict[Path, str] = {}
        args_list = [(img_path, hash_func) for img_path in image_files]

        with Pool(processes=self.config.num_workers) as pool:
            results = list(tqdm(
                pool.imap(self._compute_hash_worker, args_list),
                total=len(args_list),
                desc="Computing hashes (parallel)"
            ))

        for img_path, hash_str in results:
            if hash_str is not None:
                hash_map[img_path] = hash_str

        return hash_map

    def find_duplicates_hash(
        self,
        image_files: List[Path],
        hash_func: callable
    ) -> Dict[str, List[Path]]:
        """
        Find duplicates using hash-based method

        Args:
            image_files: List of image paths
            hash_func: Hash function to use

        Returns:
            Dictionary mapping representative hash to list of duplicate paths
        """
        print(f"\n🔍 Computing hashes for {len(image_files)} frames...")

        # Compute hashes (parallel if num_workers > 1)
        if self.config.num_workers > 1:
            path_to_hash = self.compute_hashes_parallel(image_files, hash_func)
        else:
            path_to_hash = {}
            for img_path in tqdm(image_files, desc="Computing hashes"):
                try:
                    img_hash = hash_func(img_path)
                    path_to_hash[img_path] = str(img_hash)
                except Exception as e:
                    print(f"⚠️ Error hashing {img_path.name}: {e}")
                    continue

        # Group by hash value
        hash_map: Dict[str, List[Path]] = {}
        for img_path, hash_str in path_to_hash.items():
            if hash_str not in hash_map:
                hash_map[hash_str] = []
            hash_map[hash_str].append(img_path)

        # Find groups with similar hashes (considering threshold and temporal window)
        print(f"\n🔍 Finding near-duplicates (threshold={self.config.phash_threshold})...")
        if self.config.temporal_window:
            print(f"   Using temporal window: {self.config.temporal_window} frames")

        duplicate_groups: Dict[str, List[Path]] = {}
        processed_hashes: Set[str] = set()

        hash_list = list(hash_map.keys())

        # Build index mapping for temporal window comparison
        if self.config.temporal_window:
            # Create frame index mapping based on filename sorting
            sorted_files = sorted(image_files)
            file_to_index = {f: i for i, f in enumerate(sorted_files)}

        for i, hash1_str in enumerate(tqdm(hash_list, desc="Comparing hashes")):
            if hash1_str in processed_hashes:
                continue

            hash1 = imagehash.hex_to_hash(hash1_str)
            group = hash_map[hash1_str].copy()

            # Determine comparison range based on temporal window
            if self.config.temporal_window:
                # Get indices of files with this hash
                file_indices = [file_to_index[f] for f in hash_map[hash1_str]]
                min_idx = min(file_indices) - self.config.temporal_window
                max_idx = max(file_indices) + self.config.temporal_window

                # Only compare hashes from files within temporal window
                comparison_hashes = []
                for hash2_str in hash_list[i + 1:]:
                    hash2_files = hash_map[hash2_str]
                    hash2_indices = [file_to_index[f] for f in hash2_files]
                    # Check if any file in this hash group is within window
                    if any(min_idx <= idx <= max_idx for idx in hash2_indices):
                        comparison_hashes.append(hash2_str)
            else:
                # Compare all remaining hashes
                comparison_hashes = hash_list[i + 1:]

            # Compare with hashes in range
            for hash2_str in comparison_hashes:
                if hash2_str in processed_hashes:
                    continue

                hash2 = imagehash.hex_to_hash(hash2_str)
                distance = hash1 - hash2

                if distance <= self.config.phash_threshold:
                    group.extend(hash_map[hash2_str])
                    processed_hashes.add(hash2_str)

            if len(group) > 1:
                duplicate_groups[hash1_str] = group

            processed_hashes.add(hash1_str)

        return duplicate_groups

    def find_duplicates_ssim(self, image_files: List[Path]) -> Dict[str, List[Path]]:
        """
        Find duplicates using SSIM method

        Args:
            image_files: List of image paths

        Returns:
            Dictionary mapping representative to list of duplicate paths
        """
        print(f"\n🔍 Computing SSIM for {len(image_files)} frames...")

        duplicate_groups: Dict[str, List[Path]] = {}
        processed: Set[Path] = set()

        for i, img1_path in enumerate(tqdm(image_files, desc="Comparing frames")):
            if img1_path in processed:
                continue

            group = [img1_path]

            # Compare with remaining images
            for img2_path in image_files[i + 1:]:
                if img2_path in processed:
                    continue

                similarity = self.compute_ssim(img1_path, img2_path)

                if similarity >= self.config.ssim_threshold:
                    group.append(img2_path)
                    processed.add(img2_path)

            if len(group) > 1:
                duplicate_groups[str(img1_path)] = group

            processed.add(img1_path)

        return duplicate_groups

    def select_representative(
        self,
        duplicate_group: List[Path]
    ) -> Tuple[Path, List[Path]]:
        """
        Select which frame to keep from duplicate group

        Args:
            duplicate_group: List of duplicate frame paths

        Returns:
            (representative_to_keep, duplicates_to_remove)
        """
        if self.config.keep_mode == "first":
            # Keep first (earliest in sequence)
            sorted_group = sorted(duplicate_group, key=lambda p: p.name)
            return sorted_group[0], sorted_group[1:]

        elif self.config.keep_mode == "best":
            # Keep highest quality (largest file size as proxy)
            best = max(duplicate_group, key=lambda p: p.stat().st_size)
            others = [p for p in duplicate_group if p != best]
            return best, others

        elif self.config.keep_mode == "random":
            # Keep random (for diversity)
            import random
            representative = random.choice(duplicate_group)
            others = [p for p in duplicate_group if p != representative]
            return representative, others

        else:
            # Default: first
            sorted_group = sorted(duplicate_group, key=lambda p: p.name)
            return sorted_group[0], sorted_group[1:]

    def deduplicate(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main deduplication pipeline

        Args:
            input_dir: Directory with input frames
            output_dir: Directory to save deduplicated frames

        Returns:
            Statistics dictionary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all image files
        image_files = sorted(
            list(input_dir.glob("*.jpg")) +
            list(input_dir.glob("*.png"))
        )

        print(f"\n📊 Found {len(image_files)} frames in {input_dir}")
        print(f"   Mode: {self.config.mode}")
        if self.config.temporal_window:
            print(f"   Temporal window: ±{self.config.temporal_window} frames")
        if self.config.num_workers > 1:
            print(f"   Parallel workers: {self.config.num_workers}")

        # Find duplicates based on method
        if self.config.method == "phash":
            duplicate_groups = self.find_duplicates_hash(image_files, self.compute_phash)
        elif self.config.method == "dhash":
            duplicate_groups = self.find_duplicates_hash(image_files, self.compute_dhash)
        elif self.config.method == "ahash":
            duplicate_groups = self.find_duplicates_hash(image_files, self.compute_ahash)
        elif self.config.method == "ssim":
            duplicate_groups = self.find_duplicates_ssim(image_files)
        else:
            print(f"❌ Unknown method: {self.config.method}")
            return {}

        # Process duplicates
        print(f"\n📊 Found {len(duplicate_groups)} duplicate groups")

        kept_files: List[Path] = []
        removed_files: List[Path] = []
        duplicate_map: Dict[str, List[str]] = {}

        for group in duplicate_groups.values():
            representative, duplicates = self.select_representative(group)
            kept_files.append(representative)
            removed_files.extend(duplicates)

            duplicate_map[representative.name] = [d.name for d in duplicates]

        # Copy/link unique files
        print(f"\n📁 Saving deduplicated frames to {output_dir}...")

        unique_files = [f for f in image_files if f not in removed_files]

        for src_path in tqdm(unique_files, desc="Saving frames"):
            dst_path = output_dir / src_path.name

            if self.config.create_hardlinks:
                try:
                    dst_path.hardlink_to(src_path)
                except:
                    shutil.copy2(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        # Save statistics and duplicate list
        stats = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "method": self.config.method,
            "threshold": self.config.phash_threshold if "hash" in self.config.method else self.config.ssim_threshold,
            "total_input_frames": len(image_files),
            "duplicate_groups_found": len(duplicate_groups),
            "duplicates_removed": len(removed_files),
            "unique_frames_kept": len(unique_files),
            "reduction_rate": len(removed_files) / len(image_files) if len(image_files) > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }

        # Save metadata
        metadata_path = output_dir / "deduplication_report.json"
        with open(metadata_path, 'w') as f:
            json.dump(stats, f, indent=2)

        if self.config.save_duplicates_list:
            duplicates_path = output_dir / "duplicates_mapping.json"
            with open(duplicates_path, 'w') as f:
                json.dump(duplicate_map, f, indent=2)

        print(f"\n✅ Deduplication complete!")
        print(f"   Input frames: {len(image_files)}")
        print(f"   Duplicate groups: {len(duplicate_groups)}")
        print(f"   Duplicates removed: {len(removed_files)}")
        print(f"   Unique frames kept: {len(unique_files)}")
        print(f"   Reduction: {stats['reduction_rate']:.1%}")
        print(f"\n📄 Report saved to: {metadata_path}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Remove near-duplicate frames (Film-Agnostic)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with input frames"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save deduplicated frames"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="phash",
        choices=["phash", "dhash", "ahash", "ssim"],
        help="Deduplication method (default: phash)"
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=12,
        help="pHash Hamming distance threshold (default: 12, lower=stricter)"
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.92,
        help="SSIM similarity threshold (default: 0.92, higher=stricter)"
    )
    parser.add_argument(
        "--keep-mode",
        type=str,
        default="first",
        choices=["first", "best", "random"],
        help="Which frame to keep from duplicates (default: first)"
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help="Hash size for perceptual hashing (default: 8)"
    )
    parser.add_argument(
        "--hardlinks",
        action="store_true",
        help="Use hardlinks instead of copying (saves space)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (for logging)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "thorough"],
        help="Processing mode: fast (aggressive filtering), balanced (default), thorough (conservative)"
    )
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=None,
        help="Only compare frames within N positions (for temporal locality, None=compare all)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for hash computation (default: 4)"
    )

    args = parser.parse_args()

    # Apply mode presets
    if args.mode == "fast":
        # Fast mode: aggressive thresholds, temporal window, parallel processing
        if args.phash_threshold == 12:  # If using default
            args.phash_threshold = 15  # More aggressive
        if args.temporal_window is None:
            args.temporal_window = 30  # Only compare nearby frames
        args.workers = max(args.workers, 6)  # Use more workers
    elif args.mode == "thorough":
        # Thorough mode: conservative thresholds, compare all frames
        if args.phash_threshold == 12:
            args.phash_threshold = 8  # More conservative
        args.temporal_window = None  # Compare all frames

    # Create config
    config = DeduplicationConfig(
        method=args.method,
        phash_threshold=args.phash_threshold,
        ssim_threshold=args.ssim_threshold,
        keep_mode=args.keep_mode,
        hash_size=args.hash_size,
        create_hardlinks=args.hardlinks,
        save_duplicates_list=True,
        mode=args.mode,
        temporal_window=args.temporal_window,
        num_workers=args.workers
    )

    # Run deduplication
    deduplicator = FrameDeduplicator(config)
    stats = deduplicator.deduplicate(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
