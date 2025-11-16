#!/usr/bin/env python3
"""
Temporal Context Extractor

Purpose: Extract relevant context from temporal window for inpainting
Features: Find similar frames, extract non-occluded patches, temporal consistency
Use Cases: Provide reference context for character inpainting

Usage:
    python temporal_context_extractor.py \
        --frames-dir /path/to/frames \
        --instance-path /path/to/instance.png \
        --output-dir /path/to/context \
        --window-size 10 \
        --max-references 3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import shutil


@dataclass
class ContextExtractionConfig:
    """Configuration for temporal context extraction"""
    window_size: int = 10  # Frames before/after to search
    max_references: int = 3  # Maximum reference frames to extract
    similarity_threshold: float = 0.7  # SSIM threshold for similar frames
    feature_type: str = "combined"  # histogram, edge, combined
    extract_patches: bool = True  # Extract relevant patches
    patch_size: int = 128  # Patch size for extraction
    check_occlusion: bool = True  # Check for occlusions
    save_visualization: bool = True  # Save debug visualizations


class TemporalContextExtractor:
    """Extract temporal context for inpainting"""

    def __init__(self, config: ContextExtractionConfig):
        """
        Initialize context extractor

        Args:
            config: Extraction configuration
        """
        self.config = config

    def parse_instance_filename(self, instance_path: Path) -> Optional[Tuple[str, int]]:
        """
        Parse instance filename to get source frame info

        Expected format: frame_XXXX_instance_Y.png

        Args:
            instance_path: Path to instance image

        Returns:
            (frame_name, frame_number) or None
        """
        filename = instance_path.stem

        # Try to parse frame_XXXX_instance_Y format
        parts = filename.split('_')

        for i, part in enumerate(parts):
            if part == 'frame' and i + 1 < len(parts):
                try:
                    frame_num = int(parts[i + 1])
                    frame_name = f"frame_{parts[i + 1]}"
                    return frame_name, frame_num
                except ValueError:
                    continue

        # Fallback: try to find any number sequence
        import re
        numbers = re.findall(r'\d+', filename)
        if numbers:
            frame_num = int(numbers[0])
            return filename, frame_num

        return None

    def find_temporal_window(
        self,
        frames_dir: Path,
        frame_number: int
    ) -> List[Path]:
        """
        Find frames in temporal window

        Args:
            frames_dir: Directory with all frames
            frame_number: Center frame number

        Returns:
            List of frame paths in window
        """
        frames_dir = Path(frames_dir)

        # Get all frames
        all_frames = sorted(
            list(frames_dir.glob("frame_*.jpg")) +
            list(frames_dir.glob("frame_*.png"))
        )

        if not all_frames:
            print(f"⚠️ No frames found in {frames_dir}")
            return []

        # Find frames in window
        window_frames = []

        for frame_path in all_frames:
            # Parse frame number
            try:
                fname = frame_path.stem
                fnum = int(fname.split('_')[1])

                # Check if in window
                if abs(fnum - frame_number) <= self.config.window_size:
                    if fnum != frame_number:  # Exclude the center frame itself
                        window_frames.append(frame_path)
            except:
                continue

        return window_frames

    def compute_frame_similarity(
        self,
        frame1_path: Path,
        frame2_path: Path
    ) -> float:
        """
        Compute similarity between two frames using SSIM

        Args:
            frame1_path: First frame
            frame2_path: Second frame

        Returns:
            Similarity score (0-1)
        """
        from skimage.metrics import structural_similarity as ssim

        img1 = cv2.imread(str(frame1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(frame2_path), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return 0.0

        # Resize to same size if needed
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Compute SSIM
        score, _ = ssim(img1, img2, full=True)
        return float(score)

    def extract_histogram_features(self, image_path: Path) -> np.ndarray:
        """Extract color histogram features"""
        img = cv2.imread(str(image_path))
        if img is None:
            return np.zeros(384)

        features = []
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [64], [0, 256])
            features.append(hist.flatten())

        # HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [64], [0, 256])
            features.append(hist.flatten())

        features = np.concatenate(features)
        return features / (features.sum() + 1e-8)

    def compute_feature_similarity(
        self,
        frame1_path: Path,
        frame2_path: Path
    ) -> float:
        """
        Compute similarity using feature vectors

        Args:
            frame1_path: First frame
            frame2_path: Second frame

        Returns:
            Similarity score (0-1)
        """
        feat1 = self.extract_histogram_features(frame1_path)
        feat2 = self.extract_histogram_features(frame2_path)

        # Cosine similarity
        dot = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot / (norm1 * norm2)
        return float(similarity)

    def find_similar_frames(
        self,
        source_frame_path: Path,
        candidate_frames: List[Path]
    ) -> List[Tuple[Path, float]]:
        """
        Find frames similar to source frame

        Args:
            source_frame_path: Source frame
            candidate_frames: Candidate frames to compare

        Returns:
            List of (frame_path, similarity_score) sorted by score
        """
        print(f"\n🔍 Finding similar frames to {source_frame_path.name}...")

        similarities = []

        for candidate in tqdm(candidate_frames, desc="Computing similarity"):
            if self.config.feature_type == "ssim":
                score = self.compute_frame_similarity(source_frame_path, candidate)
            else:
                score = self.compute_feature_similarity(source_frame_path, candidate)

            if score >= self.config.similarity_threshold:
                similarities.append((candidate, score))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top N
        return similarities[:self.config.max_references]

    def extract_reference_patches(
        self,
        reference_frame: Path,
        instance_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> List[np.ndarray]:
        """
        Extract relevant patches from reference frame

        Args:
            reference_frame: Reference frame path
            instance_bbox: Bounding box of instance (x, y, w, h)

        Returns:
            List of extracted patches
        """
        img = cv2.imread(str(reference_frame))
        if img is None:
            return []

        h, w = img.shape[:2]
        patches = []

        if instance_bbox:
            x, y, bw, bh = instance_bbox

            # Extract patches around the bbox area
            patch_positions = [
                (max(0, x - self.config.patch_size), max(0, y - self.config.patch_size)),
                (min(w - self.config.patch_size, x + bw), max(0, y - self.config.patch_size)),
                (max(0, x - self.config.patch_size), min(h - self.config.patch_size, y + bh)),
                (min(w - self.config.patch_size, x + bw), min(h - self.config.patch_size, y + bh)),
            ]

            for px, py in patch_positions:
                if px >= 0 and py >= 0 and px + self.config.patch_size <= w and py + self.config.patch_size <= h:
                    patch = img[py:py+self.config.patch_size, px:px+self.config.patch_size]
                    patches.append(patch)
        else:
            # Extract grid of patches
            step = self.config.patch_size
            for y in range(0, h - self.config.patch_size, step):
                for x in range(0, w - self.config.patch_size, step):
                    patch = img[y:y+self.config.patch_size, x:x+self.config.patch_size]
                    patches.append(patch)

        return patches

    def extract_context(
        self,
        frames_dir: Path,
        instance_path: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main context extraction pipeline

        Args:
            frames_dir: Directory with all frames
            instance_path: Path to instance image
            output_dir: Output directory for context

        Returns:
            Context metadata
        """
        instance_path = Path(instance_path)
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Extracting temporal context for: {instance_path.name}")

        # Parse instance filename to get source frame
        parse_result = self.parse_instance_filename(instance_path)

        if not parse_result:
            print(f"❌ Could not parse frame number from: {instance_path.name}")
            return {}

        frame_name, frame_number = parse_result
        print(f"   Source frame: {frame_name} (#{frame_number})")

        # Get source frame path
        source_frame_candidates = list(frames_dir.glob(f"{frame_name}.*"))
        if not source_frame_candidates:
            # Try alternative naming
            source_frame_candidates = list(frames_dir.glob(f"frame_{frame_number:04d}.*"))

        if not source_frame_candidates:
            print(f"❌ Source frame not found in {frames_dir}")
            return {}

        source_frame_path = source_frame_candidates[0]
        print(f"   Found source: {source_frame_path.name}")

        # Find temporal window
        window_frames = self.find_temporal_window(frames_dir, frame_number)
        print(f"   Temporal window: {len(window_frames)} frames")

        if not window_frames:
            print(f"⚠️ No frames in temporal window")
            return {}

        # Find similar frames
        similar_frames = self.find_similar_frames(source_frame_path, window_frames)
        print(f"   Similar frames: {len(similar_frames)}")

        # Save reference frames
        references = []

        for i, (ref_frame, similarity) in enumerate(similar_frames):
            ref_info = {
                "frame_path": str(ref_frame),
                "frame_name": ref_frame.name,
                "similarity": similarity,
            }

            # Copy reference frame
            dst_path = output_dir / f"reference_{i:02d}_{ref_frame.name}"
            shutil.copy2(ref_frame, dst_path)
            ref_info["saved_path"] = str(dst_path)

            # Extract patches if enabled
            if self.config.extract_patches:
                patches = self.extract_reference_patches(ref_frame)

                if patches:
                    patches_dir = output_dir / f"patches_{i:02d}"
                    patches_dir.mkdir(exist_ok=True)

                    for j, patch in enumerate(patches):
                        patch_path = patches_dir / f"patch_{j:03d}.png"
                        cv2.imwrite(str(patch_path), patch)

                    ref_info["patches_dir"] = str(patches_dir)
                    ref_info["num_patches"] = len(patches)

            references.append(ref_info)

        # Copy source frame
        source_dst = output_dir / f"source_{source_frame_path.name}"
        shutil.copy2(source_frame_path, source_dst)

        # Create context metadata
        context = {
            "instance_path": str(instance_path),
            "instance_name": instance_path.name,
            "source_frame": {
                "path": str(source_frame_path),
                "name": source_frame_path.name,
                "frame_number": frame_number,
                "saved_path": str(source_dst),
            },
            "temporal_window": {
                "size": self.config.window_size,
                "frames_found": len(window_frames),
            },
            "references": references,
            "config": {
                "window_size": self.config.window_size,
                "max_references": self.config.max_references,
                "similarity_threshold": self.config.similarity_threshold,
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Save metadata
        metadata_path = output_dir / "context.json"
        with open(metadata_path, 'w') as f:
            json.dump(context, f, indent=2)

        print(f"\n✅ Context extraction complete!")
        print(f"   Source frame: {source_frame_path.name}")
        print(f"   References found: {len(references)}")
        print(f"   Output: {output_dir}")
        print(f"   Metadata: {metadata_path}")

        return context


def main():
    parser = argparse.ArgumentParser(
        description="Extract temporal context for inpainting (Film-Agnostic)"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory with all frames"
    )
    parser.add_argument(
        "--instance-path",
        type=str,
        required=True,
        help="Path to instance image"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for context"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Temporal window size (frames before/after, default: 10)"
    )
    parser.add_argument(
        "--max-references",
        type=int,
        default=3,
        help="Maximum reference frames to extract (default: 3)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold (default: 0.7)"
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="combined",
        choices=["histogram", "ssim", "combined"],
        help="Feature type for similarity (default: combined)"
    )
    parser.add_argument(
        "--no-patches",
        action="store_true",
        help="Don't extract patches"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Patch size (default: 128)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (for logging)"
    )

    args = parser.parse_args()

    # Create config
    config = ContextExtractionConfig(
        window_size=args.window_size,
        max_references=args.max_references,
        similarity_threshold=args.similarity_threshold,
        feature_type=args.feature_type,
        extract_patches=not args.no_patches,
        patch_size=args.patch_size,
        check_occlusion=True,
        save_visualization=True,
    )

    # Run extraction
    extractor = TemporalContextExtractor(config)
    context = extractor.extract_context(
        frames_dir=Path(args.frames_dir),
        instance_path=Path(args.instance_path),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
