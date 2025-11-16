#!/usr/bin/env python3
"""
Scene Analyzer

Purpose: Analyze scene structure and detect shot boundaries
Features: Shot type classification, camera movement detection, narrative structure
Use Cases: Scene-aware dataset splitting, temporal analysis, shot-level metadata

Usage:
    python scene_analyzer.py \
        --frames-dir /path/to/frames \
        --output-dir /path/to/analysis \
        --threshold 30.0 \
        --min-scene-length 15 \
        --project luca
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


@dataclass
class SceneAnalysisConfig:
    """Configuration for scene analysis"""
    threshold: float = 30.0  # Scene detection threshold
    min_scene_length: int = 15  # Minimum frames per scene
    detect_shot_types: bool = True  # Classify shot types
    detect_camera_movement: bool = True  # Detect camera motion
    compute_shot_features: bool = True  # Extract shot-level features
    save_visualization: bool = True


class SceneAnalyzer:
    """Analyze scene structure and shot boundaries"""

    def __init__(self, config: SceneAnalysisConfig):
        """
        Initialize scene analyzer

        Args:
            config: Analysis configuration
        """
        self.config = config

    def compute_frame_difference(
        self,
        frame1_path: Path,
        frame2_path: Path
    ) -> float:
        """
        Compute difference between consecutive frames

        Args:
            frame1_path: First frame
            frame2_path: Second frame

        Returns:
            Difference score (0-100)
        """
        img1 = cv2.imread(str(frame1_path))
        img2 = cv2.imread(str(frame2_path))

        if img1 is None or img2 is None:
            return 0.0

        # Resize if needed
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Compute histogram difference
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        # Normalize
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        # Compare
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

        return float(diff)

    def detect_scene_boundaries(
        self,
        frames: List[Path]
    ) -> List[Dict]:
        """
        Detect scene boundaries using frame difference

        Args:
            frames: List of frame paths

        Returns:
            List of scene dicts
        """
        print(f"\n🎬 Detecting scene boundaries...")

        boundaries = [0]  # First frame is always a boundary

        # Compute differences
        for i in tqdm(range(1, len(frames)), desc="Analyzing frames"):
            diff = self.compute_frame_difference(frames[i-1], frames[i])

            if diff > self.config.threshold:
                # Check minimum scene length
                if i - boundaries[-1] >= self.config.min_scene_length:
                    boundaries.append(i)

        # Create scene segments
        scenes = []

        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i+1] if i+1 < len(boundaries) else len(frames)

            scene = {
                "scene_id": i,
                "start_frame": start,
                "end_frame": end - 1,
                "num_frames": end - start,
                "start_file": frames[start].name,
                "end_file": frames[end-1].name if end > start else frames[start].name,
            }

            scenes.append(scene)

        print(f"   Found {len(scenes)} scenes")

        return scenes

    def classify_shot_type(
        self,
        frames: List[Path]
    ) -> str:
        """
        Classify shot type based on composition

        Args:
            frames: Frame paths for this shot

        Returns:
            Shot type classification
        """
        # Sample middle frame
        mid_idx = len(frames) // 2
        frame_path = frames[mid_idx]

        img = cv2.imread(str(frame_path))
        if img is None:
            return "unknown"

        h, w = img.shape[:2]
        aspect = w / h

        # Simple heuristics based on image properties
        # More sophisticated: use face detection, object detection

        # Detect faces
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # Get largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                fx, fy, fw, fh = largest_face

                face_area = fw * fh
                frame_area = w * h
                face_ratio = face_area / frame_area

                # Classify based on face size
                if face_ratio > 0.15:
                    return "close-up"
                elif face_ratio > 0.05:
                    return "medium-shot"
                else:
                    return "wide-shot"
        except:
            pass

        # Fallback: analyze overall composition
        return "wide-shot"

    def detect_camera_movement(
        self,
        frames: List[Path]
    ) -> Dict:
        """
        Detect camera movement in shot

        Args:
            frames: Frame paths for this shot

        Returns:
            Movement characteristics
        """
        if len(frames) < 3:
            return {"type": "static", "magnitude": 0.0}

        # Sample frames
        sample_indices = [0, len(frames)//2, len(frames)-1]
        sample_frames = [frames[i] for i in sample_indices]

        # Compute optical flow between consecutive samples
        movements = []

        for i in range(len(sample_frames) - 1):
            img1 = cv2.imread(str(sample_frames[i]), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(sample_frames[i+1]), cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                continue

            # Resize for speed
            img1 = cv2.resize(img1, (320, 180))
            img2 = cv2.resize(img2, (320, 180))

            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                img1, img2, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Compute magnitude
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_mag = np.mean(mag)
            movements.append(avg_mag)

        if not movements:
            return {"type": "static", "magnitude": 0.0}

        avg_movement = np.mean(movements)

        # Classify movement
        if avg_movement < 1.0:
            movement_type = "static"
        elif avg_movement < 3.0:
            movement_type = "slow-pan"
        elif avg_movement < 6.0:
            movement_type = "pan"
        else:
            movement_type = "fast-motion"

        return {
            "type": movement_type,
            "magnitude": float(avg_movement)
        }

    def compute_shot_features(
        self,
        frames: List[Path]
    ) -> Dict:
        """
        Compute visual features for shot

        Args:
            frames: Frame paths for this shot

        Returns:
            Feature dictionary
        """
        # Sample middle frame
        mid_idx = len(frames) // 2
        frame_path = frames[mid_idx]

        img = cv2.imread(str(frame_path))
        if img is None:
            return {}

        # Color statistics
        mean_color = cv2.mean(img)[:3]

        # Brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # Contrast
        contrast = np.std(gray)

        # Edge density (complexity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        return {
            "mean_color": [float(c) for c in mean_color],
            "brightness": float(brightness),
            "contrast": float(contrast),
            "edge_density": float(edge_density),
        }

    def analyze_scenes(
        self,
        frames_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main scene analysis pipeline

        Args:
            frames_dir: Directory with frames
            output_dir: Output directory

        Returns:
            Analysis results
        """
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Scene Analysis")
        print(f"   Frames: {frames_dir}")
        print(f"   Output: {output_dir}")

        # Find all frames
        frames = sorted(
            list(frames_dir.glob("frame_*.jpg")) +
            list(frames_dir.glob("frame_*.png"))
        )

        print(f"   Total frames: {len(frames)}")

        # Detect scene boundaries
        scenes = self.detect_scene_boundaries(frames)

        # Analyze each scene
        print(f"\n🔍 Analyzing scene characteristics...")

        for scene in tqdm(scenes, desc="Analyzing scenes"):
            start = scene["start_frame"]
            end = scene["end_frame"] + 1
            scene_frames = frames[start:end]

            # Shot type classification
            if self.config.detect_shot_types:
                scene["shot_type"] = self.classify_shot_type(scene_frames)

            # Camera movement
            if self.config.detect_camera_movement:
                scene["camera_movement"] = self.detect_camera_movement(scene_frames)

            # Shot features
            if self.config.compute_shot_features:
                scene["features"] = self.compute_shot_features(scene_frames)

        # Compute statistics
        shot_types = {}
        camera_movements = {}

        for scene in scenes:
            if "shot_type" in scene:
                st = scene["shot_type"]
                shot_types[st] = shot_types.get(st, 0) + 1

            if "camera_movement" in scene:
                cm = scene["camera_movement"]["type"]
                camera_movements[cm] = camera_movements.get(cm, 0) + 1

        statistics = {
            "total_scenes": len(scenes),
            "total_frames": len(frames),
            "avg_scene_length": np.mean([s["num_frames"] for s in scenes]),
            "shot_types": shot_types,
            "camera_movements": camera_movements,
        }

        print(f"\n📊 Scene Statistics:")
        print(f"   Total scenes: {statistics['total_scenes']}")
        print(f"   Avg scene length: {statistics['avg_scene_length']:.1f} frames")

        if shot_types:
            print(f"   Shot types:")
            for st, count in shot_types.items():
                print(f"     {st}: {count}")

        if camera_movements:
            print(f"   Camera movements:")
            for cm, count in camera_movements.items():
                print(f"     {cm}: {count}")

        # Save results
        results = {
            "frames_dir": str(frames_dir),
            "config": {
                "threshold": self.config.threshold,
                "min_scene_length": self.config.min_scene_length,
            },
            "scenes": scenes,
            "statistics": statistics,
            "timestamp": datetime.now().isoformat(),
        }

        results_path = output_dir / "scene_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved: {results_path}")

        # Save visualization
        if self.config.save_visualization:
            self.save_visualization(scenes, len(frames), output_dir)

        return results

    def save_visualization(
        self,
        scenes: List[Dict],
        total_frames: int,
        output_dir: Path
    ):
        """
        Save scene timeline visualization

        Args:
            scenes: List of scene dicts
            total_frames: Total number of frames
            output_dir: Output directory
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

            # Scene timeline
            shot_types = list(set(s.get("shot_type", "unknown") for s in scenes))
            colors = plt.cm.Set3(np.linspace(0, 1, len(shot_types)))
            type_colors = {st: colors[i] for i, st in enumerate(shot_types)}

            for scene in scenes:
                st = scene.get("shot_type", "unknown")
                ax1.barh(
                    0,
                    scene["num_frames"],
                    left=scene["start_frame"],
                    height=0.8,
                    color=type_colors[st],
                    edgecolor='black',
                    linewidth=0.5
                )

            ax1.set_xlim(0, total_frames)
            ax1.set_ylim(-0.5, 0.5)
            ax1.set_xlabel("Frame Number", fontsize=12)
            ax1.set_yticks([])
            ax1.set_title("Scene Timeline (by Shot Type)", fontsize=14, fontweight='bold')
            ax1.grid(True, axis='x', alpha=0.3)

            # Legend
            patches = [mpatches.Patch(color=type_colors[st], label=st) for st in shot_types]
            ax1.legend(handles=patches, loc='upper right')

            # Scene lengths distribution
            lengths = [s["num_frames"] for s in scenes]
            ax2.hist(lengths, bins=20, color='steelblue', edgecolor='black')
            ax2.set_xlabel("Scene Length (frames)", fontsize=12)
            ax2.set_ylabel("Count", fontsize=12)
            ax2.set_title("Scene Length Distribution", fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            viz_path = output_dir / "scene_timeline.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   Visualization saved: {viz_path}")

        except ImportError:
            print("   ⚠️ Matplotlib not available, skipping visualization")


def main():
    parser = argparse.ArgumentParser(
        description="Scene Structure Analysis (Film-Agnostic)"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory with frames"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for analysis"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Scene detection threshold (default: 30.0)"
    )
    parser.add_argument(
        "--min-scene-length",
        type=int,
        default=15,
        help="Minimum scene length in frames (default: 15)"
    )
    parser.add_argument(
        "--no-shot-types",
        action="store_true",
        help="Disable shot type classification"
    )
    parser.add_argument(
        "--no-camera-movement",
        action="store_true",
        help="Disable camera movement detection"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name (for logging)"
    )

    args = parser.parse_args()

    # Create config
    config = SceneAnalysisConfig(
        threshold=args.threshold,
        min_scene_length=args.min_scene_length,
        detect_shot_types=not args.no_shot_types,
        detect_camera_movement=not args.no_camera_movement,
        compute_shot_features=True,
        save_visualization=True,
    )

    # Run analysis
    analyzer = SceneAnalyzer(config)
    results = analyzer.analyze_scenes(
        frames_dir=Path(args.frames_dir),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
