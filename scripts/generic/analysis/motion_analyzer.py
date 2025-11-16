#!/usr/bin/env python3
"""
Motion Analyzer

Purpose: Analyze character and camera motion in detail
Features: Optical flow, pose tracking, action recognition, trajectory analysis
Use Cases: Motion-aware sampling, action classification, dynamic scene detection

Usage:
    python motion_analyzer.py \
        --frames-dir /path/to/frames \
        --output-dir /path/to/analysis \
        --sample-rate 5 \
        --analyze-trajectories \
        --detect-poses \
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
from collections import defaultdict


@dataclass
class MotionAnalysisConfig:
    """Configuration for motion analysis"""
    sample_rate: int = 5  # Analyze every N frames
    flow_method: str = "farneback"  # farneback, lucas_kanade
    magnitude_threshold: float = 2.0  # Motion magnitude threshold
    compute_trajectories: bool = True  # Track motion trajectories
    detect_actions: bool = True  # Detect action types
    detect_poses: bool = False  # Use pose detection (requires RTM-Pose)
    analyze_camera_motion: bool = True  # Separate camera vs object motion
    save_flow_visualizations: bool = True  # Save optical flow visualizations
    trajectory_max_distance: float = 50.0  # Max distance for trajectory linking


class MotionAnalyzer:
    """Comprehensive motion analysis for video frames"""

    def __init__(self, config: MotionAnalysisConfig):
        """
        Initialize motion analyzer

        Args:
            config: Analysis configuration
        """
        self.config = config
        self.pose_detector = None

        # Load pose detector if needed
        if config.detect_poses:
            self.load_pose_detector()

    def load_pose_detector(self):
        """Load pose detection model (RTM-Pose or OpenPose)"""
        try:
            # Try to use a lightweight pose detector
            # For production, you'd use RTM-Pose or MediaPipe
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5
            )
            print("✅ Pose detector loaded (MediaPipe)")
        except ImportError:
            print("⚠️ MediaPipe not available, pose detection disabled")
            self.config.detect_poses = False

    def compute_optical_flow_farneback(
        self,
        frame1_path: Path,
        frame2_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute dense optical flow using Farneback method

        Args:
            frame1_path: First frame
            frame2_path: Second frame

        Returns:
            (flow, magnitude, angle) arrays
        """
        img1 = cv2.imread(str(frame1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(frame2_path), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            h, w = 180, 320
            return np.zeros((h, w, 2)), np.zeros((h, w)), np.zeros((h, w))

        # Resize for performance
        target_height = 360
        aspect = img1.shape[1] / img1.shape[0]
        target_width = int(target_height * aspect)

        img1_resized = cv2.resize(img1, (target_width, target_height))
        img2_resized = cv2.resize(img2, (target_width, target_height))

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            img1_resized,
            img2_resized,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Compute magnitude and angle
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        angle = np.arctan2(flow[..., 1], flow[..., 0])

        return flow, magnitude, angle

    def compute_optical_flow_lucas_kanade(
        self,
        frame1_path: Path,
        frame2_path: Path
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Compute sparse optical flow using Lucas-Kanade

        Args:
            frame1_path: First frame
            frame2_path: Second frame

        Returns:
            (old_points, new_points) lists of tracked points
        """
        img1 = cv2.imread(str(frame1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(frame2_path), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return [], []

        # Find good features to track
        feature_params = dict(
            maxCorners=200,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)

        if p0 is None:
            return [], []

        # Calculate optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        p1, status, error = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

        # Select good points
        good_old = p0[status == 1]
        good_new = p1[status == 1]

        return good_old.tolist(), good_new.tolist()

    def analyze_motion_statistics(
        self,
        flow: np.ndarray,
        magnitude: np.ndarray,
        angle: np.ndarray
    ) -> Dict:
        """
        Compute comprehensive motion statistics

        Args:
            flow: Optical flow array
            magnitude: Flow magnitude
            angle: Flow angle

        Returns:
            Statistics dictionary
        """
        # Basic statistics
        mean_mag = float(np.mean(magnitude))
        max_mag = float(np.max(magnitude))
        std_mag = float(np.std(magnitude))
        median_mag = float(np.median(magnitude))

        # Motion density (percentage of pixels with significant motion)
        motion_mask = magnitude > self.config.magnitude_threshold
        motion_density = float(np.sum(motion_mask) / magnitude.size)

        # Direction statistics
        mean_angle = float(np.mean(angle[motion_mask]) if motion_mask.any() else 0)
        angle_std = float(np.std(angle[motion_mask]) if motion_mask.any() else 0)

        # Flow direction distribution (8 bins for compass directions)
        angle_hist, _ = np.histogram(angle[motion_mask], bins=8, range=(-np.pi, np.pi))
        dominant_direction_idx = int(np.argmax(angle_hist))
        directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
        dominant_direction = directions[dominant_direction_idx]

        # Spatial distribution of motion
        h, w = magnitude.shape
        grid_h, grid_w = 4, 4
        cell_h, cell_w = h // grid_h, w // grid_w

        spatial_motion = np.zeros((grid_h, grid_w))
        for i in range(grid_h):
            for j in range(grid_w):
                cell = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                spatial_motion[i, j] = np.mean(cell)

        # Detect motion hotspots
        hotspot_threshold = np.percentile(spatial_motion, 75)
        num_hotspots = int(np.sum(spatial_motion > hotspot_threshold))

        return {
            "mean_magnitude": mean_mag,
            "max_magnitude": max_mag,
            "std_magnitude": std_mag,
            "median_magnitude": median_mag,
            "motion_density": motion_density,
            "mean_direction": mean_angle,
            "direction_std": angle_std,
            "dominant_direction": dominant_direction,
            "direction_histogram": angle_hist.tolist(),
            "num_hotspots": num_hotspots,
            "spatial_motion_variance": float(np.var(spatial_motion)),
        }

    def classify_motion_type(self, stats: Dict) -> str:
        """
        Classify motion type based on comprehensive statistics

        Args:
            stats: Motion statistics

        Returns:
            Motion type classification
        """
        mag = stats["mean_magnitude"]
        density = stats["motion_density"]
        variance = stats["spatial_motion_variance"]

        if mag < 0.5:
            return "static"
        elif mag < 1.5 and density < 0.15:
            return "minimal"
        elif mag < 2.5 and density < 0.3:
            return "slow"
        elif mag < 5.0 and variance < 2.0:
            return "moderate-uniform"
        elif mag < 5.0 and variance >= 2.0:
            return "moderate-mixed"
        elif mag >= 5.0 and density > 0.5:
            return "fast-chaotic"
        elif mag >= 5.0:
            return "fast"
        else:
            return "unknown"

    def classify_camera_motion(self, stats: Dict) -> str:
        """
        Classify camera motion type

        Args:
            stats: Motion statistics

        Returns:
            Camera motion classification
        """
        mag = stats["mean_magnitude"]
        density = stats["motion_density"]
        direction_std = stats["direction_std"]

        # High density + consistent direction = camera movement
        if density > 0.7 and direction_std < 1.0:
            if mag < 2.0:
                return "slow-pan"
            elif mag < 4.0:
                return "pan"
            else:
                return "fast-pan"

        # Low density = stationary camera with object motion
        elif density < 0.3:
            return "static"

        # Mixed
        else:
            return "complex"

    def detect_action_type(self, motion_history: List[Dict]) -> str:
        """
        Detect action type from motion history

        Args:
            motion_history: Recent motion statistics

        Returns:
            Action classification
        """
        if len(motion_history) < 3:
            return "unknown"

        # Analyze temporal pattern
        magnitudes = [m["mean_magnitude"] for m in motion_history]
        avg_mag = np.mean(magnitudes)
        mag_variance = np.var(magnitudes)

        if avg_mag < 1.0:
            return "idle"
        elif avg_mag < 2.0 and mag_variance < 0.5:
            return "walking"
        elif avg_mag < 3.0 and mag_variance < 1.0:
            return "moving"
        elif mag_variance > 2.0:
            return "dynamic-action"
        else:
            return "continuous-motion"

    def track_motion_trajectories(
        self,
        frames: List[Path]
    ) -> List[Dict]:
        """
        Track motion trajectories using Lucas-Kanade

        Args:
            frames: List of frame paths

        Returns:
            List of trajectory data
        """
        print(f"\n🎯 Tracking motion trajectories...")

        trajectories = []
        active_trajectories = []

        for i in tqdm(range(len(frames) - 1), desc="Tracking trajectories"):
            old_points, new_points = self.compute_optical_flow_lucas_kanade(
                frames[i],
                frames[i+1]
            )

            # Update active trajectories
            new_active = []

            for traj in active_trajectories:
                last_point = traj["points"][-1]

                # Try to find matching point
                matched = False
                for idx, (ox, oy) in enumerate(old_points):
                    dist = np.sqrt((ox - last_point[0])**2 + (oy - last_point[1])**2)

                    if dist < self.config.trajectory_max_distance:
                        # Match found
                        traj["points"].append(new_points[idx])
                        traj["frames"].append(i + 1)
                        new_active.append(traj)
                        matched = True
                        break

                if not matched:
                    # Trajectory ended
                    if len(traj["points"]) >= 3:
                        trajectories.append(traj)

            # Start new trajectories from unmatched new points
            matched_indices = set()
            for traj in new_active:
                if len(traj["frames"]) > 0:
                    matched_indices.add(traj["frames"][-1])

            for idx, point in enumerate(new_points):
                if idx not in matched_indices:
                    new_active.append({
                        "id": len(trajectories) + len(new_active),
                        "start_frame": i + 1,
                        "points": [point],
                        "frames": [i + 1]
                    })

            active_trajectories = new_active

        # Add remaining active trajectories
        for traj in active_trajectories:
            if len(traj["points"]) >= 3:
                trajectories.append(traj)

        # Compute trajectory statistics
        for traj in trajectories:
            points = np.array(traj["points"])
            if len(points) > 1:
                # Compute trajectory length
                diffs = np.diff(points, axis=0)
                lengths = np.sqrt(np.sum(diffs**2, axis=1))
                total_length = float(np.sum(lengths))

                # Compute average speed
                duration = len(traj["frames"])
                avg_speed = total_length / duration if duration > 0 else 0

                traj["total_length"] = total_length
                traj["avg_speed"] = avg_speed
                traj["duration"] = duration

        print(f"   Found {len(trajectories)} trajectories")

        return trajectories

    def detect_poses(
        self,
        frame_path: Path
    ) -> Optional[Dict]:
        """
        Detect human poses in frame

        Args:
            frame_path: Path to frame

        Returns:
            Pose detection results
        """
        if not self.pose_detector:
            return None

        img = cv2.imread(str(frame_path))
        if img is None:
            return None

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect poses
        results = self.pose_detector.process(img_rgb)

        if not results.pose_landmarks:
            return None

        # Extract keypoints
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })

        return {
            "keypoints": keypoints,
            "num_keypoints": len(keypoints),
        }

    def save_flow_visualization(
        self,
        flow: np.ndarray,
        magnitude: np.ndarray,
        frame_idx: int,
        output_dir: Path
    ):
        """
        Save optical flow visualization

        Args:
            flow: Optical flow
            magnitude: Flow magnitude
            frame_idx: Frame index
            output_dir: Output directory
        """
        # Create HSV visualization
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)

        # Hue = direction, Value = magnitude
        angle = np.arctan2(flow[..., 1], flow[..., 0])
        hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 180
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Save
        viz_path = output_dir / f"flow_{frame_idx:06d}.png"
        cv2.imwrite(str(viz_path), bgr)

    def analyze_motion(
        self,
        frames_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main motion analysis pipeline

        Args:
            frames_dir: Directory with frames
            output_dir: Output directory

        Returns:
            Analysis results
        """
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Motion Analysis")
        print(f"   Frames: {frames_dir}")
        print(f"   Output: {output_dir}")

        # Find frames
        frames = sorted(
            list(frames_dir.glob("frame_*.jpg")) +
            list(frames_dir.glob("frame_*.png"))
        )

        print(f"   Total frames: {len(frames)}")
        print(f"   Sample rate: 1/{self.config.sample_rate}")

        # Sample frames
        sampled_frames = frames[::self.config.sample_rate]
        print(f"   Analyzing {len(sampled_frames)} frames")

        # Create visualization directory
        if self.config.save_flow_visualizations:
            viz_dir = output_dir / "flow_visualizations"
            viz_dir.mkdir(exist_ok=True)

        # Analyze motion frame by frame
        motion_data = []
        motion_history = []

        for i in tqdm(range(len(sampled_frames) - 1), desc="Analyzing motion"):
            # Compute optical flow
            flow, magnitude, angle = self.compute_optical_flow_farneback(
                sampled_frames[i],
                sampled_frames[i+1]
            )

            # Compute statistics
            stats = self.analyze_motion_statistics(flow, magnitude, angle)
            stats["frame"] = i * self.config.sample_rate
            stats["motion_type"] = self.classify_motion_type(stats)

            # Analyze camera motion
            if self.config.analyze_camera_motion:
                stats["camera_motion"] = self.classify_camera_motion(stats)

            # Detect action type
            motion_history.append(stats)
            if len(motion_history) > 10:
                motion_history.pop(0)

            if self.config.detect_actions:
                stats["action_type"] = self.detect_action_type(motion_history)

            motion_data.append(stats)

            # Save flow visualization
            if self.config.save_flow_visualizations and i % 10 == 0:
                self.save_flow_visualization(flow, magnitude, i, viz_dir)

        # Track trajectories
        trajectories = []
        if self.config.compute_trajectories:
            trajectories = self.track_motion_trajectories(sampled_frames)

        # Aggregate statistics
        motion_types = {}
        camera_motions = {}
        action_types = {}

        for data in motion_data:
            mt = data.get("motion_type", "unknown")
            motion_types[mt] = motion_types.get(mt, 0) + 1

            if "camera_motion" in data:
                cm = data["camera_motion"]
                camera_motions[cm] = camera_motions.get(cm, 0) + 1

            if "action_type" in data:
                at = data["action_type"]
                action_types[at] = action_types.get(at, 0) + 1

        # Compute overall statistics
        all_magnitudes = [d["mean_magnitude"] for d in motion_data]
        all_densities = [d["motion_density"] for d in motion_data]

        statistics = {
            "total_frames": len(frames),
            "analyzed_frames": len(sampled_frames),
            "avg_magnitude": float(np.mean(all_magnitudes)),
            "max_magnitude": float(np.max(all_magnitudes)),
            "avg_density": float(np.mean(all_densities)),
            "motion_types": motion_types,
            "camera_motions": camera_motions,
            "action_types": action_types,
            "num_trajectories": len(trajectories),
        }

        print(f"\n📊 Motion Statistics:")
        print(f"   Avg magnitude: {statistics['avg_magnitude']:.2f}")
        print(f"   Avg density: {statistics['avg_density']:.2%}")
        print(f"   Motion types: {motion_types}")
        print(f"   Camera motions: {camera_motions}")
        if action_types:
            print(f"   Action types: {action_types}")
        print(f"   Trajectories tracked: {len(trajectories)}")

        # Save results
        results = {
            "frames_dir": str(frames_dir),
            "config": {
                "sample_rate": self.config.sample_rate,
                "flow_method": self.config.flow_method,
                "magnitude_threshold": self.config.magnitude_threshold,
            },
            "motion_data": motion_data,
            "trajectories": trajectories[:100],  # Save top 100 trajectories
            "statistics": statistics,
            "timestamp": datetime.now().isoformat(),
        }

        results_path = output_dir / "motion_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved: {results_path}")

        # Save visualizations
        self.save_motion_timeline(motion_data, output_dir)

        return results

    def save_motion_timeline(
        self,
        motion_data: List[Dict],
        output_dir: Path
    ):
        """
        Save motion timeline visualization

        Args:
            motion_data: Motion analysis data
            output_dir: Output directory
        """
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

            frames = [d["frame"] for d in motion_data]
            magnitudes = [d["mean_magnitude"] for d in motion_data]
            densities = [d["motion_density"] for d in motion_data]

            # Magnitude timeline
            ax1.plot(frames, magnitudes, linewidth=2, color='steelblue')
            ax1.fill_between(frames, magnitudes, alpha=0.3, color='steelblue')
            ax1.set_xlabel("Frame Number", fontsize=12)
            ax1.set_ylabel("Motion Magnitude", fontsize=12)
            ax1.set_title("Motion Magnitude Timeline", fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Density timeline
            ax2.plot(frames, densities, linewidth=2, color='coral')
            ax2.fill_between(frames, densities, alpha=0.3, color='coral')
            ax2.set_xlabel("Frame Number", fontsize=12)
            ax2.set_ylabel("Motion Density", fontsize=12)
            ax2.set_title("Motion Density Timeline", fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            viz_path = output_dir / "motion_timeline.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   Motion timeline saved: {viz_path}")

        except ImportError:
            print("   ⚠️ Matplotlib not available")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Motion Analysis (Film-Agnostic)"
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
        help="Output directory"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=5,
        help="Analyze every N frames (default: 5)"
    )
    parser.add_argument(
        "--flow-method",
        type=str,
        default="farneback",
        choices=["farneback", "lucas_kanade"],
        help="Optical flow method (default: farneback)"
    )
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Disable trajectory tracking"
    )
    parser.add_argument(
        "--no-actions",
        action="store_true",
        help="Disable action detection"
    )
    parser.add_argument(
        "--detect-poses",
        action="store_true",
        help="Enable pose detection (requires MediaPipe)"
    )
    parser.add_argument(
        "--save-flow-viz",
        action="store_true",
        help="Save optical flow visualizations"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name"
    )

    args = parser.parse_args()

    # Create config
    config = MotionAnalysisConfig(
        sample_rate=args.sample_rate,
        flow_method=args.flow_method,
        compute_trajectories=not args.no_trajectories,
        detect_actions=not args.no_actions,
        detect_poses=args.detect_poses,
        analyze_camera_motion=True,
        save_flow_visualizations=args.save_flow_viz,
    )

    # Run analysis
    analyzer = MotionAnalyzer(config)
    results = analyzer.analyze_motion(
        frames_dir=Path(args.frames_dir),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
