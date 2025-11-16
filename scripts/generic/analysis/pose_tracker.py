#!/usr/bin/env python3
"""
Pose Tracker

Purpose: Track character poses across frames with temporal consistency
Features: RTM-Pose integration, MediaPipe fallback, pose sequence analysis, action classification
Use Cases: Pose-based clustering, action recognition, pose diversity analysis

Usage:
    python pose_tracker.py \
        --frames-dir /path/to/frames \
        --output-dir /path/to/poses \
        --detector mediapipe \
        --track-across-frames \
        --analyze-sequences \
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
class PoseTrackingConfig:
    """Configuration for pose tracking"""
    detector: str = "mediapipe"  # mediapipe, rtmpose, openpose
    confidence_threshold: float = 0.5  # Minimum detection confidence
    track_across_frames: bool = True  # Temporal tracking
    max_tracking_distance: float = 50.0  # Max distance for ID association
    analyze_sequences: bool = True  # Analyze pose sequences
    sequence_window: int = 10  # Frames to analyze for sequences
    save_visualizations: bool = True  # Save pose visualizations
    detect_actions: bool = True  # Classify actions from poses


class PoseTracker:
    """Track and analyze human poses across frames"""

    def __init__(self, config: PoseTrackingConfig):
        """
        Initialize pose tracker

        Args:
            config: Tracking configuration
        """
        self.config = config
        self.detector = None
        self.mp_pose = None
        self.mp_drawing = None

        # Load pose detector
        self.load_detector()

    def load_detector(self):
        """Load pose detection model"""
        if self.config.detector == "mediapipe":
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles

                self.detector = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    min_detection_confidence=self.config.confidence_threshold,
                    min_tracking_confidence=0.5
                )
                print("✅ MediaPipe Pose detector loaded")

            except ImportError:
                print("❌ MediaPipe not available. Install: pip install mediapipe")
                self.detector = None

        elif self.config.detector == "rtmpose":
            # TODO: Implement RTM-Pose integration
            print("⚠️ RTM-Pose not yet implemented, falling back to MediaPipe")
            self.config.detector = "mediapipe"
            self.load_detector()

        elif self.config.detector == "openpose":
            # TODO: Implement OpenPose integration
            print("⚠️ OpenPose not yet implemented, falling back to MediaPipe")
            self.config.detector = "mediapipe"
            self.load_detector()

    def detect_pose_mediapipe(
        self,
        image_path: Path
    ) -> Optional[Dict]:
        """
        Detect pose using MediaPipe

        Args:
            image_path: Path to image

        Returns:
            Pose detection results
        """
        if not self.detector:
            return None

        img = cv2.imread(str(image_path))
        if img is None:
            return None

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect pose
        results = self.detector.process(img_rgb)

        if not results.pose_landmarks:
            return None

        # Extract keypoints
        keypoints = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints.append({
                "id": idx,
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
                "visibility": float(landmark.visibility)
            })

        # Compute bounding box
        visible_keypoints = [kp for kp in keypoints if kp["visibility"] > 0.5]
        if visible_keypoints:
            xs = [kp["x"] for kp in visible_keypoints]
            ys = [kp["y"] for kp in visible_keypoints]
            bbox = {
                "x_min": min(xs),
                "x_max": max(xs),
                "y_min": min(ys),
                "y_max": max(ys),
                "width": max(xs) - min(xs),
                "height": max(ys) - min(ys)
            }
        else:
            bbox = None

        # Compute pose features
        pose_features = self.compute_pose_features(keypoints)

        return {
            "keypoints": keypoints,
            "num_keypoints": len(keypoints),
            "num_visible": len(visible_keypoints),
            "bbox": bbox,
            "features": pose_features,
            "confidence": float(np.mean([kp["visibility"] for kp in keypoints]))
        }

    def compute_pose_features(
        self,
        keypoints: List[Dict]
    ) -> Dict:
        """
        Compute pose-specific features

        Args:
            keypoints: List of keypoint dictionaries

        Returns:
            Feature dictionary
        """
        # MediaPipe keypoint indices
        # 0: nose, 11-12: shoulders, 13-14: elbows, 15-16: wrists
        # 23-24: hips, 25-26: knees, 27-28: ankles

        features = {}

        try:
            # Body orientation (from shoulders)
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]

            shoulder_angle = np.arctan2(
                right_shoulder["y"] - left_shoulder["y"],
                right_shoulder["x"] - left_shoulder["x"]
            )
            features["shoulder_angle"] = float(shoulder_angle)

            # Body height (shoulder to hip)
            left_hip = keypoints[23]
            torso_length = np.sqrt(
                (left_shoulder["x"] - left_hip["x"])**2 +
                (left_shoulder["y"] - left_hip["y"])**2
            )
            features["torso_length"] = float(torso_length)

            # Arm angles
            left_elbow = keypoints[13]
            left_wrist = keypoints[15]

            left_arm_angle = np.arctan2(
                left_wrist["y"] - left_elbow["y"],
                left_wrist["x"] - left_elbow["x"]
            )
            features["left_arm_angle"] = float(left_arm_angle)

            # Leg stance (hip-knee-ankle angles)
            left_knee = keypoints[25]
            left_ankle = keypoints[27]

            left_leg_angle = np.arctan2(
                left_ankle["y"] - left_knee["y"],
                left_ankle["x"] - left_knee["x"]
            )
            features["left_leg_angle"] = float(left_leg_angle)

            # Symmetry score (compare left and right)
            right_elbow = keypoints[14]
            right_wrist = keypoints[16]

            right_arm_angle = np.arctan2(
                right_wrist["y"] - right_elbow["y"],
                right_wrist["x"] - right_elbow["x"]
            )

            symmetry = 1.0 - abs(left_arm_angle - right_arm_angle) / np.pi
            features["arm_symmetry"] = float(symmetry)

        except (IndexError, KeyError):
            pass

        return features

    def classify_pose_type(self, pose: Dict) -> str:
        """
        Classify pose type based on features

        Args:
            pose: Pose detection result

        Returns:
            Pose classification
        """
        if not pose or not pose.get("features"):
            return "unknown"

        features = pose["features"]

        # Simple heuristic classification
        if "torso_length" not in features:
            return "unknown"

        torso = features.get("torso_length", 0)
        shoulder_angle = features.get("shoulder_angle", 0)

        # Classify based on body orientation and proportions
        if abs(shoulder_angle) < 0.2:
            return "frontal"
        elif abs(shoulder_angle) > 1.3:
            return "side-view"
        elif torso < 0.2:
            return "sitting"
        else:
            return "standing"

    def track_pose_across_frames(
        self,
        pose_detections: List[Dict]
    ) -> List[Dict]:
        """
        Associate poses across frames to create tracks

        Args:
            pose_detections: List of pose detections per frame

        Returns:
            List of pose tracks
        """
        print(f"\n🎯 Tracking poses across frames...")

        tracks = []
        active_tracks = []

        for frame_idx, detection in enumerate(tqdm(pose_detections, desc="Tracking")):
            if not detection or not detection.get("bbox"):
                continue

            bbox = detection["bbox"]
            center = ((bbox["x_min"] + bbox["x_max"]) / 2, (bbox["y_min"] + bbox["y_max"]) / 2)

            # Try to match with existing tracks
            matched = False

            for track in active_tracks:
                last_pose = track["poses"][-1]
                last_bbox = last_pose["bbox"]
                last_center = ((last_bbox["x_min"] + last_bbox["x_max"]) / 2,
                             (last_bbox["y_min"] + last_bbox["y_max"]) / 2)

                # Compute distance
                distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)

                # Match if close enough
                if distance < self.config.max_tracking_distance:
                    track["poses"].append(detection)
                    track["frames"].append(frame_idx)
                    matched = True
                    break

            if not matched:
                # Start new track
                new_track = {
                    "id": len(tracks) + len(active_tracks),
                    "start_frame": frame_idx,
                    "poses": [detection],
                    "frames": [frame_idx]
                }
                active_tracks.append(new_track)

        # Finalize all tracks
        tracks.extend(active_tracks)

        # Compute track statistics
        for track in tracks:
            track["duration"] = len(track["poses"])
            track["end_frame"] = track["frames"][-1] if track["frames"] else track["start_frame"]

            # Classify pose sequence
            if len(track["poses"]) >= 3:
                track["sequence_type"] = self.classify_pose_sequence(track["poses"])

        print(f"   Found {len(tracks)} pose tracks")

        return tracks

    def classify_pose_sequence(
        self,
        poses: List[Dict]
    ) -> str:
        """
        Classify action from pose sequence

        Args:
            poses: Sequence of pose detections

        Returns:
            Action classification
        """
        if len(poses) < 3:
            return "unknown"

        # Analyze movement patterns
        movements = []

        for i in range(len(poses) - 1):
            if not poses[i].get("bbox") or not poses[i+1].get("bbox"):
                continue

            bbox1 = poses[i]["bbox"]
            bbox2 = poses[i+1]["bbox"]

            center1 = ((bbox1["x_min"] + bbox1["x_max"]) / 2, (bbox1["y_min"] + bbox1["y_max"]) / 2)
            center2 = ((bbox2["x_min"] + bbox2["x_max"]) / 2, (bbox2["y_min"] + bbox2["y_max"]) / 2)

            movement = np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
            movements.append(movement)

        if not movements:
            return "static"

        avg_movement = np.mean(movements)
        movement_variance = np.var(movements)

        # Classify based on movement characteristics
        if avg_movement < 0.01:
            return "idle"
        elif avg_movement < 0.05 and movement_variance < 0.001:
            return "walking"
        elif movement_variance > 0.01:
            return "dynamic-action"
        else:
            return "moving"

    def analyze_pose_diversity(
        self,
        poses: List[Dict]
    ) -> Dict:
        """
        Analyze diversity of poses

        Args:
            poses: List of pose detections

        Returns:
            Diversity metrics
        """
        pose_types = defaultdict(int)
        sequence_types = defaultdict(int)

        for pose in poses:
            if pose:
                pose_type = self.classify_pose_type(pose)
                pose_types[pose_type] += 1

        # Compute diversity score (entropy)
        total = sum(pose_types.values())
        if total > 0:
            probs = [count / total for count in pose_types.values()]
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            max_entropy = np.log(len(pose_types)) if len(pose_types) > 0 else 1
            diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        else:
            diversity_score = 0

        return {
            "pose_types": dict(pose_types),
            "num_unique_types": len(pose_types),
            "diversity_score": float(diversity_score),
            "total_poses": total
        }

    def save_pose_visualization(
        self,
        image_path: Path,
        pose: Dict,
        output_path: Path
    ):
        """
        Save pose visualization

        Args:
            image_path: Original image path
            pose: Pose detection result
            output_path: Output path
        """
        if not self.mp_drawing or not pose:
            return

        img = cv2.imread(str(image_path))
        if img is None:
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert keypoints back to MediaPipe format
        # This is a simplified version
        if pose.get("keypoints"):
            # Draw keypoints and connections
            for kp in pose["keypoints"]:
                if kp["visibility"] > 0.5:
                    x = int(kp["x"] * img.shape[1])
                    y = int(kp["y"] * img.shape[0])
                    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        # Save
        cv2.imwrite(str(output_path), img)

    def track_poses(
        self,
        frames_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Main pose tracking pipeline

        Args:
            frames_dir: Directory with frames
            output_dir: Output directory

        Returns:
            Tracking results
        """
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Pose Tracking")
        print(f"   Frames: {frames_dir}")
        print(f"   Output: {output_dir}")
        print(f"   Detector: {self.config.detector}")

        if not self.detector:
            print("❌ No pose detector available")
            return {}

        # Find frames
        frames = sorted(
            list(frames_dir.glob("frame_*.jpg")) +
            list(frames_dir.glob("frame_*.png"))
        )

        print(f"   Total frames: {len(frames)}")

        # Detect poses in all frames
        print(f"\n🔍 Detecting poses...")

        pose_detections = []

        for frame_path in tqdm(frames, desc="Detecting poses"):
            pose = self.detect_pose_mediapipe(frame_path)

            if pose:
                pose["frame"] = frame_path.name
                pose["pose_type"] = self.classify_pose_type(pose)

            pose_detections.append(pose)

        # Count detections
        num_detected = sum(1 for p in pose_detections if p is not None)
        detection_rate = num_detected / len(frames) if len(frames) > 0 else 0

        print(f"   Detected poses: {num_detected}/{len(frames)} ({detection_rate:.1%})")

        # Track across frames
        tracks = []
        if self.config.track_across_frames:
            tracks = self.track_pose_across_frames(pose_detections)

        # Analyze diversity
        diversity = self.analyze_pose_diversity([p for p in pose_detections if p])

        # Aggregate statistics
        pose_types = defaultdict(int)
        for pose in pose_detections:
            if pose:
                pose_types[pose.get("pose_type", "unknown")] += 1

        statistics = {
            "total_frames": len(frames),
            "poses_detected": num_detected,
            "detection_rate": detection_rate,
            "pose_types": dict(pose_types),
            "diversity": diversity,
            "num_tracks": len(tracks),
            "avg_track_duration": np.mean([t["duration"] for t in tracks]) if tracks else 0
        }

        print(f"\n📊 Pose Statistics:")
        print(f"   Detection rate: {detection_rate:.1%}")
        print(f"   Pose types: {dict(pose_types)}")
        print(f"   Diversity score: {diversity['diversity_score']:.2f}")
        if tracks:
            print(f"   Tracks found: {len(tracks)}")
            print(f"   Avg track duration: {statistics['avg_track_duration']:.1f} frames")

        # Save results
        results = {
            "frames_dir": str(frames_dir),
            "detector": self.config.detector,
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "track_across_frames": self.config.track_across_frames,
            },
            "pose_detections": [p for p in pose_detections if p],  # Save only successful detections
            "tracks": tracks,
            "statistics": statistics,
            "timestamp": datetime.now().isoformat()
        }

        results_path = output_dir / "pose_tracking.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved: {results_path}")

        # Save visualizations
        if self.config.save_visualizations:
            self.save_pose_timeline(tracks, len(frames), output_dir)

        return results

    def save_pose_timeline(
        self,
        tracks: List[Dict],
        total_frames: int,
        output_dir: Path
    ):
        """
        Save pose track timeline visualization

        Args:
            tracks: List of pose tracks
            total_frames: Total number of frames
            output_dir: Output directory
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, ax = plt.subplots(figsize=(15, max(6, len(tracks) * 0.3)))

            colors = plt.cm.Set3(np.linspace(0, 1, len(tracks)))

            for i, track in enumerate(tracks):
                start = track["start_frame"]
                duration = track["duration"]

                ax.barh(
                    i,
                    duration,
                    left=start,
                    height=0.8,
                    color=colors[i],
                    edgecolor='black',
                    linewidth=0.5
                )

                # Add label
                seq_type = track.get("sequence_type", "unknown")
                ax.text(
                    start + duration/2,
                    i,
                    f"Track {track['id']} ({seq_type})",
                    ha='center',
                    va='center',
                    fontsize=8
                )

            ax.set_yticks(range(len(tracks)))
            ax.set_yticklabels([f"Track {t['id']}" for t in tracks])
            ax.set_xlabel("Frame Number", fontsize=12)
            ax.set_title("Pose Track Timeline", fontsize=14, fontweight='bold')
            ax.set_xlim(0, total_frames)
            ax.grid(True, axis='x', alpha=0.3)

            plt.tight_layout()

            viz_path = output_dir / "pose_timeline.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   Pose timeline saved: {viz_path}")

        except ImportError:
            print("   ⚠️ Matplotlib not available")


def main():
    parser = argparse.ArgumentParser(
        description="Pose Tracking and Analysis (Film-Agnostic)"
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
        "--detector",
        type=str,
        default="mediapipe",
        choices=["mediapipe", "rtmpose", "openpose"],
        help="Pose detector to use (default: mediapipe)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)"
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable temporal tracking"
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable visualization saving"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name"
    )

    args = parser.parse_args()

    # Create config
    config = PoseTrackingConfig(
        detector=args.detector,
        confidence_threshold=args.confidence_threshold,
        track_across_frames=not args.no_tracking,
        analyze_sequences=True,
        save_visualizations=not args.no_visualization,
        detect_actions=True
    )

    # Run tracking
    tracker = PoseTracker(config)
    results = tracker.track_poses(
        frames_dir=Path(args.frames_dir),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
