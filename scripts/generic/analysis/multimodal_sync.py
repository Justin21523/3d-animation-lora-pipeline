#!/usr/bin/env python3
"""
Multi-Modal Synchronization Analyzer

Purpose: Synchronize and analyze audio-visual correspondences
Features: Face-voice matching, lip sync detection, event synchronization
Use Cases: Character identification, quality control, dataset validation

Usage:
    python multimodal_sync.py \
        --frames-dir /path/to/frames \
        --audio-path /path/to/audio.wav \
        --diarization-json /path/to/diarization.json \
        --face-clusters /path/to/clusters \
        --output-dir /path/to/analysis \
        --project luca
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class MultiModalSyncConfig:
    """Configuration for multi-modal synchronization"""
    fps: float = 24.0  # Video frame rate
    audio_sample_rate: int = 16000  # Audio sample rate
    lip_sync_window: float = 0.1  # Seconds around speech to check lip movement
    event_sync_threshold: float = 0.3  # Correlation threshold for events
    confidence_threshold: float = 0.5  # Minimum confidence for matches
    max_face_distance: float = 100.0  # Max pixel distance for face tracking


class MultiModalSynchronizer:
    """Analyze audio-visual synchronization"""

    def __init__(self, config: MultiModalSyncConfig):
        """
        Initialize synchronizer

        Args:
            config: Synchronization configuration
        """
        self.config = config
        self.face_detector = None
        self.landmark_detector = None

    def load_face_detector(self):
        """Load face detection and landmark models"""
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            self.landmark_detector = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                min_detection_confidence=0.5
            )
            return "mediapipe"
        except ImportError:
            print("   ⚠️ MediaPipe not available, using OpenCV cascade")
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            return "opencv"

    def parse_frame_number(self, frame_name: str) -> Optional[int]:
        """Parse frame number from filename"""
        import re
        numbers = re.findall(r'\d+', frame_name)
        if numbers:
            return int(numbers[0])
        return None

    def frame_to_timestamp(self, frame_number: int) -> float:
        """Convert frame number to timestamp in seconds"""
        return frame_number / self.config.fps

    def timestamp_to_frame(self, timestamp: float) -> int:
        """Convert timestamp to frame number"""
        return int(timestamp * self.config.fps)

    def load_diarization(self, diarization_path: Path) -> List[Dict]:
        """
        Load speaker diarization results

        Args:
            diarization_path: Path to diarization JSON

        Returns:
            List of speaker segments
        """
        with open(diarization_path, 'r') as f:
            data = json.load(f)

        segments = data.get('segments', [])
        print(f"   Loaded {len(segments)} speaker segments")
        return segments

    def load_face_clusters(self, clusters_dir: Path) -> Dict[str, List[Dict]]:
        """
        Load character face clusters with frame numbers

        Args:
            clusters_dir: Directory with character clusters

        Returns:
            Dictionary mapping character to frame info
        """
        clusters_dir = Path(clusters_dir)
        clusters = {}

        for char_dir in clusters_dir.iterdir():
            if char_dir.is_dir() and not char_dir.name.startswith('.'):
                character = char_dir.name
                frames = []

                for img in char_dir.glob("*.png"):
                    frame_num = self.parse_frame_number(img.name)
                    if frame_num is not None:
                        timestamp = self.frame_to_timestamp(frame_num)
                        frames.append({
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'path': img
                        })

                if frames:
                    clusters[character] = sorted(frames, key=lambda x: x['frame'])

        print(f"   Loaded {len(clusters)} character clusters")
        return clusters

    def detect_faces_in_frame(self, frame_path: Path) -> List[Dict]:
        """
        Detect faces in a single frame

        Args:
            frame_path: Path to frame image

        Returns:
            List of face detections with bounding boxes
        """
        if self.face_detector is None:
            self.load_face_detector()

        img = cv2.imread(str(frame_path))
        if img is None:
            return []

        faces = []

        try:
            # Try MediaPipe first
            if hasattr(self, 'mp_face_detection'):
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.face_detector.process(rgb_img)

                if results.detections:
                    h, w = img.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)

                        faces.append({
                            'bbox': [x, y, width, height],
                            'confidence': detection.score[0],
                            'center': [x + width // 2, y + height // 2]
                        })
            else:
                # Fallback to OpenCV
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detections = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                for (x, y, w, h) in detections:
                    faces.append({
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'confidence': 1.0,
                        'center': [int(x + w // 2), int(y + h // 2)]
                    })

        except Exception as e:
            print(f"   ⚠️ Face detection failed for {frame_path.name}: {e}")

        return faces

    def detect_mouth_movement(self, frame_path: Path, face_bbox: List[int]) -> Dict:
        """
        Detect mouth movement in face region

        Args:
            frame_path: Path to frame
            face_bbox: Face bounding box [x, y, w, h]

        Returns:
            Mouth movement metrics
        """
        img = cv2.imread(str(frame_path))
        if img is None:
            return {'movement': 0.0, 'area': 0, 'aspect_ratio': 0.0}

        x, y, w, h = face_bbox
        # Extract mouth region (lower third of face)
        mouth_y = y + int(h * 0.6)
        mouth_region = img[mouth_y:y+h, x:x+w]

        if mouth_region.size == 0:
            return {'movement': 0.0, 'area': 0, 'aspect_ratio': 0.0}

        # Convert to grayscale
        gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)

        # Detect edges (open mouth has more edges)
        edges = cv2.Canny(gray_mouth, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0.0

        # Compute variance (movement creates variance)
        variance = np.var(gray_mouth) if gray_mouth.size > 0 else 0.0

        return {
            'movement': float(edge_density),
            'variance': float(variance),
            'area': int(np.sum(edges > 0))
        }

    def analyze_lip_sync(
        self,
        speaker_segments: List[Dict],
        face_clusters: Dict[str, List[Dict]],
        frames_dir: Path
    ) -> Dict:
        """
        Analyze lip synchronization with speech

        Args:
            speaker_segments: Speaker diarization segments
            face_clusters: Character face appearances
            frames_dir: Directory with frames

        Returns:
            Lip sync analysis results
        """
        print(f"\n👄 Analyzing lip synchronization...")

        sync_results = {}

        for character, frames in tqdm(face_clusters.items(), desc="Characters"):
            char_sync = []

            # For each character appearance
            for frame_info in frames:
                timestamp = frame_info['timestamp']

                # Find overlapping speech segments
                overlapping_speech = [
                    seg for seg in speaker_segments
                    if seg['start'] <= timestamp <= seg['end']
                ]

                # Detect faces and mouth movement
                faces = self.detect_faces_in_frame(frame_info['path'])

                if faces:
                    # Use first/largest face
                    face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
                    mouth = self.detect_mouth_movement(frame_info['path'], face['bbox'])

                    # Determine if lip sync is present
                    is_speaking = len(overlapping_speech) > 0
                    has_mouth_movement = mouth['movement'] > 0.1

                    char_sync.append({
                        'frame': frame_info['frame'],
                        'timestamp': timestamp,
                        'is_speaking': is_speaking,
                        'mouth_movement': mouth['movement'],
                        'sync_score': 1.0 if is_speaking == has_mouth_movement else 0.0,
                        'speakers': [seg.get('speaker', 'unknown') for seg in overlapping_speech]
                    })

            if char_sync:
                # Compute aggregate statistics
                total_frames = len(char_sync)
                speaking_frames = sum(1 for s in char_sync if s['is_speaking'])
                synced_frames = sum(1 for s in char_sync if s['sync_score'] > 0.5)
                avg_sync = np.mean([s['sync_score'] for s in char_sync])

                sync_results[character] = {
                    'total_frames': total_frames,
                    'speaking_frames': speaking_frames,
                    'synced_frames': synced_frames,
                    'sync_accuracy': avg_sync,
                    'detailed': char_sync
                }

                print(f"   {character}: {synced_frames}/{speaking_frames} synced "
                      f"({avg_sync:.2%} accuracy)")

        return sync_results

    def match_faces_to_voices(
        self,
        speaker_segments: List[Dict],
        face_clusters: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Match character faces to speaker voices

        Args:
            speaker_segments: Speaker diarization segments
            face_clusters: Character face appearances

        Returns:
            Face-voice matching results
        """
        print(f"\n🎭 Matching faces to voices...")

        matches = {}

        for character, frames in tqdm(face_clusters.items(), desc="Matching"):
            # Count co-occurrences with each speaker
            speaker_cooccurrence = defaultdict(float)

            for frame_info in frames:
                timestamp = frame_info['timestamp']

                # Find overlapping speech
                for segment in speaker_segments:
                    if segment['start'] <= timestamp <= segment['end']:
                        speaker_id = segment.get('speaker', 'unknown')
                        # Weight by overlap duration
                        overlap = min(segment['end'], timestamp + 0.5) - max(segment['start'], timestamp - 0.5)
                        speaker_cooccurrence[speaker_id] += max(0, overlap)

            # Find best matching speaker
            if speaker_cooccurrence:
                best_speaker = max(speaker_cooccurrence.items(), key=lambda x: x[1])
                total_time = sum(speaker_cooccurrence.values())

                matches[character] = {
                    'primary_speaker': best_speaker[0],
                    'confidence': best_speaker[1] / total_time if total_time > 0 else 0.0,
                    'all_speakers': dict(speaker_cooccurrence),
                    'total_overlap_time': total_time
                }

                print(f"   {character} → {best_speaker[0]} "
                      f"(confidence: {matches[character]['confidence']:.2%})")

        return matches

    def detect_synchronized_events(
        self,
        frames_dir: Path,
        audio_path: Path
    ) -> List[Dict]:
        """
        Detect synchronized audio-visual events (impacts, actions)

        Args:
            frames_dir: Directory with frames
            audio_path: Path to audio file

        Returns:
            List of synchronized events
        """
        print(f"\n⚡ Detecting synchronized events...")

        events = []

        try:
            import librosa

            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=self.config.audio_sample_rate)

            # Detect audio onsets (sudden energy increases)
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr,
                units='time'
            )

            print(f"   Found {len(onsets)} audio onsets")

            # Load frames
            frames = sorted(
                list(frames_dir.glob("frame_*.jpg")) +
                list(frames_dir.glob("frame_*.png"))
            )

            # For each onset, check for visual change
            for onset_time in tqdm(onsets[:100], desc="Checking events"):  # Limit to 100
                frame_idx = self.timestamp_to_frame(onset_time)

                if frame_idx < 1 or frame_idx >= len(frames) - 1:
                    continue

                # Compute visual difference around onset
                prev_frame = frames[frame_idx - 1]
                curr_frame = frames[frame_idx]
                next_frame = frames[min(frame_idx + 1, len(frames) - 1)]

                visual_change = self.compute_frame_difference(prev_frame, curr_frame)
                visual_change_next = self.compute_frame_difference(curr_frame, next_frame)

                # Event is synchronized if visual change peaks near onset
                if visual_change > 0.2 or visual_change_next > 0.2:
                    events.append({
                        'timestamp': float(onset_time),
                        'frame': frame_idx,
                        'audio_intensity': float(onset_env[int(onset_time * sr / 512)]),
                        'visual_change': max(visual_change, visual_change_next),
                        'type': 'impact' if visual_change > 0.5 else 'action'
                    })

        except ImportError:
            print("   ⚠️ librosa not available, skipping audio analysis")
        except Exception as e:
            print(f"   ⚠️ Event detection failed: {e}")

        print(f"   Found {len(events)} synchronized events")
        return events

    def compute_frame_difference(self, frame1_path: Path, frame2_path: Path) -> float:
        """Compute visual difference between two frames"""
        img1 = cv2.imread(str(frame1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(frame2_path), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return 0.0

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Compute mean absolute difference
        diff = cv2.absdiff(img1, img2)
        mean_diff = np.mean(diff) / 255.0

        return float(mean_diff)

    def analyze_temporal_alignment(
        self,
        frames_dir: Path,
        audio_path: Path
    ) -> Dict:
        """
        Analyze temporal alignment between audio and video

        Args:
            frames_dir: Directory with frames
            audio_path: Path to audio file

        Returns:
            Temporal alignment metrics
        """
        print(f"\n⏱️ Analyzing temporal alignment...")

        frames = sorted(
            list(frames_dir.glob("frame_*.jpg")) +
            list(frames_dir.glob("frame_*.png"))
        )

        video_duration = len(frames) / self.config.fps

        try:
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=self.config.audio_sample_rate)
            audio_duration = len(audio) / sr

            alignment = {
                'video_duration': video_duration,
                'audio_duration': audio_duration,
                'duration_difference': abs(video_duration - audio_duration),
                'is_aligned': abs(video_duration - audio_duration) < 0.5,
                'video_fps': self.config.fps,
                'audio_sample_rate': sr,
                'total_frames': len(frames)
            }

            print(f"   Video: {video_duration:.2f}s, Audio: {audio_duration:.2f}s")
            print(f"   Difference: {alignment['duration_difference']:.2f}s")

            return alignment

        except ImportError:
            print("   ⚠️ librosa not available")
            return {
                'video_duration': video_duration,
                'total_frames': len(frames),
                'video_fps': self.config.fps
            }

    def analyze_multimodal_sync(
        self,
        frames_dir: Path,
        audio_path: Path,
        diarization_path: Optional[Path],
        face_clusters_dir: Optional[Path],
        output_dir: Path
    ) -> Dict:
        """
        Main multi-modal synchronization analysis

        Args:
            frames_dir: Directory with frames
            audio_path: Path to audio file
            diarization_path: Path to speaker diarization JSON (optional)
            face_clusters_dir: Path to face clusters (optional)
            output_dir: Output directory

        Returns:
            Analysis results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🎬 Multi-Modal Synchronization Analysis")
        print(f"   Frames: {frames_dir}")
        print(f"   Audio: {audio_path}")
        if diarization_path:
            print(f"   Diarization: {diarization_path}")
        if face_clusters_dir:
            print(f"   Face Clusters: {face_clusters_dir}")
        print(f"   Output: {output_dir}")

        results = {
            'frames_dir': str(frames_dir),
            'audio_path': str(audio_path),
            'timestamp': datetime.now().isoformat(),
        }

        # Temporal alignment
        results['temporal_alignment'] = self.analyze_temporal_alignment(
            frames_dir,
            audio_path
        )

        # Load speaker segments if available
        speaker_segments = []
        if diarization_path and diarization_path.exists():
            speaker_segments = self.load_diarization(diarization_path)
            results['speaker_segments_count'] = len(speaker_segments)

        # Load face clusters if available
        face_clusters = {}
        if face_clusters_dir and face_clusters_dir.exists():
            face_clusters = self.load_face_clusters(face_clusters_dir)
            results['character_count'] = len(face_clusters)

        # Face-voice matching
        if speaker_segments and face_clusters:
            results['face_voice_matches'] = self.match_faces_to_voices(
                speaker_segments,
                face_clusters
            )

        # Lip sync analysis
        if speaker_segments and face_clusters:
            results['lip_sync'] = self.analyze_lip_sync(
                speaker_segments,
                face_clusters,
                frames_dir
            )

        # Event synchronization
        if audio_path.exists():
            results['synchronized_events'] = self.detect_synchronized_events(
                frames_dir,
                audio_path
            )

        # Save results
        results_path = output_dir / "multimodal_sync.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved: {results_path}")

        # Save visualizations
        if results.get('face_voice_matches'):
            self.save_voice_matching_visualization(
                results['face_voice_matches'],
                output_dir
            )

        if results.get('synchronized_events'):
            self.save_event_timeline(
                results['synchronized_events'],
                output_dir
            )

        return results

    def save_voice_matching_visualization(
        self,
        matches: Dict,
        output_dir: Path
    ):
        """Save face-voice matching visualization"""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, max(6, len(matches) * 0.8)))

            characters = sorted(matches.keys())
            y_positions = range(len(characters))

            # Plot confidence bars
            confidences = [matches[char]['confidence'] for char in characters]
            speakers = [matches[char]['primary_speaker'] for char in characters]

            bars = ax.barh(y_positions, confidences, height=0.6)

            # Color code by speaker
            unique_speakers = list(set(speakers))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_speakers)))
            speaker_colors = {spk: colors[i] for i, spk in enumerate(unique_speakers)}

            for i, (bar, speaker) in enumerate(zip(bars, speakers)):
                bar.set_color(speaker_colors[speaker])

            ax.set_yticks(y_positions)
            ax.set_yticklabels(characters)
            ax.set_xlabel("Confidence", fontsize=12)
            ax.set_title("Face-Voice Matching Confidence", fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1.0)
            ax.grid(True, axis='x', alpha=0.3)

            # Add speaker labels
            for i, (char, speaker) in enumerate(zip(characters, speakers)):
                ax.text(
                    confidences[i] + 0.02,
                    i,
                    speaker,
                    va='center',
                    fontsize=10
                )

            plt.tight_layout()

            viz_path = output_dir / "face_voice_matching.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   Voice matching visualization saved: {viz_path}")

        except ImportError:
            print("   ⚠️ Matplotlib not available")

    def save_event_timeline(
        self,
        events: List[Dict],
        output_dir: Path
    ):
        """Save synchronized events timeline"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

            timestamps = [e['timestamp'] for e in events]
            audio_intensity = [e['audio_intensity'] for e in events]
            visual_change = [e['visual_change'] for e in events]

            # Audio intensity
            ax1.scatter(timestamps, audio_intensity, alpha=0.6, c='blue', s=50)
            ax1.set_ylabel("Audio Intensity", fontsize=12)
            ax1.set_title("Synchronized Events Timeline", fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Visual change
            ax2.scatter(timestamps, visual_change, alpha=0.6, c='red', s=50)
            ax2.set_xlabel("Time (seconds)", fontsize=12)
            ax2.set_ylabel("Visual Change", fontsize=12)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            viz_path = output_dir / "event_timeline.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"   Event timeline saved: {viz_path}")

        except ImportError:
            print("   ⚠️ Matplotlib not available")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal Synchronization Analysis (Film-Agnostic)"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory with frames"
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--diarization-json",
        type=str,
        help="Path to speaker diarization JSON (optional)"
    )
    parser.add_argument(
        "--face-clusters",
        type=str,
        help="Path to face clusters directory (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Video frame rate"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name"
    )

    args = parser.parse_args()

    config = MultiModalSyncConfig(
        fps=args.fps,
    )

    analyzer = MultiModalSynchronizer(config)
    results = analyzer.analyze_multimodal_sync(
        frames_dir=Path(args.frames_dir),
        audio_path=Path(args.audio_path),
        diarization_path=Path(args.diarization_json) if args.diarization_json else None,
        face_clusters_dir=Path(args.face_clusters) if args.face_clusters else None,
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")

    # Print summary
    print(f"\n📊 Summary:")
    if 'temporal_alignment' in results:
        print(f"   Temporal alignment: {'✓' if results['temporal_alignment'].get('is_aligned') else '✗'}")
    if 'face_voice_matches' in results:
        print(f"   Face-voice matches: {len(results['face_voice_matches'])}")
    if 'synchronized_events' in results:
        print(f"   Synchronized events: {len(results['synchronized_events'])}")


if __name__ == "__main__":
    main()
