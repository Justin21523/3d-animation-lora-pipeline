#!/usr/bin/env python3
"""
Universal Frame Extraction Tool for Anime/Video Series

This script provides a flexible, generalized framework for extracting frames
from video files with multiple extraction strategies:
  1. Scene-based: Extract frames at scene boundaries
  2. Interval-based: Extract frames at fixed time/frame intervals
  3. Hybrid: Combine both strategies for comprehensive coverage

Designed to work with any anime series or video collection.
"""

import re
import argparse
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum
import cv2
import numpy as np
from scenedetect import detect, ContentDetector
import subprocess
import os


# ============================================================================
# Configuration Classes
# ============================================================================

class ExtractionMode(Enum):
    """Defines available frame extraction strategies"""
    SCENE_BASED = "scene"      # Extract based on scene changes
    INTERVAL_BASED = "interval"  # Extract at fixed intervals
    HYBRID = "hybrid"          # Combine both strategies


@dataclass
class ExtractionConfig:
    """
    Configuration for frame extraction process

    Attributes:
        mode: Which extraction strategy to use
        scene_threshold: Sensitivity for scene detection (lower = more sensitive)
        frames_per_scene: How many frames to extract from each scene
        skip_scene_boundaries: Whether to skip first/last portions of scenes
        interval_seconds: Extract one frame every N seconds (for interval mode)
        interval_frames: Extract one frame every N frames (for interval mode)
        min_scene_length: Minimum scene duration in frames to process
        jpeg_quality: Output JPEG quality (1-100, higher = better)
    """
    mode: ExtractionMode = ExtractionMode.SCENE_BASED
    scene_threshold: float = 27.0
    frames_per_scene: int = 3
    skip_scene_boundaries: bool = True
    interval_seconds: Optional[float] = None
    interval_frames: Optional[int] = None
    min_scene_length: int = 10
    jpeg_quality: int = 95


@dataclass
class VideoMetadata:
    """
    Stores video file metadata

    Attributes:
        path: Path to video file
        episode_number: Episode number (if applicable)
        fps: Frames per second
        total_frames: Total frame count
        duration_seconds: Video duration
    """
    path: Path
    episode_number: int
    fps: float
    total_frames: int
    duration_seconds: float


# ============================================================================
# Video Processing Utilities
# ============================================================================

def extract_episode_number(filename: str, pattern: Optional[str] = r'(\d+)'):
    """
    Extract episode number or use filename

    If pattern is None, returns the filename (without extension) to preserve
    all information and avoid file overwrites.

    Args:
        filename: Video filename
        pattern: Regex pattern to extract episode number, or None to use filename

    Returns:
        Episode number (int) if pattern matches, filename (str) if pattern is None, or 0 if no match

    Examples:
        With pattern:
            "episode_01.mp4" with r'(\d+)' -> 1
            "S01E05.mkv" with r'E(\d+)' -> 5

        Without pattern (None):
            "S1.01.mp4" -> "S1.01"
            "妖怪手表.mp4" -> "妖怪手表"
    """
    if pattern is None:
        # Return filename without extension
        return Path(filename).stem

    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return 0


def convert_to_mp4(input_path: Path, output_path: Path,
                   preset: str = "fast", crf: int = 23) -> bool:
    """
    Convert video to MP4 format using ffmpeg

    MP4 is preferred because:
      - Better random access performance (crucial for frame extraction)
      - Widely supported by OpenCV
      - Good compression with H.264

    Args:
        input_path: Source video file
        output_path: Destination MP4 file
        preset: ffmpeg encoding preset (ultrafast/fast/medium/slow)
                faster = larger file, slower = better compression
        crf: Constant Rate Factor (0-51, lower = better quality)
             18 = visually lossless, 23 = default, 28 = acceptable

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        print(f"  [Convert] {input_path.name} -> MP4 (preset={preset}, crf={crf})")

        cmd = [
            'ffmpeg',
            '-i', str(input_path),           # Input file
            '-c:v', 'libx264',               # H.264 video codec
            '-preset', preset,               # Encoding speed/compression trade-off
            '-crf', str(crf),                # Quality level
            '-c:a', 'aac',                   # AAC audio codec
            '-b:a', '128k',                  # Audio bitrate
            '-movflags', '+faststart',       # Enable fast streaming
            '-y',                            # Overwrite output file
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  [Convert] ✓ Success: {output_path.name}")
            return True
        else:
            print(f"  [Convert] ✗ Failed: {result.stderr[:200]}")
            return False

    except Exception as e:
        print(f"  [Convert] ✗ Error: {e}")
        return False


def get_video_metadata(video_path: Path, episode_num: Union[int, str]) -> Optional[VideoMetadata]:
    """
    Extract metadata from video file using OpenCV

    Args:
        video_path: Path to video file
        episode_num: Episode number identifier

    Returns:
        VideoMetadata object, or None if video cannot be read
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  [Metadata] ✗ Cannot open: {video_path.name}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    if fps == 0 or total_frames == 0:
        print(f"  [Metadata] ✗ Invalid video properties")
        return None

    return VideoMetadata(
        path=video_path,
        episode_number=episode_num,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration
    )


# ============================================================================
# Frame Extraction Strategies
# ============================================================================

class FrameExtractor:
    """
    Base class for frame extraction strategies

    This implements the Strategy pattern, allowing different algorithms
    to be swapped at runtime.
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config

    def extract(self, video_path: Path, output_dir: Path,
                metadata: VideoMetadata) -> int:
        """
        Extract frames from video

        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            metadata: Video metadata

        Returns:
            Number of frames extracted
        """
        raise NotImplementedError


class SceneBasedExtractor(FrameExtractor):
    """
    Extract frames based on scene change detection

    This strategy:
      1. Detects scene boundaries using content analysis
      2. Extracts N frames from each scene
      3. Optionally skips scene transition frames (first/last 10%)

    Best for:
      - Character-focused datasets (captures different poses/expressions)
      - Action sequences (scene changes often = important moments)
      - Efficient storage (avoids redundant similar frames)
    """

    def detect_scenes(self, video_path: Path) -> List[Tuple]:
        """
        Detect scene boundaries using PySceneDetect

        Scene detection works by analyzing frame-to-frame differences:
          - High difference = scene change (cut or transition)
          - Threshold controls sensitivity

        Args:
            video_path: Path to video file

        Returns:
            List of (start_time, end_time) tuples for each scene
        """
        print(f"  [SceneDetect] Analyzing with threshold={self.config.scene_threshold}")

        try:
            scene_list = detect(
                str(video_path),
                ContentDetector(threshold=self.config.scene_threshold)
            )
            print(f"  [SceneDetect] ✓ Found {len(scene_list)} scenes")
            return scene_list
        except Exception as e:
            print(f"  [SceneDetect] ✗ Error: {e}")
            return []

    def calculate_frame_positions(self, start_frame: int, end_frame: int,
                                  num_frames: int) -> List[int]:
        """
        Calculate which frames to extract from a scene

        Strategy:
          1. Skip first/last 10% of scene (transition frames)
          2. Distribute remaining frames evenly

        Visual representation:
          Scene: [===========================================]
                 ^     ^              ^                ^    ^
                 skip  frame1         frame2          frame3 skip
                 10%                                        10%

        Args:
            start_frame: Scene start frame index
            end_frame: Scene end frame index
            num_frames: Number of frames to extract

        Returns:
            List of frame indices to extract
        """
        scene_length = end_frame - start_frame

        # Skip short scenes
        if scene_length < self.config.min_scene_length:
            return []

        # Calculate extraction boundaries
        if self.config.skip_scene_boundaries and scene_length > num_frames + 2:
            # Skip 10% at start and end to avoid transition artifacts
            start_offset = max(1, scene_length // 10)
            end_offset = max(1, scene_length // 10)

            effective_start = start_frame + start_offset
            effective_end = end_frame - end_offset

            # Distribute frames evenly across effective range
            if num_frames == 1:
                # Single frame: take middle
                return [effective_start + (effective_end - effective_start) // 2]
            else:
                # Multiple frames: evenly distributed
                return [
                    effective_start + (effective_end - effective_start) * i // (num_frames - 1)
                    for i in range(num_frames)
                ]
        else:
            # Scene too short to skip boundaries, use full range
            if num_frames == 1:
                return [start_frame + scene_length // 2]
            else:
                return [
                    start_frame + scene_length * i // (num_frames - 1)
                    for i in range(num_frames)
                ]

    def extract(self, video_path: Path, output_dir: Path,
                metadata: VideoMetadata) -> int:
        """
        Extract frames using scene-based strategy

        Process:
          1. Detect all scenes in video
          2. For each scene:
             a. Calculate frame positions
             b. Extract frames at those positions
             c. Save with descriptive filenames

        Args:
            video_path: Path to video file
            output_dir: Output directory
            metadata: Video metadata

        Returns:
            Total number of frames extracted
        """
        # Step 1: Detect scenes
        scene_list = self.detect_scenes(video_path)
        if not scene_list:
            return 0

        # Step 2: Open video for frame extraction
        cap = cv2.VideoCapture(str(video_path))
        extracted_count = 0

        print(f"  [Extract] Processing {len(scene_list)} scenes...")

        # Step 3: Process each scene
        for scene_idx, scene in enumerate(scene_list):
            # Get scene boundaries in frame numbers
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()

            # Calculate which frames to extract
            frame_positions = self.calculate_frame_positions(
                start_frame, end_frame, self.config.frames_per_scene
            )

            # Extract each frame
            for pos_idx, frame_idx in enumerate(frame_positions):
                # Seek to specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # Calculate timestamp for filename
                    timestamp = frame_idx / metadata.fps

                    # Generate descriptive filename:
                    # Format: scene{scene_num}_pos{position}_frame{global_count}_t{timestamp}s.jpg
                    # Example: scene0042_pos1_frame002156_t847.25s.jpg
                    filename = (
                        f"scene{scene_idx:04d}_"
                        f"pos{pos_idx}_"
                        f"frame{extracted_count:06d}_"
                        f"t{timestamp:.2f}s.jpg"
                    )

                    output_path = output_dir / filename

                    # Save frame with specified quality
                    cv2.imwrite(
                        str(output_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                    )
                    extracted_count += 1

        cap.release()
        print(f"  [Extract] ✓ Extracted {extracted_count} frames from {len(scene_list)} scenes")
        return extracted_count


class IntervalBasedExtractor(FrameExtractor):
    """
    Extract frames at fixed time or frame intervals

    This strategy:
      1. Extracts one frame every N seconds OR every N frames
      2. Provides uniform temporal coverage
      3. Simpler but may include redundant frames

    Best for:
      - Complete coverage of video content
      - When scene detection misses important frames
      - Creating time-lapse style datasets
      - Backgrounds and environments (less focused on action)
    """

    def calculate_interval_positions(self, metadata: VideoMetadata) -> List[int]:
        """
        Calculate frame positions based on interval settings

        Two modes:
          1. Time-based: Extract every N seconds
          2. Frame-based: Extract every N frames

        Args:
            metadata: Video metadata

        Returns:
            List of frame indices to extract
        """
        positions = []

        if self.config.interval_seconds is not None:
            # Time-based interval
            # Example: interval_seconds=5 means extract one frame every 5 seconds
            interval_frames = int(self.config.interval_seconds * metadata.fps)
            print(f"  [Interval] Time-based: every {self.config.interval_seconds}s "
                  f"({interval_frames} frames)")

        elif self.config.interval_frames is not None:
            # Frame-based interval
            # Example: interval_frames=100 means extract every 100th frame
            interval_frames = self.config.interval_frames
            print(f"  [Interval] Frame-based: every {interval_frames} frames")

        else:
            # Default: 1 frame per second
            interval_frames = int(metadata.fps)
            print(f"  [Interval] Default: 1 frame/second ({interval_frames} frames)")

        # Generate positions from start to end
        current_frame = 0
        while current_frame < metadata.total_frames:
            positions.append(current_frame)
            current_frame += interval_frames

        return positions

    def extract(self, video_path: Path, output_dir: Path,
                metadata: VideoMetadata) -> int:
        """
        Extract frames using interval-based strategy

        Process:
          1. Calculate frame positions based on interval
          2. Extract frame at each position
          3. Save with timestamp information

        Args:
            video_path: Path to video file
            output_dir: Output directory
            metadata: Video metadata

        Returns:
            Total number of frames extracted
        """
        # Step 1: Calculate positions
        frame_positions = self.calculate_interval_positions(metadata)
        print(f"  [Extract] Will extract {len(frame_positions)} frames")

        # Step 2: Open video
        cap = cv2.VideoCapture(str(video_path))
        extracted_count = 0

        # Step 3: Extract each frame
        for idx, frame_idx in enumerate(frame_positions):
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                timestamp = frame_idx / metadata.fps

                # Filename format: interval{index}_frame{frame_num}_t{timestamp}s.jpg
                # Example: interval0042_frame012500_t520.83s.jpg
                filename = (
                    f"interval{idx:05d}_"
                    f"frame{frame_idx:06d}_"
                    f"t{timestamp:.2f}s.jpg"
                )

                output_path = output_dir / filename
                cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                )
                extracted_count += 1

        cap.release()
        print(f"  [Extract] ✓ Extracted {extracted_count} frames at regular intervals")
        return extracted_count


class HybridExtractor(FrameExtractor):
    """
    Combine scene-based and interval-based extraction

    This strategy:
      1. Runs both scene detection and interval extraction
      2. Merges results, removing duplicates
      3. Provides comprehensive coverage

    Best for:
      - Maximum dataset coverage
      - When you want both semantic (scenes) and temporal (intervals) coverage
      - Initial exploration of unknown content

    Trade-off:
      - More frames = larger dataset
      - May include redundant information
    """

    def extract(self, video_path: Path, output_dir: Path,
                metadata: VideoMetadata) -> int:
        """
        Extract frames using hybrid strategy

        Process:
          1. Extract scene-based frames
          2. Extract interval-based frames
          3. Results are automatically separated by filename prefix

        Args:
            video_path: Path to video file
            output_dir: Output directory
            metadata: Video metadata

        Returns:
            Total number of frames extracted
        """
        print(f"  [Hybrid] Running both scene-based and interval-based extraction")

        # Create sub-extractors
        scene_extractor = SceneBasedExtractor(self.config)
        interval_extractor = IntervalBasedExtractor(self.config)

        # Run both strategies
        scene_count = scene_extractor.extract(video_path, output_dir, metadata)
        interval_count = interval_extractor.extract(video_path, output_dir, metadata)

        total = scene_count + interval_count
        print(f"  [Hybrid] ✓ Total: {total} frames "
              f"(scene={scene_count}, interval={interval_count})")

        return total


# ============================================================================
# Multi-Process Worker
# ============================================================================

def process_single_video(args_tuple) -> Tuple[int, int]:
    """
    Worker function for parallel video processing

    This function is designed to be called by multiprocessing.Pool.
    Each worker independently processes one video file.

    Args:
        args_tuple: Tuple containing:
            - video_path: Path to video file
            - episode_num: Episode number
            - output_base_dir: Base output directory
            - temp_dir: Temporary directory for conversions
            - config: ExtractionConfig object

    Returns:
        Tuple of (episode_number, frames_extracted)
    """
    video_path, episode_num, output_base_dir, temp_dir, config = args_tuple

    try:
        video_path = Path(video_path)
        print(f"\n{'='*80}")
        # Format episode identifier based on type
        # If episode_num is the video filename (no episode pattern), output directly to base dir
        if isinstance(episode_num, int):
            episode_id = f"{episode_num:03d}"
            episode_dir_name = f"episode_{episode_id}"
            episode_dir = output_base_dir / episode_dir_name
        elif isinstance(episode_num, str) and episode_num != video_path.stem:
            # String episode number but not just the filename
            episode_id = str(episode_num)
            episode_dir_name = episode_id
            episode_dir = output_base_dir / episode_dir_name
        else:
            # No episode pattern or episode_num is just the filename - use base dir directly
            episode_id = video_path.stem
            episode_dir = output_base_dir

        print(f"[Episode {episode_id}] Processing: {video_path.name}")
        print(f"{'='*80}")

        # Create output directory
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Convert to MP4 if needed (for better OpenCV compatibility)
        if video_path.suffix.lower() in ['.flv', '.wmv', '.avi']:
            mp4_path = temp_dir / f"{episode_id}.mp4"
            if not mp4_path.exists():
                if not convert_to_mp4(video_path, mp4_path):
                    print(f"  [Episode {episode_id}] ✗ Conversion failed")
                    return (episode_num, 0)
            video_to_process = mp4_path
        else:
            video_to_process = video_path

        # Get video metadata
        metadata = get_video_metadata(video_to_process, episode_num)
        if metadata is None:
            return (episode_num, 0)

        print(f"  [Metadata] FPS: {metadata.fps:.2f}, "
              f"Frames: {metadata.total_frames}, "
              f"Duration: {metadata.duration_seconds:.1f}s")

        # Select appropriate extraction strategy
        if config.mode == ExtractionMode.SCENE_BASED:
            extractor = SceneBasedExtractor(config)
        elif config.mode == ExtractionMode.INTERVAL_BASED:
            extractor = IntervalBasedExtractor(config)
        elif config.mode == ExtractionMode.HYBRID:
            extractor = HybridExtractor(config)
        else:
            raise ValueError(f"Unknown extraction mode: {config.mode}")

        # Extract frames
        count = extractor.extract(video_to_process, episode_dir, metadata)

        print(f"  ✓ [Episode {episode_id}] Completed: {count} frames extracted")
        return (episode_num, count)

    except Exception as e:
        # Create episode_id if not defined (error occurred before initialization)
        if 'episode_id' not in locals():
            episode_id = f"{episode_num:03d}" if isinstance(episode_num, int) else str(episode_num)
        print(f"  ✗ [Episode {episode_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return (episode_num, 0)


# ============================================================================
# Main Processing Function
# ============================================================================

def process_video_collection(
    input_dir: Path,
    output_dir: Path,
    temp_dir: Path,
    config: ExtractionConfig,
    episode_pattern: str = r'(\d+)',
    num_workers: Optional[int] = None,
    start_episode: Optional[int] = None,
    end_episode: Optional[int] = None,
    video_extensions: List[str] = None
) -> Dict[str, any]:
    """
    Process an entire collection of videos

    Args:
        input_dir: Directory containing video files
        output_dir: Directory for extracted frames
        temp_dir: Directory for temporary files (MP4 conversions)
        config: Extraction configuration
        episode_pattern: Regex pattern to extract episode numbers
        num_workers: Number of parallel workers (default: auto)
        start_episode: Only process episodes >= this number
        end_episode: Only process episodes <= this number
        video_extensions: List of video file extensions to process

    Returns:
        Dictionary with processing statistics
    """
    # Default video extensions
    if video_extensions is None:
        video_extensions = ['.mp4', '.mkv', '.avi', '.flv', '.mov', '.wmv', '.ts', '.m2ts']

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f'*{ext}'))
        video_files.extend(input_dir.glob(f'*{ext.upper()}'))

    if not video_files:
        print(f"✗ No video files found in: {input_dir}")
        print(f"  Searched for extensions: {video_extensions}")
        return {"error": "No video files found"}

    # Build episode map (handles deduplication)
    episode_map = {}
    for video_path in video_files:
        episode_num = extract_episode_number(video_path.name, episode_pattern)
        # For int: skip if 0, for str: skip if empty
        if (isinstance(episode_num, int) and episode_num > 0) or \
           (isinstance(episode_num, str) and episode_num):
            # If duplicate, keep larger file (assume better quality)
            if episode_num not in episode_map or \
               video_path.stat().st_size > episode_map[episode_num].stat().st_size:
                episode_map[episode_num] = video_path

    # Filter by episode range if specified (only for integer episode numbers)
    if (start_episode is not None or end_episode is not None) and \
       episode_map and isinstance(next(iter(episode_map.keys())), int):
        start = start_episode if start_episode is not None else min(episode_map.keys())
        end = end_episode if end_episode is not None else max(episode_map.keys())
        episode_map = {k: v for k, v in episode_map.items() if start <= k <= end}

    sorted_episodes = sorted(episode_map.items())

    # Print processing info
    print(f"\n{'='*80}")
    print(f"Universal Frame Extraction")
    print(f"{'='*80}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Temp directory: {temp_dir}")
    print(f"Episodes to process: {len(sorted_episodes)}")
    if sorted_episodes:
        print(f"Episode range: {sorted_episodes[0][0]} - {sorted_episodes[-1][0]}")
    print(f"\nExtraction Configuration:")
    print(f"  Mode: {config.mode.value}")
    if config.mode in [ExtractionMode.SCENE_BASED, ExtractionMode.HYBRID]:
        print(f"  Scene threshold: {config.scene_threshold}")
        print(f"  Frames per scene: {config.frames_per_scene}")
        print(f"  Skip boundaries: {config.skip_scene_boundaries}")
    if config.mode in [ExtractionMode.INTERVAL_BASED, ExtractionMode.HYBRID]:
        if config.interval_seconds:
            print(f"  Interval: {config.interval_seconds} seconds")
        elif config.interval_frames:
            print(f"  Interval: {config.interval_frames} frames")
    print(f"  JPEG quality: {config.jpeg_quality}%")

    # Determine worker count
    if num_workers is None:
        num_workers = min(cpu_count(), len(sorted_episodes))
    else:
        num_workers = min(num_workers, cpu_count(), len(sorted_episodes))

    print(f"  Workers: {num_workers} (CPU cores: {cpu_count()})")
    print(f"{'='*80}\n")

    # Prepare processing arguments
    process_args = [
        (video_path, episode_num, output_dir, temp_dir, config)
        for episode_num, video_path in sorted_episodes
    ]

    # Process videos in parallel
    results = []
    if num_workers > 1:
        print(f"Starting parallel processing with {num_workers} workers...\n")
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_single_video, process_args)
    else:
        print(f"Processing videos sequentially...\n")
        for args_tuple in process_args:
            result = process_single_video(args_tuple)
            results.append(result)

    # Calculate statistics
    total_frames = sum(count for _, count in results)
    successful = sum(1 for _, count in results if count > 0)

    # Print summary
    print(f"\n{'='*80}")
    print(f"✓ Extraction Complete!")
    print(f"{'='*80}")
    print(f"Episodes processed: {successful}/{len(sorted_episodes)}")
    print(f"Total frames extracted: {total_frames:,}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    return {
        "total_episodes": len(sorted_episodes),
        "successful_episodes": successful,
        "total_frames": total_frames,
        "output_dir": str(output_dir),
        "config": config
    }


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Universal frame extraction tool for video collections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scene-based extraction (default)
  python universal_frame_extractor.py /path/to/videos --mode scene

  # Extract every 5 seconds
  python universal_frame_extractor.py /path/to/videos --mode interval --interval-seconds 5

  # Extract every 100 frames
  python universal_frame_extractor.py /path/to/videos --mode interval --interval-frames 100

  # Hybrid mode (both scene and interval)
  python universal_frame_extractor.py /path/to/videos --mode hybrid --interval-seconds 10

  # Custom scene sensitivity
  python universal_frame_extractor.py /path/to/videos --scene-threshold 20 --frames-per-scene 5

  # Process specific episode range
  python universal_frame_extractor.py /path/to/videos --start 10 --end 20

  # Use custom episode number pattern (e.g., "S01E05" format)
  python universal_frame_extractor.py /path/to/videos --episode-pattern "E(\d+)"

  # Multi-season anime: preserve full filename (avoids S1.01/S2.01 overwrites)
  python universal_frame_extractor.py /path/to/videos --episode-pattern "none"
        """
    )

    # Required arguments
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing video files'
    )

    # Extraction mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['scene', 'interval', 'hybrid'],
        default='scene',
        help='Extraction strategy (default: scene)'
    )

    # Scene-based parameters
    parser.add_argument(
        '--scene-threshold',
        type=float,
        default=27.0,
        help='Scene detection threshold (default: 27.0, lower=more sensitive)'
    )
    parser.add_argument(
        '--frames-per-scene',
        type=int,
        default=3,
        help='Frames to extract per scene (default: 3)'
    )
    parser.add_argument(
        '--no-skip-boundaries',
        action='store_true',
        help='Do not skip scene boundary frames'
    )

    # Interval-based parameters
    parser.add_argument(
        '--interval-seconds',
        type=float,
        default=None,
        help='Extract one frame every N seconds'
    )
    parser.add_argument(
        '--interval-frames',
        type=int,
        default=None,
        help='Extract one frame every N frames'
    )

    # Output parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: input_dir/extracted_frames)'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default=None,
        help='Temporary directory (default: input_dir/temp)'
    )
    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=95,
        help='JPEG quality 1-100 (default: 95)'
    )

    # Processing parameters
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: auto, max {cpu_count()})'
    )
    parser.add_argument(
        '--episode-pattern',
        type=str,
        default=r'(\d+)',
        help='Regex pattern to extract episode number (default: any digits). '
             'Use "none" to preserve full filename (avoids overwrites for multi-season anime)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=None,
        help='Start episode number'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End episode number'
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"✗ Input directory not found: {input_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "extracted_frames"
    temp_dir = Path(args.temp_dir) if args.temp_dir else input_dir / "temp"

    # Convert "none" pattern to None (use full filename)
    episode_pattern = None if args.episode_pattern.lower() == "none" else args.episode_pattern

    # Create extraction config
    config = ExtractionConfig(
        mode=ExtractionMode(args.mode),
        scene_threshold=args.scene_threshold,
        frames_per_scene=args.frames_per_scene,
        skip_scene_boundaries=not args.no_skip_boundaries,
        interval_seconds=args.interval_seconds,
        interval_frames=args.interval_frames,
        jpeg_quality=args.jpeg_quality
    )

    # Process videos
    results = process_video_collection(
        input_dir=input_dir,
        output_dir=output_dir,
        temp_dir=temp_dir,
        config=config,
        episode_pattern=episode_pattern,
        num_workers=args.workers,
        start_episode=args.start,
        end_episode=args.end
    )

    # Save results to JSON
    results_file = output_dir / "extraction_results.json"
    with open(results_file, 'w') as f:
        # Convert non-serializable objects
        results_copy = results.copy()
        if 'config' in results_copy:
            results_copy['config'] = {
                'mode': results_copy['config'].mode.value,
                'scene_threshold': results_copy['config'].scene_threshold,
                'frames_per_scene': results_copy['config'].frames_per_scene,
                'interval_seconds': results_copy['config'].interval_seconds,
                'interval_frames': results_copy['config'].interval_frames,
                'jpeg_quality': results_copy['config'].jpeg_quality
            }
        json.dump(results_copy, f, indent=2)

    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()
