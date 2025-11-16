#!/usr/bin/env python3
"""
Video Synthesizer
Combine frames into video files with optional audio
"""

import subprocess
from pathlib import Path
import argparse
from typing import Optional, List
import json
from datetime import datetime


class VideoSynthesizer:
    """Synthesize videos from frame sequences"""

    def __init__(
        self,
        fps: int = 30,
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "medium",
        audio_codec: str = "aac",
        audio_bitrate: str = "192k"
    ):
        """
        Initialize video synthesizer

        Args:
            fps: Frames per second
            codec: Video codec (libx264, libx265, etc.)
            crf: Constant Rate Factor (0-51, lower=better quality)
            preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
            audio_codec: Audio codec
            audio_bitrate: Audio bitrate
        """
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate

        # Check if ffmpeg is available
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            print("✓ ffmpeg found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    def create_video(
        self,
        input_pattern: str,
        output_path: Path,
        audio_path: Optional[Path] = None,
        start_number: int = 0,
        pixel_format: str = "yuv420p",
        resolution: Optional[str] = None,
        loop: int = 0
    ) -> dict:
        """
        Create video from frame sequence

        Args:
            input_pattern: Input pattern (e.g., "frame_%06d.jpg")
            output_path: Output video path
            audio_path: Optional audio file to add
            start_number: Starting frame number
            pixel_format: Pixel format (yuv420p for compatibility)
            resolution: Optional resolution (e.g., "1920x1080", "1280x720")
            loop: Number of times to loop (0 = no loop)

        Returns:
            Synthesis statistics
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(self.fps),
            "-start_number", str(start_number),
            "-i", input_pattern,
        ]

        # Add audio if provided
        if audio_path and audio_path.exists():
            cmd.extend(["-i", str(audio_path)])

        # Video encoding options
        cmd.extend([
            "-c:v", self.codec,
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-pix_fmt", pixel_format
        ])

        # Resolution scaling
        if resolution:
            cmd.extend(["-s", resolution])

        # Audio encoding options
        if audio_path and audio_path.exists():
            cmd.extend([
                "-c:a", self.audio_codec,
                "-b:a", self.audio_bitrate,
                "-shortest"  # Match shortest stream duration
            ])

        # Loop option
        if loop > 0:
            cmd.extend(["-stream_loop", str(loop)])

        # Output file
        cmd.append(str(output_path))

        print(f"Creating video: {output_path.name}")
        print(f"Command: {' '.join(cmd)}")

        # Run ffmpeg
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            # Parse output for statistics
            stats = {
                'success': True,
                'output_path': str(output_path),
                'fps': self.fps,
                'codec': self.codec,
                'crf': self.crf,
                'has_audio': audio_path is not None and audio_path.exists(),
                'audio_path': str(audio_path) if audio_path else None,
                'resolution': resolution,
                'timestamp': datetime.now().isoformat()
            }

            # Get file size
            if output_path.exists():
                stats['file_size_mb'] = output_path.stat().st_size / (1024 * 1024)

            print(f"✓ Video created: {output_path}")
            if 'file_size_mb' in stats:
                print(f"  Size: {stats['file_size_mb']:.2f} MB")

            return stats

        except subprocess.CalledProcessError as e:
            print(f"✗ FFmpeg error: {e.stderr}")
            return {
                'success': False,
                'error': str(e),
                'stderr': e.stderr
            }

    def batch_create_videos(
        self,
        frame_directories: List[Path],
        output_dir: Path,
        frame_pattern: str = "frame_%06d.jpg",
        audio_dir: Optional[Path] = None,
        audio_pattern: Optional[str] = None
    ) -> List[dict]:
        """
        Create multiple videos from frame directories

        Args:
            frame_directories: List of directories containing frames
            output_dir: Output directory for videos
            frame_pattern: Pattern for frame files
            audio_dir: Optional directory containing audio files
            audio_pattern: Pattern to match audio files (by episode name)

        Returns:
            List of synthesis statistics for each video
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for frame_dir in frame_directories:
            if not frame_dir.is_dir():
                continue

            # Determine output filename
            video_name = f"{frame_dir.name}.mp4"
            output_path = output_dir / video_name

            # Find matching audio file if audio_dir provided
            audio_path = None
            if audio_dir and audio_dir.exists():
                if audio_pattern:
                    # Try to find matching audio file
                    audio_files = list(audio_dir.glob(f"*{frame_dir.name}*"))
                    if audio_files:
                        audio_path = audio_files[0]
                else:
                    # Look for audio file with same name
                    for ext in ['.wav', '.mp3', '.aac', '.m4a']:
                        potential_audio = audio_dir / f"{frame_dir.name}{ext}"
                        if potential_audio.exists():
                            audio_path = potential_audio
                            break

            # Create video
            input_pattern = str(frame_dir / frame_pattern)
            stats = self.create_video(
                input_pattern=input_pattern,
                output_path=output_path,
                audio_path=audio_path
            )
            stats['frame_directory'] = str(frame_dir)
            results.append(stats)

        return results

    def create_comparison_grid(
        self,
        video_paths: List[Path],
        output_path: Path,
        grid_layout: str = "2x2",
        add_labels: bool = True
    ) -> dict:
        """
        Create side-by-side comparison video grid

        Args:
            video_paths: List of video files to compare
            output_path: Output video path
            grid_layout: Grid layout (e.g., "2x2", "3x1", "1x4")
            add_labels: Add text labels for each video

        Returns:
            Synthesis statistics
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Parse grid layout
        rows, cols = map(int, grid_layout.split('x'))
        if len(video_paths) > rows * cols:
            raise ValueError(f"Grid {grid_layout} can only fit {rows * cols} videos, but {len(video_paths)} provided")

        # Build filter complex for grid
        inputs = []
        for vp in video_paths:
            inputs.extend(["-i", str(vp)])

        # Create xstack filter
        # For simplicity, we'll use hstack/vstack for 2-video comparisons
        if len(video_paths) == 2 and grid_layout == "2x1":
            filter_complex = "[0:v][1:v]hstack=inputs=2[v]"
        elif len(video_paths) == 2 and grid_layout == "1x2":
            filter_complex = "[0:v][1:v]vstack=inputs=2[v]"
        elif len(video_paths) == 4 and grid_layout == "2x2":
            filter_complex = "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]"
        else:
            raise NotImplementedError(f"Grid layout {grid_layout} not implemented for {len(video_paths)} videos")

        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-c:v", self.codec,
            "-crf", str(self.crf),
            "-preset", self.preset,
            str(output_path)
        ]

        print(f"Creating comparison grid: {grid_layout}")
        print(f"Videos: {[vp.name for vp in video_paths]}")

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Comparison video created: {output_path}")
            return {
                'success': True,
                'output_path': str(output_path),
                'grid_layout': grid_layout,
                'num_videos': len(video_paths)
            }
        except subprocess.CalledProcessError as e:
            print(f"✗ FFmpeg error: {e.stderr.decode()}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    parser = argparse.ArgumentParser(description="Synthesize videos from frame sequences")

    parser.add_argument("input_dir", type=Path, help="Directory containing frames")
    parser.add_argument("--output", type=Path, required=True, help="Output video path")
    parser.add_argument("--pattern", type=str, default="frame_%06d.jpg",
                       help="Frame filename pattern")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--audio", type=Path, help="Optional audio file")
    parser.add_argument("--codec", type=str, default="libx264", help="Video codec")
    parser.add_argument("--crf", type=int, default=18, help="Constant Rate Factor (0-51)")
    parser.add_argument("--preset", type=str, default="medium",
                       choices=['ultrafast', 'fast', 'medium', 'slow', 'veryslow'],
                       help="Encoding preset")
    parser.add_argument("--resolution", type=str, help="Output resolution (e.g., 1920x1080)")
    parser.add_argument("--start-number", type=int, default=0, help="Starting frame number")

    # Batch mode
    parser.add_argument("--batch", action="store_true", help="Batch process multiple frame directories")
    parser.add_argument("--output-dir", type=Path, help="Output directory for batch mode")
    parser.add_argument("--audio-dir", type=Path, help="Directory containing audio files for batch mode")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Video Synthesis")
    print(f"{'='*80}\n")

    # Initialize synthesizer
    synthesizer = VideoSynthesizer(
        fps=args.fps,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset
    )

    if args.batch:
        # Batch mode: process multiple directories
        if not args.output_dir:
            raise ValueError("--output-dir required for batch mode")

        # Find all subdirectories in input_dir
        frame_dirs = [d for d in args.input_dir.iterdir() if d.is_dir()]
        print(f"Found {len(frame_dirs)} frame directories")

        results = synthesizer.batch_create_videos(
            frame_directories=frame_dirs,
            output_dir=args.output_dir,
            frame_pattern=args.pattern,
            audio_dir=args.audio_dir
        )

        # Save batch metadata
        metadata_path = args.output_dir / "synthesis_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_videos': len(results),
                'successful': sum(1 for r in results if r.get('success')),
                'results': results
            }, f, indent=2)

        successful = sum(1 for r in results if r.get('success'))
        print(f"\n✅ Batch complete: {successful}/{len(results)} videos created")
        print(f"Output: {args.output_dir}")

    else:
        # Single video mode
        input_pattern = str(args.input_dir / args.pattern)

        stats = synthesizer.create_video(
            input_pattern=input_pattern,
            output_path=args.output,
            audio_path=args.audio,
            start_number=args.start_number,
            resolution=args.resolution
        )

        if stats['success']:
            print(f"\n✅ Video created successfully!")
            print(f"Output: {args.output}")
        else:
            print(f"\n❌ Video creation failed")
            print(f"Error: {stats.get('error')}")


if __name__ == "__main__":
    main()
