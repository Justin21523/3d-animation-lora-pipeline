#!/usr/bin/env python3
"""
Frame Interpolation Tool
Generate smooth intermediate frames using RIFE (Real-Time Intermediate Flow Estimation)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import argparse
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import json
from datetime import datetime


try:
    # Try importing RIFE if available
    from rife_ncnn_vulkan_python import Rife

    RIFE_AVAILABLE = True
except ImportError:
    RIFE_AVAILABLE = False
    print("⚠️  RIFE not available. Will use basic interpolation.")


class FrameInterpolator:
    """
    Interpolate frames to create smooth animations
    Supports RIFE for high-quality interpolation or basic averaging fallback
    """

    def __init__(
        self,
        model: str = "rife-v4.6",
        num_threads: int = 4,
        gpu_id: int = 0,
        use_rife: bool = True,
    ):
        """
        Initialize frame interpolator

        Args:
            model: RIFE model version
            num_threads: Number of threads for processing
            gpu_id: GPU device ID
            use_rife: Use RIFE if available, else basic interpolation
        """
        self.use_rife = use_rife and RIFE_AVAILABLE

        if self.use_rife:
            try:
                self.rife = Rife(gpuid=gpu_id, model=model, num_threads=num_threads)
                print(f"✓ RIFE {model} initialized on GPU {gpu_id}")
            except Exception as e:
                print(f"⚠️  RIFE initialization failed: {e}")
                print("Falling back to basic interpolation")
                self.use_rife = False
        else:
            print("Using basic frame interpolation (linear blending)")

    def interpolate_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_intermediate: int = 1,
        time_step: Optional[float] = None,
    ) -> List[np.ndarray]:
        """
        Interpolate between two frames

        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)
            num_intermediate: Number of intermediate frames to generate
            time_step: Custom time step (0-1), overrides num_intermediate

        Returns:
            List of interpolated frames (does not include input frames)
        """
        if self.use_rife:
            return self._interpolate_rife(frame1, frame2, num_intermediate, time_step)
        else:
            return self._interpolate_basic(frame1, frame2, num_intermediate, time_step)

    def _interpolate_rife(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_intermediate: int = 1,
        time_step: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Interpolate using RIFE"""
        interpolated = []

        if time_step is not None:
            # Single interpolation at specific time
            result = self.rife.process(frame1, frame2, timestep=time_step)
            interpolated.append(result)
        else:
            # Multiple evenly-spaced interpolations
            for i in range(1, num_intermediate + 1):
                t = i / (num_intermediate + 1)
                result = self.rife.process(frame1, frame2, timestep=t)
                interpolated.append(result)

        return interpolated

    def _interpolate_basic(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_intermediate: int = 1,
        time_step: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Basic linear interpolation (fallback)"""
        interpolated = []

        if time_step is not None:
            # Single blend at specific ratio
            blended = cv2.addWeighted(frame1, 1 - time_step, frame2, time_step, 0)
            interpolated.append(blended)
        else:
            # Multiple evenly-spaced blends
            for i in range(1, num_intermediate + 1):
                alpha = i / (num_intermediate + 1)
                blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
                interpolated.append(blended)

        return interpolated

    def process_sequence(
        self, frames: List[np.ndarray], multiplier: int = 2
    ) -> List[np.ndarray]:
        """
        Process entire frame sequence, interpolating between each pair

        Args:
            frames: List of input frames
            multiplier: Frame rate multiplier (2x, 4x, etc.)

        Returns:
            List of frames with interpolations inserted
        """
        if multiplier < 2:
            return frames

        # Calculate how many intermediate frames per pair
        num_intermediate = multiplier - 1

        output_frames = []

        for i in tqdm(range(len(frames) - 1), desc="Interpolating frames"):
            # Add original frame
            output_frames.append(frames[i])

            # Add interpolated frames
            interpolated = self.interpolate_pair(
                frames[i], frames[i + 1], num_intermediate=num_intermediate
            )
            output_frames.extend(interpolated)

        # Add final frame
        output_frames.append(frames[-1])

        return output_frames

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        multiplier: int = 2,
        file_pattern: str = "*.jpg",
        save_format: str = "jpg",
        jpeg_quality: int = 95,
    ) -> Dict:
        """
        Process all frames in a directory

        Args:
            input_dir: Input directory with sequential frames
            output_dir: Output directory for interpolated frames
            multiplier: Frame rate multiplier
            file_pattern: Pattern to match input files
            save_format: Output format (jpg/png)
            jpeg_quality: JPEG quality (if using jpg)

        Returns:
            Processing statistics
        """
        # Load frames
        frame_files = sorted(input_dir.glob(file_pattern))
        if not frame_files:
            raise ValueError(f"No frames found in {input_dir} matching {file_pattern}")

        print(f"Loading {len(frame_files)} frames from {input_dir}")
        frames = []
        for f in tqdm(frame_files, desc="Loading frames"):
            img = cv2.imread(str(f))
            if img is not None:
                frames.append(img)

        print(f"✓ Loaded {len(frames)} frames")

        # Interpolate
        print(f"Interpolating with {multiplier}x multiplier...")
        interpolated_frames = self.process_sequence(frames, multiplier)

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(interpolated_frames)} frames to {output_dir}")

        for idx, frame in enumerate(tqdm(interpolated_frames, desc="Saving frames")):
            if save_format.lower() == "jpg":
                filename = f"frame_{idx:06d}.jpg"
                filepath = output_dir / filename
                cv2.imwrite(
                    str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                )
            else:
                filename = f"frame_{idx:06d}.png"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), frame)

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "input_frames": len(frames),
            "output_frames": len(interpolated_frames),
            "multiplier": multiplier,
            "method": "RIFE" if self.use_rife else "basic",
            "save_format": save_format,
        }

        metadata_path = output_dir / "interpolation_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to: {metadata_path}")

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Frame interpolation for smooth animation"
    )

    parser.add_argument(
        "input_dir", type=Path, help="Input directory with sequential frames"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=2,
        choices=[2, 4, 8],
        help="Frame rate multiplier (2x, 4x, 8x)",
    )
    parser.add_argument(
        "--pattern", type=str, default="*.jpg", help="Input file pattern"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Output format",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95, help="JPEG quality (1-100)"
    )
    parser.add_argument(
        "--model", type=str, default="rife-v4.6", help="RIFE model version"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument(
        "--no-rife", action="store_true", help="Disable RIFE, use basic interpolation"
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    print(f"\n{'='*80}")
    print(f"Frame Interpolation")
    print(f"{'='*80}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Multiplier: {args.multiplier}x")
    print(f"Method: {'RIFE' if not args.no_rife and RIFE_AVAILABLE else 'Basic'}")
    print(f"{'='*80}\n")

    # Initialize interpolator
    interpolator = FrameInterpolator(
        model=args.model,
        num_threads=args.threads,
        gpu_id=args.gpu,
        use_rife=not args.no_rife,
    )

    # Process directory
    stats = interpolator.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        multiplier=args.multiplier,
        file_pattern=args.pattern,
        save_format=args.format,
        jpeg_quality=args.jpeg_quality,
    )

    print(f"\n✅ Complete!")
    print(f"Input frames:  {stats['input_frames']}")
    print(f"Output frames: {stats['output_frames']}")
    print(f"Multiplier:    {stats['multiplier']}x")
    print(f"Output:        {stats['output_dir']}")


if __name__ == "__main__":
    main()
