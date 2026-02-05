#!/usr/bin/env python3
"""
Unified Enhancement Pipeline

Orchestrates multiple enhancement steps:
1. Face Restoration (CodeFormer)
2. Super-Resolution (Real-ESRGAN) - optional
3. Deblurring (NAFNet) - optional
4. Background Cleanup
5. Quality Filtering

Usage:
  python enhancement_pipeline.py INPUT_DIR --output-dir OUTPUT_DIR --preset 3d_character
"""

import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, List
import shutil


class EnhancementPipeline:
    """
    Orchestrates image enhancement pipeline
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        device: str = "cuda",
        preset: str = "3d_character"
    ):
        """
        Initialize enhancement pipeline

        Args:
            input_dir: Directory with instances to enhance
            output_dir: Output directory
            device: cuda or cpu
            preset: Enhancement preset (3d_character, 2d_anime, photo)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        self.preset = preset

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load preset configuration
        self.config = self._load_preset(preset)

        print(f"🎨 Enhancement Pipeline Initialized")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")
        print(f"   Preset: {preset}")
        print(f"   Device: {device}")
        print()

    def _load_preset(self, preset: str) -> dict:
        """Load preset configuration"""
        presets = {
            "3d_character": {
                "face_restoration": {
                    "enabled": True,
                    "fidelity": 0.7,
                    "upscale": 2
                },
                "super_resolution": {
                    "enabled": False,  # Only if needed
                    "min_size": 512,
                    "model": "RealESRGAN_x2plus"
                },
                "deblur": {
                    "enabled": False,  # Only if motion blur detected
                    "blur_threshold": 80,
                    "strength": 0.7
                },
                "background_cleanup": {
                    "enabled": True,
                    "edge_smoothing": True,
                    "alpha_threshold": 10
                },
                "quality_filter": {
                    "enabled": True,
                    "min_face_size": 64,
                    "min_blur_score": 50
                }
            },
            "2d_anime": {
                "face_restoration": {
                    "enabled": True,
                    "fidelity": 0.6,
                    "upscale": 2
                },
                "super_resolution": {
                    "enabled": True,
                    "min_size": 512,
                    "model": "RealESRGAN_x4plus_anime_6B"
                },
                "deblur": {
                    "enabled": True,
                    "blur_threshold": 100,
                    "strength": 0.8
                },
                "background_cleanup": {
                    "enabled": True,
                    "edge_smoothing": True,
                    "alpha_threshold": 15
                },
                "quality_filter": {
                    "enabled": True,
                    "min_face_size": 48,
                    "min_blur_score": 60
                }
            }
        }

        return presets.get(preset, presets["3d_character"])

    def run(self, stages: Optional[List[str]] = None):
        """
        Run enhancement pipeline

        Args:
            stages: List of stages to run (None = all enabled stages)
        """
        current_dir = self.input_dir
        pipeline_log = []

        print("🚀 Starting Enhancement Pipeline")
        print("=" * 60)

        # Stage 1: Face Restoration
        if (stages is None or "face_restoration" in stages) and self.config["face_restoration"]["enabled"]:
            print("\n📍 Stage 1: Face Restoration")
            print("-" * 60)

            stage_output = self.output_dir / "01_face_restored"
            success = self._run_face_restoration(current_dir, stage_output)

            if success:
                pipeline_log.append({
                    "stage": "face_restoration",
                    "input": str(current_dir),
                    "output": str(stage_output),
                    "status": "success"
                })
                current_dir = stage_output / "restored"
            else:
                print("⚠️  Face restoration failed, continuing with original images...")
                pipeline_log.append({
                    "stage": "face_restoration",
                    "status": "failed"
                })

        # Stage 2: Super-Resolution (optional)
        if (stages is None or "super_resolution" in stages) and self.config["super_resolution"]["enabled"]:
            print("\n📍 Stage 2: Super-Resolution")
            print("-" * 60)

            stage_output = self.output_dir / "02_upscaled"
            success = self._run_super_resolution(current_dir, stage_output)

            if success:
                pipeline_log.append({
                    "stage": "super_resolution",
                    "input": str(current_dir),
                    "output": str(stage_output),
                    "status": "success"
                })
                current_dir = stage_output / "upscaled"
            else:
                print("⚠️  Super-resolution failed, skipping...")
                pipeline_log.append({
                    "stage": "super_resolution",
                    "status": "skipped"
                })

        # Stage 3: Deblurring (optional)
        if (stages is None or "deblur" in stages) and self.config["deblur"]["enabled"]:
            print("\n📍 Stage 3: Deblurring")
            print("-" * 60)

            stage_output = self.output_dir / "03_deblurred"
            success = self._run_deblur(current_dir, stage_output)

            if success:
                pipeline_log.append({
                    "stage": "deblur",
                    "input": str(current_dir),
                    "output": str(stage_output),
                    "status": "success"
                })
                current_dir = stage_output / "deblurred"
            else:
                print("⚠️  Deblurring failed, skipping...")
                pipeline_log.append({
                    "stage": "deblur",
                    "status": "skipped"
                })

        # Stage 4: Background Cleanup
        if (stages is None or "background_cleanup" in stages) and self.config["background_cleanup"]["enabled"]:
            print("\n📍 Stage 4: Background Cleanup")
            print("-" * 60)

            stage_output = self.output_dir / "04_cleaned"
            success = self._run_background_cleanup(current_dir, stage_output)

            if success:
                pipeline_log.append({
                    "stage": "background_cleanup",
                    "input": str(current_dir),
                    "output": str(stage_output),
                    "status": "success"
                })
                current_dir = stage_output / "cleaned"
            else:
                print("⚠️  Background cleanup failed, skipping...")
                pipeline_log.append({
                    "stage": "background_cleanup",
                    "status": "skipped"
                })

        # Create final output symlink
        final_dir = self.output_dir / "final"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.symlink_to(current_dir.resolve(), target_is_directory=True)

        print("\n" + "=" * 60)
        print("✅ Enhancement Pipeline Complete!")
        print(f"📁 Final output: {final_dir}")

        # Save pipeline log
        log_path = self.output_dir / "pipeline_log.json"
        with open(log_path, 'w') as f:
            json.dump({
                'pipeline_log': pipeline_log,
                'final_output': str(final_dir),
                'preset': self.preset,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        print(f"📊 Pipeline log saved: {log_path}")

    def _run_face_restoration(self, input_dir: Path, output_dir: Path) -> bool:
        """Run face restoration stage"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))

            from face_restoration import process_instances

            config = self.config["face_restoration"]

            process_instances(
                input_dir=input_dir,
                output_dir=output_dir,
                device=self.device,
                fidelity_weight=config["fidelity"],
                upscale=config["upscale"],
                save_comparison=False
            )

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def _run_super_resolution(self, input_dir: Path, output_dir: Path) -> bool:
        """Run super-resolution stage"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))

            from super_resolution import process_instances

            config = self.config["super_resolution"]

            process_instances(
                input_dir=input_dir,
                output_dir=output_dir,
                model_name=config["model"],
                device=self.device,
                min_size=config["min_size"]
            )

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def _run_deblur(self, input_dir: Path, output_dir: Path) -> bool:
        """Run deblurring stage"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))

            from deblur import process_instances

            config = self.config["deblur"]

            process_instances(
                input_dir=input_dir,
                output_dir=output_dir,
                device=self.device,
                blur_threshold=config["blur_threshold"],
                strength=config["strength"]
            )

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def _run_background_cleanup(self, input_dir: Path, output_dir: Path) -> bool:
        """Run background cleanup stage"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))

            from background_cleanup import process_instances

            config = self.config["background_cleanup"]

            process_instances(
                input_dir=input_dir,
                output_dir=output_dir,
                background_color=(0, 0, 0, 0),
                edge_smoothing=config["edge_smoothing"],
                alpha_threshold=config["alpha_threshold"]
            )

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Unified image enhancement pipeline"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory with character instances"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for enhanced instances"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="3d_character",
        choices=["3d_character", "2d_anime"],
        help="Enhancement preset (default: 3d_character)"
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs='+',
        choices=["face_restoration", "super_resolution", "deblur", "background_cleanup"],
        help="Specific stages to run (default: all enabled stages)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = EnhancementPipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        preset=args.preset
    )

    pipeline.run(stages=args.stages)


if __name__ == "__main__":
    main()
