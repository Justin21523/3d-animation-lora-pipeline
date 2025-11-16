#!/usr/bin/env python3
"""
Context-Aware Inpainting

Purpose: Inpaint instances using scene context and temporal information
Features: Multi-reference inpainting, scene context integration, character prompts
Use Cases: High-quality inpainting for character instances with temporal coherence

Usage:
    python inpaint_context_aware.py \
        --instances-dir /path/to/instances \
        --frames-dir /path/to/frames \
        --output-dir /path/to/inpainted \
        --method lama \
        --use-temporal-context \
        --window-size 10 \
        --project luca
"""

import argparse
import json
import subprocess
import sys
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
class InpaintingConfig:
    """Configuration for context-aware inpainting"""
    method: str = "lama"  # lama, opencv, sd (stable diffusion)
    use_temporal_context: bool = True
    window_size: int = 10
    max_references: int = 3
    use_prompts: bool = True
    prompt_file: Optional[Path] = None
    default_prompt: str = "3d animated character, pixar style, smooth shading"
    blend_mode: str = "weighted"  # weighted, average, best
    quality: str = "high"  # low, medium, high
    device: str = "cuda"
    batch_size: int = 4


class ContextAwareInpainter:
    """Inpaint instances using temporal context"""

    def __init__(self, config: InpaintingConfig):
        """
        Initialize inpainter

        Args:
            config: Inpainting configuration
        """
        self.config = config
        self.script_dir = Path(__file__).parent

        # Load prompts if available
        self.prompts = {}
        if config.use_prompts and config.prompt_file:
            self.load_prompts(config.prompt_file)

    def load_prompts(self, prompt_file: Path):
        """Load character prompts from file"""
        if prompt_file.exists():
            with open(prompt_file) as f:
                self.prompts = json.load(f)
            print(f"✅ Loaded {len(self.prompts)} character prompts")

    def extract_temporal_context(
        self,
        instance_path: Path,
        frames_dir: Path,
        output_dir: Path
    ) -> Optional[Dict]:
        """
        Extract temporal context for instance

        Args:
            instance_path: Instance image path
            frames_dir: Frames directory
            output_dir: Context output directory

        Returns:
            Context metadata dict
        """
        context_dir = output_dir / instance_path.stem / "context"
        context_dir.mkdir(parents=True, exist_ok=True)

        # Run temporal context extractor
        extractor_script = self.script_dir / "temporal_context_extractor.py"

        cmd = [
            sys.executable,
            str(extractor_script),
            "--frames-dir", str(frames_dir),
            "--instance-path", str(instance_path),
            "--output-dir", str(context_dir),
            "--window-size", str(self.config.window_size),
            "--max-references", str(self.config.max_references),
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            # Load context metadata
            context_file = context_dir / "context.json"
            if context_file.exists():
                with open(context_file) as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Error extracting context for {instance_path.name}: {e}")

        return None

    def inpaint_lama(
        self,
        instance_path: Path,
        mask_path: Path,
        output_path: Path,
        context: Optional[Dict] = None
    ):
        """
        Inpaint using LaMa (Fast Fourier Convolution)

        Args:
            instance_path: Instance image
            mask_path: Inpainting mask
            output_path: Output path
            context: Temporal context (optional)
        """
        # Load instance and create mask
        instance = cv2.imread(str(instance_path))
        if instance is None:
            return

        # Create mask from alpha channel or use provided mask
        if instance.shape[2] == 4:
            mask = instance[:, :, 3]
            mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)[1]
        elif mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Create mask from transparency
            gray = cv2.cvtColor(instance, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)[1]

        # Use LaMa inpainting
        try:
            # Try to use lama-cleaner or simple-lama-inpainting
            # For now, use OpenCV as fallback
            result = cv2.inpaint(instance, mask, 3, cv2.INPAINT_TELEA)

            # If we have context, blend with reference frames
            if context and context.get("references"):
                reference_images = []

                for ref in context["references"]:
                    ref_path = Path(ref["saved_path"])
                    if ref_path.exists():
                        ref_img = cv2.imread(str(ref_path))
                        if ref_img is not None and ref_img.shape == instance.shape:
                            reference_images.append((ref_img, ref["similarity"]))

                if reference_images:
                    # Weighted blend with references
                    if self.config.blend_mode == "weighted":
                        total_weight = sum(sim for _, sim in reference_images)
                        blended = np.zeros_like(instance, dtype=np.float32)

                        for ref_img, sim in reference_images:
                            weight = sim / total_weight
                            blended += ref_img.astype(np.float32) * weight

                        # Blend inpainted result with references
                        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        mask_3ch = mask_3ch.astype(np.float32) / 255.0

                        result = result.astype(np.float32) * (1 - mask_3ch * 0.3) + \
                                blended * (mask_3ch * 0.3)
                        result = result.astype(np.uint8)

            cv2.imwrite(str(output_path), result)

        except Exception as e:
            print(f"⚠️ LaMa inpainting failed: {e}, using OpenCV fallback")
            result = cv2.inpaint(instance, mask, 3, cv2.INPAINT_TELEA)
            cv2.imwrite(str(output_path), result)

    def inpaint_opencv(
        self,
        instance_path: Path,
        mask_path: Path,
        output_path: Path,
        context: Optional[Dict] = None
    ):
        """
        Inpaint using OpenCV (Telea or NS)

        Args:
            instance_path: Instance image
            mask_path: Inpainting mask
            output_path: Output path
            context: Temporal context (optional)
        """
        instance = cv2.imread(str(instance_path))
        if instance is None:
            return

        # Create mask
        if instance.shape[2] == 4:
            mask = instance[:, :, 3]
            mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)[1]
        elif mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            gray = cv2.cvtColor(instance, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)[1]

        # Use Telea method
        result = cv2.inpaint(instance, mask, 3, cv2.INPAINT_TELEA)

        cv2.imwrite(str(output_path), result)

    def inpaint_stable_diffusion(
        self,
        instance_path: Path,
        mask_path: Path,
        output_path: Path,
        context: Optional[Dict] = None
    ):
        """
        Inpaint using Stable Diffusion

        Args:
            instance_path: Instance image
            mask_path: Inpainting mask
            output_path: Output path
            context: Temporal context (optional)
        """
        # Get character prompt
        prompt = self.config.default_prompt

        if self.config.use_prompts and self.prompts:
            # Try to match instance to character
            instance_name = instance_path.stem.lower()

            for char_name, char_prompt in self.prompts.items():
                if char_name.lower() in instance_name:
                    prompt = char_prompt
                    break

        # TODO: Implement SD inpainting
        # For now, use fallback
        print(f"⚠️ SD inpainting not yet implemented, using OpenCV fallback")
        self.inpaint_opencv(instance_path, mask_path, output_path, context)

    def inpaint_single(
        self,
        instance_path: Path,
        frames_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Inpaint single instance

        Args:
            instance_path: Instance image path
            frames_dir: Frames directory
            output_dir: Output directory

        Returns:
            Result metadata
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract temporal context if enabled
        context = None
        if self.config.use_temporal_context:
            context = self.extract_temporal_context(
                instance_path,
                frames_dir,
                output_dir
            )

        # Create output path
        output_path = output_dir / instance_path.name

        # Create mask path (if exists)
        mask_dir = instance_path.parent.parent / "masks"
        mask_path = mask_dir / instance_path.name if mask_dir.exists() else None

        # Inpaint based on method
        if self.config.method == "lama":
            self.inpaint_lama(instance_path, mask_path, output_path, context)
        elif self.config.method == "opencv":
            self.inpaint_opencv(instance_path, mask_path, output_path, context)
        elif self.config.method == "sd":
            self.inpaint_stable_diffusion(instance_path, mask_path, output_path, context)
        else:
            print(f"⚠️ Unknown method: {self.config.method}, using opencv")
            self.inpaint_opencv(instance_path, mask_path, output_path, context)

        result = {
            "instance": str(instance_path),
            "output": str(output_path),
            "method": self.config.method,
            "used_context": context is not None,
            "num_references": len(context.get("references", [])) if context else 0,
        }

        return result

    def inpaint_batch(
        self,
        instances_dir: Path,
        frames_dir: Path,
        output_dir: Path
    ) -> Dict:
        """
        Batch inpaint all instances

        Args:
            instances_dir: Directory with instances
            frames_dir: Directory with frames
            output_dir: Output directory

        Returns:
            Statistics dictionary
        """
        instances_dir = Path(instances_dir)
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all instances
        instances = sorted(
            list(instances_dir.glob("*.png")) +
            list(instances_dir.glob("*.jpg"))
        )

        print(f"\n📊 Found {len(instances)} instances to inpaint")
        print(f"   Method: {self.config.method}")
        print(f"   Temporal context: {'Yes' if self.config.use_temporal_context else 'No'}")

        results = []
        failed = []

        for instance_path in tqdm(instances, desc="Inpainting"):
            try:
                result = self.inpaint_single(
                    instance_path,
                    frames_dir,
                    output_dir
                )
                results.append(result)
            except Exception as e:
                print(f"\n⚠️ Failed to inpaint {instance_path.name}: {e}")
                failed.append(str(instance_path))

        # Statistics
        stats = {
            "instances_dir": str(instances_dir),
            "frames_dir": str(frames_dir),
            "output_dir": str(output_dir),
            "method": self.config.method,
            "config": {
                "use_temporal_context": self.config.use_temporal_context,
                "window_size": self.config.window_size,
                "max_references": self.config.max_references,
            },
            "total_instances": len(instances),
            "successful": len(results),
            "failed": len(failed),
            "failed_instances": failed,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

        # Save report
        report_path = output_dir / "inpainting_report.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✅ Inpainting complete!")
        print(f"   Total: {len(instances)}")
        print(f"   Successful: {len(results)}")
        print(f"   Failed: {len(failed)}")
        print(f"   Output: {output_dir}")
        print(f"   Report: {report_path}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Context-Aware Inpainting (Film-Agnostic)"
    )
    parser.add_argument(
        "--instances-dir",
        type=str,
        required=True,
        help="Directory with instance images"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory with source frames"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for inpainted instances"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="lama",
        choices=["lama", "opencv", "sd"],
        help="Inpainting method (default: lama)"
    )
    parser.add_argument(
        "--no-temporal-context",
        action="store_true",
        help="Disable temporal context extraction"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Temporal window size (default: 10)"
    )
    parser.add_argument(
        "--max-references",
        type=int,
        default=3,
        help="Maximum reference frames (default: 3)"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="JSON file with character prompts"
    )
    parser.add_argument(
        "--default-prompt",
        type=str,
        default="3d animated character, pixar style, smooth shading",
        help="Default prompt for SD inpainting"
    )
    parser.add_argument(
        "--blend-mode",
        type=str,
        default="weighted",
        choices=["weighted", "average", "best"],
        help="Blending mode for references (default: weighted)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project/film name"
    )

    args = parser.parse_args()

    # Create config
    config = InpaintingConfig(
        method=args.method,
        use_temporal_context=not args.no_temporal_context,
        window_size=args.window_size,
        max_references=args.max_references,
        use_prompts=args.prompt_file is not None,
        prompt_file=Path(args.prompt_file) if args.prompt_file else None,
        default_prompt=args.default_prompt,
        blend_mode=args.blend_mode,
        device=args.device,
    )

    # Run inpainting
    inpainter = ContextAwareInpainter(config)
    stats = inpainter.inpaint_batch(
        instances_dir=Path(args.instances_dir),
        frames_dir=Path(args.frames_dir),
        output_dir=Path(args.output_dir)
    )

    if args.project:
        print(f"\n💡 Project: {args.project}")


if __name__ == "__main__":
    main()
