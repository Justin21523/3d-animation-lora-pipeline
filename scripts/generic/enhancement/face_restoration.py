#!/usr/bin/env python3
"""
Face Restoration for Character Instances

Uses CodeFormer to enhance facial details in character instances.
Optimized for 3D animated characters with conservative fidelity settings.

Key Features:
- Automatic face detection
- Preserves 3D animation style
- Batch processing with resume capability
- Before/after comparison
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import Optional, Tuple, List
from tqdm import tqdm
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class FaceRestorer:
    """
    Face restoration using CodeFormer
    """

    def __init__(
        self,
        device: str = "cuda",
        fidelity_weight: float = 0.7,
        upscale: int = 2,
        face_detector: str = "retinaface"
    ):
        """
        Initialize face restoration model

        Args:
            device: cuda or cpu
            fidelity_weight: Balance between restoration and original (0-1)
                           0.5 = more restoration, 1.0 = preserve original
                           For 3D anime, use 0.7-0.8 to maintain style
            upscale: Upscale factor (1, 2, or 4)
            face_detector: Face detection backend (retinaface, yolov5)
        """
        self.device = device
        self.fidelity_weight = fidelity_weight
        self.upscale = upscale
        self.face_detector_name = face_detector

        print(f"🔧 Initializing Face Restorer...")
        print(f"   Fidelity: {fidelity_weight} (higher = preserve original style)")
        print(f"   Upscale: {upscale}x")

        self._init_codeformer()
        self._init_face_detector()

    def _init_codeformer(self):
        """Initialize CodeFormer model"""
        try:
            from basicsr.archs.codeformer_arch import CodeFormer
            from basicsr.utils.download_util import load_file_from_url

            # Model path
            model_dir = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/face_restoration")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "codeformer.pth"

            # Download if not exists
            if not model_path.exists():
                print("📥 Downloading CodeFormer model...")
                url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
                load_file_from_url(url, str(model_dir), file_name="codeformer.pth")

            # Load model
            self.codeformer = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=['32', '64', '128', '256']
            ).to(self.device)

            checkpoint = torch.load(model_path, map_location=self.device)
            self.codeformer.load_state_dict(checkpoint['params_ema'])
            self.codeformer.eval()

            print("✓ CodeFormer model loaded")

        except ImportError:
            print("⚠️ CodeFormer not installed. Falling back to GFPGAN...")
            self._init_gfpgan_fallback()

    def _init_gfpgan_fallback(self):
        """Fallback to GFPGAN if CodeFormer not available"""
        try:
            from gfpgan import GFPGANer

            model_dir = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/face_restoration")
            model_path = model_dir / "GFPGANv1.4.pth"

            if not model_path.exists():
                print("📥 Downloading GFPGAN model...")
                import urllib.request
                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
                urllib.request.urlretrieve(url, model_path)

            self.gfpgan = GFPGANer(
                model_path=str(model_path),
                upscale=self.upscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )

            self.codeformer = None  # Use GFPGAN instead
            print("✓ GFPGAN model loaded")

        except Exception as e:
            print(f"❌ Failed to load face restoration model: {e}")
            raise

    def _init_face_detector(self):
        """Initialize face detection"""
        try:
            if self.face_detector_name == "retinaface":
                from retinaface import RetinaFace
                self.face_detector = RetinaFace
                print("✓ RetinaFace detector loaded")

            elif self.face_detector_name == "yolov5":
                # YOLOv5-face as alternative
                import insightface
                self.face_detector = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.face_detector.prepare(ctx_id=0 if self.device == 'cuda' else -1)
                print("✓ InsightFace (YOLOv5) detector loaded")

        except ImportError:
            print("⚠️ Face detector not available, will process all images")
            self.face_detector = None

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in image

        Args:
            image: RGB image (numpy array)

        Returns:
            List of face bounding boxes
        """
        if self.face_detector is None:
            # Return full image as single "face"
            h, w = image.shape[:2]
            return [{'bbox': [0, 0, w, h]}]

        try:
            if self.face_detector_name == "retinaface":
                faces = self.face_detector.detect_faces(image)
                if not faces:
                    return []

                face_list = []
                for key, face_info in faces.items():
                    bbox = face_info['facial_area']
                    face_list.append({'bbox': bbox})
                return face_list

            elif self.face_detector_name == "yolov5":
                faces = self.face_detector.get(image)
                if not faces:
                    return []

                face_list = []
                for face in faces:
                    bbox = face.bbox.astype(int)
                    face_list.append({'bbox': bbox})
                return face_list

        except Exception as e:
            print(f"⚠️ Face detection failed: {e}")
            return []

    def restore_face(
        self,
        image: Image.Image,
        return_comparison: bool = False
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """
        Restore faces in image

        Args:
            image: PIL Image
            return_comparison: Return side-by-side comparison

        Returns:
            (restored_image, comparison_image or None)
        """
        # Convert to numpy
        image_np = np.array(image)
        if image_np.shape[2] == 4:  # RGBA
            image_rgb = image_np[:, :, :3]
            alpha = image_np[:, :, 3]
        else:
            image_rgb = image_np
            alpha = None

        # Detect faces
        faces = self.detect_faces(image_rgb)

        if not faces:
            # No faces, return original
            if return_comparison:
                return image, None
            return image, None

        # Restore using CodeFormer or GFPGAN
        if self.codeformer is not None:
            restored_np = self._restore_with_codeformer(image_rgb)
        else:
            restored_np = self._restore_with_gfpgan(image_rgb)

        # Restore alpha channel if existed
        if alpha is not None:
            restored_rgba = np.concatenate([
                restored_np,
                alpha[:, :, None]
            ], axis=2)
            restored_img = Image.fromarray(restored_rgba.astype(np.uint8))
        else:
            restored_img = Image.fromarray(restored_np.astype(np.uint8))

        # Create comparison if requested
        if return_comparison and len(faces) > 0:
            comparison = self._create_comparison(image, restored_img)
            return restored_img, comparison

        return restored_img, None

    def _restore_with_codeformer(self, image_rgb: np.ndarray) -> np.ndarray:
        """Restore with CodeFormer"""
        # Normalize to [-1, 1]
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Restore
        with torch.no_grad():
            output = self.codeformer(
                image_tensor,
                w=self.fidelity_weight,
                adain=True
            )[0]

        # Denormalize
        output = (output * 0.5 + 0.5).clamp(0, 1)
        output = (output * 255.0).permute(1, 2, 0).cpu().numpy()

        return output.astype(np.uint8)

    def _restore_with_gfpgan(self, image_rgb: np.ndarray) -> np.ndarray:
        """Restore with GFPGAN"""
        # GFPGAN expects BGR
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Restore
        _, _, restored_bgr = self.gfpgan.enhance(
            image_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=self.fidelity_weight
        )

        # Convert back to RGB
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        return restored_rgb

    def _create_comparison(
        self,
        original: Image.Image,
        restored: Image.Image
    ) -> Image.Image:
        """Create side-by-side comparison"""
        w, h = original.size

        # Create canvas
        comparison = Image.new('RGB', (w * 2, h))
        comparison.paste(original.convert('RGB'), (0, 0))
        comparison.paste(restored.convert('RGB'), (w, 0))

        return comparison


def process_instances(
    input_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    fidelity_weight: float = 0.7,
    upscale: int = 2,
    save_comparison: bool = False,
    face_detector: str = "retinaface"
) -> dict:
    """
    Process all character instances with face restoration

    Args:
        input_dir: Directory with character instances
        output_dir: Output directory for restored instances
        device: cuda or cpu
        fidelity_weight: Restoration fidelity (0.7 recommended for 3D)
        upscale: Upscale factor
        save_comparison: Save before/after comparisons
        face_detector: Face detection backend

    Returns:
        Processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output structure
    restored_dir = output_dir / "restored"
    restored_dir.mkdir(parents=True, exist_ok=True)

    if save_comparison:
        comparison_dir = output_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True)

    # Initialize restorer
    restorer = FaceRestorer(
        device=device,
        fidelity_weight=fidelity_weight,
        upscale=upscale,
        face_detector=face_detector
    )

    # Find all instances
    image_files = sorted(
        list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    )

    print(f"\n📊 Processing {len(image_files)} instances...")

    # Check for already processed files (resume capability)
    processed_files = set()
    if restored_dir.exists():
        for f in restored_dir.glob("*.png"):
            processed_files.add(f.name)

    print(f"📊 Found {len(processed_files)} already processed, will skip them...")

    stats = {
        'total_instances': len(image_files),
        'processed': 0,
        'skipped': 0,
        'faces_detected': 0,
        'no_faces': 0
    }

    for img_path in tqdm(image_files, desc="Restoring faces"):
        # Skip if already processed
        if img_path.name in processed_files:
            stats['skipped'] += 1
            continue

        # Load image
        image = Image.open(img_path)

        # Restore
        restored, comparison = restorer.restore_face(
            image,
            return_comparison=save_comparison
        )

        # Save restored
        output_path = restored_dir / img_path.name
        restored.save(output_path)

        # Save comparison if available
        if comparison is not None:
            comp_path = comparison_dir / f"{img_path.stem}_compare.jpg"
            comparison.save(comp_path, quality=95)
            stats['faces_detected'] += 1
        else:
            stats['no_faces'] += 1

        stats['processed'] += 1

    # Save statistics
    stats_path = output_dir / "restoration_stats.json"
    with open(stats_path, 'w') as f:
        json.dump({
            'statistics': stats,
            'parameters': {
                'fidelity_weight': fidelity_weight,
                'upscale': upscale,
                'face_detector': face_detector
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n✅ Face restoration complete!")
    print(f"   Processed: {stats['processed']}")
    print(f"   Faces detected: {stats['faces_detected']}")
    print(f"   No faces: {stats['no_faces']}")
    print(f"   Skipped: {stats['skipped']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Face restoration for character instances"
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
        help="Output directory for restored instances"
    )
    parser.add_argument(
        "--fidelity",
        type=float,
        default=0.7,
        help="Fidelity weight (0-1, higher = preserve original style, default: 0.7)"
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=2,
        choices=[1, 2, 4],
        help="Upscale factor (default: 2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save before/after comparison images"
    )
    parser.add_argument(
        "--face-detector",
        type=str,
        default="retinaface",
        choices=["retinaface", "yolov5"],
        help="Face detection backend"
    )

    args = parser.parse_args()

    # Process instances
    stats = process_instances(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        fidelity_weight=args.fidelity,
        upscale=args.upscale,
        save_comparison=args.save_comparison,
        face_detector=args.face_detector
    )

    print(f"\n📁 Output saved to: {args.output_dir}")
    print(f"   Restored: {args.output_dir}/restored/")
    if args.save_comparison:
        print(f"   Comparisons: {args.output_dir}/comparisons/")


if __name__ == "__main__":
    main()
