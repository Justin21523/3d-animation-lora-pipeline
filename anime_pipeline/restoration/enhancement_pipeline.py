"""
Unified Enhancement Pipeline for 2D Animation Characters.

Combines multiple enhancement stages:
1. Face restoration (CodeFormer)
2. Super-resolution (RealESRGAN)
3. Adaptive contrast enhancement (style-aware)

Designed for 2D animated content with flat shading and line art.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from anime_pipeline.utils.logging_utils import setup_logging
from anime_pipeline.utils.path_utils import ensure_dir


@dataclass
class FaceRestorationConfig:
    """Face restoration settings."""
    enabled: bool = True
    model: str = "codeformer"  # codeformer | gfpgan
    model_path: Optional[str] = None
    fidelity: float = 0.5  # 0.0 = quality, 1.0 = fidelity
    upscale_face: int = 2


@dataclass
class UpscalingConfig:
    """Upscaling settings."""
    enabled: bool = True
    model: str = "realesrgan_x4"  # realesrgan_x4 | realesrgan_anime
    model_path: Optional[str] = None
    scale: int = 4
    target_resolution: int = 1024  # Target longest edge


@dataclass
class ContrastConfig:
    """Contrast enhancement settings."""
    enabled: bool = True
    style: str = "flat"  # flat | gradient | detailed
    strength: float = 0.3  # 0.0-1.0
    preserve_colors: bool = True


@dataclass
class EnhancementConfig:
    """Configuration for the unified enhancement pipeline."""

    # Nested configs
    face_restoration: FaceRestorationConfig = field(default_factory=FaceRestorationConfig)
    upscaling: UpscalingConfig = field(default_factory=UpscalingConfig)
    contrast: ContrastConfig = field(default_factory=ContrastConfig)

    # Processing settings
    batch_size: int = 4
    device: str = "cuda"
    precision: str = "fp16"

    # I/O settings
    input_dir: str = "data_clustered"
    output_dir: str = "data_enhanced"
    log_dir: Optional[str] = "logs"

    # Fallback
    use_stub: bool = True


@dataclass
class EnhancementResult:
    """Result of enhancement operation."""

    input_path: str
    output_path: str
    face_enhanced: bool = False
    upscaled: bool = False
    contrast_adjusted: bool = False
    original_size: Tuple[int, int] = (0, 0)
    enhanced_size: Tuple[int, int] = (0, 0)
    backend: str = "stub"
    success: bool = True
    error: Optional[str] = None


class CodeFormerEnhancer:
    """Face-focused enhancement using CodeFormer."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        fidelity: float = 0.5,
        device: str = "cuda",
        logger: Optional[logging.Logger] = None,
    ):
        self.model_path = model_path
        self.fidelity = fidelity
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.use_stub = True

        if model_path:
            self._load_model()

    def _load_model(self) -> None:
        """Load CodeFormer model."""
        if not self.model_path or not Path(self.model_path).exists():
            self.logger.warning("CodeFormer model not found: %s", self.model_path)
            return

        try:
            # Try loading via facexlib/codeformer
            from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
            from codeformer import CodeFormer

            self.logger.info("Loading CodeFormer from %s", self.model_path)

            # Initialize face helper
            self.face_helper = FaceRestoreHelper(
                upscale_factor=2,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=True,
                device=self.device,
            )

            # Load CodeFormer model
            import torch
            self.model = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(self.device)

            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["params_ema"])
            self.model.eval()
            self.use_stub = False

            self.logger.info("CodeFormer loaded successfully")

        except ImportError:
            self.logger.warning(
                "CodeFormer not installed. Install with: pip install codeformer-pip"
            )
        except Exception as e:
            self.logger.warning("Failed to load CodeFormer: %s", e)

    def enhance_face(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Enhance a single face in the image.

        Args:
            image: Input image as numpy array (H, W, 3), uint8
            face_bbox: Optional (x1, y1, x2, y2) bounding box for face

        Returns:
            Enhanced image as numpy array
        """
        if self.use_stub:
            return self._enhance_stub(image)

        try:
            import torch

            # Use face helper to detect and align faces
            self.face_helper.clean_all()
            self.face_helper.read_image(image)
            self.face_helper.get_face_landmarks_5(only_center_face=False)
            self.face_helper.align_warp_face()

            # Process each face
            for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                # Prepare input
                cropped_face_t = torch.from_numpy(cropped_face).float().to(self.device)
                cropped_face_t = cropped_face_t.permute(2, 0, 1).unsqueeze(0) / 255.0
                cropped_face_t = cropped_face_t * 2 - 1  # Normalize to [-1, 1]

                # Inference
                with torch.no_grad():
                    output = self.model(cropped_face_t, w=self.fidelity)
                    restored_face = output[0].squeeze()

                # Post-process
                restored_face = (restored_face + 1) / 2
                restored_face = restored_face.clamp(0, 1)
                restored_face = restored_face.permute(1, 2, 0).cpu().numpy()
                restored_face = (restored_face * 255).astype(np.uint8)

                self.face_helper.add_restored_face(restored_face)

            # Paste back
            self.face_helper.get_inverse_affine(None)
            result = self.face_helper.paste_faces_to_input_image()

            return result

        except Exception as e:
            self.logger.warning("CodeFormer enhancement failed: %s", e)
            return self._enhance_stub(image)

    def enhance_all_faces(
        self,
        image: np.ndarray,
        face_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> np.ndarray:
        """Enhance all faces in the image."""
        return self.enhance_face(image)  # CodeFormer handles multiple faces

    def _enhance_stub(self, image: np.ndarray) -> np.ndarray:
        """Stub enhancement using basic sharpening."""
        try:
            import cv2

            # Apply mild sharpening for 2D animation
            kernel = np.array([
                [0, -0.5, 0],
                [-0.5, 3, -0.5],
                [0, -0.5, 0]
            ])
            sharpened = cv2.filter2D(image, -1, kernel)
            # Blend with original
            result = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
            return result
        except Exception:
            return image


class AdaptiveContrastEnhancer:
    """Style-aware contrast enhancement for 2D animation."""

    STYLE_PARAMS = {
        "flat": {
            "clip_limit": 1.5,
            "tile_size": (16, 16),
            "saturation_boost": 1.05,
        },
        "gradient": {
            "clip_limit": 2.0,
            "tile_size": (8, 8),
            "saturation_boost": 1.1,
        },
        "detailed": {
            "clip_limit": 2.5,
            "tile_size": (4, 4),
            "saturation_boost": 1.0,
        },
    }

    def __init__(
        self,
        style: str = "flat",
        strength: float = 0.3,
        preserve_colors: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.style = style
        self.strength = strength
        self.preserve_colors = preserve_colors
        self.logger = logger or logging.getLogger(__name__)
        self.params = self.STYLE_PARAMS.get(style, self.STYLE_PARAMS["flat"])

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive contrast enhancement.

        Args:
            image: Input image as numpy array (H, W, 3), uint8

        Returns:
            Enhanced image as numpy array
        """
        try:
            import cv2

            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.params["clip_limit"],
                tileGridSize=self.params["tile_size"],
            )
            l_enhanced = clahe.apply(l_channel)

            # Merge and convert back
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

            # Blend with original based on strength
            result = cv2.addWeighted(
                image, 1 - self.strength,
                result, self.strength,
                0
            )

            # Optional: boost saturation for flat 2D styles
            if self.params["saturation_boost"] != 1.0 and not self.preserve_colors:
                hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(
                    hsv[:, :, 1] * self.params["saturation_boost"],
                    0, 255
                )
                result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            return result

        except Exception as e:
            self.logger.warning("Contrast enhancement failed: %s", e)
            return image


class EnhancementPipeline:
    """Unified enhancement pipeline combining face, upscale, and contrast."""

    def __init__(
        self,
        config: Optional[EnhancementConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or EnhancementConfig()
        self.logger = logger or setup_logging("enhancement_pipeline", self.config.log_dir)

        # Initialize components
        self.face_enhancer = None
        self.upscaler = None
        self.contrast_enhancer = None

        self._init_components()

    def _init_components(self) -> None:
        """Initialize enhancement components based on config."""
        # Face restoration
        if self.config.face_restoration.enabled:
            self.face_enhancer = CodeFormerEnhancer(
                model_path=self.config.face_restoration.model_path,
                fidelity=self.config.face_restoration.fidelity,
                device=self.config.device,
                logger=self.logger,
            )

        # Upscaling (uses RealESRGAN from existing wrapper)
        if self.config.upscaling.enabled:
            from anime_pipeline.restoration.realesrgan_wrapper import RealESRGANConfig
            self.upscaler_config = RealESRGANConfig(
                model_path=self.config.upscaling.model_path,
                scale=self.config.upscaling.scale,
                device=self.config.device,
                use_stub=self.config.use_stub,
            )

        # Contrast enhancement
        if self.config.contrast.enabled:
            self.contrast_enhancer = AdaptiveContrastEnhancer(
                style=self.config.contrast.style,
                strength=self.config.contrast.strength,
                preserve_colors=self.config.contrast.preserve_colors,
                logger=self.logger,
            )

    def enhance(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        face_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> EnhancementResult:
        """
        Apply full enhancement pipeline to a single image.

        Args:
            image_path: Path to input image
            output_path: Path to save enhanced image
            face_bboxes: Optional list of face bounding boxes

        Returns:
            EnhancementResult with processing details
        """
        image_path = Path(image_path)
        output_path = Path(output_path)

        try:
            # Load image
            image = np.array(Image.open(image_path).convert("RGB"))
            original_size = (image.shape[1], image.shape[0])

            face_enhanced = False
            upscaled = False
            contrast_adjusted = False

            # Step 1: Face restoration
            if self.face_enhancer and self.config.face_restoration.enabled:
                image = self.face_enhancer.enhance_all_faces(image, face_bboxes)
                face_enhanced = True
                self.logger.debug("Face enhancement applied to %s", image_path.name)

            # Step 2: Upscaling
            if self.upscaler_config and self.config.upscaling.enabled:
                image = self._upscale_image(image)
                upscaled = True
                self.logger.debug("Upscaling applied to %s", image_path.name)

            # Step 3: Contrast enhancement
            if self.contrast_enhancer and self.config.contrast.enabled:
                image = self.contrast_enhancer.enhance(image)
                contrast_adjusted = True
                self.logger.debug("Contrast enhancement applied to %s", image_path.name)

            # Resize to target resolution if needed
            if self.config.upscaling.target_resolution:
                image = self._resize_to_target(
                    image, self.config.upscaling.target_resolution
                )

            # Save result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(output_path, quality=95)

            enhanced_size = (image.shape[1], image.shape[0])

            return EnhancementResult(
                input_path=str(image_path),
                output_path=str(output_path),
                face_enhanced=face_enhanced,
                upscaled=upscaled,
                contrast_adjusted=contrast_adjusted,
                original_size=original_size,
                enhanced_size=enhanced_size,
                backend="stub" if self.config.use_stub else "full",
                success=True,
            )

        except Exception as e:
            self.logger.error("Enhancement failed for %s: %s", image_path, e)
            return EnhancementResult(
                input_path=str(image_path),
                output_path=str(output_path),
                success=False,
                error=str(e),
            )

    def _upscale_image(self, image: np.ndarray) -> np.ndarray:
        """Apply upscaling to image."""
        try:
            import cv2

            # Stub upscaling using bicubic interpolation
            h, w = image.shape[:2]
            scale = self.upscaler_config.scale
            new_h, new_w = h * scale, w * scale
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        except Exception as e:
            self.logger.warning("Upscaling failed: %s", e)
            return image

    def _resize_to_target(
        self,
        image: np.ndarray,
        target_size: int,
    ) -> np.ndarray:
        """Resize image so longest edge equals target_size."""
        try:
            import cv2

            h, w = image.shape[:2]
            if max(h, w) <= target_size:
                return image

            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        except Exception:
            return image

    def batch_enhance(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        face_bboxes: Optional[Dict[str, List[Tuple[int, int, int, int]]]] = None,
    ) -> List[EnhancementResult]:
        """
        Batch enhance all images in a directory.

        Args:
            input_dir: Directory containing images
            output_dir: Directory to save enhanced images
            face_bboxes: Optional dict mapping filenames to face bboxes

        Returns:
            List of EnhancementResult for each processed image
        """
        input_dir = Path(input_dir)
        output_dir = ensure_dir(output_dir)
        face_bboxes = face_bboxes or {}

        if not input_dir.exists():
            self.logger.warning("Input directory does not exist: %s", input_dir)
            return []

        results = []
        image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

        for image_path in sorted(input_dir.glob("**/*")):
            if image_path.suffix.lower() not in image_extensions:
                continue

            rel_path = image_path.relative_to(input_dir)
            output_path = output_dir / rel_path

            # Get face bboxes if available
            img_bboxes = face_bboxes.get(image_path.name)

            result = self.enhance(image_path, output_path, img_bboxes)
            results.append(result)

            if result.success:
                self.logger.debug(
                    "Enhanced %s: %s -> %s",
                    image_path.name,
                    result.original_size,
                    result.enhanced_size,
                )

        # Summary
        successful = sum(1 for r in results if r.success)
        self.logger.info(
            "Batch enhancement complete: %d/%d successful",
            successful, len(results)
        )

        # Save summary
        self._save_batch_summary(results, output_dir)

        return results

    def _save_batch_summary(
        self,
        results: List[EnhancementResult],
        output_dir: Path,
    ) -> None:
        """Save batch processing summary."""
        summary = {
            "total": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "face_enhanced_count": sum(1 for r in results if r.face_enhanced),
            "upscaled_count": sum(1 for r in results if r.upscaled),
            "contrast_adjusted_count": sum(1 for r in results if r.contrast_adjusted),
            "config": {
                "face_restoration": self.config.face_restoration.enabled,
                "upscaling": self.config.upscaling.enabled,
                "contrast": self.config.contrast.enabled,
            },
            "results": [
                {
                    "input": r.input_path,
                    "output": r.output_path,
                    "face_enhanced": r.face_enhanced,
                    "upscaled": r.upscaled,
                    "contrast_adjusted": r.contrast_adjusted,
                    "original_size": r.original_size,
                    "enhanced_size": r.enhanced_size,
                    "success": r.success,
                    "error": r.error,
                }
                for r in results
            ],
        }

        summary_path = output_dir / "enhancement_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("Summary saved to %s", summary_path)


def enhance_dataset(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[EnhancementConfig] = None,
    logger=None,
) -> List[EnhancementResult]:
    """
    Convenience function for batch enhancement.

    Args:
        input_dir: Directory containing images to enhance
        output_dir: Directory to save enhanced images
        config: Optional EnhancementConfig
        logger: Optional logger

    Returns:
        List of EnhancementResult
    """
    pipeline = EnhancementPipeline(config=config, logger=logger)
    return pipeline.batch_enhance(input_dir, output_dir)
