"""
LaMa (Large Mask Inpainting) wrapper for background filling.

LaMa is a resolution-robust inpainting model that works well with
large irregular masks - ideal for filling backgrounds after character extraction.

References:
- Paper: "Resolution-robust Large Mask Inpainting with Fourier Convolutions" (WACV 2022)
- Repository: https://github.com/advimman/lama
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
class LaMaConfig:
    """Configuration for LaMa inpainting."""

    # Model settings
    model_path: Optional[str] = None  # Path to LaMa checkpoint (.ckpt or .pt)
    backend: str = "stub"  # stub | pytorch | onnx
    device: str = "cuda"
    precision: str = "fp32"  # fp32 | fp16

    # Processing settings
    batch_size: int = 4
    max_resolution: int = 1024  # Resize larger images
    pad_to_modulo: int = 8  # LaMa requires dimensions divisible by 8

    # Quality settings
    quality_threshold: float = 25.0  # PSNR threshold for quality check
    save_comparison: bool = True  # Save before/after comparison

    # I/O settings
    input_dir: str = "data_segmented/background_with_holes"
    mask_dir: str = "data_segmented/masks"
    output_dir: str = "data_inpainted/backgrounds"
    log_dir: Optional[str] = "logs"

    # Fallback
    use_stub: bool = True  # Use stub when model unavailable


@dataclass
class InpaintingResult:
    """Result of inpainting operation."""

    input_path: str
    mask_path: str
    output_path: str
    psnr: float = 0.0
    ssim: float = 0.0
    backend: str = "stub"
    success: bool = True
    error: Optional[str] = None


class LaMaInpainter:
    """LaMa-based background inpainting."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        config: Optional[LaMaConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or LaMaConfig()
        if model_path:
            self.config.model_path = model_path
        self.config.device = device

        self.logger = logger or setup_logging("lama_inpainter", self.config.log_dir)
        self.model = None
        self.use_stub = self.config.use_stub or self.config.backend == "stub"

        if not self.use_stub:
            self._load_model()

    def _load_model(self) -> None:
        """Load LaMa model."""
        if not self.config.model_path or not Path(self.config.model_path).exists():
            self.logger.warning(
                "LaMa model_path missing or invalid: %s. Using stub mode.",
                self.config.model_path
            )
            self.use_stub = True
            return

        try:
            if self.config.backend == "pytorch":
                self._load_pytorch_model()
            elif self.config.backend == "onnx":
                self._load_onnx_model()
            else:
                self.logger.warning("Unknown backend %s, using stub.", self.config.backend)
                self.use_stub = True
        except Exception as e:
            self.logger.warning("Failed to load LaMa model: %s. Using stub mode.", e)
            self.use_stub = True

    def _load_pytorch_model(self) -> None:
        """Load PyTorch LaMa model."""
        try:
            import torch

            self.logger.info("Loading LaMa PyTorch model from %s", self.config.model_path)

            # LaMa uses a specific checkpoint format
            checkpoint = torch.load(self.config.model_path, map_location=self.config.device)

            # Try to extract model from different checkpoint formats
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # For now, we'll use the simple-lama library if available
            try:
                from simple_lama_inpainting import SimpleLama
                self.model = SimpleLama()
                self.logger.info("Loaded LaMa via simple-lama-inpainting")
            except ImportError:
                # Fallback: try to load raw state dict
                self.logger.warning(
                    "simple-lama-inpainting not installed. "
                    "Install with: pip install simple-lama-inpainting"
                )
                self.model = ("pytorch", state_dict)

            self.use_stub = False

        except Exception as e:
            self.logger.warning("PyTorch LaMa load failed: %s", e)
            self.use_stub = True

    def _load_onnx_model(self) -> None:
        """Load ONNX LaMa model."""
        try:
            import onnxruntime as ort

            self.logger.info("Loading LaMa ONNX model from %s", self.config.model_path)

            providers = (
                ["CPUExecutionProvider"]
                if self.config.device == "cpu"
                else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            session = ort.InferenceSession(self.config.model_path, providers=providers)
            self.model = ("onnx", session)
            self.use_stub = False
            self.logger.info("LaMa ONNX model loaded successfully")

        except Exception as e:
            self.logger.warning("ONNX LaMa load failed: %s", e)
            self.use_stub = True

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Inpaint a single image.

        Args:
            image: Input image as numpy array (H, W, 3) or (H, W, 4), uint8
            mask: Binary mask where 1/255 indicates regions to inpaint (H, W), uint8

        Returns:
            Inpainted image as numpy array (H, W, 3), uint8
        """
        # Ensure correct formats
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]

        # Normalize mask to binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        if self.use_stub:
            return self._inpaint_stub(image, mask)

        try:
            if isinstance(self.model, tuple):
                backend, model = self.model
                if backend == "onnx":
                    return self._inpaint_onnx(image, mask, model)
                else:
                    return self._inpaint_stub(image, mask)
            else:
                # SimpleLama instance
                return self._inpaint_simple_lama(image, mask)
        except Exception as e:
            self.logger.warning("Inpainting failed: %s. Using stub.", e)
            return self._inpaint_stub(image, mask)

    def _inpaint_simple_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using simple-lama-inpainting library."""
        from PIL import Image as PILImage

        img_pil = PILImage.fromarray(image)
        mask_pil = PILImage.fromarray(mask * 255)

        result = self.model(img_pil, mask_pil)
        return np.array(result)

    def _inpaint_onnx(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        session,
    ) -> np.ndarray:
        """Inpaint using ONNX model."""
        # Preprocess
        h, w = image.shape[:2]

        # Pad to modulo
        pad_h = (self.config.pad_to_modulo - h % self.config.pad_to_modulo) % self.config.pad_to_modulo
        pad_w = (self.config.pad_to_modulo - w % self.config.pad_to_modulo) % self.config.pad_to_modulo

        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

        # Normalize to [0, 1] and add batch dimension
        img_tensor = image.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))  # HWC -> CHW
        img_tensor = np.expand_dims(img_tensor, 0)  # Add batch

        mask_tensor = mask.astype(np.float32)
        mask_tensor = np.expand_dims(np.expand_dims(mask_tensor, 0), 0)  # 1, 1, H, W

        # Run inference
        input_names = [inp.name for inp in session.get_inputs()]
        output = session.run(None, {
            input_names[0]: img_tensor,
            input_names[1]: mask_tensor,
        })[0]

        # Postprocess
        output = output[0]  # Remove batch
        output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
        output = np.clip(output * 255, 0, 255).astype(np.uint8)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:h, :w]

        return output

    def _inpaint_stub(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Stub inpainting using OpenCV's telea algorithm.

        This provides reasonable results for small masks but is not
        suitable for production use with large masks.
        """
        try:
            import cv2

            # Ensure mask is uint8
            mask_cv = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)

            # OpenCV telea inpainting
            result = cv2.inpaint(image, mask_cv, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return result

        except Exception as e:
            self.logger.warning("CV2 inpainting failed: %s. Returning original.", e)
            # Fallback: just fill mask region with mean color
            result = image.copy()
            if mask.any():
                mean_color = image[~mask.astype(bool)].mean(axis=0).astype(np.uint8)
                result[mask.astype(bool)] = mean_color
            return result

    def inpaint_file(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> InpaintingResult:
        """
        Inpaint a single file.

        Args:
            image_path: Path to input image
            mask_path: Path to mask image
            output_path: Path to save result

        Returns:
            InpaintingResult with quality metrics
        """
        image_path = Path(image_path)
        mask_path = Path(mask_path)
        output_path = Path(output_path)

        try:
            # Load images
            image = np.array(Image.open(image_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))

            # Inpaint
            result = self.inpaint(image, mask)

            # Save result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result).save(output_path)

            # Compute quality metrics
            psnr = self.compute_psnr(image, result, mask)
            ssim = self.compute_ssim(image, result, mask)

            # Save comparison if requested
            if self.config.save_comparison:
                self._save_comparison(image, mask, result, output_path)

            return InpaintingResult(
                input_path=str(image_path),
                mask_path=str(mask_path),
                output_path=str(output_path),
                psnr=psnr,
                ssim=ssim,
                backend="stub" if self.use_stub else self.config.backend,
                success=True,
            )

        except Exception as e:
            self.logger.error("Failed to inpaint %s: %s", image_path, e)
            return InpaintingResult(
                input_path=str(image_path),
                mask_path=str(mask_path),
                output_path=str(output_path),
                success=False,
                error=str(e),
            )

    def batch_inpaint(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        output_dir: Union[str, Path],
        mask_suffix: str = "_mask",
    ) -> List[InpaintingResult]:
        """
        Batch inpaint all images in a directory.

        Args:
            image_dir: Directory containing images to inpaint
            mask_dir: Directory containing corresponding masks
            output_dir: Directory to save results
            mask_suffix: Suffix to identify mask files (e.g., "image_mask.png")

        Returns:
            List of InpaintingResult for each processed image
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        output_dir = ensure_dir(output_dir)

        if not image_dir.exists():
            self.logger.warning("Input directory does not exist: %s", image_dir)
            return []

        results = []
        image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

        for image_path in sorted(image_dir.glob("**/*")):
            if image_path.suffix.lower() not in image_extensions:
                continue

            # Skip mask files
            if mask_suffix in image_path.stem:
                continue

            # Find corresponding mask
            rel_path = image_path.relative_to(image_dir)
            mask_path = mask_dir / f"{rel_path.stem}{mask_suffix}{rel_path.suffix}"

            if not mask_path.exists():
                # Try without suffix
                mask_path = mask_dir / rel_path.with_suffix(".png")

            if not mask_path.exists():
                self.logger.debug("No mask found for %s, skipping", image_path)
                continue

            # Output path
            output_path = output_dir / rel_path

            # Process
            result = self.inpaint_file(image_path, mask_path, output_path)
            results.append(result)

            if result.success:
                self.logger.debug(
                    "Inpainted %s (PSNR: %.2f, SSIM: %.3f)",
                    image_path.name, result.psnr, result.ssim
                )

        # Summary
        successful = sum(1 for r in results if r.success)
        avg_psnr = np.mean([r.psnr for r in results if r.success]) if successful else 0

        self.logger.info(
            "Batch inpainting complete: %d/%d successful, avg PSNR: %.2f",
            successful, len(results), avg_psnr
        )

        # Save summary
        self._save_batch_summary(results, output_dir)

        return results

    def compute_psnr(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Peak Signal-to-Noise Ratio.

        If mask is provided, only computes PSNR in non-masked regions
        (to measure how well the boundary blends).
        """
        try:
            if mask is not None:
                # Only compare non-masked regions (boundary quality)
                inv_mask = ~mask.astype(bool)
                if not inv_mask.any():
                    return 0.0
                orig_vals = original[inv_mask]
                inp_vals = inpainted[inv_mask]
            else:
                orig_vals = original.flatten()
                inp_vals = inpainted.flatten()

            mse = np.mean((orig_vals.astype(float) - inp_vals.astype(float)) ** 2)
            if mse == 0:
                return float("inf")

            return 10 * np.log10(255.0 ** 2 / mse)

        except Exception:
            return 0.0

    def compute_ssim(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute Structural Similarity Index."""
        try:
            from skimage.metrics import structural_similarity as ssim

            # Convert to grayscale for SSIM
            if original.ndim == 3:
                original_gray = np.mean(original, axis=-1)
                inpainted_gray = np.mean(inpainted, axis=-1)
            else:
                original_gray = original
                inpainted_gray = inpainted

            return ssim(original_gray, inpainted_gray, data_range=255)

        except ImportError:
            self.logger.debug("skimage not available for SSIM computation")
            return 0.0
        except Exception:
            return 0.0

    def _save_comparison(
        self,
        original: np.ndarray,
        mask: np.ndarray,
        result: np.ndarray,
        output_path: Path,
    ) -> None:
        """Save side-by-side comparison image."""
        try:
            # Create comparison: original | masked | result
            h, w = original.shape[:2]

            # Create masked version (show mask overlay)
            masked = original.copy()
            mask_3ch = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
            masked = np.where(mask_3ch > 0, mask_3ch, masked)

            # Concatenate horizontally
            comparison = np.concatenate([original, masked, result], axis=1)

            # Save
            comp_path = output_path.parent / f"{output_path.stem}_comparison{output_path.suffix}"
            Image.fromarray(comparison).save(comp_path)

        except Exception as e:
            self.logger.debug("Failed to save comparison: %s", e)

    def _save_batch_summary(
        self,
        results: List[InpaintingResult],
        output_dir: Path,
    ) -> None:
        """Save batch processing summary."""
        summary = {
            "total": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "avg_psnr": float(np.mean([r.psnr for r in results if r.success])) if results else 0,
            "avg_ssim": float(np.mean([r.ssim for r in results if r.success])) if results else 0,
            "backend": "stub" if self.use_stub else self.config.backend,
            "results": [
                {
                    "input": r.input_path,
                    "output": r.output_path,
                    "psnr": r.psnr,
                    "ssim": r.ssim,
                    "success": r.success,
                    "error": r.error,
                }
                for r in results
            ],
        }

        summary_path = output_dir / "inpainting_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("Summary saved to %s", summary_path)


def inpaint_backgrounds(config: LaMaConfig, logger=None) -> List[InpaintingResult]:
    """
    Convenience function for batch inpainting.

    Args:
        config: LaMaConfig with paths and settings
        logger: Optional logger

    Returns:
        List of InpaintingResult
    """
    inpainter = LaMaInpainter(config=config, logger=logger)
    return inpainter.batch_inpaint(
        image_dir=config.input_dir,
        mask_dir=config.mask_dir,
        output_dir=config.output_dir,
    )
