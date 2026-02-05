"""
PowerPaint wrapper for text-guided, character-aware background inpainting.

PowerPaint (ECCV 2024) is a versatile image inpainting model that supports:
- Text-guided inpainting with natural language prompts
- Object removal with context-aware filling
- Shape-guided generation

This is particularly useful for animation pipelines where we want to:
- Remove characters while maintaining consistent background style
- Fill backgrounds with content matching the scene context

References:
- Paper: "A Task is Worth One Word: Learning with Task Prompts for High-Quality Versatile Image Inpainting" (ECCV 2024)
- Repository: https://github.com/open-mmlab/PowerPaint
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
class PowerPaintConfig:
    """Configuration for PowerPaint inpainting."""

    # Model settings
    model_path: Optional[str] = None  # Path to PowerPaint checkpoint
    brushnet_path: Optional[str] = None  # Path to BrushNet adapter
    backend: str = "stub"  # stub | diffusers
    device: str = "cuda"
    precision: str = "fp16"  # fp32 | fp16 | bf16

    # Generation settings
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    strength: float = 1.0  # Inpainting strength
    negative_prompt: str = "blurry, low quality, artifacts, distorted"

    # Task prompt settings
    task_prompt: str = "P_obj"  # P_obj (object removal) | P_ctxt (context-aware) | P_shape (shape-guided)

    # Processing settings
    batch_size: int = 1
    max_resolution: int = 1024
    seed: Optional[int] = None  # For reproducibility

    # Quality settings
    quality_threshold: float = 25.0  # PSNR threshold
    save_comparison: bool = True

    # I/O settings
    input_dir: str = "data_segmented/background_with_holes"
    mask_dir: str = "data_segmented/masks"
    output_dir: str = "data_inpainted/backgrounds"
    character_info_dir: Optional[str] = None  # Directory with character descriptions
    log_dir: Optional[str] = "logs"

    # Fallback
    use_stub: bool = True


@dataclass
class PowerPaintResult:
    """Result of PowerPaint inpainting operation."""

    input_path: str
    mask_path: str
    output_path: str
    prompt: Optional[str] = None
    psnr: float = 0.0
    ssim: float = 0.0
    backend: str = "stub"
    success: bool = True
    error: Optional[str] = None


class PowerPaintInpainter:
    """PowerPaint-based text-guided background inpainting."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        config: Optional[PowerPaintConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or PowerPaintConfig()
        if model_path:
            self.config.model_path = model_path
        self.config.device = device

        self.logger = logger or setup_logging("powerpaint_inpainter", self.config.log_dir)
        self.pipeline = None
        self.use_stub = self.config.use_stub or self.config.backend == "stub"

        if not self.use_stub:
            self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load PowerPaint pipeline."""
        if not self.config.model_path or not Path(self.config.model_path).exists():
            self.logger.warning(
                "PowerPaint model_path missing or invalid: %s. Using stub mode.",
                self.config.model_path
            )
            self.use_stub = True
            return

        try:
            self._load_diffusers_pipeline()
        except Exception as e:
            self.logger.warning("Failed to load PowerPaint pipeline: %s. Using stub mode.", e)
            self.use_stub = True

    def _load_diffusers_pipeline(self) -> None:
        """Load PowerPaint using diffusers."""
        try:
            import torch
            from diffusers import (
                StableDiffusionInpaintPipeline,
                AutoencoderKL,
                UNet2DConditionModel,
            )
            from transformers import CLIPTextModel, CLIPTokenizer

            self.logger.info("Loading PowerPaint from %s", self.config.model_path)

            # PowerPaint is based on SD inpainting with BrushNet adapter
            # Try to load as standard inpainting pipeline first
            try:
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    self.config.model_path,
                    torch_dtype=self._get_torch_dtype(),
                    safety_checker=None,
                )
                self.pipeline = self.pipeline.to(self.config.device)

                # Enable optimizations
                if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                    try:
                        self.pipeline.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass

                self.use_stub = False
                self.logger.info("PowerPaint pipeline loaded successfully")

            except Exception as e:
                self.logger.warning("Standard pipeline load failed: %s", e)

                # Try loading as safetensors checkpoint
                if self.config.model_path.endswith(".safetensors"):
                    self.logger.info("Attempting to load as safetensors checkpoint...")
                    # Would need custom loading logic here
                    raise NotImplementedError("Safetensors loading not yet implemented")

                raise

        except ImportError as e:
            self.logger.warning("Diffusers not available: %s", e)
            self.use_stub = True
        except Exception as e:
            self.logger.warning("Pipeline load failed: %s", e)
            self.use_stub = True

    def _get_torch_dtype(self):
        """Get torch dtype based on precision config."""
        import torch

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(self.config.precision, torch.float16)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: Optional[str] = None,
    ) -> np.ndarray:
        """
        Inpaint a single image with optional text guidance.

        Args:
            image: Input image as numpy array (H, W, 3) or (H, W, 4), uint8
            mask: Binary mask where 1/255 indicates regions to inpaint (H, W), uint8
            prompt: Optional text prompt describing desired background

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
            return self._inpaint_diffusers(image, mask, prompt)
        except Exception as e:
            self.logger.warning("PowerPaint inpainting failed: %s. Using stub.", e)
            return self._inpaint_stub(image, mask)

    def _inpaint_diffusers(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: Optional[str] = None,
    ) -> np.ndarray:
        """Inpaint using diffusers pipeline."""
        import torch
        from PIL import Image as PILImage

        # Convert to PIL
        image_pil = PILImage.fromarray(image)
        mask_pil = PILImage.fromarray(mask * 255)

        # Default prompt for animation backgrounds
        if prompt is None:
            prompt = self._get_default_prompt()

        # Set seed for reproducibility
        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device=self.config.device).manual_seed(self.config.seed)

        # Run inference
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=self.config.negative_prompt,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                strength=self.config.strength,
                generator=generator,
            ).images[0]

        return np.array(result)

    def _get_default_prompt(self) -> str:
        """Get default prompt based on task type."""
        task_prompts = {
            "P_obj": "clean background, no objects, seamless fill, high quality",
            "P_ctxt": "consistent background, matching style, seamless, high quality",
            "P_shape": "coherent fill, matching context, high quality",
        }
        return task_prompts.get(self.config.task_prompt, task_prompts["P_obj"])

    def inpaint_with_context(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        character_description: Optional[str] = None,
        scene_description: Optional[str] = None,
    ) -> np.ndarray:
        """
        Inpaint with character-aware context.

        This method generates a prompt that considers:
        - What character was removed (to avoid regenerating them)
        - What the scene context is (to maintain consistency)

        Args:
            image: Input image
            mask: Inpainting mask
            character_description: Description of removed character (e.g., "blue-haired boy")
            scene_description: Description of scene (e.g., "sunny outdoor park")

        Returns:
            Inpainted image
        """
        # Build context-aware prompt
        prompt_parts = []

        if scene_description:
            prompt_parts.append(scene_description)
        else:
            prompt_parts.append("clean background")

        prompt_parts.extend([
            "seamless fill",
            "no characters",
            "consistent style",
            "high quality",
            "2d animation background",
        ])

        # Add negative elements if character was described
        negative_parts = [self.config.negative_prompt]
        if character_description:
            negative_parts.append(f"no {character_description}")
            negative_parts.append("no people")
            negative_parts.append("no characters")

        prompt = ", ".join(prompt_parts)

        # For stub mode, just use basic inpainting
        if self.use_stub:
            return self._inpaint_stub(image, mask)

        # Temporarily override negative prompt
        original_negative = self.config.negative_prompt
        self.config.negative_prompt = ", ".join(negative_parts)

        try:
            result = self.inpaint(image, mask, prompt)
        finally:
            self.config.negative_prompt = original_negative

        return result

    def _inpaint_stub(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Stub inpainting using OpenCV with Navier-Stokes method.

        This provides better results than telea for larger masks
        but is still not suitable for production use.
        """
        try:
            import cv2

            # Ensure mask is uint8
            mask_cv = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)

            # Use Navier-Stokes for potentially better results on larger masks
            result = cv2.inpaint(image, mask_cv, inpaintRadius=5, flags=cv2.INPAINT_NS)
            return result

        except Exception as e:
            self.logger.warning("CV2 inpainting failed: %s. Returning original.", e)
            # Fallback: blend with Gaussian blurred version
            result = image.copy()
            if mask.any():
                try:
                    import cv2
                    blurred = cv2.GaussianBlur(image, (31, 31), 0)
                    mask_3ch = np.stack([mask] * 3, axis=-1).astype(bool)
                    result = np.where(mask_3ch, blurred, image)
                except Exception:
                    # Ultimate fallback: fill with mean color
                    mean_color = image[~mask.astype(bool)].mean(axis=0).astype(np.uint8)
                    result[mask.astype(bool)] = mean_color
            return result

    def inpaint_file(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
        output_path: Union[str, Path],
        prompt: Optional[str] = None,
        character_info_path: Optional[Union[str, Path]] = None,
    ) -> PowerPaintResult:
        """
        Inpaint a single file.

        Args:
            image_path: Path to input image
            mask_path: Path to mask image
            output_path: Path to save result
            prompt: Optional text prompt
            character_info_path: Optional path to JSON with character info

        Returns:
            PowerPaintResult with quality metrics
        """
        image_path = Path(image_path)
        mask_path = Path(mask_path)
        output_path = Path(output_path)

        try:
            # Load images
            image = np.array(Image.open(image_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))

            # Load character info if available
            char_desc = None
            scene_desc = None
            if character_info_path and Path(character_info_path).exists():
                with open(character_info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                    char_desc = info.get("character_description")
                    scene_desc = info.get("scene_description")

            # Choose inpainting method
            if char_desc or scene_desc:
                result = self.inpaint_with_context(image, mask, char_desc, scene_desc)
            else:
                result = self.inpaint(image, mask, prompt)

            # Save result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result).save(output_path)

            # Compute quality metrics
            psnr = self._compute_psnr(image, result, mask)
            ssim = self._compute_ssim(image, result, mask)

            # Save comparison if requested
            if self.config.save_comparison:
                self._save_comparison(image, mask, result, output_path)

            return PowerPaintResult(
                input_path=str(image_path),
                mask_path=str(mask_path),
                output_path=str(output_path),
                prompt=prompt,
                psnr=psnr,
                ssim=ssim,
                backend="stub" if self.use_stub else self.config.backend,
                success=True,
            )

        except Exception as e:
            self.logger.error("Failed to inpaint %s: %s", image_path, e)
            return PowerPaintResult(
                input_path=str(image_path),
                mask_path=str(mask_path),
                output_path=str(output_path),
                prompt=prompt,
                success=False,
                error=str(e),
            )

    def batch_inpaint(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        output_dir: Union[str, Path],
        mask_suffix: str = "_mask",
        prompt: Optional[str] = None,
    ) -> List[PowerPaintResult]:
        """
        Batch inpaint all images in a directory.

        Args:
            image_dir: Directory containing images to inpaint
            mask_dir: Directory containing corresponding masks
            output_dir: Directory to save results
            mask_suffix: Suffix to identify mask files
            prompt: Optional global prompt for all images

        Returns:
            List of PowerPaintResult for each processed image
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
                mask_path = mask_dir / rel_path.with_suffix(".png")

            if not mask_path.exists():
                self.logger.debug("No mask found for %s, skipping", image_path)
                continue

            # Find character info if available
            char_info_path = None
            if self.config.character_info_dir:
                potential_info = Path(self.config.character_info_dir) / f"{rel_path.stem}.json"
                if potential_info.exists():
                    char_info_path = potential_info

            # Output path
            output_path = output_dir / rel_path

            # Process
            result = self.inpaint_file(
                image_path,
                mask_path,
                output_path,
                prompt=prompt,
                character_info_path=char_info_path,
            )
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

    def _compute_psnr(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        try:
            if mask is not None:
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

    def _compute_ssim(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute Structural Similarity Index."""
        try:
            from skimage.metrics import structural_similarity as ssim

            if original.ndim == 3:
                original_gray = np.mean(original, axis=-1)
                inpainted_gray = np.mean(inpainted, axis=-1)
            else:
                original_gray = original
                inpainted_gray = inpainted

            return ssim(original_gray, inpainted_gray, data_range=255)
        except ImportError:
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
            masked = original.copy()
            mask_3ch = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
            masked = np.where(mask_3ch > 0, mask_3ch, masked)

            comparison = np.concatenate([original, masked, result], axis=1)

            comp_path = output_path.parent / f"{output_path.stem}_comparison{output_path.suffix}"
            Image.fromarray(comparison).save(comp_path)
        except Exception as e:
            self.logger.debug("Failed to save comparison: %s", e)

    def _save_batch_summary(
        self,
        results: List[PowerPaintResult],
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
            "task_prompt": self.config.task_prompt,
            "results": [
                {
                    "input": r.input_path,
                    "output": r.output_path,
                    "prompt": r.prompt,
                    "psnr": r.psnr,
                    "ssim": r.ssim,
                    "success": r.success,
                    "error": r.error,
                }
                for r in results
            ],
        }

        summary_path = output_dir / "powerpaint_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("Summary saved to %s", summary_path)


def inpaint_backgrounds_powerpaint(
    config: PowerPaintConfig,
    logger=None,
) -> List[PowerPaintResult]:
    """
    Convenience function for batch PowerPaint inpainting.

    Args:
        config: PowerPaintConfig with paths and settings
        logger: Optional logger

    Returns:
        List of PowerPaintResult
    """
    inpainter = PowerPaintInpainter(config=config, logger=logger)
    return inpainter.batch_inpaint(
        image_dir=config.input_dir,
        mask_dir=config.mask_dir,
        output_dir=config.output_dir,
    )
