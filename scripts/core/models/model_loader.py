#!/usr/bin/env python3
"""
Unified Model Loader for Intelligent Frame Processing

Manages loading and caching of:
- SAM2 (instance segmentation)
- LaMa (inpainting)
- RealESRGAN (upscaling)
- CodeFormer (face enhancement)
- Qwen2-VL (captioning)

Features:
- Lazy loading (only load models when needed)
- Model caching (reuse loaded models)
- Graceful fallbacks if models unavailable
- Memory management (unload unused models)
"""

import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import numpy as np
import cv2

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


class ModelLoader:
    """
    Centralized model loading and management
    """

    def __init__(self, device: str = "cuda", cache_dir: Optional[Path] = None):
        """
        Initialize model loader

        Args:
            device: 'cuda' or 'cpu'
            cache_dir: Directory for model caches
        """
        self.device = device
        self.cache_dir = cache_dir or Path.home() / ".cache" / "intelligent_processor"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model caches
        self._sam2_model = None
        self._lama_model = None
        self._realesrgan_model = None
        self._codeformer_model = None
        self._caption_model = None

        # Model availability flags
        self.sam2_available = self._check_sam2()
        self.lama_available = self._check_lama()
        self.realesrgan_available = self._check_realesrgan()
        self.codeformer_available = self._check_codeformer()

        print(f"📦 Model Loader initialized (device: {device})")
        print(f"   SAM2:       {'✓' if self.sam2_available else '✗'}")
        print(f"   LaMa:       {'✓' if self.lama_available else '✗'}")
        print(f"   RealESRGAN: {'✓' if self.realesrgan_available else '✗'}")
        print(f"   CodeFormer: {'✓' if self.codeformer_available else '✗'}")

    # ==================== Availability Checks ====================

    def _check_sam2(self) -> bool:
        """Check if SAM2 is available"""
        try:
            from sam2.build_sam import build_sam2
            return True
        except ImportError:
            return False

    def _check_lama(self) -> bool:
        """Check if LaMa is available"""
        try:
            from lama_cleaner.model_manager import ModelManager
            return True
        except ImportError:
            return False

    def _check_realesrgan(self) -> bool:
        """Check if RealESRGAN is available"""
        try:
            from realesrgan import RealESRGANer
            return True
        except ImportError:
            return False

    def _check_codeformer(self) -> bool:
        """Check if CodeFormer is available"""
        try:
            # CodeFormer is typically bundled with face restoration packages
            import basicsr
            return True
        except ImportError:
            return False

    # ==================== SAM2 Segmentation ====================

    def get_sam2_model(
        self,
        model_type: str = "sam2_hiera_large",
        config: Optional[Dict] = None
    ):
        """
        Get SAM2 model (cached)

        Args:
            model_type: Model variant
            config: Optional configuration

        Returns:
            SAM2 mask generator or None if unavailable
        """
        if not self.sam2_available:
            print("⚠️ SAM2 not available, using fallback")
            return None

        if self._sam2_model is not None:
            return self._sam2_model

        try:
            print(f"🔧 Loading SAM2 ({model_type})...")

            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            # Get checkpoint and config paths
            checkpoint_path = self._get_sam2_checkpoint(model_type)
            config_file = self._get_sam2_config(model_type)

            # Build model
            sam2_model = build_sam2(
                config_file,
                checkpoint_path,
                device=self.device
            )

            # Create mask generator
            config = config or {}
            self._sam2_model = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=config.get('points_per_side', 20),
                pred_iou_thresh=config.get('pred_iou_thresh', 0.76),
                stability_score_thresh=config.get('stability_score_thresh', 0.86),
                crop_n_layers=config.get('crop_n_layers', 0),
                min_mask_region_area=config.get('min_mask_size', 64 * 64),
                points_per_batch=config.get('points_per_batch', 192)
            )

            print("✓ SAM2 loaded successfully")
            return self._sam2_model

        except Exception as e:
            print(f"❌ Failed to load SAM2: {e}")
            return None

    def _get_sam2_checkpoint(self, model_type: str) -> str:
        """Get SAM2 checkpoint path"""
        # Try warehouse first
        warehouse_path = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/segmentation")

        checkpoint_names = {
            "sam2_hiera_large": "sam2_hiera_large.pt",
            "sam2_hiera_base": "sam2_hiera_base_plus.pt",
            "sam2_hiera_small": "sam2_hiera_small.pt"
        }

        checkpoint_file = warehouse_path / checkpoint_names.get(model_type, "sam2_hiera_large.pt")

        if checkpoint_file.exists():
            return str(checkpoint_file)

        # Fallback to cache directory
        cache_checkpoint = self.cache_dir / checkpoint_names.get(model_type, "sam2_hiera_large.pt")
        if cache_checkpoint.exists():
            return str(cache_checkpoint)

        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_file}")

    def _get_sam2_config(self, model_type: str) -> str:
        """Get SAM2 config file path"""
        # These are typically in the SAM2 package
        config_names = {
            "sam2_hiera_large": "sam2_hiera_l.yaml",
            "sam2_hiera_base": "sam2_hiera_b+.yaml",
            "sam2_hiera_small": "sam2_hiera_s.yaml"
        }

        # SAM2 configs are usually in the package
        try:
            import sam2
            sam2_dir = Path(sam2.__file__).parent
            config_file = sam2_dir / "configs" / config_names.get(model_type, "sam2_hiera_l.yaml")

            if config_file.exists():
                return str(config_file)
        except:
            pass

        raise FileNotFoundError(f"SAM2 config not found for {model_type}")

    # ==================== LaMa Inpainting ====================

    def get_lama_model(self, config: Optional[Dict] = None):
        """
        Get LaMa inpainting model (cached)

        Args:
            config: Optional configuration

        Returns:
            LaMa model or None if unavailable
        """
        if not self.lama_available:
            print("⚠️ LaMa not available, using OpenCV fallback")
            return None

        if self._lama_model is not None:
            return self._lama_model

        try:
            print("🔧 Loading LaMa inpainting model...")

            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config

            self._lama_model = {
                'manager': ModelManager(name="lama", device=self.device),
                'config': Config(
                    ldm_steps=config.get('ldm_steps', 25) if config else 25,
                    ldm_sampler=config.get('ldm_sampler', 'plms') if config else 'plms',
                    hd_strategy="Resize",
                    hd_strategy_resize_limit=config.get('resize_limit', 1024) if config else 1024
                )
            }

            print("✓ LaMa loaded successfully")
            return self._lama_model

        except Exception as e:
            print(f"❌ Failed to load LaMa: {e}")
            return None

    def inpaint_with_lama(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        config: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Inpaint using LaMa (or OpenCV fallback)

        Args:
            image: RGB image
            mask: Binary mask (255 = inpaint, 0 = keep)
            config: Optional config

        Returns:
            Inpainted image
        """
        lama_model = self.get_lama_model(config)

        if lama_model is not None:
            # Use LaMa
            try:
                result = lama_model['manager'](
                    image,
                    mask,
                    lama_model['config']
                )
                return result
            except Exception as e:
                print(f"⚠️ LaMa failed, using OpenCV: {e}")

        # Fallback to OpenCV inpainting
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    # ==================== RealESRGAN Enhancement ====================

    def get_realesrgan_model(self, config: Optional[Dict] = None):
        """
        Get RealESRGAN model (cached)

        Args:
            config: Optional configuration

        Returns:
            RealESRGAN upsampler or None if unavailable
        """
        if not self.realesrgan_available:
            print("⚠️ RealESRGAN not available")
            return None

        if self._realesrgan_model is not None:
            return self._realesrgan_model

        try:
            print("🔧 Loading RealESRGAN...")

            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            # Get model path
            model_path = self._get_realesrgan_checkpoint()

            # Define architecture
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=config.get('upscale', 2) if config else 2
            )

            # Create upsampler
            self._realesrgan_model = RealESRGANer(
                scale=config.get('upscale', 2) if config else 2,
                model_path=model_path,
                model=model,
                tile=config.get('tile_size', 512) if config else 512,
                tile_pad=config.get('tile_pad', 10) if config else 10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                device=self.device
            )

            print("✓ RealESRGAN loaded successfully")
            return self._realesrgan_model

        except Exception as e:
            print(f"❌ Failed to load RealESRGAN: {e}")
            return None

    def _get_realesrgan_checkpoint(self) -> str:
        """Get RealESRGAN checkpoint path"""
        # Try warehouse
        warehouse_path = Path("/mnt/c/AI_LLM_projects/ai_warehouse/models/enhancement")
        checkpoint_file = warehouse_path / "RealESRGAN_x4plus_anime_6B.pth"

        if checkpoint_file.exists():
            return str(checkpoint_file)

        # Try cache
        cache_checkpoint = self.cache_dir / "RealESRGAN_x4plus_anime_6B.pth"
        if cache_checkpoint.exists():
            return str(cache_checkpoint)

        raise FileNotFoundError(f"RealESRGAN checkpoint not found: {checkpoint_file}")

    def enhance_with_realesrgan(
        self,
        image: np.ndarray,
        config: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Enhance image with RealESRGAN

        Args:
            image: RGB image
            config: Optional config

        Returns:
            Enhanced image (or original if unavailable)
        """
        realesrgan = self.get_realesrgan_model(config)

        if realesrgan is None:
            print("⚠️ RealESRGAN not available, returning original")
            return image

        try:
            output, _ = realesrgan.enhance(image, outscale=config.get('upscale', 2) if config else 2)
            return output
        except Exception as e:
            print(f"⚠️ RealESRGAN enhancement failed: {e}")
            return image

    # ==================== Memory Management ====================

    def unload_all(self):
        """Unload all models to free memory"""
        print("🗑️ Unloading all models...")

        if self._sam2_model is not None:
            del self._sam2_model
            self._sam2_model = None

        if self._lama_model is not None:
            del self._lama_model
            self._lama_model = None

        if self._realesrgan_model is not None:
            del self._realesrgan_model
            self._realesrgan_model = None

        if self._codeformer_model is not None:
            del self._codeformer_model
            self._codeformer_model = None

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✓ All models unloaded")

    def unload_model(self, model_name: str):
        """Unload specific model"""
        if model_name == "sam2" and self._sam2_model is not None:
            del self._sam2_model
            self._sam2_model = None
        elif model_name == "lama" and self._lama_model is not None:
            del self._lama_model
            self._lama_model = None
        elif model_name == "realesrgan" and self._realesrgan_model is not None:
            del self._realesrgan_model
            self._realesrgan_model = None
        elif model_name == "codeformer" and self._codeformer_model is not None:
            del self._codeformer_model
            self._codeformer_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✓ {model_name} unloaded")

    # ==================== Utility Methods ====================

    def check_model_availability(self) -> Dict[str, bool]:
        """Check which models are available"""
        return {
            'sam2': self.sam2_available,
            'lama': self.lama_available,
            'realesrgan': self.realesrgan_available,
            'codeformer': self.codeformer_available
        }

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        if not torch.cuda.is_available():
            return {'device': 'cpu', 'memory': 'N/A'}

        return {
            'device': self.device,
            'allocated': f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            'cached': f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
            'max_allocated': f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        }


# ==================== Singleton Instance ====================

_global_loader = None


def get_model_loader(device: str = "cuda", force_reload: bool = False) -> ModelLoader:
    """
    Get global model loader instance (singleton)

    Args:
        device: Device to use
        force_reload: Force create new loader

    Returns:
        ModelLoader instance
    """
    global _global_loader

    if _global_loader is None or force_reload:
        _global_loader = ModelLoader(device=device)

    return _global_loader


# ==================== CLI Testing ====================

def main():
    """Test model loader"""
    import argparse

    parser = argparse.ArgumentParser(description="Test model loader")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--test-model', type=str, choices=['sam2', 'lama', 'realesrgan', 'all'])

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  MODEL LOADER TEST")
    print("="*60 + "\n")

    # Initialize loader
    loader = ModelLoader(device=args.device)

    # Check availability
    availability = loader.check_model_availability()
    print("\n📊 Model Availability:")
    for model, available in availability.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"   {model:12s}: {status}")

    # Test loading
    if args.test_model == 'sam2' or args.test_model == 'all':
        print("\n🔧 Testing SAM2...")
        sam2 = loader.get_sam2_model()
        if sam2:
            print("✓ SAM2 loaded and ready")

    if args.test_model == 'lama' or args.test_model == 'all':
        print("\n🔧 Testing LaMa...")
        lama = loader.get_lama_model()
        if lama:
            print("✓ LaMa loaded and ready")

    if args.test_model == 'realesrgan' or args.test_model == 'all':
        print("\n🔧 Testing RealESRGAN...")
        realesrgan = loader.get_realesrgan_model()
        if realesrgan:
            print("✓ RealESRGAN loaded and ready")

    # Show memory usage
    if args.device == 'cuda':
        print("\n💾 Memory Usage:")
        mem = loader.get_memory_usage()
        for key, value in mem.items():
            print(f"   {key}: {value}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
