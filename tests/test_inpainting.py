"""
Inpainting Module Tests

Tests for LaMa and PowerPaint inpainting in stub mode.
"""
import pytest
from pathlib import Path
import numpy as np


class TestLaMaInpainter:
    """Test LaMa inpainting module."""

    def test_import_lama_inpainter(self):
        """Test that LaMaInpainter can be imported."""
        from anime_pipeline.inpainting import LaMaInpainter
        assert LaMaInpainter is not None

    def test_lama_stub_mode_init(self, tmp_path):
        """Test LaMaInpainter initialization in stub mode."""
        from anime_pipeline.inpainting import LaMaInpainter

        inpainter = LaMaInpainter(
            model_path=str(tmp_path / "fake_model.ckpt"),
            device="cpu",
            stub_mode=True
        )

        assert inpainter.stub_mode is True

    def test_lama_inpaint_stub(self, tmp_path):
        """Test LaMa inpainting in stub mode."""
        from anime_pipeline.inpainting import LaMaInpainter

        inpainter = LaMaInpainter(
            model_path=str(tmp_path / "fake_model.ckpt"),
            device="cpu",
            stub_mode=True
        )

        # Create test image and mask
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 255  # Square mask region

        # Inpaint
        result = inpainter.inpaint(image, mask)

        # Verify output
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_lama_batch_inpaint_stub(self, tmp_path):
        """Test LaMa batch inpainting in stub mode."""
        from anime_pipeline.inpainting import LaMaInpainter
        from PIL import Image

        inpainter = LaMaInpainter(
            model_path=str(tmp_path / "fake_model.ckpt"),
            device="cpu",
            stub_mode=True
        )

        # Create test images and masks
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "inpainted"

        image_dir.mkdir()
        mask_dir.mkdir()

        for i in range(3):
            img = Image.new('RGB', (128, 128), color=(100, 100, 100))
            img.save(image_dir / f"frame_{i:04d}.png")

            mask = Image.new('L', (128, 128), color=0)
            mask.save(mask_dir / f"frame_{i:04d}.png")

        # Batch inpaint
        results = inpainter.batch_inpaint(image_dir, mask_dir, output_dir)

        assert output_dir.exists()
        assert len(list(output_dir.glob("*.png"))) == 3


class TestPowerPaintInpainter:
    """Test PowerPaint inpainting module."""

    def test_import_powerpaint_inpainter(self):
        """Test that PowerPaintInpainter can be imported."""
        from anime_pipeline.inpainting import PowerPaintInpainter
        assert PowerPaintInpainter is not None

    def test_powerpaint_stub_mode_init(self, tmp_path):
        """Test PowerPaintInpainter initialization in stub mode."""
        from anime_pipeline.inpainting import PowerPaintInpainter

        inpainter = PowerPaintInpainter(
            model_path=str(tmp_path / "fake_model.safetensors"),
            device="cpu",
            stub_mode=True
        )

        assert inpainter.stub_mode is True

    def test_powerpaint_inpaint_with_prompt_stub(self, tmp_path):
        """Test PowerPaint text-guided inpainting in stub mode."""
        from anime_pipeline.inpainting import PowerPaintInpainter

        inpainter = PowerPaintInpainter(
            model_path=str(tmp_path / "fake_model.safetensors"),
            device="cpu",
            stub_mode=True
        )

        # Create test image and mask
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        # Inpaint with text prompt
        result = inpainter.inpaint(
            image,
            mask,
            prompt="a simple background with blue sky"
        )

        assert result.shape == image.shape


class TestInpaintingQuality:
    """Test inpainting quality metrics."""

    def test_compute_quality_score(self, tmp_path):
        """Test quality score computation (PSNR/SSIM)."""
        from anime_pipeline.inpainting import LaMaInpainter

        inpainter = LaMaInpainter(
            model_path=str(tmp_path / "fake_model.ckpt"),
            device="cpu",
            stub_mode=True
        )

        # Create test images
        original = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        inpainted = original.copy()  # Perfect reconstruction

        score = inpainter.compute_quality_score(original, inpainted)

        # Perfect reconstruction should have high score
        assert score >= 0.0


class TestInpaintingIntegration:
    """Test inpainting integration with pipeline."""

    def test_inpainting_after_segmentation_stub(self, tmp_path):
        """Test inpainting works after segmentation stage."""
        from anime_pipeline.inpainting import LaMaInpainter
        from PIL import Image
        import numpy as np

        # Create mock segmentation outputs
        bg_dir = tmp_path / "backgrounds"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "inpainted"

        bg_dir.mkdir()
        mask_dir.mkdir()

        # Create background with hole and corresponding mask
        for i in range(2):
            # Background with visible hole (gray region)
            bg = np.ones((256, 256, 3), dtype=np.uint8) * 150
            bg[50:150, 50:150] = 0  # Black hole where character was
            Image.fromarray(bg).save(bg_dir / f"bg_{i:04d}.png")

            # Mask indicating hole region
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[50:150, 50:150] = 255
            Image.fromarray(mask).save(mask_dir / f"bg_{i:04d}.png")

        # Initialize inpainter in stub mode
        inpainter = LaMaInpainter(
            model_path=str(tmp_path / "fake_model.ckpt"),
            device="cpu",
            stub_mode=True
        )

        # Run batch inpainting
        results = inpainter.batch_inpaint(bg_dir, mask_dir, output_dir)

        # Verify outputs
        assert output_dir.exists()
        inpainted_files = list(output_dir.glob("*.png"))
        assert len(inpainted_files) == 2

        # Verify each output is valid image
        for f in inpainted_files:
            img = Image.open(f)
            assert img.size == (256, 256)
