"""
Tests for OpenAICaptioner module.

Tests stub mode functionality without requiring actual API calls.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anime_pipeline.captioning.openai_captioner import (
    OpenAICaptioner,
    CaptionResult,
    BatchCaptionResult,
)


class TestOpenAICaptionerStubMode:
    """Test OpenAICaptioner in stub mode (no API calls)."""

    def test_init_stub_mode(self):
        """Test initialization in stub mode."""
        captioner = OpenAICaptioner(use_stub=True)
        assert captioner.use_stub is True
        assert captioner.model == "gpt-4o-mini"
        assert captioner.max_tokens == 150

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        captioner = OpenAICaptioner(
            model="gpt-4-vision-preview",
            max_tokens=200,
            temperature=0.5,
            detail="high",
            concurrent_requests=5,
            use_stub=True,
        )
        assert captioner.model == "gpt-4-vision-preview"
        assert captioner.max_tokens == 200
        assert captioner.temperature == 0.5
        assert captioner.detail == "high"
        assert captioner.concurrent_requests == 5

    def test_stub_caption_generation(self, tmp_path):
        """Test stub caption generation returns valid CaptionResult."""
        captioner = OpenAICaptioner(use_stub=True)

        # Create a dummy image file
        image_path = tmp_path / "test_image.png"
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="red")
        img.save(image_path)

        result = captioner.generate_caption(
            image_path=image_path,
            prefix="a 2d animated character",
            style="2d",
        )

        assert isinstance(result, CaptionResult)
        assert result.caption.startswith("a 2d animated character")
        assert result.tokens >= 0
        assert result.generation_time >= 0
        assert result.model == "stub"

    def test_stub_batch_caption(self, tmp_path):
        """Test stub batch caption generation."""
        captioner = OpenAICaptioner(use_stub=True)

        # Create dummy images
        image_paths = []
        from PIL import Image
        for i in range(3):
            image_path = tmp_path / f"test_image_{i}.png"
            img = Image.new("RGB", (100, 100), color=f"#{i:02x}{i:02x}{i:02x}")
            img.save(image_path)
            image_paths.append(image_path)

        result = captioner.batch_caption(
            image_paths=image_paths,
            prefix="a 2d animated character",
            style="2d",
        )

        assert isinstance(result, BatchCaptionResult)
        assert len(result.results) == 3
        assert result.total_images == 3
        assert result.successful == 3
        assert result.failed == 0
        assert result.avg_tokens >= 0
        assert result.total_time >= 0

    def test_cost_tracking_stub(self):
        """Test cost report in stub mode returns zeros."""
        captioner = OpenAICaptioner(use_stub=True)

        report = captioner.get_cost_report()

        assert report["total_cost_usd"] == 0.0
        assert report["total_input_tokens"] == 0
        assert report["total_output_tokens"] == 0
        assert report["total_images"] == 0
        assert report["model"] == "gpt-4o-mini"


class TestCaptionResult:
    """Test CaptionResult dataclass."""

    def test_caption_result_creation(self):
        """Test CaptionResult can be created with required fields."""
        result = CaptionResult(
            image_path="/path/to/image.png",
            caption="test caption",
            tokens=50,
            model="gpt-4o-mini",
            generation_time=0.5,
        )

        assert result.image_path == "/path/to/image.png"
        assert result.caption == "test caption"
        assert result.tokens == 50
        assert result.generation_time == 0.5
        assert result.model == "gpt-4o-mini"


class TestBatchCaptionResult:
    """Test BatchCaptionResult dataclass."""

    def test_batch_result_creation(self):
        """Test BatchCaptionResult can be created."""
        caption_results = [
            CaptionResult(
                image_path="/path/to/image1.png",
                caption="caption 1",
                tokens=50,
                model="gpt-4o-mini",
                generation_time=0.5,
            ),
            CaptionResult(
                image_path="/path/to/image2.png",
                caption="caption 2",
                tokens=60,
                model="gpt-4o-mini",
                generation_time=0.6,
            ),
        ]

        batch_result = BatchCaptionResult(
            total_images=2,
            successful=2,
            failed=0,
            avg_tokens=55.0,
            total_time=1.1,
            results=caption_results,
        )

        assert len(batch_result.results) == 2
        assert batch_result.total_images == 2
        assert batch_result.successful == 2
        assert batch_result.failed == 0
        assert batch_result.avg_tokens == 55.0
        assert batch_result.total_time == 1.1


class TestOpenAICaptionerAPIKeyHandling:
    """Test API key handling."""

    def test_api_key_from_param(self):
        """Test API key can be passed as parameter."""
        captioner = OpenAICaptioner(
            api_key="test-key-123",
            use_stub=True,
        )
        assert captioner.api_key == "test-key-123"

    def test_api_key_from_env(self):
        """Test API key can be read from environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-456"}):
            captioner = OpenAICaptioner(use_stub=True)
            assert captioner.api_key == "env-key-456"


class TestPromptGeneration:
    """Test prompt generation for different styles."""

    def test_2d_style_prompt(self):
        """Test 2D animation style prompt."""
        captioner = OpenAICaptioner(use_stub=True)
        
        prompt = captioner._build_prompt(
            prefix="a 2d animated character",
            style="2d",
        )
        
        assert "2d" in prompt.lower() or "2D" in prompt
        assert "character" in prompt.lower()

    def test_3d_style_prompt(self):
        """Test 3D animation style prompt."""
        captioner = OpenAICaptioner(use_stub=True)
        
        prompt = captioner._build_prompt(
            prefix="a 3d animated character, pixar style",
            style="3d",
        )
        
        assert "3d" in prompt.lower() or "3D" in prompt

    def test_custom_prompt(self):
        """Test custom prompt override."""
        captioner = OpenAICaptioner(use_stub=True)
        
        custom = "Describe this anime character in detail."
        prompt = captioner._build_prompt(
            prefix="",
            style="2d",
            custom_prompt=custom,
        )
        
        assert custom in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
