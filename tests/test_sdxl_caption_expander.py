"""
Tests for SDXLCaptionExpander module.

Tests stub mode functionality without requiring actual API calls.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anime_pipeline.captioning.sdxl_caption_expander import (
    SDXLCaptionExpander,
    ExpandedCaption,
    BatchExpandResult,
    SDXL_QUALITY_PREFIXES,
    SDXL_TECHNICAL_DETAILS,
    SDXL_NEGATIVE_PROMPTS,
)


class TestExpandedCaption:
    """Test ExpandedCaption dataclass."""

    def test_expanded_caption_creation(self):
        """Test ExpandedCaption can be created with required fields."""
        result = ExpandedCaption(
            original="a boy with brown hair",
            main_prompt="a young boy with brown hair, smiling, soft lighting",
            quality_prefix="masterpiece, best quality",
            technical_details="clean lines, vibrant colors",
            negative_prompt="blurry, low quality",
            full_caption="masterpiece, best quality, a young boy, clean lines",
            token_count=50,
            style="2d",
            model="gpt-4o-mini",
            generation_time=0.5,
        )

        assert result.original == "a boy with brown hair"
        assert result.main_prompt.startswith("a young boy")
        assert result.token_count == 50
        assert result.style == "2d"

    def test_to_dict(self):
        """Test ExpandedCaption to_dict method."""
        result = ExpandedCaption(
            original="test",
            main_prompt="expanded test",
            quality_prefix="quality",
            technical_details="details",
            negative_prompt="negative",
            full_caption="full",
            token_count=10,
            style="2d",
            model="stub",
            generation_time=0.1,
        )

        d = result.to_dict()
        assert d["original"] == "test"
        assert d["main_prompt"] == "expanded test"
        assert d["token_count"] == 10
        assert "generation_time" in d


class TestBatchExpandResult:
    """Test BatchExpandResult dataclass."""

    def test_batch_result_creation(self):
        """Test BatchExpandResult can be created."""
        expanded1 = ExpandedCaption(
            original="caption1",
            main_prompt="expanded1",
            quality_prefix="quality",
            technical_details="details",
            negative_prompt="negative",
            full_caption="full1",
            token_count=50,
            style="2d",
            model="stub",
            generation_time=0.1,
        )
        expanded2 = ExpandedCaption(
            original="caption2",
            main_prompt="expanded2",
            quality_prefix="quality",
            technical_details="details",
            negative_prompt="negative",
            full_caption="full2",
            token_count=60,
            style="2d",
            model="stub",
            generation_time=0.2,
        )

        batch_result = BatchExpandResult(
            total_captions=2,
            successful=2,
            failed=0,
            avg_token_count=55.0,
            total_time=0.3,
            results=[expanded1, expanded2],
        )

        assert batch_result.total_captions == 2
        assert batch_result.successful == 2
        assert batch_result.failed == 0
        assert batch_result.avg_token_count == 55.0
        assert len(batch_result.results) == 2


class TestSDXLCaptionExpanderStubMode:
    """Test SDXLCaptionExpander in stub mode (no API calls)."""

    def test_init_stub_mode(self):
        """Test initialization in stub mode."""
        expander = SDXLCaptionExpander(use_stub=True)
        assert expander.use_stub is True
        assert expander.model == "gpt-4o-mini"
        assert expander.max_tokens == 200

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        expander = SDXLCaptionExpander(
            model="gpt-4o",
            max_tokens=300,
            temperature=0.5,
            concurrent_requests=5,
            use_stub=True,
        )
        assert expander.model == "gpt-4o"
        assert expander.max_tokens == 300
        assert expander.temperature == 0.5
        assert expander.concurrent_requests == 5

    def test_stub_caption_expansion(self):
        """Test stub caption expansion returns valid ExpandedCaption."""
        expander = SDXLCaptionExpander(use_stub=True)

        result = expander.expand_caption(
            caption="a young boy with brown hair, smiling",
            style="2d",
        )

        assert isinstance(result, ExpandedCaption)
        assert result.original == "a young boy with brown hair, smiling"
        assert len(result.main_prompt) > len(result.original)
        assert result.token_count > 0
        assert result.model == "stub"
        assert result.style == "2d"
        assert len(result.quality_prefix) > 0
        assert len(result.technical_details) > 0
        assert len(result.negative_prompt) > 0

    def test_stub_batch_expand(self):
        """Test stub batch caption expansion."""
        expander = SDXLCaptionExpander(use_stub=True)

        captions = [
            "a boy with brown hair",
            "a girl in a red dress",
            "a robot character",
        ]

        result = expander.batch_expand(
            captions=captions,
            style="2d",
        )

        assert isinstance(result, BatchExpandResult)
        assert result.total_captions == 3
        assert result.successful == 3
        assert result.failed == 0
        assert result.avg_token_count > 0
        assert result.total_time >= 0
        assert len(result.results) == 3

    def test_different_styles(self):
        """Test caption expansion with different animation styles."""
        expander = SDXLCaptionExpander(use_stub=True)

        styles = ["2d", "3d", "anime", "realistic"]

        for style in styles:
            result = expander.expand_caption(
                caption="a character standing",
                style=style,
            )

            assert result.style == style
            assert result.quality_prefix == SDXL_QUALITY_PREFIXES[style]
            assert result.technical_details == SDXL_TECHNICAL_DETAILS[style]
            assert result.negative_prompt == SDXL_NEGATIVE_PROMPTS[style]

    def test_disable_quality_prefix(self):
        """Test caption expansion without quality prefix."""
        expander = SDXLCaptionExpander(use_stub=True)

        result = expander.expand_caption(
            caption="a character",
            style="2d",
            include_quality_prefix=False,
        )

        assert result.quality_prefix == ""
        # Full caption should not start with quality prefix
        assert not result.full_caption.startswith("masterpiece")

    def test_disable_technical_details(self):
        """Test caption expansion without technical details."""
        expander = SDXLCaptionExpander(use_stub=True)

        result = expander.expand_caption(
            caption="a character",
            style="2d",
            include_technical=False,
        )

        assert result.technical_details == ""

    def test_custom_negative_prompt(self):
        """Test caption expansion with custom negative prompt."""
        expander = SDXLCaptionExpander(use_stub=True)

        custom_negative = "bad art, ugly, deformed hands"
        result = expander.expand_caption(
            caption="a character",
            style="2d",
            custom_negative=custom_negative,
        )

        assert result.negative_prompt == custom_negative

    def test_cost_tracking_stub(self):
        """Test cost report in stub mode returns zeros."""
        expander = SDXLCaptionExpander(use_stub=True)

        report = expander.get_cost_report()

        assert report["total_cost_usd"] == 0.0
        assert report["total_input_tokens"] == 0
        assert report["total_output_tokens"] == 0
        assert report["total_captions"] == 0
        assert report["model"] == "gpt-4o-mini"

    def test_reset_cost_tracking(self):
        """Test cost tracking reset."""
        expander = SDXLCaptionExpander(use_stub=True)

        # Generate some captions
        expander.expand_caption("test", style="2d")

        # Reset
        expander.reset_cost_tracking()

        report = expander.get_cost_report()
        assert report["total_cost_usd"] == 0.0
        assert report["total_captions"] == 0


class TestSDXLCaptionExpanderDirectory:
    """Test directory expansion functionality."""

    def test_expand_directory_stub(self, tmp_path):
        """Test directory expansion in stub mode."""
        expander = SDXLCaptionExpander(use_stub=True)

        # Create input directory with caption files
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        captions = [
            ("image1.txt", "a boy with brown hair"),
            ("image2.txt", "a girl in a blue dress"),
            ("image3.txt", "a robot character with red eyes"),
        ]

        for filename, caption in captions:
            (input_dir / filename).write_text(caption, encoding="utf-8")

        # Run expansion
        output_dir = tmp_path / "output"
        result = expander.expand_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            style="2d",
        )

        # Check results
        assert result.total_captions == 3
        assert result.successful == 3

        # Check output files exist
        assert (output_dir / "image1.txt").exists()
        assert (output_dir / "image2.txt").exists()
        assert (output_dir / "image3.txt").exists()

        # Check negative prompt files exist
        assert (output_dir / "image1_negative.txt").exists()

        # Check metadata file exists
        assert (output_dir / "expansion_metadata.json").exists()

        # Check expanded content is longer
        original = "a boy with brown hair"
        expanded = (output_dir / "image1.txt").read_text(encoding="utf-8")
        assert len(expanded) > len(original)

    def test_expand_empty_directory(self, tmp_path):
        """Test expansion of empty directory."""
        expander = SDXLCaptionExpander(use_stub=True)

        input_dir = tmp_path / "empty"
        input_dir.mkdir()

        result = expander.expand_directory(
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            style="2d",
        )

        assert result.total_captions == 0
        assert result.successful == 0


class TestSDXLCaptionExpanderAPIKeyHandling:
    """Test API key handling."""

    def test_api_key_from_param(self):
        """Test API key can be passed as parameter."""
        expander = SDXLCaptionExpander(
            api_key="test-key-123",
            use_stub=True,
        )
        assert expander.api_key == "test-key-123"

    def test_api_key_from_env(self):
        """Test API key can be read from environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-456"}):
            expander = SDXLCaptionExpander(use_stub=True)
            assert expander.api_key == "env-key-456"


class TestSDXLQualityPrefixes:
    """Test quality prefix configurations."""

    def test_all_styles_have_prefix(self):
        """Test all styles have quality prefixes defined."""
        styles = ["2d", "3d", "anime", "realistic", "default"]
        for style in styles:
            assert style in SDXL_QUALITY_PREFIXES
            assert len(SDXL_QUALITY_PREFIXES[style]) > 0

    def test_all_styles_have_technical_details(self):
        """Test all styles have technical details defined."""
        styles = ["2d", "3d", "anime", "realistic", "default"]
        for style in styles:
            assert style in SDXL_TECHNICAL_DETAILS
            assert len(SDXL_TECHNICAL_DETAILS[style]) > 0

    def test_all_styles_have_negative_prompts(self):
        """Test all styles have negative prompts defined."""
        styles = ["2d", "3d", "anime", "realistic", "default"]
        for style in styles:
            assert style in SDXL_NEGATIVE_PROMPTS
            assert len(SDXL_NEGATIVE_PROMPTS[style]) > 0


class TestTokenEstimation:
    """Test token estimation functionality."""

    def test_token_estimation(self):
        """Test token count estimation."""
        expander = SDXLCaptionExpander(use_stub=True)

        # Test various text lengths
        short_text = "hello world"  # ~3 tokens
        medium_text = "a young boy with brown hair standing in a park"  # ~12 tokens
        long_text = "masterpiece, best quality, highly detailed 2d animation, " * 5

        assert expander._estimate_tokens(short_text) > 0
        assert expander._estimate_tokens(medium_text) > expander._estimate_tokens(short_text)
        assert expander._estimate_tokens(long_text) > expander._estimate_tokens(medium_text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
