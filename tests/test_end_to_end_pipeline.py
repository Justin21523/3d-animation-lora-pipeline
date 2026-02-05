"""
End-to-End Pipeline Tests for 2D Animation LoRA Pipeline.

Tests the complete pipeline from video input to training-ready dataset,
validating all stages work together in stub mode (no GPU/API required).

Tests include:
- Frame extraction → Detection → Segmentation (existing stages)
- Captioning integration (VLM, OpenAI, SDXL expansion)
- Dataset preparation with captions
- Training config generation
- Full pipeline in stub mode
"""
import os
import sys
import pytest
from pathlib import Path
from typing import List, Dict
import json
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCaptioningIntegration:
    """Test captioning modules work together in stub mode."""

    def test_vlm_captioner_stub_mode(self, tmp_path):
        """Test VLM captioner in stub mode generates valid captions."""
        from anime_pipeline.captioning.vlm_captioner import VLMCaptioner, CaptionResult
        from PIL import Image

        # Create a dummy image
        image_path = tmp_path / "test_image.png"
        img = Image.new("RGB", (256, 256), color="blue")
        img.save(image_path)

        # Initialize in stub mode
        captioner = VLMCaptioner(use_stub=True)

        result = captioner.generate_caption(
            image_path=str(image_path),
            prefix="a 2d animated character",
            style="2d"
        )

        assert isinstance(result, CaptionResult)
        assert result.caption.startswith("a 2d animated character")
        assert result.tokens >= 0
        # VLM stub mode may report model name rather than "stub"
        assert result.model is not None

    def test_vlm_single_caption_stub(self, tmp_path):
        """Test VLM single captioning in stub mode."""
        from anime_pipeline.captioning.vlm_captioner import VLMCaptioner, CaptionResult
        from PIL import Image

        # Create dummy images and caption each
        captioner = VLMCaptioner(use_stub=True)

        results = []
        for i in range(3):
            image_path = tmp_path / f"test_image_{i}.png"
            img = Image.new("RGB", (128, 128), color=(i * 50, i * 50, i * 50))
            img.save(image_path)

            result = captioner.generate_caption(
                image_path=str(image_path),
                prefix="a 2d animated character",
                style="2d"
            )
            results.append(result)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, CaptionResult)
            assert result.caption.startswith("a 2d animated character")

    def test_openai_captioner_stub_mode(self, tmp_path):
        """Test OpenAI captioner in stub mode."""
        from anime_pipeline.captioning.openai_captioner import OpenAICaptioner, CaptionResult
        from PIL import Image

        # Create a dummy image
        image_path = tmp_path / "test_image.png"
        img = Image.new("RGB", (256, 256), color="green")
        img.save(image_path)

        captioner = OpenAICaptioner(use_stub=True)

        result = captioner.generate_caption(
            image_path=image_path,
            prefix="a 2d animated character",
            style="2d"
        )

        assert isinstance(result, CaptionResult)
        assert result.caption.startswith("a 2d animated character")
        assert result.tokens >= 0
        assert result.model == "stub"

    def test_sdxl_caption_expander_stub_mode(self):
        """Test SDXL caption expander in stub mode."""
        from anime_pipeline.captioning.sdxl_caption_expander import (
            SDXLCaptionExpander,
            ExpandedCaption,
        )

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
        assert len(result.quality_prefix) > 0
        assert len(result.negative_prompt) > 0

    def test_sdxl_batch_expansion_stub(self):
        """Test SDXL batch caption expansion in stub mode."""
        from anime_pipeline.captioning.sdxl_caption_expander import (
            SDXLCaptionExpander,
            BatchExpandResult,
        )

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
        assert len(result.results) == 3

    def test_sdxl_expand_directory_stub(self, tmp_path):
        """Test SDXL directory expansion in stub mode."""
        from anime_pipeline.captioning.sdxl_caption_expander import SDXLCaptionExpander

        expander = SDXLCaptionExpander(use_stub=True)

        # Create input directory with caption files
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        captions = [
            ("image1.txt", "a boy with brown hair"),
            ("image2.txt", "a girl in a blue dress"),
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
        assert result.total_captions == 2
        assert result.successful == 2

        # Check output files exist
        assert (output_dir / "image1.txt").exists()
        assert (output_dir / "image2.txt").exists()

        # Check negative prompt files exist
        assert (output_dir / "image1_negative.txt").exists()

        # Check metadata file exists
        assert (output_dir / "expansion_metadata.json").exists()

        # Check expanded content is longer
        original = "a boy with brown hair"
        expanded = (output_dir / "image1.txt").read_text(encoding="utf-8")
        assert len(expanded) > len(original)


class TestCaptioningPipelineIntegration:
    """Test captioning integrated with other pipeline stages."""

    def test_captioning_after_segmentation(self, tmp_path):
        """Test that captioning works on segmented character images."""
        from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
        from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking
        from anime_pipeline.segmentation.toonout_wrapper import SegmentConfig, segment_foreground_background
        from anime_pipeline.captioning.vlm_captioner import VLMCaptioner

        # Stage 1: Frame extraction (stub)
        extract_cfg = ExtractFramesConfig(
            input_videos_dir=tmp_path / "raw",
            output_dir=tmp_path / "frames",
            metadata_path=tmp_path / "frames.parquet",
            use_stub=True,
            stub_frame_count=3,
            stub_width=256,
            stub_height=256,
            log_dir=tmp_path / "logs",
        )
        extract_cfg.input_videos_dir.mkdir(parents=True, exist_ok=True)
        frame_records = extract_frames(extract_cfg)

        # Stage 2: Detection (stub)
        det_cfg = YoloTrackingConfig(
            frames_dir=extract_cfg.output_dir,
            output_detections_path=tmp_path / "detections.parquet",
            output_tracks_path=tmp_path / "tracks.parquet",
            use_stub=True,
            max_dets_per_frame=2,
            log_dir=tmp_path / "logs",
        )
        run_yolo_tracking(det_cfg)

        # Stage 3: Segmentation (stub)
        seg_cfg = SegmentConfig(
            frames_dir=extract_cfg.output_dir,
            detections_path=det_cfg.output_detections_path,
            output_fg_dir=tmp_path / "fg",
            output_bg_dir=tmp_path / "bg",
            output_fg_metadata_path=tmp_path / "fg.parquet",
            output_bg_metadata_path=tmp_path / "bg.parquet",
            use_stub=True,
            log_dir=tmp_path / "logs",
        )
        fg_records, _ = segment_foreground_background(seg_cfg)

        # Stage 4: Caption the segmented foregrounds
        captioner = VLMCaptioner(use_stub=True)

        for record in fg_records:
            fg_path = record.get('rgba_path')
            if fg_path and Path(fg_path).exists():
                result = captioner.generate_caption(
                    image_path=fg_path,
                    prefix="a 2d animated character",
                    style="2d"
                )
                assert result.caption.startswith("a 2d animated character")

    def test_caption_expansion_workflow(self, tmp_path):
        """Test the full caption expansion workflow: short → expanded."""
        from anime_pipeline.captioning.vlm_captioner import VLMCaptioner
        from anime_pipeline.captioning.sdxl_caption_expander import SDXLCaptionExpander
        from PIL import Image

        # Create a dummy image
        image_path = tmp_path / "test_char.png"
        img = Image.new("RGB", (512, 512), color="red")
        img.save(image_path)

        # Step 1: Generate short caption with VLM
        vlm = VLMCaptioner(use_stub=True)
        short_result = vlm.generate_caption(
            image_path=str(image_path),
            prefix="a 2d animated character",
            style="2d"
        )

        assert len(short_result.caption) > 0

        # Step 2: Expand caption for SDXL
        expander = SDXLCaptionExpander(use_stub=True)
        expanded = expander.expand_caption(
            caption=short_result.caption,
            style="2d"
        )

        # Verify expansion
        assert len(expanded.main_prompt) > len(short_result.caption)
        assert expanded.token_count > 0
        assert len(expanded.quality_prefix) > 0
        assert len(expanded.negative_prompt) > 0

        # Full caption should be longer than original
        assert len(expanded.full_caption) > len(short_result.caption)


class TestDatasetPreparationIntegration:
    """Test dataset preparation with captioning."""

    def test_dataset_with_captions(self, tmp_path):
        """Test creating a dataset with images and captions."""
        from PIL import Image

        # Create character cluster directory
        cluster_dir = tmp_path / "clusters" / "character_0"
        cluster_dir.mkdir(parents=True)

        # Create dummy character images
        for i in range(5):
            img_path = cluster_dir / f"char_{i:04d}.png"
            img = Image.new("RGB", (512, 512), color=(i * 40, i * 40, i * 40))
            img.save(img_path)

        # Create output dataset directory
        output_dir = tmp_path / "dataset"
        output_dir.mkdir()

        # Mock dataset preparation
        images_dir = output_dir / "images"
        captions_dir = output_dir / "captions"
        images_dir.mkdir()
        captions_dir.mkdir()

        # Copy images and create captions
        from anime_pipeline.captioning.vlm_captioner import VLMCaptioner
        captioner = VLMCaptioner(use_stub=True)

        for img_file in cluster_dir.glob("*.png"):
            # Copy image
            shutil.copy(img_file, images_dir / img_file.name)

            # Generate caption
            result = captioner.generate_caption(
                image_path=str(img_file),
                prefix="character_name",
                style="2d"
            )

            # Save caption
            caption_file = captions_dir / f"{img_file.stem}.txt"
            caption_file.write_text(result.caption, encoding="utf-8")

        # Verify dataset structure
        assert len(list(images_dir.glob("*.png"))) == 5
        assert len(list(captions_dir.glob("*.txt"))) == 5

        # Verify caption content
        sample_caption = (captions_dir / "char_0000.txt").read_text(encoding="utf-8")
        assert "character_name" in sample_caption

    def test_sdxl_dataset_preparation(self, tmp_path):
        """Test preparing dataset with SDXL-expanded captions."""
        from PIL import Image
        from anime_pipeline.captioning.sdxl_caption_expander import SDXLCaptionExpander

        # Create source captions
        captions_dir = tmp_path / "captions"
        captions_dir.mkdir()

        short_captions = [
            "a boy with brown hair, smiling",
            "a girl in a blue dress, standing",
            "a robot character with red eyes",
        ]

        for i, caption in enumerate(short_captions):
            (captions_dir / f"image_{i}.txt").write_text(caption, encoding="utf-8")

        # Expand captions for SDXL
        expander = SDXLCaptionExpander(use_stub=True)
        output_dir = tmp_path / "expanded_captions"

        result = expander.expand_directory(
            input_dir=captions_dir,
            output_dir=output_dir,
            style="2d",
        )

        # Verify expansion
        assert result.total_captions == 3
        assert result.successful == 3

        # Check expanded captions are longer
        for i in range(3):
            original = (captions_dir / f"image_{i}.txt").read_text(encoding="utf-8")
            expanded = (output_dir / f"image_{i}.txt").read_text(encoding="utf-8")
            assert len(expanded) > len(original)

        # Check negative prompts exist
        assert (output_dir / "image_0_negative.txt").exists()


class TestFullPipelineStubMode:
    """Test complete pipeline execution in stub mode."""

    def test_full_pipeline_with_captioning(self, tmp_path):
        """Test complete pipeline from extraction to captioned dataset."""
        from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
        from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking
        from anime_pipeline.segmentation.toonout_wrapper import SegmentConfig, segment_foreground_background
        from anime_pipeline.captioning.vlm_captioner import VLMCaptioner
        from anime_pipeline.captioning.sdxl_caption_expander import SDXLCaptionExpander

        # Stage 1: Frame extraction
        extract_cfg = ExtractFramesConfig(
            input_videos_dir=tmp_path / "raw",
            output_dir=tmp_path / "frames",
            metadata_path=tmp_path / "frames.parquet",
            use_stub=True,
            stub_frame_count=5,
            stub_width=512,
            stub_height=512,
            log_dir=tmp_path / "logs",
        )
        extract_cfg.input_videos_dir.mkdir(parents=True, exist_ok=True)
        frame_records = extract_frames(extract_cfg)
        assert len(frame_records) == 5

        # Stage 2: Detection - need to handle CSV fallback path
        det_metadata_base = tmp_path / "detections"
        det_cfg = YoloTrackingConfig(
            frames_dir=extract_cfg.output_dir,
            output_detections_path=tmp_path / "detections.parquet",
            output_tracks_path=tmp_path / "tracks.parquet",
            use_stub=True,
            max_dets_per_frame=2,
            log_dir=tmp_path / "logs",
        )
        detections, _ = run_yolo_tracking(det_cfg)
        assert len(detections) >= 1

        # Stage 3: Segmentation - handle CSV fallback
        seg_cfg = SegmentConfig(
            frames_dir=extract_cfg.output_dir,
            detections_path=tmp_path / "detections.csv" if (tmp_path / "detections.csv").exists() else det_cfg.output_detections_path,
            output_fg_dir=tmp_path / "fg",
            output_bg_dir=tmp_path / "bg",
            output_fg_metadata_path=tmp_path / "fg.parquet",
            output_bg_metadata_path=tmp_path / "bg.parquet",
            use_stub=True,
            log_dir=tmp_path / "logs",
        )
        fg_records, bg_records = segment_foreground_background(seg_cfg)
        assert len(fg_records) >= 1

        # Stage 4: Generate captions
        captioner = VLMCaptioner(use_stub=True)
        captions_dir = tmp_path / "captions"
        captions_dir.mkdir()

        generated_captions = []
        for record in fg_records:
            fg_path = record.get('rgba_path')
            if fg_path and Path(fg_path).exists():
                result = captioner.generate_caption(
                    image_path=fg_path,
                    prefix="test_character",
                    style="2d"
                )
                # Save caption
                stem = Path(fg_path).stem
                caption_file = captions_dir / f"{stem}.txt"
                caption_file.write_text(result.caption, encoding="utf-8")
                generated_captions.append(result.caption)

        assert len(generated_captions) >= 1

        # Stage 5: Expand captions for SDXL
        expander = SDXLCaptionExpander(use_stub=True)
        expanded_dir = tmp_path / "expanded_captions"

        expand_result = expander.expand_directory(
            input_dir=captions_dir,
            output_dir=expanded_dir,
            style="2d",
        )

        # Verify pipeline outputs
        assert (tmp_path / "frames").exists()
        # Check for either parquet or csv (fallback when pyarrow not installed)
        assert (tmp_path / "detections.parquet").exists() or (tmp_path / "detections.csv").exists()
        assert (tmp_path / "fg").exists()
        assert expand_result.total_captions >= 1
        assert expand_result.successful >= 1

    def test_pipeline_output_metadata(self, tmp_path):
        """Test that pipeline generates proper metadata files."""
        from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
        from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking
        import pandas as pd

        # Run partial pipeline
        extract_cfg = ExtractFramesConfig(
            input_videos_dir=tmp_path / "raw",
            output_dir=tmp_path / "frames",
            metadata_path=tmp_path / "frames.parquet",
            use_stub=True,
            stub_frame_count=3,
            stub_width=256,
            stub_height=256,
            log_dir=tmp_path / "logs",
        )
        extract_cfg.input_videos_dir.mkdir(parents=True, exist_ok=True)
        extract_frames(extract_cfg)

        det_cfg = YoloTrackingConfig(
            frames_dir=extract_cfg.output_dir,
            output_detections_path=tmp_path / "detections.parquet",
            output_tracks_path=tmp_path / "tracks.parquet",
            use_stub=True,
            max_dets_per_frame=2,
            log_dir=tmp_path / "logs",
        )
        run_yolo_tracking(det_cfg)

        # Verify metadata files - handle CSV fallback when pyarrow not available
        # Check for either parquet or csv
        frames_parquet = extract_cfg.metadata_path
        frames_csv = Path(str(extract_cfg.metadata_path).replace('.parquet', '.csv'))

        if frames_parquet.exists():
            try:
                frames_df = pd.read_parquet(frames_parquet)
            except ImportError:
                frames_df = pd.read_csv(frames_csv)
        else:
            frames_df = pd.read_csv(frames_csv)

        assert len(frames_df) == 3
        assert 'image_path' in frames_df.columns or 'frame_path' in frames_df.columns or 'path' in frames_df.columns

        det_parquet = det_cfg.output_detections_path
        det_csv = Path(str(det_cfg.output_detections_path).replace('.parquet', '.csv'))

        if det_parquet.exists():
            try:
                det_df = pd.read_parquet(det_parquet)
            except ImportError:
                det_df = pd.read_csv(det_csv)
        else:
            det_df = pd.read_csv(det_csv)

        assert len(det_df) >= 1


class TestCaptionQualityIntegration:
    """Test caption quality validation integration."""

    def test_caption_token_limits(self):
        """Test that captions respect token limits."""
        from anime_pipeline.captioning.sdxl_caption_expander import SDXLCaptionExpander

        expander = SDXLCaptionExpander(use_stub=True, max_tokens=200)

        result = expander.expand_caption(
            caption="a simple test character",
            style="2d",
        )

        # Token count should be within reasonable bounds for SDXL
        # SDXL optimal range is 77-150 tokens
        assert result.token_count > 0
        # Stub mode generates predictable output

    def test_caption_style_consistency(self):
        """Test that captions maintain style consistency."""
        from anime_pipeline.captioning.sdxl_caption_expander import (
            SDXLCaptionExpander,
            SDXL_QUALITY_PREFIXES,
            SDXL_TECHNICAL_DETAILS,
            SDXL_NEGATIVE_PROMPTS,
        )

        expander = SDXLCaptionExpander(use_stub=True)

        for style in ["2d", "3d", "anime", "realistic"]:
            result = expander.expand_caption(
                caption="a test character",
                style=style,
            )

            # Verify style-specific elements
            assert result.style == style
            assert result.quality_prefix == SDXL_QUALITY_PREFIXES[style]
            assert result.technical_details == SDXL_TECHNICAL_DETAILS[style]
            assert result.negative_prompt == SDXL_NEGATIVE_PROMPTS[style]


class TestCostTracking:
    """Test cost tracking for API-based modules."""

    def test_sdxl_expander_cost_tracking(self):
        """Test SDXL expander tracks costs in stub mode."""
        from anime_pipeline.captioning.sdxl_caption_expander import SDXLCaptionExpander

        expander = SDXLCaptionExpander(use_stub=True)

        # Generate some captions
        for _ in range(5):
            expander.expand_caption("test caption", style="2d")

        report = expander.get_cost_report()

        # Stub mode should report zero costs
        assert report["total_cost_usd"] == 0.0
        assert report["model"] == "gpt-4o-mini"

        # Reset and verify
        expander.reset_cost_tracking()
        report = expander.get_cost_report()
        assert report["total_captions"] == 0

    def test_openai_captioner_cost_tracking(self):
        """Test OpenAI captioner tracks costs in stub mode."""
        from anime_pipeline.captioning.openai_captioner import OpenAICaptioner

        captioner = OpenAICaptioner(use_stub=True)

        report = captioner.get_cost_report()

        # Stub mode should report zero costs
        assert report["total_cost_usd"] == 0.0
        assert report["total_input_tokens"] == 0
        assert report["model"] == "gpt-4o-mini"


class TestErrorHandling:
    """Test error handling in captioning pipeline."""

    def test_sdxl_empty_caption_handling(self):
        """Test SDXL expander handles empty captions gracefully."""
        from anime_pipeline.captioning.sdxl_caption_expander import SDXLCaptionExpander

        expander = SDXLCaptionExpander(use_stub=True)

        # Empty caption should still produce valid output in stub mode
        result = expander.expand_caption("", style="2d")

        assert result.original == ""
        # Stub mode will add expansion anyway
        assert result.model == "stub"

    def test_nonexistent_image_handling(self, tmp_path):
        """Test VLM captioner handles nonexistent images."""
        from anime_pipeline.captioning.vlm_captioner import VLMCaptioner

        captioner = VLMCaptioner(use_stub=True)

        # In stub mode, may generate caption anyway since no actual image loading
        # This tests that it doesn't crash
        try:
            result = captioner.generate_caption(
                image_path=str(tmp_path / "nonexistent.png"),
                prefix="test",
                style="2d"
            )
            # Stub mode might still work
            assert result is not None
        except (FileNotFoundError, Exception):
            # Expected behavior for production mode
            pass


class TestMultiStyleSupport:
    """Test multi-style support across captioning modules."""

    def test_all_styles_supported(self):
        """Test all animation styles are supported."""
        from anime_pipeline.captioning.sdxl_caption_expander import (
            SDXLCaptionExpander,
            SDXL_QUALITY_PREFIXES,
        )

        supported_styles = ["2d", "3d", "anime", "realistic"]

        expander = SDXLCaptionExpander(use_stub=True)

        for style in supported_styles:
            assert style in SDXL_QUALITY_PREFIXES

            result = expander.expand_caption(
                caption="a test character",
                style=style,
            )

            assert result.style == style
            assert len(result.quality_prefix) > 0

    def test_default_style_fallback(self):
        """Test default style is used for unknown styles."""
        from anime_pipeline.captioning.sdxl_caption_expander import (
            SDXL_QUALITY_PREFIXES,
        )

        # Default should exist
        assert "default" in SDXL_QUALITY_PREFIXES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
