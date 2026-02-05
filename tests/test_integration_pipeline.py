"""
Integration Pipeline Tests

Tests for full stub-mode end-to-end pipeline execution.
Validates that all stages can run together and produce expected outputs.
"""
import pytest
from pathlib import Path
import json


class TestPipelineStages:
    """Test individual pipeline stages in stub mode."""

    def test_frame_extraction_stub(self, tmp_path):
        """Test frame extraction in stub mode."""
        from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames

        config = ExtractFramesConfig(
            input_videos_dir=tmp_path / "raw",
            output_dir=tmp_path / "frames",
            metadata_path=tmp_path / "frames.parquet",
            use_stub=True,
            stub_frame_count=5,
            stub_width=128,
            stub_height=128,
            log_dir=tmp_path / "logs",
        )
        config.input_videos_dir.mkdir(parents=True, exist_ok=True)

        records = extract_frames(config)

        assert len(records) == 5
        assert config.output_dir.exists()
        assert config.metadata_path.exists()

    def test_detection_stub(self, tmp_path):
        """Test YOLO detection in stub mode."""
        from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
        from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking

        # First create frames
        extract_cfg = ExtractFramesConfig(
            input_videos_dir=tmp_path / "raw",
            output_dir=tmp_path / "frames",
            metadata_path=tmp_path / "frames.parquet",
            use_stub=True,
            stub_frame_count=3,
            stub_width=128,
            stub_height=128,
            log_dir=tmp_path / "logs",
        )
        extract_cfg.input_videos_dir.mkdir(parents=True, exist_ok=True)
        extract_frames(extract_cfg)

        # Then run detection
        det_cfg = YoloTrackingConfig(
            frames_dir=extract_cfg.output_dir,
            output_detections_path=tmp_path / "detections.parquet",
            output_tracks_path=tmp_path / "tracks.parquet",
            use_stub=True,
            max_dets_per_frame=2,
            log_dir=tmp_path / "logs",
        )
        detections, tracks = run_yolo_tracking(det_cfg)

        assert len(detections) >= 1
        assert det_cfg.output_detections_path.exists()

    def test_segmentation_stub(self, tmp_path):
        """Test segmentation in stub mode."""
        from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
        from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking
        from anime_pipeline.segmentation.toonout_wrapper import SegmentConfig, segment_foreground_background

        # Setup frames
        extract_cfg = ExtractFramesConfig(
            input_videos_dir=tmp_path / "raw",
            output_dir=tmp_path / "frames",
            metadata_path=tmp_path / "frames.parquet",
            use_stub=True,
            stub_frame_count=2,
            stub_width=128,
            stub_height=128,
            log_dir=tmp_path / "logs",
        )
        extract_cfg.input_videos_dir.mkdir(parents=True, exist_ok=True)
        extract_frames(extract_cfg)

        # Setup detections
        det_cfg = YoloTrackingConfig(
            frames_dir=extract_cfg.output_dir,
            output_detections_path=tmp_path / "detections.parquet",
            output_tracks_path=tmp_path / "tracks.parquet",
            use_stub=True,
            max_dets_per_frame=1,
            log_dir=tmp_path / "logs",
        )
        run_yolo_tracking(det_cfg)

        # Run segmentation
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
        fg_records, bg_records = segment_foreground_background(seg_cfg)

        assert len(fg_records) >= 1
        assert len(bg_records) >= 1
        assert seg_cfg.output_fg_dir.exists()
        assert seg_cfg.output_bg_dir.exists()


class TestEndToEndPipeline:
    """Test complete pipeline execution in stub mode."""

    def test_full_pipeline_stub_mode(self, tmp_path):
        """Test complete pipeline from extraction to dataset preparation."""
        from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
        from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking
        from anime_pipeline.segmentation.toonout_wrapper import SegmentConfig, segment_foreground_background

        # Stage 1: Frame extraction
        extract_cfg = ExtractFramesConfig(
            input_videos_dir=tmp_path / "raw",
            output_dir=tmp_path / "frames",
            metadata_path=tmp_path / "frames.parquet",
            use_stub=True,
            stub_frame_count=5,
            stub_width=256,
            stub_height=256,
            log_dir=tmp_path / "logs",
        )
        extract_cfg.input_videos_dir.mkdir(parents=True, exist_ok=True)
        frame_records = extract_frames(extract_cfg)
        assert len(frame_records) == 5, "Should extract 5 stub frames"

        # Stage 2: Detection
        det_cfg = YoloTrackingConfig(
            frames_dir=extract_cfg.output_dir,
            output_detections_path=tmp_path / "detections.parquet",
            output_tracks_path=tmp_path / "tracks.parquet",
            use_stub=True,
            max_dets_per_frame=2,
            log_dir=tmp_path / "logs",
        )
        detections, tracks = run_yolo_tracking(det_cfg)
        assert len(detections) >= 1, "Should have at least 1 detection"

        # Stage 3: Segmentation
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
        fg_records, bg_records = segment_foreground_background(seg_cfg)
        assert len(fg_records) >= 1, "Should have foreground outputs"
        assert len(bg_records) >= 1, "Should have background outputs"

        # Verify output structure
        assert (tmp_path / "frames").exists()
        assert (tmp_path / "detections.parquet").exists()
        assert (tmp_path / "fg").exists()
        assert (tmp_path / "bg").exists()


class TestCheckpointAndResume:
    """Test checkpoint and resume functionality."""

    def test_stage_checkpoint_creation(self, tmp_path):
        """Test that stages create valid checkpoints."""
        from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        config = ExtractFramesConfig(
            input_videos_dir=tmp_path / "raw",
            output_dir=tmp_path / "frames",
            metadata_path=tmp_path / "frames.parquet",
            use_stub=True,
            stub_frame_count=3,
            stub_width=64,
            stub_height=64,
            log_dir=tmp_path / "logs",
            checkpoint_dir=checkpoint_dir,
        )
        config.input_videos_dir.mkdir(parents=True, exist_ok=True)

        extract_frames(config)

        # Check metadata was saved (acts as checkpoint)
        assert config.metadata_path.exists()


class TestOutputContracts:
    """Test that outputs meet expected contracts."""

    def test_segmentation_output_structure(self, tmp_path):
        """Test segmentation outputs follow expected structure."""
        from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
        from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking
        from anime_pipeline.segmentation.toonout_wrapper import SegmentConfig, segment_foreground_background

        # Setup pipeline
        extract_cfg = ExtractFramesConfig(
            input_videos_dir=tmp_path / "raw",
            output_dir=tmp_path / "frames",
            metadata_path=tmp_path / "frames.parquet",
            use_stub=True,
            stub_frame_count=2,
            stub_width=128,
            stub_height=128,
            log_dir=tmp_path / "logs",
        )
        extract_cfg.input_videos_dir.mkdir(parents=True, exist_ok=True)
        extract_frames(extract_cfg)

        det_cfg = YoloTrackingConfig(
            frames_dir=extract_cfg.output_dir,
            output_detections_path=tmp_path / "detections.parquet",
            output_tracks_path=tmp_path / "tracks.parquet",
            use_stub=True,
            max_dets_per_frame=1,
            log_dir=tmp_path / "logs",
        )
        run_yolo_tracking(det_cfg)

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
        fg_records, bg_records = segment_foreground_background(seg_cfg)

        # Verify foreground record structure
        for record in fg_records:
            assert 'rgba_path' in record
            assert 'mask_path' in record
            assert Path(record['rgba_path']).exists()

        # Verify background record structure
        for record in bg_records:
            assert 'with_holes_path' in record
            assert Path(record['with_holes_path']).exists()


class TestMetadataIntegrity:
    """Test metadata file integrity across stages."""

    def test_parquet_metadata_readable(self, tmp_path):
        """Test that parquet metadata files are readable."""
        from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
        import pandas as pd

        config = ExtractFramesConfig(
            input_videos_dir=tmp_path / "raw",
            output_dir=tmp_path / "frames",
            metadata_path=tmp_path / "frames.parquet",
            use_stub=True,
            stub_frame_count=3,
            stub_width=64,
            stub_height=64,
            log_dir=tmp_path / "logs",
        )
        config.input_videos_dir.mkdir(parents=True, exist_ok=True)
        extract_frames(config)

        # Read and validate parquet
        df = pd.read_parquet(config.metadata_path)

        assert len(df) == 3
        assert 'frame_path' in df.columns or 'path' in df.columns

    def test_metadata_cross_stage_consistency(self, tmp_path):
        """Test metadata consistency between stages."""
        from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
        from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking
        import pandas as pd

        # Extract frames
        extract_cfg = ExtractFramesConfig(
            input_videos_dir=tmp_path / "raw",
            output_dir=tmp_path / "frames",
            metadata_path=tmp_path / "frames.parquet",
            use_stub=True,
            stub_frame_count=3,
            stub_width=128,
            stub_height=128,
            log_dir=tmp_path / "logs",
        )
        extract_cfg.input_videos_dir.mkdir(parents=True, exist_ok=True)
        extract_frames(extract_cfg)

        # Run detection
        det_cfg = YoloTrackingConfig(
            frames_dir=extract_cfg.output_dir,
            output_detections_path=tmp_path / "detections.parquet",
            output_tracks_path=tmp_path / "tracks.parquet",
            use_stub=True,
            max_dets_per_frame=1,
            log_dir=tmp_path / "logs",
        )
        run_yolo_tracking(det_cfg)

        # Verify detection metadata references valid frames
        det_df = pd.read_parquet(det_cfg.output_detections_path)

        if 'frame_path' in det_df.columns:
            for frame_path in det_df['frame_path'].unique():
                assert Path(frame_path).exists(), f"Detection references non-existent frame: {frame_path}"
