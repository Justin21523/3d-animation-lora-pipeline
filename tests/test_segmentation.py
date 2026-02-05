from pathlib import Path

from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking
from anime_pipeline.segmentation.toonout_wrapper import SegmentConfig, segment_foreground_background


def test_segment_fg_bg_stub(tmp_path):
    # Stub frames
    extract_cfg = ExtractFramesConfig(
        input_videos_dir=tmp_path / "raw",
        output_dir=tmp_path / "frames",
        metadata_path=tmp_path / "frames.parquet",
        use_stub=True,
        stub_frame_count=1,
        stub_width=64,
        stub_height=64,
        log_dir=tmp_path / "logs",
    )
    extract_cfg.input_videos_dir.mkdir(parents=True, exist_ok=True)
    extract_frames(extract_cfg)

    # Stub detections
    det_cfg = YoloTrackingConfig(
        frames_dir=extract_cfg.output_dir,
        output_detections_path=tmp_path / "detections.parquet",
        output_tracks_path=tmp_path / "tracks.parquet",
        use_stub=True,
        max_dets_per_frame=1,
        log_dir=tmp_path / "logs",
    )
    run_yolo_tracking(det_cfg)

    # Segmentation
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

    assert len(fg_records) == 1
    assert len(bg_records) == 1
    assert Path(fg_records[0]["rgba_path"]).exists()
    assert Path(bg_records[0]["with_holes_path"]).exists()
