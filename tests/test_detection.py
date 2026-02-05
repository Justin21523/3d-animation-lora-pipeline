from pathlib import Path

from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking


def test_run_yolo_tracking_stub(tmp_path):
    # Prepare stub frames
    extract_cfg = ExtractFramesConfig(
        input_videos_dir=tmp_path / "raw",
        output_dir=tmp_path / "frames",
        metadata_path=tmp_path / "frames.parquet",
        use_stub=True,
        stub_frame_count=2,
        stub_width=64,
        stub_height=64,
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
    detections, tracks = run_yolo_tracking(det_cfg)

    assert len(detections) == 2  # 2 frames * 1 det per frame
    assert len(tracks) == 1
    for det in detections:
        assert Path(det["image_path"]).exists()
        assert det["bbox_x2"] > det["bbox_x1"]
        assert det["bbox_y2"] > det["bbox_y1"]
