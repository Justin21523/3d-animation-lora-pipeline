from pathlib import Path

from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking
from anime_pipeline.pose.dwpose_wrapper import PoseExtractConfig, extract_poses


def test_extract_pose_stub(tmp_path):
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

    det_cfg = YoloTrackingConfig(
        frames_dir=extract_cfg.output_dir,
        output_detections_path=tmp_path / "detections.parquet",
        output_tracks_path=tmp_path / "tracks.parquet",
        use_stub=True,
        max_dets_per_frame=1,
        log_dir=tmp_path / "logs",
    )
    run_yolo_tracking(det_cfg)

    pose_cfg = PoseExtractConfig(
        frames_dir=extract_cfg.output_dir,
        detections_path=det_cfg.output_detections_path,
        output_pose_dir=tmp_path / "poses",
        output_metadata_path=tmp_path / "poses.parquet",
        pose_type="dwpose",
        use_stub=True,
        log_dir=tmp_path / "logs",
    )
    poses = extract_poses(pose_cfg)

    assert len(poses) == 1
    assert Path(poses[0]["pose_image_path"]).exists()
    keypoints = poses[0]["keypoints"]
    assert isinstance(keypoints, str)
