from pathlib import Path

from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking
from anime_pipeline.segmentation.toonout_wrapper import SegmentConfig, segment_foreground_background
from anime_pipeline.pose.dwpose_wrapper import PoseExtractConfig, extract_poses
from anime_pipeline.inference.lora_controlnet import InferenceConfig, run_inference
from anime_pipeline.animation.animatediff_runner import AnimationConfig, generate_animation


def test_inference_and_animation_stub(tmp_path):
    # Frames
    extract_cfg = ExtractFramesConfig(
        input_videos_dir=tmp_path / "raw",
        output_dir=tmp_path / "frames",
        metadata_path=tmp_path / "frames.parquet",
        use_stub=True,
        stub_frame_count=2,
        stub_width=32,
        stub_height=32,
        log_dir=tmp_path / "logs",
    )
    extract_cfg.input_videos_dir.mkdir(parents=True, exist_ok=True)
    extract_frames(extract_cfg)

    # Detections
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
    segment_foreground_background(seg_cfg)

    # Pose
    pose_cfg = PoseExtractConfig(
        frames_dir=extract_cfg.output_dir,
        detections_path=det_cfg.output_detections_path,
        output_pose_dir=tmp_path / "poses",
        output_metadata_path=tmp_path / "poses.parquet",
        use_stub=True,
        log_dir=tmp_path / "logs",
    )
    extract_poses(pose_cfg)

    # Inference
    infer_cfg = InferenceConfig(
        pose_metadata_path=pose_cfg.output_metadata_path,
        output_dir=tmp_path / "inference",
        output_metadata_path=tmp_path / "inference/metadata.parquet",
        num_samples=2,
        image_size=64,
        use_stub=True,
        log_dir=tmp_path / "logs",
    )
    samples = run_inference(infer_cfg)
    assert len(samples) == 2
    for s in samples:
        assert Path(s["output_image_path"]).exists()

    # Animation
    anim_cfg = AnimationConfig(
        pose_metadata_path=pose_cfg.output_metadata_path,
        output_dir=tmp_path / "animation",
        output_metadata_path=tmp_path / "animation/metadata.parquet",
        num_frames=3,
        frame_size=64,
        use_stub=True,
        log_dir=tmp_path / "logs",
    )
    frames = generate_animation(anim_cfg)
    assert len(frames) == 3
    for f in frames:
        assert Path(f["frame_path"]).exists()
