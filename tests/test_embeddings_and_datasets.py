from pathlib import Path

from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking
from anime_pipeline.segmentation.toonout_wrapper import SegmentConfig, segment_foreground_background
from anime_pipeline.pose.dwpose_wrapper import PoseExtractConfig, extract_poses
from anime_pipeline.embeddings.builders import EmbeddingConfig, build_embeddings
from anime_pipeline.datasets.character_builder import CharacterDatasetConfig, build_character_dataset
from anime_pipeline.datasets.controlnet_pose_builder import ControlNetPoseDatasetConfig, build_controlnet_pose_dataset


def test_embeddings_and_datasets_stub(tmp_path):
    # Prepare frames
    extract_cfg = ExtractFramesConfig(
        input_videos_dir=tmp_path / "raw",
        output_dir=tmp_path / "frames",
        metadata_path=tmp_path / "frames.parquet",
        use_stub=True,
        stub_frame_count=1,
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

    # Embeddings
    emb_cfg = EmbeddingConfig(
        target_type="frame",
        input_metadata_path=extract_cfg.metadata_path,
        output_dir=tmp_path / "embeddings",
        output_metadata_path=tmp_path / "embeddings.parquet",
        dim=8,
        use_stub=True,
        log_dir=tmp_path / "logs",
    )
    embeddings = build_embeddings(emb_cfg)
    assert len(embeddings) == 1
    assert Path(embeddings[0]["vector_path"]).exists()

    # Character dataset
    char_cfg = CharacterDatasetConfig(
        fg_metadata_path=seg_cfg.output_fg_metadata_path,
        output_dir=tmp_path / "lora_chars",
        output_metadata_path=tmp_path / "lora_chars/metadata.parquet",
        use_stub=True,
        log_dir=tmp_path / "logs",
    )
    chars = build_character_dataset(char_cfg)
    assert len(chars) == 1
    assert Path(chars[0]["image_path"]).exists()

    # ControlNet pose dataset
    cn_cfg = ControlNetPoseDatasetConfig(
        poses_metadata_path=pose_cfg.output_metadata_path,
        output_dir=tmp_path / "cn_pose",
        output_metadata_path=tmp_path / "cn_pose/metadata.parquet",
        use_stub=True,
        log_dir=tmp_path / "logs",
    )
    cn_samples = build_controlnet_pose_dataset(cn_cfg)
    assert len(cn_samples) == 1
