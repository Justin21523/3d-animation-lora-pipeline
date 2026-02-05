#!/usr/bin/env python3
"""
2D Animation Pipeline Stage Definitions

Defines all pipeline stages and their execution functions.

Stages:
1. frame_extraction - Extract frames from video
2. yolo_tracking - YOLO detection + multi-object tracking
3. toonout_segmentation - ToonOut per-track segmentation
4. dwpose_extraction - DWpose keypoint extraction (optional)
5. identity_clustering - Cluster instances by character identity
6. dataset_building - Build LoRA training dataset
7. lora_training - Train LoRA adapter

Author: Created for 2D pipeline
Date: 2025-01-XX
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional


def execute_frame_extraction(
    project: str,
    config: Dict,
    stage_config: Optional[Dict],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Stage: Extract frames from video.

    Args:
        project: Project name
        config: Full pipeline config
        stage_config: Stage-specific config
        logger: Logger instance

    Returns:
        Result dict with success status
    """
    from anime_pipeline.frames.sampling import extract_frames, ExtractFramesConfig

    logger.info("Extracting frames from video...")

    # Build config from stage_config
    if stage_config:
        cfg = ExtractFramesConfig(**stage_config)
    else:
        # Use defaults from config
        cfg = ExtractFramesConfig(
            input_path=config.get('input_video'),
            output_dir=config.get('frames_dir'),
            mode=config.get('extraction_mode', 'scene'),
            fps=config.get('fps', None),
        )

    # Execute extraction
    try:
        frames_extracted = extract_frames(cfg, logger)
        return {
            'success': True,
            'frames_extracted': frames_extracted
        }
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        return {'success': False, 'error': str(e)}


def execute_yolo_tracking(
    project: str,
    config: Dict,
    stage_config: Optional[Dict],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Stage: YOLO detection + multi-object tracking.

    Args:
        project: Project name
        config: Full pipeline config
        stage_config: Stage-specific config
        logger: Logger instance

    Returns:
        Result dict with success status
    """
    from anime_pipeline.detection.yolo_detector import run_yolo_tracking

    logger.info("Running YOLO detection and tracking...")

    # This will be implemented later when we integrate YOLO tracking
    # For now, return a stub
    logger.warning("YOLO tracking stage not yet implemented (stub)")

    return {
        'success': True,
        'num_tracks': 0,
        'stub': True
    }


def execute_toonout_segmentation(
    project: str,
    config: Dict,
    stage_config: Optional[Dict],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Stage: ToonOut per-track segmentation.

    Args:
        project: Project name
        config: Full pipeline config
        stage_config: Stage-specific config
        logger: Logger instance

    Returns:
        Result dict with success status
    """
    from anime_pipeline.segmentation.toonout_wrapper import segment_foreground_background

    logger.info("Running ToonOut segmentation...")

    # This will be implemented later when we adapt ToonOut for per-track processing
    # For now, return a stub
    logger.warning("ToonOut segmentation stage not yet implemented (stub)")

    return {
        'success': True,
        'instances_segmented': 0,
        'stub': True
    }


def execute_dwpose_extraction(
    project: str,
    config: Dict,
    stage_config: Optional[Dict],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Stage: DWpose keypoint extraction (optional for ControlNet).

    Args:
        project: Project name
        config: Full pipeline config
        stage_config: Stage-specific config
        logger: Logger instance

    Returns:
        Result dict with success status
    """
    from anime_pipeline.pose.dwpose_wrapper import extract_poses

    logger.info("Extracting DWpose keypoints...")

    # This is optional, used for ControlNet pose conditioning
    logger.info("DWpose extraction stage (optional for ControlNet)")

    return {
        'success': True,
        'poses_extracted': 0,
        'optional': True
    }


def execute_identity_clustering(
    project: str,
    config: Dict,
    stage_config: Optional[Dict],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Stage: Cluster instances by character identity.

    This will merge tracks of same character using face embeddings.

    Args:
        project: Project name
        config: Full pipeline config
        stage_config: Stage-specific config
        logger: Logger instance

    Returns:
        Result dict with success status
    """
    from anime_pipeline.clustering.identity_clustering import cluster_by_identity

    logger.info("Clustering by character identity...")

    # This will be implemented when we have track_segments from segmentation stage
    logger.warning("Identity clustering stage not yet fully integrated (pending track_segments)")

    return {
        'success': True,
        'num_characters': 0,
        'stub': True
    }


def execute_multi_character_extraction(
    project: str,
    config: Dict,
    stage_config: Optional[Dict],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Integrated multi-character extraction stage (Phase 3.4).

    Complete pipeline for multi-character handling:
    1. YOLO detection + tracking → track groups
    2. Per-track ToonOut segmentation → character instances
    3. Face-based identity clustering → merge tracks of same character

    This is the NEW unified stage that replaces separate yolo_tracking,
    toonout_segmentation, and identity_clustering stages for multi-character workflows.

    Args:
        project: Project name
        config: Full pipeline config
        stage_config: Stage-specific config
        logger: Logger instance

    Returns:
        Result dict with success status and statistics
    """
    from anime_pipeline.detection.yolo_detector import (
        run_yolo_tracking_with_grouping,
        YoloTrackingConfig
    )
    from anime_pipeline.segmentation.toonout_wrapper import (
        segment_foreground_background_per_track,
        SegmentConfig
    )
    from anime_pipeline.clustering.identity_clustering import cluster_by_identity

    logger.info("=" * 80)
    logger.info("MULTI-CHARACTER EXTRACTION PIPELINE")
    logger.info("=" * 80)

    # Step 1: YOLO + Tracking
    logger.info("\n[1/3] YOLO Detection + Multi-Object Tracking...")

    tracking_config_dict = stage_config.get("tracking", {}) if stage_config else {}
    track_cfg = YoloTrackingConfig(**tracking_config_dict)

    track_groups = run_yolo_tracking_with_grouping(track_cfg, logger)

    logger.info(f"✓ Found {len(track_groups)} valid tracks")

    # Step 2: Per-track segmentation
    logger.info("\n[2/3] Per-Track ToonOut Segmentation...")

    seg_config_dict = stage_config.get("segmentation", {}) if stage_config else {}
    seg_cfg = SegmentConfig(**seg_config_dict)

    track_segments = segment_foreground_background_per_track(seg_cfg, track_groups, logger)

    total_fg = sum(len(fg) for fg, _ in track_segments.values())
    total_bg = sum(len(bg) for _, bg in track_segments.values())

    logger.info(f"✓ Segmented {total_fg} foreground instances, {total_bg} backgrounds")

    # Step 3: Identity clustering (merge tracks of same character)
    logger.info("\n[3/3] Face-Based Identity Clustering...")

    cluster_config_dict = stage_config.get("clustering", {}) if stage_config else {}
    min_cluster_size = cluster_config_dict.get("min_cluster_size", 20)
    device = cluster_config_dict.get("device", "cuda")

    character_clusters = cluster_by_identity(
        track_segments,
        min_cluster_size=min_cluster_size,
        device=device,
        save_face_crops=True,
        output_dir=Path(config.get("paths", {}).get("clustered", "data/clustered"))
    )

    num_characters = len([k for k in character_clusters.keys() if k != "noise"])

    logger.info("=" * 80)
    logger.info("MULTI-CHARACTER EXTRACTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Tracks detected: {len(track_groups)}")
    logger.info(f"Characters identified: {num_characters}")
    logger.info(f"Total instances: {total_fg}")

    return {
        "success": True,
        "num_tracks": len(track_groups),
        "num_characters": num_characters,
        "total_instances": total_fg,
        "character_clusters": {k: len(v) for k, v in character_clusters.items()}
    }


def execute_dataset_building(
    project: str,
    config: Dict,
    stage_config: Optional[Dict],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Stage: Build LoRA training dataset.

    Organizes clustered data into training format with captions.

    Args:
        project: Project name
        config: Full pipeline config
        stage_config: Stage-specific config
        logger: Logger instance

    Returns:
        Result dict with success status
    """
    from anime_pipeline.datasets.character_dataset import build_character_dataset

    logger.info("Building LoRA training dataset...")

    # This will use existing dataset builders
    logger.warning("Dataset building stage not yet fully implemented (stub)")

    return {
        'success': True,
        'dataset_size': 0,
        'stub': True
    }


def execute_lora_training(
    project: str,
    config: Dict,
    stage_config: Optional[Dict],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Stage: Train LoRA adapter.

    Args:
        project: Project name
        config: Full pipeline config
        stage_config: Stage-specific config
        logger: Logger instance

    Returns:
        Result dict with success status
    """
    from anime_pipeline.training.lora_trainer_sd import train_lora

    logger.info("Training LoRA adapter...")

    # This will use existing LoRA training infrastructure
    logger.warning("LoRA training stage not yet fully implemented (stub)")

    return {
        'success': True,
        'checkpoints_saved': 0,
        'stub': True
    }


# Stage registry - maps stage name to execution function
STAGE_REGISTRY = {
    'frame_extraction': execute_frame_extraction,
    'yolo_tracking': execute_yolo_tracking,
    'toonout_segmentation': execute_toonout_segmentation,
    'dwpose_extraction': execute_dwpose_extraction,
    'identity_clustering': execute_identity_clustering,
    'dataset_building': execute_dataset_building,
    'lora_training': execute_lora_training,
    # NEW: Phase 3.4 - Integrated multi-character extraction
    'multi_character_extraction': execute_multi_character_extraction,
}


def get_stage_executor(stage_name: str):
    """
    Get executor function for a stage.

    Args:
        stage_name: Name of stage

    Returns:
        Executor function

    Raises:
        ValueError: If stage not found
    """
    if stage_name not in STAGE_REGISTRY:
        raise ValueError(
            f"Unknown stage: {stage_name}. "
            f"Available stages: {list(STAGE_REGISTRY.keys())}"
        )
    return STAGE_REGISTRY[stage_name]
