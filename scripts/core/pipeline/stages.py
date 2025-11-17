#!/usr/bin/env python3
"""
Pipeline Stage Executors

Wrapper functions for each pipeline stage.
These functions interface with existing scripts in scripts/generic/.

Author: Claude Code
Date: 2025-01-17
"""

import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys


def execute_frame_extraction(**kwargs) -> Dict[str, Any]:
    """
    Execute frame extraction stage.

    Args:
        **kwargs: project, config, stage_config, logger

    Returns:
        Result dictionary with success status and metadata
    """
    logger = kwargs.get('logger', logging.getLogger(__name__))
    config = kwargs['config']
    stage_config = kwargs.get('stage_config', {})
    project = kwargs.get('project')

    logger.info("Executing frame extraction...")

    # Get paths from config
    paths = config.get('paths', {})
    video_input = paths.get('video_input')
    frames_output = paths.get('frames')

    if not video_input:
        return {
            'success': False,
            'stage': 'frame_extraction',
            'error': 'No video_input path specified in config'
        }

    # Get extraction config
    extraction_cfg = stage_config or config.get('frame_extraction', {})
    mode = extraction_cfg.get('mode', 'scene')
    scene_threshold = extraction_cfg.get('scene_threshold', 0.3)
    min_scene_length = extraction_cfg.get('min_scene_length', 15)
    quality = extraction_cfg.get('quality', 'high')

    # Build command
    script_path = Path(__file__).parent.parent.parent / 'generic' / 'video' / 'universal_frame_extractor.py'

    cmd = [
        sys.executable,
        str(script_path),
        '--input', str(video_input),
        '--output', str(frames_output),
        '--mode', mode,
        '--scene-threshold', str(scene_threshold),
        '--min-scene-length', str(min_scene_length),
        '--quality', quality
    ]

    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Count extracted frames
        frames_dir = Path(frames_output)
        if frames_dir.exists():
            frames_count = len(list(frames_dir.glob('*.jpg'))) + len(list(frames_dir.glob('*.png')))
        else:
            frames_count = 0

        logger.info(f"Frame extraction completed: {frames_count} frames extracted")

        return {
            'success': True,
            'stage': 'frame_extraction',
            'message': 'Frame extraction completed',
            'frames_extracted': frames_count,
            'output_dir': str(frames_output)
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Frame extraction failed: {e.stderr}")
        return {
            'success': False,
            'stage': 'frame_extraction',
            'error': str(e),
            'stderr': e.stderr
        }


def execute_instance_segmentation(**kwargs) -> Dict[str, Any]:
    """
    Execute instance segmentation stage.

    Args:
        **kwargs: project, config, stage_config, logger

    Returns:
        Result dictionary
    """
    logger = kwargs.get('logger', logging.getLogger(__name__))
    config = kwargs['config']
    stage_config = kwargs.get('stage_config', {})

    logger.info("Executing instance segmentation...")

    # Get paths from config
    paths = config.get('paths', {})
    frames_dir = paths.get('frames')
    instances_output = paths.get('instances')

    if not frames_dir or not Path(frames_dir).exists():
        return {
            'success': False,
            'stage': 'instance_segmentation',
            'error': f'Frames directory not found: {frames_dir}'
        }

    # Get segmentation config
    seg_cfg = stage_config or config.get('segmentation', {})
    model_type = seg_cfg.get('model_type', 'sam2_hiera_large')
    min_size = seg_cfg.get('min_instance_size', 4096)
    device = seg_cfg.get('device', config.get('hardware', {}).get('gpu', {}).get('device', 'cuda'))

    # Build command
    script_path = Path(__file__).parent.parent.parent / 'generic' / 'segmentation' / 'instance_segmentation.py'

    cmd = [
        sys.executable,
        str(script_path),
        '--input-dir', str(frames_dir),
        '--output-dir', str(instances_output),
        '--model-type', model_type,
        '--device', device,
        '--min-instance-size', str(min_size),
        '--save-masks',
        '--save-backgrounds',
        '--context-mode', 'transparent'
    ]

    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Count extracted instances
        instances_dir = Path(instances_output) / 'instances'
        if instances_dir.exists():
            instances_count = len(list(instances_dir.glob('*.png')))
        else:
            instances_count = 0

        # Try to load metadata
        metadata_file = Path(instances_output) / 'metadata.json'
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        logger.info(f"Instance segmentation completed: {instances_count} instances extracted")

        return {
            'success': True,
            'stage': 'instance_segmentation',
            'message': 'Instance segmentation completed',
            'instances_extracted': instances_count,
            'output_dir': str(instances_output),
            'metadata': metadata
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Instance segmentation failed: {e.stderr}")
        return {
            'success': False,
            'stage': 'instance_segmentation',
            'error': str(e),
            'stderr': e.stderr
        }


def execute_identity_clustering(**kwargs) -> Dict[str, Any]:
    """
    Execute identity clustering stage.

    Args:
        **kwargs: project, config, stage_config, logger

    Returns:
        Result dictionary
    """
    logger = kwargs.get('logger', logging.getLogger(__name__))
    config = kwargs['config']
    stage_config = kwargs.get('stage_config', {})

    logger.info("Executing identity clustering...")

    # Get paths from config
    paths = config.get('paths', {})
    instances_dir = paths.get('instances')
    clustered_output = paths.get('clustered')

    if not instances_dir or not Path(instances_dir).exists():
        return {
            'success': False,
            'stage': 'identity_clustering',
            'error': f'Instances directory not found: {instances_dir}'
        }

    # Get clustering config
    cluster_cfg = stage_config or config.get('clustering', {})
    min_cluster_size = cluster_cfg.get('min_cluster_size', 12)
    min_samples = cluster_cfg.get('min_samples', 2)
    device = cluster_cfg.get('device', config.get('hardware', {}).get('gpu', {}).get('device', 'cuda'))

    # Use face-centric identity clustering (best for multi-character scenes)
    script_path = Path(__file__).parent.parent.parent / 'generic' / 'clustering' / 'face_identity_clustering.py'

    # Build instances path (handle both direct instances/ and instances_sampled/)
    instances_path = Path(instances_dir)
    if (instances_path / 'instances').exists():
        instances_path = instances_path / 'instances'

    cmd = [
        sys.executable,
        str(script_path),
        '--input-dir', str(instances_path),
        '--output-dir', str(clustered_output),
        '--min-cluster-size', str(min_cluster_size),
        '--min-samples', str(min_samples),
        '--device', device,
        '--visualize'
    ]

    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Count clusters
        clustered_path = Path(clustered_output)
        if clustered_path.exists():
            clusters = [d for d in clustered_path.iterdir() if d.is_dir() and d.name.startswith('character_')]
            clusters_found = len(clusters)
        else:
            clusters_found = 0

        # Try to load cluster report
        report_file = clustered_path / 'cluster_report.json'
        metadata = {}
        if report_file.exists():
            with open(report_file, 'r') as f:
                metadata = json.load(f)

        logger.info(f"Identity clustering completed: {clusters_found} character clusters found")

        return {
            'success': True,
            'stage': 'identity_clustering',
            'message': 'Identity clustering completed',
            'clusters_found': clusters_found,
            'output_dir': str(clustered_output),
            'metadata': metadata
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Identity clustering failed: {e.stderr}")
        return {
            'success': False,
            'stage': 'identity_clustering',
            'error': str(e),
            'stderr': e.stderr
        }


def execute_interactive_review(**kwargs) -> Dict[str, Any]:
    """
    Execute interactive review stage.

    Args:
        **kwargs: project, config, stage_config, logger

    Returns:
        Result dictionary
    """
    logger = kwargs.get('logger', logging.getLogger(__name__))
    config = kwargs['config']

    logger.info("Interactive review stage - manual intervention required")
    logger.info("Please launch the interactive review UI manually:")

    paths = config.get('paths', {})
    clustered_dir = paths.get('clustered')
    clustered_refined = paths.get('clustered_refined')

    logger.info(f"  python scripts/generic/clustering/launch_interactive_review.py \\")
    logger.info(f"    --cluster-dir {clustered_dir} \\")
    logger.info(f"    --output-dir {clustered_refined}")

    # This stage must be completed manually
    return {
        'success': True,
        'stage': 'interactive_review',
        'message': 'Interactive review step (manual intervention required)',
        'requires_manual_completion': True,
        'input_dir': str(clustered_dir),
        'output_dir': str(clustered_refined)
    }


def execute_pose_subclustering(**kwargs) -> Dict[str, Any]:
    """
    Execute pose subclustering stage.

    Args:
        **kwargs: project, config, stage_config, logger

    Returns:
        Result dictionary
    """
    logger = kwargs.get('logger', logging.getLogger(__name__))
    config = kwargs['config']
    stage_config = kwargs.get('stage_config', {})

    logger.info("Executing pose subclustering...")

    # Get paths from config
    paths = config.get('paths', {})
    clustered_dir = paths.get('clustered_refined') or paths.get('clustered')
    pose_output = paths.get('pose_subclusters')

    if not clustered_dir or not Path(clustered_dir).exists():
        return {
            'success': False,
            'stage': 'pose_subclustering',
            'error': f'Clustered directory not found: {clustered_dir}'
        }

    # Get pose config
    pose_cfg = stage_config or config.get('pose_subclustering', {})
    min_cluster_size = pose_cfg.get('min_cluster_size', 5)
    pose_model = pose_cfg.get('pose_model', 'rtmpose-m')
    device = pose_cfg.get('device', config.get('hardware', {}).get('gpu', {}).get('device', 'cuda'))

    script_path = Path(__file__).parent.parent.parent / 'generic' / 'clustering' / 'pose_subclustering.py'

    cmd = [
        sys.executable,
        str(script_path),
        str(clustered_dir),
        '--output-dir', str(pose_output),
        '--pose-model', pose_model,
        '--device', device,
        '--min-cluster-size', str(min_cluster_size),
        '--method', 'umap_hdbscan',
        '--visualize'
    ]

    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Count subclusters
        pose_path = Path(pose_output)
        if pose_path.exists():
            subclusters = [d for d in pose_path.rglob('*') if d.is_dir() and 'pose_' in d.name]
            subclusters_count = len(subclusters)
        else:
            subclusters_count = 0

        logger.info(f"Pose subclustering completed: {subclusters_count} pose subclusters created")

        return {
            'success': True,
            'stage': 'pose_subclustering',
            'message': 'Pose subclustering completed',
            'subclusters_created': subclusters_count,
            'output_dir': str(pose_output)
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Pose subclustering failed: {e.stderr}")
        return {
            'success': False,
            'stage': 'pose_subclustering',
            'error': str(e),
            'stderr': e.stderr
        }


def execute_caption_generation(**kwargs) -> Dict[str, Any]:
    """
    Execute caption generation stage.

    Args:
        **kwargs: project, config, stage_config, logger

    Returns:
        Result dictionary
    """
    logger = kwargs.get('logger', logging.getLogger(__name__))
    config = kwargs['config']
    stage_config = kwargs.get('stage_config', {})

    logger.info("Executing caption generation...")

    # Get paths
    paths = config.get('paths', {})
    character_dirs = paths.get('pose_subclusters') or paths.get('clustered_refined') or paths.get('clustered')

    if not character_dirs or not Path(character_dirs).exists():
        return {
            'success': False,
            'stage': 'caption_generation',
            'error': f'Character directory not found: {character_dirs}'
        }

    # Get caption config
    caption_cfg = stage_config or config.get('captioning', {})
    model_name = caption_cfg.get('model', 'qwen2_vl')
    device = caption_cfg.get('device', config.get('hardware', {}).get('gpu', {}).get('device', 'cuda'))
    character_name = caption_cfg.get('character_name', project)
    caption_prefix = caption_cfg.get('prefix', 'a 3d animated character, pixar style')

    # Use qwen_caption_generator_robust.py script
    script_path = Path(__file__).parent.parent.parent / 'generic' / 'training' / 'qwen_caption_generator_robust.py'

    if not script_path.exists():
        logger.error(f"Caption generator script not found: {script_path}")
        return {
            'success': False,
            'stage': 'caption_generation',
            'error': f'Caption generator script not found: {script_path}'
        }

    # Build character directories list
    character_path = Path(character_dirs)
    if (character_path / 'character_0').exists():
        # Multiple characters - generate captions for all
        char_dirs = sorted([d for d in character_path.iterdir() if d.is_dir() and d.name.startswith('character_')])
    else:
        char_dirs = [character_path]

    total_captions = 0
    total_images = 0

    for char_dir in char_dirs:
        logger.info(f"Generating captions for {char_dir.name}...")

        # Count images
        images = list(char_dir.glob('*.jpg')) + list(char_dir.glob('*.png')) + list(char_dir.glob('*.jpeg'))
        total_images += len(images)

        if len(images) == 0:
            logger.warning(f"No images found in {char_dir}")
            continue

        cmd = [
            sys.executable,
            str(script_path),
            '--input-dir', str(char_dir),
            '--character-name', character_name,
            '--device', device,
            '--batch-size', '1'  # Safe default for VLM
        ]

        try:
            logger.info(f"Running caption generation for {len(images)} images...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                # Count generated captions
                captions = list(char_dir.glob('*.txt'))
                total_captions += len(captions)
                logger.info(f"Generated {len(captions)} captions for {char_dir.name}")
            else:
                logger.error(f"Caption generation failed: {result.stderr}")
                return {
                    'success': False,
                    'stage': 'caption_generation',
                    'error': result.stderr
                }

        except subprocess.TimeoutExpired:
            logger.error("Caption generation timed out (>1 hour)")
            return {
                'success': False,
                'stage': 'caption_generation',
                'error': 'Caption generation timeout'
            }
        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            return {
                'success': False,
                'stage': 'caption_generation',
                'error': str(e)
            }

    return {
        'success': True,
        'stage': 'caption_generation',
        'message': f'Generated {total_captions} captions for {total_images} images',
        'captions_generated': total_captions,
        'total_images': total_images,
        'model': model_name,
        'character_dirs_processed': len(char_dirs)
    }


def execute_dataset_preparation(**kwargs) -> Dict[str, Any]:
    """
    Execute dataset preparation stage.

    Args:
        **kwargs: project, config, stage_config, logger

    Returns:
        Result dictionary
    """
    logger = kwargs.get('logger', logging.getLogger(__name__))
    config = kwargs['config']
    stage_config = kwargs.get('stage_config', {})
    project = kwargs.get('project')

    logger.info("Executing dataset preparation...")

    # Get paths
    paths = config.get('paths', {})
    character_dirs = paths.get('pose_subclusters') or paths.get('clustered_refined') or paths.get('clustered')
    training_output = paths.get('training_data')

    if not character_dirs or not Path(character_dirs).exists():
        return {
            'success': False,
            'stage': 'dataset_preparation',
            'error': f'Character directory not found: {character_dirs}'
        }

    # Get dataset config
    dataset_cfg = stage_config or config.get('dataset_preparation', {})
    target_size = dataset_cfg.get('target_size', 400)
    character_name = dataset_cfg.get('character_name', project)

    script_path = Path(__file__).parent.parent.parent / 'generic' / 'training' / 'prepare_training_data.py'

    # Build character directories list
    character_path = Path(character_dirs)
    if (character_path / 'character_0').exists():
        # Multiple characters - prepare first one for now
        char_dir = character_path / 'character_0'
    else:
        char_dir = character_path

    cmd = [
        sys.executable,
        str(script_path),
        '--character-dirs', str(char_dir),
        '--output-dir', str(training_output),
        '--character-name', character_name,
        '--target-size', str(target_size)
    ]

    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Count dataset images
        training_path = Path(training_output)
        if (training_path / 'images').exists():
            dataset_size = len(list((training_path / 'images').glob('*.[pj][np]g')))
        else:
            dataset_size = 0

        logger.info(f"Dataset preparation completed: {dataset_size} images prepared")

        return {
            'success': True,
            'stage': 'dataset_preparation',
            'message': 'Dataset preparation completed',
            'dataset_size': dataset_size,
            'output_dir': str(training_output)
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Dataset preparation failed: {e.stderr}")
        return {
            'success': False,
            'stage': 'dataset_preparation',
            'error': str(e),
            'stderr': e.stderr
        }


def execute_lora_training(**kwargs) -> Dict[str, Any]:
    """
    Execute LoRA training stage.

    Args:
        **kwargs: project, config, stage_config, logger

    Returns:
        Result dictionary
    """
    logger = kwargs.get('logger', logging.getLogger(__name__))
    config = kwargs['config']
    stage_config = kwargs.get('stage_config', {})
    project = kwargs.get('project')

    logger.info("Executing LoRA training...")

    # Get paths
    paths = config.get('paths', {})
    training_data = paths.get('training_data')
    lora_output = paths.get('lora_output')

    if not training_data or not Path(training_data).exists():
        return {
            'success': False,
            'stage': 'lora_training',
            'error': f'Training data directory not found: {training_data}'
        }

    # Get training config
    training_cfg = stage_config or config.get('training', {})
    config_file = training_cfg.get('config_file')

    if not config_file:
        logger.error("No training config file specified")
        logger.info("Please create a training config file and specify it in the config")
        return {
            'success': False,
            'stage': 'lora_training',
            'error': 'No training config file specified'
        }

    # TODO: Integrate with Kohya_ss sd-scripts
    logger.warning("LoRA training integration pending")
    logger.info("For now, run training manually:")
    logger.info(f"  conda run -n kohya_ss accelerate launch sd-scripts/sdxl_train_network.py \\")
    logger.info(f"    --config_file {config_file}")

    return {
        'success': True,
        'stage': 'lora_training',
        'message': 'LoRA training placeholder (run manually)',
        'epochs_trained': 0,
        'note': 'Manual training required'
    }


# Additional utility functions for stage integration

def validate_stage_inputs(stage_name: str, config: Dict, logger: logging.Logger) -> bool:
    """
    Validate that required inputs exist for a stage.

    Args:
        stage_name: Name of stage
        config: Pipeline configuration
        logger: Logger instance

    Returns:
        True if inputs are valid
    """
    # TODO: Implement input validation logic
    return True


def get_stage_output_path(stage_name: str, config: Dict) -> Optional[Path]:
    """
    Get output path for a stage.

    Args:
        stage_name: Name of stage
        config: Pipeline configuration

    Returns:
        Output path or None
    """
    paths = config.get('paths', {})

    path_mapping = {
        'frame_extraction': paths.get('frames'),
        'instance_segmentation': paths.get('instances'),
        'identity_clustering': paths.get('clustered'),
        'interactive_review': paths.get('clustered_refined'),
        'pose_subclustering': paths.get('pose_subclusters'),
        'dataset_preparation': paths.get('training_data')
    }

    output_path = path_mapping.get(stage_name)
    return Path(output_path) if output_path else None
