#!/usr/bin/env python3
"""
Pipeline Orchestrator

Main coordinator for end-to-end pipeline execution.
Integrates ResourceMonitor and StageManager with unified configuration.

Author: Claude Code
Date: 2025-01-17
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from scripts.core.utils.config_loader import get_config
from .resource_monitor import ResourceMonitor
from .stage_manager import StageManager, PipelineStage, StageStatus


class PipelineOrchestrator:
    """
    Main pipeline coordinator.

    Orchestrates end-to-end processing from video to trained LoRA:
    1. Frame extraction
    2. Instance segmentation
    3. Character clustering
    4. Caption generation
    5. Dataset preparation
    6. LoRA training
    7. Evaluation

    Features:
    - Unified configuration management
    - Resource-aware execution
    - Progress tracking
    - Checkpoint/resume support
    - Error recovery
    """

    def __init__(self,
                 project: str,
                 character: Optional[str] = None,
                 config: Optional[Dict] = None,
                 device: str = "cuda",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize pipeline orchestrator.

        Args:
            project: Project name (e.g., 'luca')
            character: Character name (optional, for character-specific config)
            config: Configuration dict (if None, loads from project)
            device: Device for processing ('cuda' or 'cpu')
            logger: Logger instance
        """
        self.project = project
        self.character = character
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # Load configuration
        if config is None:
            self.logger.info(f"Loading configuration for project: {project}")
            self.config = get_config(project=project, character=character)
        else:
            self.config = config

        # Initialize components
        self.resource_monitor = ResourceMonitor(device=device, logger=self.logger)
        self.stage_manager = StageManager(
            project=project,
            config=self.config,
            logger=self.logger
        )

        # Pipeline state
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.checkpoint_path: Optional[Path] = None

        self.logger.info(f"Pipeline Orchestrator initialized for project: {project}")

    def setup_standard_pipeline(self, stages_enabled: Optional[Dict[str, bool]] = None) -> List[str]:
        """
        Setup standard pipeline stages based on configuration.

        Args:
            stages_enabled: Override stage enable/disable (None = use config)

        Returns:
            List of registered stage names
        """
        # Get pipeline stages from config
        pipeline_config = self.config.get('pipeline_stages', [])

        if not pipeline_config:
            self.logger.warning("No pipeline_stages in config, using default stages")
            pipeline_config = self._get_default_pipeline_stages()

        # Register each stage
        for stage_config in pipeline_config:
            stage_name = stage_config['name']

            # Check if stage is enabled
            enabled = stage_config.get('enabled', True)
            if stages_enabled and stage_name in stages_enabled:
                enabled = stages_enabled[stage_name]

            # Create stage based on name
            stage = self._create_pipeline_stage(stage_name, enabled)

            if stage:
                self.stage_manager.register_stage(stage)
                self.logger.debug(
                    f"Registered stage: {stage_name} "
                    f"({'enabled' if enabled else 'disabled'})"
                )

        # Compute execution order
        execution_order = self.stage_manager.compute_execution_order()

        self.logger.info(
            f"Pipeline setup complete: {len(execution_order)} stages registered"
        )

        return execution_order

    def _get_default_pipeline_stages(self) -> List[Dict]:
        """Get default pipeline stage definitions."""
        return [
            {'name': 'frame_extraction', 'enabled': True},
            {'name': 'instance_segmentation', 'enabled': True},
            {'name': 'identity_clustering', 'enabled': True},
            {'name': 'interactive_review', 'enabled': False},  # Manual step
            {'name': 'pose_subclustering', 'enabled': False},  # Optional
            {'name': 'caption_generation', 'enabled': True},
            {'name': 'dataset_preparation', 'enabled': True},
            {'name': 'lora_training', 'enabled': True},
        ]

    def _create_pipeline_stage(self, stage_name: str, enabled: bool = True) -> Optional[PipelineStage]:
        """
        Create a pipeline stage by name.

        Args:
            stage_name: Name of stage to create
            enabled: Whether stage is enabled

        Returns:
            PipelineStage object or None if unknown stage
        """
        # Import stage executors (placeholder - will be implemented with actual functions)
        from .stages import (
            execute_frame_extraction,
            execute_instance_segmentation,
            execute_identity_clustering,
            execute_interactive_review,
            execute_pose_subclustering,
            execute_caption_generation,
            execute_dataset_preparation,
            execute_lora_training
        )

        # Stage definitions
        stage_definitions = {
            'frame_extraction': {
                'description': 'Extract frames from video',
                'execute_fn': execute_frame_extraction,
                'dependencies': [],
                'config_key': 'frame_extraction'
            },
            'instance_segmentation': {
                'description': 'Segment character instances with SAM2',
                'execute_fn': execute_instance_segmentation,
                'dependencies': ['frame_extraction'],
                'config_key': 'segmentation'
            },
            'identity_clustering': {
                'description': 'Cluster instances by character identity',
                'execute_fn': execute_identity_clustering,
                'dependencies': ['instance_segmentation'],
                'config_key': 'clustering'
            },
            'interactive_review': {
                'description': 'Manual review and refinement of clusters',
                'execute_fn': execute_interactive_review,
                'dependencies': ['identity_clustering'],
                'optional': True
            },
            'pose_subclustering': {
                'description': 'Sub-cluster by pose and view angle',
                'execute_fn': execute_pose_subclustering,
                'dependencies': ['identity_clustering'],
                'optional': True,
                'config_key': 'pose_subclustering'
            },
            'caption_generation': {
                'description': 'Generate captions with VLM',
                'execute_fn': execute_caption_generation,
                'dependencies': ['identity_clustering'],
                'config_key': 'captioning'
            },
            'dataset_preparation': {
                'description': 'Prepare final training dataset',
                'execute_fn': execute_dataset_preparation,
                'dependencies': ['caption_generation']
            },
            'lora_training': {
                'description': 'Train LoRA adapter',
                'execute_fn': execute_lora_training,
                'dependencies': ['dataset_preparation'],
                'config_key': 'lora_training'
            }
        }

        if stage_name not in stage_definitions:
            self.logger.error(f"Unknown stage: {stage_name}")
            return None

        stage_def = stage_definitions[stage_name]

        # Create stage
        stage = PipelineStage(
            name=stage_name,
            description=stage_def['description'],
            execute_fn=stage_def['execute_fn'],
            dependencies=stage_def.get('dependencies', []),
            enabled=enabled,
            optional=stage_def.get('optional', False),
            config_key=stage_def.get('config_key')
        )

        # Add input/output paths from config
        stage = self._populate_stage_paths(stage)

        return stage

    def _populate_stage_paths(self, stage: PipelineStage) -> PipelineStage:
        """
        Populate stage input/output paths from configuration.

        Args:
            stage: Pipeline stage

        Returns:
            Stage with populated paths
        """
        paths = self.config.get('paths', {})

        # Map stage names to path keys
        path_mapping = {
            'frame_extraction': {
                'outputs': [paths.get('frames')]
            },
            'instance_segmentation': {
                'required_inputs': [paths.get('frames')],
                'outputs': [paths.get('instances')]
            },
            'identity_clustering': {
                'required_inputs': [paths.get('instances')],
                'outputs': [paths.get('clustered')]
            },
            'interactive_review': {
                'required_inputs': [paths.get('clustered')],
                'outputs': [paths.get('clustered_refined')]
            },
            'pose_subclustering': {
                'required_inputs': [paths.get('clustered')],
                'outputs': [paths.get('pose_subclusters')]
            },
            'caption_generation': {
                'required_inputs': [paths.get('clustered')],
                'outputs': [paths.get('clustered')]  # Adds captions in-place
            },
            'dataset_preparation': {
                'required_inputs': [paths.get('clustered')],
                'outputs': [paths.get('training_data')]
            },
            'lora_training': {
                'required_inputs': [paths.get('training_data')],
            }
        }

        if stage.name in path_mapping:
            stage_paths = path_mapping[stage.name]

            # Add required inputs
            if 'required_inputs' in stage_paths:
                stage.required_inputs = [
                    Path(p) for p in stage_paths['required_inputs'] if p
                ]

            # Add outputs
            if 'outputs' in stage_paths:
                stage.outputs = [
                    Path(p) for p in stage_paths['outputs'] if p
                ]

        return stage

    def run_full_pipeline(self,
                         start_from: Optional[str] = None,
                         stop_at: Optional[str] = None) -> bool:
        """
        Run complete pipeline from start to finish.

        Args:
            start_from: Stage name to start from (None = start from beginning)
            stop_at: Stage name to stop at (None = run to end)

        Returns:
            True if pipeline completed successfully
        """
        self.start_time = datetime.now()

        self.logger.info("=" * 80)
        self.logger.info(f"Starting full pipeline for project: {self.project}")
        self.logger.info("=" * 80)

        # Log initial resource state
        self.resource_monitor.log_current_stats()

        # Setup pipeline if not already done
        if not self.stage_manager.stages:
            self.setup_standard_pipeline()

        # Filter stages if start_from/stop_at specified
        if start_from or stop_at:
            execution_order = self._filter_execution_range(start_from, stop_at)
        else:
            execution_order = self.stage_manager.execution_order

        self.logger.info(f"Executing {len(execution_order)} stages")

        # Execute pipeline
        success = self.stage_manager.execute_all(skip_failed_dependencies=True)

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        # Final summary
        self.logger.info("=" * 80)
        if success:
            self.logger.info(f"✓ Pipeline completed successfully ({duration:.1f}s)")
        else:
            self.logger.error(f"✗ Pipeline failed ({duration:.1f}s)")
        self.logger.info("=" * 80)

        # Log final resource state
        self.resource_monitor.log_current_stats()

        return success

    def run_partial_pipeline(self, stages: List[str]) -> bool:
        """
        Run specific pipeline stages.

        Args:
            stages: List of stage names to execute

        Returns:
            True if all stages completed successfully
        """
        self.logger.info(f"Running partial pipeline: {', '.join(stages)}")

        # Setup pipeline if needed
        if not self.stage_manager.stages:
            self.setup_standard_pipeline()

        # Execute each stage
        success_count = 0
        for stage_name in stages:
            if stage_name not in self.stage_manager.stages:
                self.logger.error(f"Unknown stage: {stage_name}")
                continue

            success = self.stage_manager.execute_stage(stage_name)
            if success:
                success_count += 1

        return success_count == len(stages)

    def _filter_execution_range(self,
                                start_from: Optional[str],
                                stop_at: Optional[str]) -> List[str]:
        """
        Filter execution order to run only stages in range.

        Args:
            start_from: First stage to execute
            stop_at: Last stage to execute

        Returns:
            Filtered list of stage names
        """
        full_order = self.stage_manager.execution_order

        # Find indices
        start_idx = 0
        stop_idx = len(full_order)

        if start_from:
            try:
                start_idx = full_order.index(start_from)
            except ValueError:
                self.logger.warning(f"Stage '{start_from}' not found, starting from beginning")

        if stop_at:
            try:
                stop_idx = full_order.index(stop_at) + 1
            except ValueError:
                self.logger.warning(f"Stage '{stop_at}' not found, running to end")

        filtered = full_order[start_idx:stop_idx]

        self.logger.info(
            f"Filtered execution range: {filtered[0]} → {filtered[-1]} "
            f"({len(filtered)} stages)"
        )

        return filtered

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current pipeline progress.

        Returns:
            Dictionary with progress information
        """
        summary = self.stage_manager.get_pipeline_summary()

        # Add overall progress
        total_stages = len(self.stage_manager.stages)
        completed = summary['status_counts'].get('completed', 0)
        failed = summary['status_counts'].get('failed', 0)
        running = summary['status_counts'].get('running', 0)

        progress = {
            'project': self.project,
            'character': self.character,
            'total_stages': total_stages,
            'completed': completed,
            'failed': failed,
            'running': running,
            'progress_percent': (completed / total_stages * 100) if total_stages > 0 else 0,
            'stages': summary['stages'],
            'resource_stats': self.resource_monitor.get_current_stats().__dict__
        }

        if self.start_time:
            progress['start_time'] = self.start_time.isoformat()
            if self.end_time:
                progress['end_time'] = self.end_time.isoformat()
                progress['duration_seconds'] = (self.end_time - self.start_time).total_seconds()
            else:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                progress['elapsed_seconds'] = elapsed

        return progress

    def save_checkpoint(self, checkpoint_path: Optional[Path] = None):
        """
        Save pipeline checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint (None = auto-generate)
        """
        if checkpoint_path is None:
            checkpoint_dir = Path(self.config.get('paths', {}).get('base_dir', '.')) / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{self.project}_checkpoint.json"

        progress = self.get_progress()

        with open(checkpoint_path, 'w') as f:
            json.dump(progress, f, indent=2, default=str)

        self.checkpoint_path = checkpoint_path
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def resume_from_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Resume pipeline from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if successfully resumed
        """
        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False

        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        # Find last completed stage
        last_completed = None
        for stage_name, stage_info in checkpoint.get('stages', {}).items():
            if stage_info['status'] == 'completed':
                last_completed = stage_name

        if last_completed:
            self.logger.info(f"Resuming after stage: {last_completed}")
            # Find next stage in execution order
            execution_order = self.stage_manager.execution_order
            try:
                last_idx = execution_order.index(last_completed)
                if last_idx + 1 < len(execution_order):
                    next_stage = execution_order[last_idx + 1]
                    return self.run_full_pipeline(start_from=next_stage)
                else:
                    self.logger.info("All stages already completed")
                    return True
            except ValueError:
                self.logger.error(f"Stage '{last_completed}' not in current pipeline")
                return False
        else:
            self.logger.info("No completed stages found, starting from beginning")
            return self.run_full_pipeline()


def main():
    """Test pipeline orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline Orchestrator")
    parser.add_argument('--project', required=True, help='Project name (e.g., luca)')
    parser.add_argument('--character', help='Character name (optional)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--start-from', help='Stage to start from')
    parser.add_argument('--stop-at', help='Stage to stop at')
    parser.add_argument('--stages', help='Comma-separated list of stages to run')
    parser.add_argument('--dry-run', action='store_true', help='Show pipeline without executing')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        project=args.project,
        character=args.character,
        device=args.device
    )

    # Setup pipeline
    orchestrator.setup_standard_pipeline()

    if args.dry_run:
        # Just show pipeline
        progress = orchestrator.get_progress()
        print("\n=== Pipeline Configuration ===\n")
        print(json.dumps(progress, indent=2, default=str))
        return 0

    # Run pipeline
    if args.stages:
        # Run specific stages
        stages = [s.strip() for s in args.stages.split(',')]
        success = orchestrator.run_partial_pipeline(stages)
    else:
        # Run full pipeline
        success = orchestrator.run_full_pipeline(
            start_from=args.start_from,
            stop_at=args.stop_at
        )

    # Save final summary
    output_dir = Path('outputs') / args.project
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'pipeline_summary.json'

    progress = orchestrator.get_progress()
    with open(summary_path, 'w') as f:
        json.dump(progress, f, indent=2, default=str)

    print(f"\nPipeline summary saved to: {summary_path}")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
