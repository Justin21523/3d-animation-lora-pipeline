#!/usr/bin/env python3
"""
Stage Manager for Pipeline Orchestrator

Manages individual pipeline stages with dependencies, validation, and error handling.

Author: Claude Code
Date: 2025-01-17
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime


class StageStatus(Enum):
    """Pipeline stage status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage:
    """
    Definition of a pipeline processing stage.

    Attributes:
        name: Stage identifier (e.g., 'frame_extraction')
        description: Human-readable description
        execute_fn: Function to execute this stage
        dependencies: List of stage names that must complete first
        required_inputs: Required input paths/files
        outputs: Expected output paths/files
        enabled: Whether stage is enabled
        optional: Whether stage can be skipped if inputs missing
        config_key: Key in config for stage-specific settings
    """
    name: str
    description: str
    execute_fn: Callable
    dependencies: List[str] = field(default_factory=list)
    required_inputs: List[Path] = field(default_factory=list)
    outputs: List[Path] = field(default_factory=list)
    enabled: bool = True
    optional: bool = False
    config_key: Optional[str] = None
    status: StageStatus = StageStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    result_metadata: Dict[str, Any] = field(default_factory=dict)


class StageManager:
    """
    Manages pipeline stages with dependency resolution and execution.

    Features:
    - Dependency-based execution order
    - Input/output validation
    - Progress tracking
    - Error handling and recovery
    - Stage result metadata
    """

    def __init__(self,
                 project: str,
                 config: Dict,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize stage manager.

        Args:
            project: Project name (e.g., 'luca')
            config: Pipeline configuration
            logger: Logger instance
        """
        self.project = project
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self.stages: Dict[str, PipelineStage] = {}
        self.execution_order: List[str] = []

    def register_stage(self, stage: PipelineStage):
        """
        Register a pipeline stage.

        Args:
            stage: PipelineStage to register
        """
        self.stages[stage.name] = stage
        self.logger.debug(f"Registered stage: {stage.name}")

    def compute_execution_order(self) -> List[str]:
        """
        Compute stage execution order based on dependencies.

        Returns:
            List of stage names in execution order

        Raises:
            ValueError: If circular dependencies detected
        """
        # Topological sort using Kahn's algorithm
        in_degree = {name: 0 for name in self.stages}
        adj_list = {name: [] for name in self.stages}

        # Build adjacency list and in-degrees
        for name, stage in self.stages.items():
            for dep in stage.dependencies:
                if dep not in self.stages:
                    raise ValueError(f"Stage '{name}' depends on unknown stage '{dep}'")
                adj_list[dep].append(name)
                in_degree[name] += 1

        # Find stages with no dependencies
        queue = [name for name, deg in in_degree.items() if deg == 0]
        execution_order = []

        while queue:
            current = queue.pop(0)
            execution_order.append(current)

            # Reduce in-degree for dependent stages
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for circular dependencies
        if len(execution_order) != len(self.stages):
            raise ValueError("Circular dependency detected in pipeline stages")

        self.execution_order = execution_order
        self.logger.info(f"Execution order: {' → '.join(execution_order)}")

        return execution_order

    def validate_stage_inputs(self, stage: PipelineStage) -> bool:
        """
        Validate that required inputs exist for a stage.

        Args:
            stage: Stage to validate

        Returns:
            True if all required inputs exist
        """
        missing_inputs = []

        for input_path in stage.required_inputs:
            if not Path(input_path).exists():
                missing_inputs.append(str(input_path))

        if missing_inputs:
            if stage.optional:
                self.logger.info(
                    f"Stage '{stage.name}' optional, missing inputs: "
                    f"{', '.join(missing_inputs)}"
                )
                return False
            else:
                self.logger.error(
                    f"Stage '{stage.name}' missing required inputs: "
                    f"{', '.join(missing_inputs)}"
                )
                return False

        return True

    def execute_stage(self, stage_name: str) -> bool:
        """
        Execute a single pipeline stage.

        Args:
            stage_name: Name of stage to execute

        Returns:
            True if stage completed successfully
        """
        stage = self.stages[stage_name]

        # Check if stage is enabled
        if not stage.enabled:
            self.logger.info(f"Stage '{stage_name}' is disabled, skipping")
            stage.status = StageStatus.SKIPPED
            return True

        # Validate inputs
        if not self.validate_stage_inputs(stage):
            if stage.optional:
                stage.status = StageStatus.SKIPPED
                return True
            else:
                stage.status = StageStatus.FAILED
                stage.error_message = "Required inputs missing"
                return False

        # Execute stage
        self.logger.info(f"Executing stage: {stage_name} - {stage.description}")
        stage.status = StageStatus.RUNNING
        stage.start_time = datetime.now()

        try:
            # Get stage-specific config if available
            stage_config = None
            if stage.config_key and stage.config_key in self.config:
                stage_config = self.config[stage.config_key]

            # Execute stage function
            result = stage.execute_fn(
                project=self.project,
                config=self.config,
                stage_config=stage_config,
                logger=self.logger
            )

            # Check result
            if isinstance(result, dict):
                stage.result_metadata = result
                success = result.get('success', True)
            elif isinstance(result, bool):
                success = result
            else:
                success = True

            if success:
                stage.status = StageStatus.COMPLETED
                stage.end_time = datetime.now()
                duration = (stage.end_time - stage.start_time).total_seconds()
                self.logger.info(
                    f"✓ Stage '{stage_name}' completed successfully "
                    f"({duration:.1f}s)"
                )
                return True
            else:
                raise RuntimeError(
                    f"Stage function returned failure: "
                    f"{stage.result_metadata.get('error', 'Unknown error')}"
                )

        except Exception as e:
            stage.status = StageStatus.FAILED
            stage.end_time = datetime.now()
            stage.error_message = str(e)

            self.logger.error(
                f"✗ Stage '{stage_name}' failed: {e}",
                exc_info=True
            )
            return False

    def execute_all(self, skip_failed_dependencies: bool = False) -> bool:
        """
        Execute all stages in dependency order.

        Args:
            skip_failed_dependencies: If True, skip stages whose dependencies failed

        Returns:
            True if all stages completed successfully
        """
        if not self.execution_order:
            self.compute_execution_order()

        success_count = 0
        failed_count = 0
        skipped_count = 0

        for stage_name in self.execution_order:
            stage = self.stages[stage_name]

            # Check dependencies
            dependencies_met = True
            for dep_name in stage.dependencies:
                dep_stage = self.stages[dep_name]
                if dep_stage.status != StageStatus.COMPLETED:
                    dependencies_met = False
                    if dep_stage.status == StageStatus.FAILED:
                        self.logger.warning(
                            f"Stage '{stage_name}' skipped: "
                            f"dependency '{dep_name}' failed"
                        )
                    break

            if not dependencies_met:
                if skip_failed_dependencies:
                    stage.status = StageStatus.SKIPPED
                    skipped_count += 1
                    continue
                else:
                    self.logger.error(
                        f"Cannot execute '{stage_name}': dependencies not met"
                    )
                    return False

            # Execute stage
            success = self.execute_stage(stage_name)

            if success:
                success_count += 1
            else:
                failed_count += 1
                if not skip_failed_dependencies:
                    self.logger.error("Pipeline execution stopped due to stage failure")
                    return False

        # Summary
        self.logger.info(
            f"\nPipeline execution summary: "
            f"{success_count} completed, {failed_count} failed, {skipped_count} skipped"
        )

        return failed_count == 0

    def get_stage_status(self, stage_name: str) -> StageStatus:
        """Get status of a stage."""
        return self.stages[stage_name].status

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get summary of pipeline execution.

        Returns:
            Dictionary with pipeline summary
        """
        summary = {
            'project': self.project,
            'total_stages': len(self.stages),
            'stages': {}
        }

        for name, stage in self.stages.items():
            stage_info = {
                'description': stage.description,
                'status': stage.status.value,
                'enabled': stage.enabled,
                'dependencies': stage.dependencies
            }

            if stage.start_time:
                stage_info['start_time'] = stage.start_time.isoformat()
            if stage.end_time:
                stage_info['end_time'] = stage.end_time.isoformat()
                duration = (stage.end_time - stage.start_time).total_seconds()
                stage_info['duration_seconds'] = duration
            if stage.error_message:
                stage_info['error'] = stage.error_message
            if stage.result_metadata:
                stage_info['metadata'] = stage.result_metadata

            summary['stages'][name] = stage_info

        # Count by status
        summary['status_counts'] = {
            status.value: sum(1 for s in self.stages.values() if s.status == status)
            for status in StageStatus
        }

        return summary

    def save_summary(self, output_path: Path):
        """
        Save pipeline summary to JSON file.

        Args:
            output_path: Path to save summary JSON
        """
        summary = self.get_pipeline_summary()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Pipeline summary saved to: {output_path}")

    def reset_stages(self):
        """Reset all stages to pending status."""
        for stage in self.stages.values():
            stage.status = StageStatus.PENDING
            stage.start_time = None
            stage.end_time = None
            stage.error_message = None
            stage.result_metadata = {}

        self.logger.info("All stages reset to pending status")


def main():
    """Test stage manager."""
    import time

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Mock config
    config = {'project': 'test'}

    # Create stage manager
    manager = StageManager(project='test', config=config, logger=logger)

    # Define mock stages
    def mock_stage_1(**kwargs):
        logger = kwargs['logger']
        logger.info("Executing stage 1...")
        time.sleep(0.5)
        return {'success': True, 'items_processed': 100}

    def mock_stage_2(**kwargs):
        logger = kwargs['logger']
        logger.info("Executing stage 2...")
        time.sleep(0.3)
        return True

    def mock_stage_3(**kwargs):
        logger = kwargs['logger']
        logger.info("Executing stage 3...")
        time.sleep(0.2)
        return {'success': True}

    # Register stages
    manager.register_stage(PipelineStage(
        name='stage1',
        description='First processing stage',
        execute_fn=mock_stage_1
    ))

    manager.register_stage(PipelineStage(
        name='stage2',
        description='Second processing stage',
        execute_fn=mock_stage_2,
        dependencies=['stage1']
    ))

    manager.register_stage(PipelineStage(
        name='stage3',
        description='Third processing stage',
        execute_fn=mock_stage_3,
        dependencies=['stage1', 'stage2']
    ))

    # Execute pipeline
    print("\n=== Testing Pipeline Execution ===\n")
    success = manager.execute_all()

    # Print summary
    print("\n=== Pipeline Summary ===\n")
    summary = manager.get_pipeline_summary()
    print(json.dumps(summary, indent=2, default=str))

    # Test execution order
    print("\n=== Execution Order ===")
    print(" → ".join(manager.execution_order))

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
