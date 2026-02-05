#!/usr/bin/env python3
"""
Unified Batch Processor for 2D Animation LoRA Pipeline.

Single entry point for all batch operations with:
- YAML-based configuration
- Stage dependencies and execution order
- Checkpoint/resume support
- Progress tracking and reporting
- Parallel execution where possible

AI_WAREHOUSE 3.0 compliant paths.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import subprocess

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of a stage execution."""
    stage_name: str
    status: StageStatus
    start_time: str
    end_time: str
    duration_seconds: float
    output_dir: Optional[str] = None
    metrics: Dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    name: str
    projects: List[Dict]
    stages: List[str]
    output_root: Path
    parallel_projects: bool = False
    max_parallel: int = 2
    checkpoint_dir: Optional[Path] = None
    resume: bool = False
    dry_run: bool = False
    device: str = "cuda"
    animation_mode: str = "2d"


@dataclass
class BatchResult:
    """Result of batch processing."""
    config_name: str
    total_projects: int
    completed_projects: int
    failed_projects: int
    total_stages: int
    project_results: Dict[str, List[StageResult]]
    start_time: str
    end_time: str
    total_duration: float

    def to_dict(self) -> Dict:
        return {
            "config_name": self.config_name,
            "total_projects": self.total_projects,
            "completed_projects": self.completed_projects,
            "failed_projects": self.failed_projects,
            "total_stages": self.total_stages,
            "project_results": {
                k: [asdict(r) for r in v]
                for k, v in self.project_results.items()
            },
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration": self.total_duration,
        }


class UnifiedBatchProcessor:
    """
    Unified batch processor for pipeline operations.

    Supports:
    - Multiple projects in sequence or parallel
    - Stage-based processing with dependencies
    - Checkpoint/resume functionality
    - Progress tracking and reporting
    """

    # Available stages and their dependencies
    STAGE_DEFINITIONS = {
        "extraction": {
            "description": "Extract frames from video",
            "dependencies": [],
            "gpu_required": False,
        },
        "tracking": {
            "description": "YOLO detection and tracking",
            "dependencies": ["extraction"],
            "gpu_required": True,
        },
        "segmentation": {
            "description": "Character segmentation",
            "dependencies": ["tracking"],
            "gpu_required": True,
        },
        "clustering": {
            "description": "Identity clustering",
            "dependencies": ["segmentation"],
            "gpu_required": True,
        },
        "dataset_prep": {
            "description": "Prepare training dataset",
            "dependencies": ["clustering"],
            "gpu_required": False,
        },
        "captioning": {
            "description": "Generate captions",
            "dependencies": ["dataset_prep"],
            "gpu_required": True,
        },
        "training": {
            "description": "Train LoRA",
            "dependencies": ["captioning"],
            "gpu_required": True,
        },
        "evaluation": {
            "description": "Evaluate checkpoints",
            "dependencies": ["training"],
            "gpu_required": True,
        },
    }

    # AI_WAREHOUSE 3.0 paths
    DEFAULT_OUTPUT_ROOT = Path("/mnt/data/training/lora")
    DEFAULT_CHECKPOINT_DIR = Path("/mnt/data/training/checkpoints")

    def __init__(
        self,
        num_workers: int = 4,
    ):
        self.num_workers = num_workers
        self._checkpoint_data: Dict = {}

    def load_config(self, config_path: Union[str, Path]) -> BatchConfig:
        """
        Load batch configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            BatchConfig object
        """
        config_path = Path(config_path)

        try:
            import yaml
            with open(config_path) as f:
                raw_config = yaml.safe_load(f)
        except ImportError:
            # Fallback to JSON
            with open(config_path) as f:
                raw_config = json.load(f)

        return BatchConfig(
            name=raw_config.get("name", config_path.stem),
            projects=raw_config.get("projects", []),
            stages=raw_config.get("stages", list(self.STAGE_DEFINITIONS.keys())),
            output_root=Path(raw_config.get("output_root", self.DEFAULT_OUTPUT_ROOT)),
            parallel_projects=raw_config.get("parallel_projects", False),
            max_parallel=raw_config.get("max_parallel", 2),
            checkpoint_dir=Path(raw_config.get("checkpoint_dir", self.DEFAULT_CHECKPOINT_DIR)),
            resume=raw_config.get("resume", False),
            dry_run=raw_config.get("dry_run", False),
            device=raw_config.get("device", "cuda"),
            animation_mode=raw_config.get("animation_mode", "2d"),
        )

    def process_batch(self, config: BatchConfig) -> BatchResult:
        """
        Process a batch of projects.

        Args:
            config: BatchConfig with all settings

        Returns:
            BatchResult with processing outcomes
        """
        start_time = datetime.now()
        logger.info(f"Starting batch processing: {config.name}")
        logger.info(f"Projects: {len(config.projects)}, Stages: {len(config.stages)}")

        # Load checkpoint if resuming
        if config.resume and config.checkpoint_dir:
            self._load_checkpoint(config)

        # Validate stages
        stages = self._validate_stages(config.stages)
        logger.info(f"Execution order: {' -> '.join(stages)}")

        project_results: Dict[str, List[StageResult]] = {}
        completed = 0
        failed = 0

        if config.parallel_projects and len(config.projects) > 1:
            # Process projects in parallel
            with ThreadPoolExecutor(max_workers=config.max_parallel) as executor:
                futures = {
                    executor.submit(
                        self._process_project,
                        project,
                        stages,
                        config,
                    ): project["name"]
                    for project in config.projects
                }

                for future in as_completed(futures):
                    project_name = futures[future]
                    try:
                        results = future.result()
                        project_results[project_name] = results

                        if all(r.status == StageStatus.COMPLETED for r in results):
                            completed += 1
                        else:
                            failed += 1

                    except Exception as e:
                        logger.error(f"Project {project_name} failed: {e}")
                        project_results[project_name] = [
                            StageResult(
                                stage_name="unknown",
                                status=StageStatus.FAILED,
                                start_time=datetime.now().isoformat(),
                                end_time=datetime.now().isoformat(),
                                duration_seconds=0,
                                error=str(e),
                            )
                        ]
                        failed += 1
        else:
            # Process projects sequentially
            for project in config.projects:
                project_name = project["name"]
                logger.info(f"\n{'='*60}\nProcessing project: {project_name}\n{'='*60}")

                try:
                    results = self._process_project(project, stages, config)
                    project_results[project_name] = results

                    if all(r.status == StageStatus.COMPLETED for r in results):
                        completed += 1
                    else:
                        failed += 1

                    # Save checkpoint after each project
                    self._save_checkpoint(config, project_results)

                except Exception as e:
                    logger.error(f"Project {project_name} failed: {e}")
                    project_results[project_name] = [
                        StageResult(
                            stage_name="unknown",
                            status=StageStatus.FAILED,
                            start_time=datetime.now().isoformat(),
                            end_time=datetime.now().isoformat(),
                            duration_seconds=0,
                            error=str(e),
                        )
                    ]
                    failed += 1

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        result = BatchResult(
            config_name=config.name,
            total_projects=len(config.projects),
            completed_projects=completed,
            failed_projects=failed,
            total_stages=len(stages),
            project_results=project_results,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration=total_duration,
        )

        # Save final report
        self._save_report(config, result)

        logger.info(f"\nBatch processing complete:")
        logger.info(f"  Completed: {completed}/{len(config.projects)}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Duration: {total_duration:.1f}s")

        return result

    def _process_project(
        self,
        project: Dict,
        stages: List[str],
        config: BatchConfig,
    ) -> List[StageResult]:
        """Process a single project through all stages."""
        project_name = project["name"]
        results = []

        # Check for already completed stages (resume)
        completed_stages = self._get_completed_stages(project_name)

        for stage in stages:
            # Skip if already completed
            if stage in completed_stages:
                logger.info(f"Skipping {stage} (already completed)")
                results.append(StageResult(
                    stage_name=stage,
                    status=StageStatus.SKIPPED,
                    start_time="",
                    end_time="",
                    duration_seconds=0,
                ))
                continue

            # Dry run mode
            if config.dry_run:
                logger.info(f"[DRY RUN] Would execute: {stage}")
                results.append(StageResult(
                    stage_name=stage,
                    status=StageStatus.SKIPPED,
                    start_time=datetime.now().isoformat(),
                    end_time=datetime.now().isoformat(),
                    duration_seconds=0,
                ))
                continue

            # Execute stage
            result = self._execute_stage(
                stage,
                project,
                config,
            )
            results.append(result)

            # Stop on failure
            if result.status == StageStatus.FAILED:
                logger.error(f"Stage {stage} failed, stopping project")
                break

        return results

    def _execute_stage(
        self,
        stage_name: str,
        project: Dict,
        config: BatchConfig,
    ) -> StageResult:
        """Execute a single stage."""
        start_time = datetime.now()
        logger.info(f"Executing stage: {stage_name}")

        try:
            # Get stage executor
            executor = self._get_stage_executor(stage_name)

            # Build stage config
            stage_config = self._build_stage_config(
                stage_name,
                project,
                config,
            )

            # Execute
            output_dir, metrics = executor(stage_config)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info(f"Stage {stage_name} completed in {duration:.1f}s")

            return StageResult(
                stage_name=stage_name,
                status=StageStatus.COMPLETED,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                output_dir=str(output_dir) if output_dir else None,
                metrics=metrics or {},
            )

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.error(f"Stage {stage_name} failed: {e}")

            return StageResult(
                stage_name=stage_name,
                status=StageStatus.FAILED,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                error=str(e),
            )

    def _get_stage_executor(self, stage_name: str) -> Callable:
        """Get the executor function for a stage."""
        executors = {
            "extraction": self._exec_extraction,
            "tracking": self._exec_tracking,
            "segmentation": self._exec_segmentation,
            "clustering": self._exec_clustering,
            "dataset_prep": self._exec_dataset_prep,
            "captioning": self._exec_captioning,
            "training": self._exec_training,
            "evaluation": self._exec_evaluation,
        }

        if stage_name not in executors:
            raise ValueError(f"Unknown stage: {stage_name}")

        return executors[stage_name]

    def _build_stage_config(
        self,
        stage_name: str,
        project: Dict,
        config: BatchConfig,
    ) -> Dict:
        """Build configuration for a stage."""
        project_name = project["name"]
        base_dir = Path(project.get("base_dir", f"/mnt/data/datasets/general/{project_name}"))

        return {
            "project_name": project_name,
            "base_dir": base_dir,
            "output_root": config.output_root,
            "device": config.device,
            "animation_mode": config.animation_mode,
            "video_path": project.get("video_path"),
            "characters": project.get("characters", []),
            "stage_overrides": project.get(stage_name, {}),
        }

    # Stage executors
    def _exec_extraction(self, cfg: Dict) -> Tuple[Path, Dict]:
        """Execute frame extraction stage."""
        output_dir = cfg["base_dir"] / "frames"
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = {"stage": "extraction", "status": "simulated"}

        # In real implementation, would call frame extractor
        logger.info(f"Extracting frames to {output_dir}")

        return output_dir, metrics

    def _exec_tracking(self, cfg: Dict) -> Tuple[Path, Dict]:
        """Execute YOLO tracking stage."""
        output_dir = cfg["base_dir"] / "detections"
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = {"stage": "tracking", "status": "simulated"}

        logger.info(f"Running YOLO tracking to {output_dir}")

        return output_dir, metrics

    def _exec_segmentation(self, cfg: Dict) -> Tuple[Path, Dict]:
        """Execute segmentation stage."""
        output_dir = cfg["base_dir"] / "segmented"
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = {"stage": "segmentation", "status": "simulated"}

        logger.info(f"Running segmentation to {output_dir}")

        return output_dir, metrics

    def _exec_clustering(self, cfg: Dict) -> Tuple[Path, Dict]:
        """Execute clustering stage."""
        output_dir = cfg["base_dir"] / "clustered"
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = {"stage": "clustering", "status": "simulated"}

        logger.info(f"Running clustering to {output_dir}")

        return output_dir, metrics

    def _exec_dataset_prep(self, cfg: Dict) -> Tuple[Path, Dict]:
        """Execute dataset preparation stage."""
        output_dir = cfg["output_root"] / cfg["project_name"]
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = {"stage": "dataset_prep", "status": "simulated"}

        logger.info(f"Preparing dataset to {output_dir}")

        return output_dir, metrics

    def _exec_captioning(self, cfg: Dict) -> Tuple[Path, Dict]:
        """Execute captioning stage."""
        output_dir = cfg["output_root"] / cfg["project_name"]

        metrics = {"stage": "captioning", "status": "simulated"}

        logger.info(f"Generating captions in {output_dir}")

        return output_dir, metrics

    def _exec_training(self, cfg: Dict) -> Tuple[Path, Dict]:
        """Execute training stage."""
        output_dir = Path("/mnt/c/ai_models/lora_sdxl") / cfg["project_name"]
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = {"stage": "training", "status": "simulated"}

        logger.info(f"Training LoRA to {output_dir}")

        return output_dir, metrics

    def _exec_evaluation(self, cfg: Dict) -> Tuple[Path, Dict]:
        """Execute evaluation stage."""
        output_dir = Path("/mnt/data/training/lora/evaluation") / cfg["project_name"]
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = {"stage": "evaluation", "status": "simulated"}

        logger.info(f"Evaluating checkpoints in {output_dir}")

        return output_dir, metrics

    def _validate_stages(self, stages: List[str]) -> List[str]:
        """Validate and order stages by dependencies."""
        # Check all stages exist
        for stage in stages:
            if stage not in self.STAGE_DEFINITIONS:
                raise ValueError(f"Unknown stage: {stage}")

        # Topological sort based on dependencies
        ordered = []
        visited = set()

        def visit(stage: str):
            if stage in visited:
                return
            visited.add(stage)

            deps = self.STAGE_DEFINITIONS[stage]["dependencies"]
            for dep in deps:
                if dep in stages:
                    visit(dep)

            if stage in stages:
                ordered.append(stage)

        for stage in stages:
            visit(stage)

        return ordered

    def _get_completed_stages(self, project_name: str) -> Set[str]:
        """Get completed stages for a project from checkpoint."""
        if project_name not in self._checkpoint_data:
            return set()

        completed = set()
        for result in self._checkpoint_data.get(project_name, []):
            if result.get("status") == "completed":
                completed.add(result.get("stage_name"))

        return completed

    def _load_checkpoint(self, config: BatchConfig) -> None:
        """Load checkpoint data."""
        checkpoint_path = config.checkpoint_dir / f"{config.name}_checkpoint.json"

        if checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    data = json.load(f)
                self._checkpoint_data = data.get("project_results", {})
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

    def _save_checkpoint(
        self,
        config: BatchConfig,
        project_results: Dict[str, List[StageResult]],
    ) -> None:
        """Save checkpoint data."""
        if not config.checkpoint_dir:
            return

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = config.checkpoint_dir / f"{config.name}_checkpoint.json"

        data = {
            "config_name": config.name,
            "timestamp": datetime.now().isoformat(),
            "project_results": {
                k: [asdict(r) for r in v]
                for k, v in project_results.items()
            },
        }

        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _save_report(self, config: BatchConfig, result: BatchResult) -> None:
        """Save final batch report."""
        report_dir = config.output_root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"{config.name}_{timestamp}_report.json"

        with open(report_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(f"Report saved to {report_path}")


def create_example_config(output_path: Path) -> None:
    """Create an example batch configuration file."""
    example_config = {
        "name": "2d_animation_batch",
        "description": "Example batch processing configuration",
        "output_root": "/mnt/data/training/lora/2d_characters",
        "checkpoint_dir": "/mnt/data/training/checkpoints",
        "device": "cuda",
        "animation_mode": "2d",
        "parallel_projects": False,
        "max_parallel": 2,
        "resume": False,
        "dry_run": False,
        "stages": [
            "extraction",
            "tracking",
            "segmentation",
            "clustering",
            "dataset_prep",
            "captioning",
            "training",
            "evaluation",
        ],
        "projects": [
            {
                "name": "simpsons",
                "base_dir": "/mnt/data/datasets/general/simpsons",
                "video_path": "/mnt/data/videos/simpsons_episode.mp4",
                "characters": [
                    {"name": "homer", "trigger": "homer_simpson"},
                    {"name": "bart", "trigger": "bart_simpson"},
                ],
            },
            {
                "name": "futurama",
                "base_dir": "/mnt/data/datasets/general/futurama",
                "video_path": "/mnt/data/videos/futurama_episode.mp4",
                "characters": [
                    {"name": "fry", "trigger": "philip_fry"},
                    {"name": "bender", "trigger": "bender_robot"},
                ],
            },
        ],
    }

    import yaml
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)

    print(f"Example config created: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified batch processor for 2D animation pipeline"
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to batch config YAML file"
    )
    parser.add_argument(
        "--create-example",
        type=Path,
        help="Create example config at specified path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--stages",
        help="Comma-separated list of stages to run"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.create_example:
        create_example_config(args.create_example)
        exit(0)

    if not args.config:
        parser.print_help()
        exit(1)

    processor = UnifiedBatchProcessor()
    config = processor.load_config(args.config)

    # Override config with CLI args
    if args.dry_run:
        config.dry_run = True
    if args.resume:
        config.resume = True
    if args.stages:
        config.stages = [s.strip() for s in args.stages.split(",")]

    result = processor.process_batch(config)

    print(f"\nBatch Processing Summary:")
    print(f"  Name: {result.config_name}")
    print(f"  Projects: {result.completed_projects}/{result.total_projects} completed")
    print(f"  Failed: {result.failed_projects}")
    print(f"  Duration: {result.total_duration:.1f}s")
