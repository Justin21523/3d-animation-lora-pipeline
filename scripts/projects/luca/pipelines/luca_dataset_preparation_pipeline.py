#!/usr/bin/env python3
"""
Dataset Preparation Pipeline (Project-Agnostic)
================================================

Configuration-driven pipeline from SAM2 multi-instance results to training-ready dataset.
Works with any character/project by specifying --project-config parameter.

Pipeline Stages:
1. Face-based pre-filtering (ArcFace vs reference faces)
2. 3D quality filtering (sharpness, completeness, diversity)
3. Comprehensive augmentation (3D-safe transforms)
4. Diversity analysis & 400-image auto-selection (RTM-Pose + multi-modal metrics)
5. Caption generation (Qwen2-VL for both datasets)
6. Training data preparation (Kohya_ss format)
7. Interactive review tool setup

Usage:
  # Default (Luca):
  python luca_dataset_preparation_pipeline.py --config configs/projects/luca_dataset_prep_v2.yaml

  # Different project:
  python luca_dataset_preparation_pipeline.py \\
    --config configs/projects/alberto_dataset_prep.yaml \\
    --project-config configs/projects/alberto.yaml

Author: Claude Code
Date: 2025-11-13
Version: 3.0.0 (Configuration-driven, project-agnostic)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Core imports
from scripts.core.utils.logger import setup_logger
from scripts.core.utils.config_loader import load_config
from scripts.core.pipeline.base_pipeline import BasePipeline

# Stage implementations
from scripts.pipelines.stages.face_prefilter import FacePrefilterStage
from scripts.pipelines.stages.quality_filter import QualityFilterStage
from scripts.pipelines.stages.augmentation import AugmentationStage
from scripts.pipelines.stages.diversity_selection import DiversitySelectionStage
from scripts.pipelines.stages.captioning import CaptioningStage
from scripts.pipelines.stages.training_prep import TrainingPrepStage
from scripts.pipelines.stages.interactive_review import InteractiveReviewStage


class LucaDatasetPreparationPipeline(BasePipeline):
    """Complete pipeline for dataset preparation from SAM2 results (project-agnostic)."""

    def __init__(self, config_path: str, resume: bool = True, project_config_path: Optional[str] = None):
        """
        Initialize pipeline with configuration.

        Args:
            config_path: Path to pipeline configuration YAML
            resume: Whether to resume from checkpoint if available
            project_config_path: Path to project configuration (optional, defaults to luca)
        """
        super().__init__(config_path, resume)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load project config to get project name
        if project_config_path:
            with open(project_config_path, 'r') as f:
                project_config = yaml.safe_load(f)
                self.project_name = project_config.get('project', {}).get('name', 'luca')
        else:
            # Default to luca for backward compatibility
            self.project_name = 'luca'

        # Setup logging
        log_dir = Path(self.config['logging']['output_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"

        self.logger = setup_logger(
            name=f"{self.project_name.title()}DatasetPipeline",
            log_file=str(log_file),
            level=self.config['logging']['level']
        )

        # Initialize stage instances
        self.stages = []
        self._initialize_stages()

        # Pipeline state
        self.start_time = None
        self.stage_times = {}

        self.logger.info("="*80)
        self.logger.info(f"{self.project_name.title()} Dataset Preparation Pipeline v3.0")
        self.logger.info("="*80)
        self.logger.info(f"Configuration: {config_path}")
        self.logger.info(f"Project: {self.project_name}")
        self.logger.info(f"Resume mode: {resume}")

    def _initialize_stages(self):
        """Initialize all pipeline stages based on configuration."""

        # Stage 1: Face-based pre-filtering
        if self.config['face_prefilter']['enabled']:
            self.stages.append(FacePrefilterStage(
                config=self.config['face_prefilter'],
                input_dir=self.config['input']['sam2_results'],
                reference_dir=self.config['input']['reference_faces'],
                logger=self.logger
            ))

        # Stage 2: Identity clustering (optional)
        if self.config.get('identity_clustering', {}).get('enabled', False):
            from scripts.pipelines.stages.identity_clustering import IdentityClusteringStage
            self.stages.append(IdentityClusteringStage(
                config=self.config['identity_clustering'],
                logger=self.logger
            ))

        # Stage 3: Quality filtering
        if self.config['quality_filtering']['enabled']:
            self.stages.append(QualityFilterStage(
                config=self.config['quality_filtering'],
                logger=self.logger
            ))

        # Stage 4: Comprehensive augmentation
        if self.config['augmentation']['enabled']:
            self.stages.append(AugmentationStage(
                config=self.config['augmentation'],
                logger=self.logger
            ))

        # Stage 5: Diversity selection (400-image auto-selection)
        if self.config['diversity_selection']['enabled']:
            self.stages.append(DiversitySelectionStage(
                config=self.config['diversity_selection'],
                logger=self.logger
            ))

        # Stage 6: Caption generation
        if self.config['captioning']['enabled']:
            self.stages.append(CaptioningStage(
                config=self.config['captioning'],
                logger=self.logger
            ))

        # Stage 7: Training data preparation
        if self.config['training_prep']['enabled']:
            self.stages.append(TrainingPrepStage(
                config=self.config['training_prep'],
                logger=self.logger
            ))

        # Stage 8: Interactive review tool setup
        if self.config['interactive_review']['enabled']:
            self.stages.append(InteractiveReviewStage(
                config=self.config['interactive_review'],
                logger=self.logger
            ))

        self.logger.info(f"Initialized {len(self.stages)} pipeline stages")

    def run(self):
        """Execute complete pipeline end-to-end."""

        self.start_time = datetime.now()
        self.logger.info(f"Pipeline started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("-"*80)

        # Execute stages sequentially
        for idx, stage in enumerate(self.stages, 1):
            stage_name = stage.__class__.__name__
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"STAGE {idx}/{len(self.stages)}: {stage_name}")
            self.logger.info(f"{'='*80}\n")

            stage_start = datetime.now()

            try:
                # Execute stage
                result = stage.execute()

                # Record timing
                stage_duration = (datetime.now() - stage_start).total_seconds()
                self.stage_times[stage_name] = stage_duration

                # Log results
                self.logger.info(f"\n✓ {stage_name} completed in {stage_duration:.1f}s")
                self.logger.info(f"Result: {result.get('summary', 'Success')}")

                # Save stage report
                if self.config['logging']['save_stage_reports']:
                    self._save_stage_report(stage_name, result)

                # Save checkpoint
                if self.config['resume']['enabled']:
                    self._save_checkpoint(stage_name, result)

            except Exception as e:
                self.logger.error(f"✗ {stage_name} failed: {str(e)}", exc_info=True)

                # Handle error based on strategy
                if self.config['error_handling']['strategy'] == 'fail_fast':
                    raise
                elif self.config['error_handling']['strategy'] == 'skip_and_log':
                    self.logger.warning(f"Skipping {stage_name} and continuing...")
                    continue

        # Pipeline completed
        total_duration = (datetime.now() - self.start_time).total_seconds()

        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("="*80)
        self.logger.info(f"Total duration: {total_duration/3600:.2f} hours")
        self.logger.info("\nStage timings:")
        for stage_name, duration in self.stage_times.items():
            self.logger.info(f"  {stage_name}: {duration/60:.1f} minutes")

        # Generate final summary
        self._generate_final_summary()

        return {
            'success': True,
            'total_duration': total_duration,
            'stage_times': self.stage_times
        }

    def _save_stage_report(self, stage_name: str, result: Dict):
        """Save detailed stage report to JSON."""
        report_dir = Path(self.config['logging']['output_dir']) / 'stage_reports'
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"{stage_name}_{timestamp}.json"

        with open(report_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        self.logger.debug(f"Stage report saved: {report_file}")

    def _save_checkpoint(self, stage_name: str, result: Dict):
        """Save pipeline checkpoint for resume capability."""
        checkpoint_file = Path(self.config['resume']['checkpoint_file'])
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'last_completed_stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'result': result
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        self.logger.debug(f"Checkpoint saved: {checkpoint_file}")

    def _generate_final_summary(self):
        """Generate comprehensive pipeline summary report."""
        summary_file = Path(self.config['logging']['output_dir']) / 'pipeline_summary.json'

        summary = {
            'pipeline': self.config['pipeline_name'],
            'version': self.config['version'],
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'stages_completed': len(self.stage_times),
            'stage_timings': self.stage_times,
            'configuration': self.config
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"\nFinal summary saved: {summary_file}")


def main():
    """Main entry point for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Dataset Preparation Pipeline (Project-Agnostic) - SAM2 to Training Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (default: Luca)
  python scripts/pipelines/luca_dataset_preparation_pipeline.py \\
    --config configs/projects/luca_dataset_prep_v2.yaml

  # Different project
  python scripts/pipelines/luca_dataset_preparation_pipeline.py \\
    --config configs/projects/alberto_dataset_prep.yaml \\
    --project-config configs/projects/alberto.yaml

  # Resume from checkpoint
  python scripts/pipelines/luca_dataset_preparation_pipeline.py \\
    --config configs/projects/luca_dataset_prep_v2.yaml \\
    --resume

  # Dry run (check configuration)
  python scripts/pipelines/luca_dataset_preparation_pipeline.py \\
    --config configs/projects/luca_dataset_prep_v2.yaml \\
    --dry-run
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to pipeline configuration YAML file'
    )

    parser.add_argument(
        '--project-config',
        type=str,
        default='configs/projects/luca.yaml',
        help='Path to project configuration (defines project name, paths, characters)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint if available'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without executing pipeline'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Validate configuration file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    # Dry run mode
    if args.dry_run:
        print("Validating configuration...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration valid")
        print(f"  Pipeline: {config['pipeline_name']} v{config['version']}")
        print(f"  Enabled stages: {sum(1 for k, v in config.items() if isinstance(v, dict) and v.get('enabled', False))}")
        sys.exit(0)

    # Initialize and run pipeline
    try:
        pipeline = LucaDatasetPreparationPipeline(
            config_path=str(config_path),
            resume=args.resume,
            project_config_path=args.project_config
        )

        result = pipeline.run()

        print("\n" + "="*80)
        print("✓ Pipeline completed successfully!")
        print(f"  Duration: {result['total_duration']/3600:.2f} hours")
        print(f"  Check logs: {pipeline.config['logging']['output_dir']}")
        print("="*80)

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
