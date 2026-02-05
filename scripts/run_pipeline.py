#!/usr/bin/env python3
"""
Main CLI Entry Point for 2D Animation LoRA Pipeline

Provides user-friendly interface for running the complete pipeline orchestrator.
Supports full pipeline execution, partial execution, checkpoint/resume, and dry-run mode.

Usage Examples:
    # Run full pipeline for a 2D animation project
    python scripts/run_pipeline.py --project simpsons --character homer --mode 2d

    # Run specific stages only
    python scripts/run_pipeline.py --project simpsons --stages yolo_tracking,toonout_segmentation

    # Resume from a checkpoint
    python scripts/run_pipeline.py --project simpsons --start-from identity_clustering

    # Dry-run to see pipeline configuration
    python scripts/run_pipeline.py --project simpsons --dry-run

    # Run with 3D parameter conversion (for porting 3D projects)
    python scripts/run_pipeline.py --project luca --mode 3d

Author: Ported from 3D pipeline for 2D animation workflow
Date: 2025-01-XX
"""

import argparse
import sys
import json
import logging
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from anime_pipeline.core.orchestrator import PipelineOrchestrator


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="2D Animation LoRA Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  %(prog)s --project simpsons --character homer

  # Run specific stages
  %(prog)s --project simpsons --stages yolo_tracking,identity_clustering

  # Resume from checkpoint
  %(prog)s --project simpsons --start-from dataset_building

  # Dry-run to preview configuration
  %(prog)s --project familyguy --dry-run

  # Use 3D parameter conversion
  %(prog)s --project luca --mode 3d --character alberto
        """
    )

    # Required arguments
    parser.add_argument(
        '--project',
        required=True,
        help='Project name (e.g., simpsons, familyguy, ricknmorty)'
    )

    # Optional arguments
    parser.add_argument(
        '--character',
        help='Character name for character-specific configuration'
    )

    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for processing (default: cuda)'
    )

    parser.add_argument(
        '--mode',
        default='2d',
        choices=['2d', '3d'],
        help='Animation style - automatically adjusts parameters (default: 2d)'
    )

    parser.add_argument(
        '--start-from',
        help='Stage name to start from (for resuming or partial execution)'
    )

    parser.add_argument(
        '--stop-at',
        help='Stage name to stop at (for partial execution)'
    )

    parser.add_argument(
        '--stages',
        help='Comma-separated list of specific stages to run (e.g., "yolo_tracking,toonout_segmentation")'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show pipeline configuration without executing'
    )

    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Custom output directory for pipeline results (default: outputs/<project>)'
    )

    return parser.parse_args()


def print_pipeline_summary(progress: dict):
    """
    Pretty-print pipeline summary.

    Args:
        progress: Progress dictionary from orchestrator
    """
    print("\n" + "=" * 80)
    print("PIPELINE CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\nProject: {progress['project']}")
    if progress.get('character'):
        print(f"Character: {progress['character']}")
    print(f"Animation Mode: {progress['animation_mode']}")
    print(f"Total Stages: {progress['total_stages']}")
    print(f"Progress: {progress['progress_percent']:.1f}%")

    print("\n" + "-" * 80)
    print("STAGE STATUS")
    print("-" * 80)

    for stage_name, stage_info in progress['stages'].items():
        status = stage_info['status']
        enabled = stage_info['enabled']

        # Status icon
        if status == 'completed':
            icon = '✓'
        elif status == 'failed':
            icon = '✗'
        elif status == 'running':
            icon = '▶'
        elif status == 'skipped':
            icon = '⊘'
        else:  # pending
            icon = '○'

        # Enabled/disabled indicator
        state = "" if enabled else " [DISABLED]"

        print(f"{icon} {stage_name:25} {status:10}{state}")

        if stage_info.get('dependencies'):
            print(f"  └─ Dependencies: {', '.join(stage_info['dependencies'])}")

        if stage_info.get('duration_seconds'):
            duration = stage_info['duration_seconds']
            print(f"  └─ Duration: {duration:.1f}s")

        if stage_info.get('error'):
            print(f"  └─ Error: {stage_info['error']}")

    print("\n" + "=" * 80)

    # Resource stats
    if progress.get('resource_stats'):
        print("\nRESOURCE STATISTICS")
        print("-" * 80)
        stats = progress['resource_stats']
        if stats.get('gpu_available'):
            print(f"GPU Memory: {stats.get('gpu_memory_used_mb', 0):.0f} MB used / "
                  f"{stats.get('gpu_memory_total_mb', 0):.0f} MB total")
        print(f"CPU Usage: {stats.get('cpu_percent', 0):.1f}%")
        print(f"RAM: {stats.get('memory_used_gb', 0):.1f} GB used / "
              f"{stats.get('memory_total_gb', 0):.1f} GB total")
        print("=" * 80)


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info(f"Starting 2D Animation Pipeline for project: {args.project}")

    # Create orchestrator
    try:
        orchestrator = PipelineOrchestrator(
            project=args.project,
            character=args.character,
            device=args.device,
            animation_mode=args.mode,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline orchestrator: {e}", exc_info=True)
        return 1

    # Setup pipeline stages
    try:
        orchestrator.setup_standard_pipeline()
    except Exception as e:
        logger.error(f"Failed to setup pipeline stages: {e}", exc_info=True)
        return 1

    # Dry-run mode: show configuration and exit
    if args.dry_run:
        logger.info("Dry-run mode: displaying configuration without execution")
        progress = orchestrator.get_progress()
        print_pipeline_summary(progress)

        # Save to file
        output_dir = args.output_dir or Path('outputs') / args.project
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / 'pipeline_config.json'

        with open(config_path, 'w') as f:
            json.dump(progress, f, indent=2, default=str)

        print(f"\nConfiguration saved to: {config_path}")
        return 0

    # Execute pipeline
    success = False

    try:
        if args.stages:
            # Run specific stages
            stages_list = [s.strip() for s in args.stages.split(',')]
            logger.info(f"Running specific stages: {', '.join(stages_list)}")
            success = orchestrator.run_partial_pipeline(stages_list)
        else:
            # Run full pipeline (with optional start/stop points)
            logger.info("Running full pipeline")
            success = orchestrator.run_full_pipeline(
                start_from=args.start_from,
                stop_at=args.stop_at
            )
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        success = False
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        success = False

    # Save final summary
    output_dir = args.output_dir or Path('outputs') / args.project
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'pipeline_summary.json'

    progress = orchestrator.get_progress()

    try:
        with open(summary_path, 'w') as f:
            json.dump(progress, f, indent=2, default=str)

        logger.info(f"Pipeline summary saved to: {summary_path}")
    except Exception as e:
        logger.error(f"Failed to save pipeline summary: {e}")

    # Print final summary
    print_pipeline_summary(progress)

    # Return exit code
    if success:
        logger.info("Pipeline completed successfully")
        return 0
    else:
        logger.error("Pipeline completed with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
