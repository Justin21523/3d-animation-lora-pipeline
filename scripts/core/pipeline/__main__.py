#!/usr/bin/env python3
"""
Pipeline CLI Entry Point

Provides command-line interface for pipeline operations.

Usage:
    python -m scripts.core.pipeline run --project luca
    python -m scripts.core.pipeline status --project luca
    python -m scripts.core.pipeline resume --checkpoint path/to/checkpoint.json

Author: Claude Code
Date: 2025-01-17
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List

from .orchestrator import PipelineOrchestrator
from ..utils.config_loader import get_config


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_run(args):
    """Execute pipeline run command"""
    logger = logging.getLogger(__name__)

    logger.info(f"Starting pipeline for project: {args.project}")

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(
        project=args.project,
        character=args.character,
        device=args.device
    )

    # Setup pipeline
    if args.stages:
        # Run specific stages
        stages = [s.strip() for s in args.stages.split(',')]
        logger.info(f"Running stages: {', '.join(stages)}")
        success = orchestrator.run_partial_pipeline(stages)
    else:
        # Run full pipeline
        logger.info("Running full pipeline")
        orchestrator.setup_standard_pipeline()
        success = orchestrator.run_full_pipeline(
            start_from=args.start_from,
            stop_at=args.stop_at
        )

    # Save checkpoint if requested
    if args.save_checkpoint:
        checkpoint_path = args.save_checkpoint
        orchestrator.save_checkpoint(checkpoint_path)
        logger.info(f"Checkpoint saved to: {checkpoint_path}")

    # Exit with appropriate code
    if success:
        logger.info("✓ Pipeline completed successfully")
        return 0
    else:
        logger.error("✗ Pipeline failed")
        return 1


def cmd_resume(args):
    """Resume pipeline from checkpoint"""
    logger = logging.getLogger(__name__)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return 1

    logger.info(f"Resuming pipeline from checkpoint: {checkpoint_path}")

    # Load checkpoint to get project info
    import json
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)

    project = checkpoint_data.get('project')
    if not project:
        logger.error("Checkpoint does not contain project information")
        return 1

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(
        project=project,
        device=args.device
    )

    # Resume from checkpoint
    success = orchestrator.resume_from_checkpoint(checkpoint_path)

    if success:
        logger.info("✓ Pipeline resumed and completed successfully")
        return 0
    else:
        logger.error("✗ Pipeline resume failed")
        return 1


def cmd_status(args):
    """Show pipeline status"""
    logger = logging.getLogger(__name__)

    logger.info(f"Checking pipeline status for project: {args.project}")

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(
        project=args.project,
        character=args.character,
        device='cpu'  # No GPU needed for status check
    )

    # Setup pipeline to get stage definitions
    orchestrator.setup_standard_pipeline()

    # Get progress
    progress = orchestrator.get_progress()

    # Display status
    print("\n" + "="*60)
    print(f"Pipeline Status: {args.project}")
    print("="*60)
    print(f"Progress: {progress['progress_percent']:.1f}%")
    print(f"Completed: {progress['completed_stages']}/{progress['total_stages']} stages")
    print()

    # Show stage details
    print("Stage Status:")
    print("-"*60)

    for stage_name in orchestrator.stage_manager.execution_order:
        stage = orchestrator.stage_manager.stages[stage_name]
        status_icon = {
            'pending': '⏳',
            'running': '🔄',
            'completed': '✓',
            'failed': '✗',
            'skipped': '⊘'
        }.get(stage.status.value, '?')

        print(f"{status_icon} {stage_name:25s} {stage.status.value:10s}")

        if stage.status.value == 'failed' and stage.error_message:
            print(f"   Error: {stage.error_message}")

    print("="*60)
    print()

    return 0


def cmd_list_stages(args):
    """List available pipeline stages"""
    logger = logging.getLogger(__name__)

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(
        project=args.project or 'default',
        device='cpu'
    )

    # Setup pipeline
    orchestrator.setup_standard_pipeline()

    print("\n" + "="*60)
    print("Available Pipeline Stages")
    print("="*60)
    print()

    for stage_name in orchestrator.stage_manager.execution_order:
        stage = orchestrator.stage_manager.stages[stage_name]

        enabled = "✓" if stage.enabled else "✗"
        optional = "(optional)" if stage.optional else ""

        print(f"{enabled} {stage_name:25s} {optional}")
        print(f"   {stage.description}")

        if stage.dependencies:
            print(f"   Dependencies: {', '.join(stage.dependencies)}")
        print()

    print("="*60)
    print()

    return 0


def cmd_validate(args):
    """Validate pipeline configuration"""
    logger = logging.getLogger(__name__)

    logger.info(f"Validating configuration for project: {args.project}")

    try:
        # Load config
        config = get_config(project=args.project, character=args.character)

        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(
            project=args.project,
            character=args.character,
            config=config,
            device='cpu'
        )

        # Setup pipeline
        orchestrator.setup_standard_pipeline()

        # Validate paths
        paths = config.get('paths', {})
        missing_paths = []

        for key, path in paths.items():
            if path and not Path(path).exists():
                missing_paths.append(f"{key}: {path}")

        if missing_paths:
            logger.warning("Missing paths:")
            for missing in missing_paths:
                logger.warning(f"  - {missing}")
        else:
            logger.info("✓ All configured paths exist")

        logger.info("✓ Configuration is valid")
        return 0

    except Exception as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='3D Animation LoRA Pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python -m scripts.core.pipeline run --project luca

  # Run specific stages
  python -m scripts.core.pipeline run --project luca --stages segmentation,clustering

  # Resume from checkpoint
  python -m scripts.core.pipeline resume --checkpoint outputs/luca/checkpoint.json

  # Check status
  python -m scripts.core.pipeline status --project luca

  # List available stages
  python -m scripts.core.pipeline list-stages --project luca

  # Validate configuration
  python -m scripts.core.pipeline validate --project luca
        """
    )

    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run pipeline')
    run_parser.add_argument('--project', required=True,
                           help='Project name (e.g., luca)')
    run_parser.add_argument('--character', default=None,
                           help='Character name (optional, defaults to project name)')
    run_parser.add_argument('--stages', default=None,
                           help='Comma-separated list of stages to run')
    run_parser.add_argument('--start-from', default=None,
                           help='Start from specific stage')
    run_parser.add_argument('--stop-at', default=None,
                           help='Stop at specific stage')
    run_parser.add_argument('--device', default='cuda',
                           choices=['cuda', 'cpu'],
                           help='Device to use (cuda or cpu)')
    run_parser.add_argument('--save-checkpoint', default=None,
                           help='Save checkpoint to specified path')
    run_parser.set_defaults(func=cmd_run)

    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume from checkpoint')
    resume_parser.add_argument('--checkpoint', required=True,
                              help='Path to checkpoint file')
    resume_parser.add_argument('--device', default='cuda',
                              choices=['cuda', 'cpu'],
                              help='Device to use (cuda or cpu)')
    resume_parser.set_defaults(func=cmd_resume)

    # Status command
    status_parser = subparsers.add_parser('status', help='Show pipeline status')
    status_parser.add_argument('--project', required=True,
                              help='Project name')
    status_parser.add_argument('--character', default=None,
                              help='Character name (optional)')
    status_parser.set_defaults(func=cmd_status)

    # List stages command
    list_parser = subparsers.add_parser('list-stages', help='List available stages')
    list_parser.add_argument('--project', default=None,
                            help='Project name (optional)')
    list_parser.set_defaults(func=cmd_list_stages)

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--project', required=True,
                                help='Project name')
    validate_parser.add_argument('--character', default=None,
                                help='Character name (optional)')
    validate_parser.set_defaults(func=cmd_validate)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    if not args.command:
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
