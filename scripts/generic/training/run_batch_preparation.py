#!/usr/bin/env python3
"""
Batch LoRA data preparation CLI.

Usage examples:

1. From batch config file:
   python run_batch_preparation.py --config batch_config.json

2. From directory structure:
   python run_batch_preparation.py \
     --input-root /data/characters \
     --output-root /data/lora_datasets \
     --preparer-type character \
     --preset character

3. With custom settings:
   python run_batch_preparation.py \
     --config batch_config.json \
     --max-workers 4 \
     --output-dir ./batch_output
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from orchestration import BatchOrchestrator, load_batch_config, create_batch_config_from_directory
from config import get_preset


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('batch_preparation.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description='Batch LoRA data preparation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--config',
        type=Path,
        help='Path to batch configuration file (JSON or YAML)'
    )
    input_group.add_argument(
        '--input-root',
        type=Path,
        help='Root directory containing character subdirectories (for auto-discovery)'
    )

    # Auto-discovery options (when using --input-root)
    parser.add_argument(
        '--output-root',
        type=Path,
        help='Root output directory (required with --input-root)'
    )
    parser.add_argument(
        '--preparer-type',
        choices=['character', 'pose', 'expression', 'background', 'style'],
        default='character',
        help='Preparer type for auto-discovered jobs'
    )
    parser.add_argument(
        '--preset',
        type=str,
        help='Config preset to use for all jobs (character, pose, expression, etc.)'
    )
    parser.add_argument(
        '--name-pattern',
        type=str,
        default='*',
        help='Glob pattern for subdirectory names in auto-discovery'
    )

    # Execution options
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for batch results and logs (default: ./batch_output)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Maximum number of parallel workers (1 = sequential, default: 1)'
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        default=True,
        help='Continue processing remaining jobs if one fails (default: True)'
    )
    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        help='Disable checkpoint saving after each job'
    )

    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without executing'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load or create batch configuration
        if args.config:
            logger.info(f"Loading batch config from: {args.config}")
            batch_config = load_batch_config(args.config)
        else:
            # Auto-discovery mode
            if not args.output_root:
                parser.error("--output-root is required when using --input-root")

            logger.info(f"Auto-discovering jobs in: {args.input_root}")

            # Get base config from preset if specified
            base_config = get_preset(args.preset) if args.preset else {}

            batch_config = create_batch_config_from_directory(
                input_root=args.input_root,
                output_root=args.output_root,
                preparer_type=args.preparer_type,
                base_config=base_config,
                name_pattern=args.name_pattern
            )

        logger.info(f"Batch config loaded with {len(batch_config)} jobs")

        # Dry run: show jobs without executing
        if args.dry_run:
            print("\n" + "="*60)
            print("DRY RUN - Jobs to be processed:")
            print("="*60)
            for i, job_config in enumerate(batch_config.jobs, 1):
                print(f"\n{i}. {job_config.get('job_id', f'job_{i}')}")
                print(f"   Type: {job_config['preparer_type']}")
                print(f"   Name: {job_config['name']}")
                print(f"   Input: {job_config['input_dir']}")
                print(f"   Output: {job_config['output_dir']}")
            print("\n" + "="*60)
            return 0

        # Create orchestrator
        orchestrator = BatchOrchestrator(
            batch_config=batch_config,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            continue_on_error=args.continue_on_error,
            save_checkpoints=not args.no_checkpoints
        )

        # Execute batch
        results = orchestrator.run()

        # Check if all jobs succeeded
        if results['successful']:
            logger.info("All jobs completed successfully!")
            return 0
        else:
            failed_count = results['status_counts'].get('failed', 0)
            logger.warning(f"{failed_count} jobs failed")
            return 1

    except KeyboardInterrupt:
        logger.info("Batch execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Batch execution failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
