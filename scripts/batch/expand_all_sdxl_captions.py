#!/usr/bin/env python3
"""
Batch SDXL Caption Expansion

Expands captions for all characters across all films to SDXL-optimized format.

Processes all characters:
- Luca: alberto, giulia
- Onward: ian, barley
- Turning Red: tyler
- Up: russell
- Orion: orion
- Elio: elio, bryce, caleb, glordon, miguel (when ready)
- Coco: miguel

Author: LLMProvider Tooling
Date: 2025-11-22
"""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'generic' / 'training'))
from sdxl_caption_expander import SDXLCaptionExpander

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchSDXLExpander:
    """
    Batch expand captions for all characters to SDXL format.
    """

    # Character configuration with film context
    CHARACTER_CONFIGS = {
        # Luca characters
        'alberto': {
            'film': 'luca',
            'style': 'pixar',
            'training_data_path': 'alberto_identity',
            'character_name': 'alberto scorfano',
        },
        'giulia': {
            'film': 'luca',
            'style': 'pixar',
            'training_data_path': 'giulia_identity',
            'character_name': 'giulia marcovaldo',
        },

        # Onward characters
        'ian': {
            'film': 'onward',
            'style': 'pixar',
            'training_data_path': 'ian_lightfoot_identity',
            'character_name': 'ian lightfoot',
        },
        'barley': {
            'film': 'onward',
            'style': 'pixar',
            'training_data_path': 'barley_lightfoot_identity',
            'character_name': 'barley lightfoot',
        },

        # Turning Red
        'tyler': {
            'film': 'turning_red',
            'style': 'pixar',
            'training_data_path': 'tyler_identity',
            'character_name': 'tyler',
        },

        # Up
        'russell': {
            'film': 'up',
            'style': 'pixar',
            'training_data_path': 'russell_identity',
            'character_name': 'russell',
        },

        # Orion and the Dark
        'orion': {
            'film': 'orion',
            'style': 'dreamworks',
            'training_data_path': 'orion_identity',
            'character_name': 'orion',
        },

        # Elio characters
        'elio': {
            'film': 'elio',
            'style': 'pixar',
            'training_data_path': 'elio_identity',
            'character_name': 'elio solis',
        },
        'bryce': {
            'film': 'elio',
            'style': 'pixar',
            'training_data_path': 'bryce_identity',
            'character_name': 'bryce markwell',
        },
        'caleb': {
            'film': 'elio',
            'style': 'pixar',
            'training_data_path': 'caleb_identity',
            'character_name': 'caleb',
        },
        'glordon': {
            'film': 'elio',
            'style': 'pixar',
            'training_data_path': 'glordon_identity',
            'character_name': 'glordon',
        },

        # Coco
        'miguel': {
            'film': 'coco',
            'style': 'pixar',
            'training_data_path': 'miguel_identity',
            'character_name': 'miguel rivera',
        },
    }

    def __init__(
        self,
        base_data_dir: Path = Path('/mnt/data/ai_data/datasets/3d-anime'),
        dry_run: bool = False
    ):
        """
        Initialize batch SDXL expander.

        Args:
            base_data_dir: Base directory for all film datasets
            dry_run: If True, only scan and report, don't expand
        """
        self.base_data_dir = Path(base_data_dir)
        self.dry_run = dry_run
        self.results = []

    def scan_characters(self, characters: Optional[List[str]] = None) -> List[Dict]:
        """
        Scan for characters that have SD1.5 training data.

        Args:
            characters: List of character IDs to process (None = all)

        Returns:
            List of character jobs to process
        """
        jobs = []

        # Filter characters if specified
        if characters:
            configs = {k: v for k, v in self.CHARACTER_CONFIGS.items() if k in characters}
        else:
            configs = self.CHARACTER_CONFIGS

        for char_id, config in configs.items():
            film = config['film']
            training_path = config['training_data_path']

            # Check if SD1.5 training data exists
            sd15_dir = self.base_data_dir / film / 'lora_data' / 'training_data' / training_path

            if not sd15_dir.exists():
                logger.warning(f"⚠️  {char_id}: SD1.5 training data not found at {sd15_dir}")
                continue

            # Check for caption files (including in subdirectories for Kohya format)
            caption_files = list(sd15_dir.rglob('*.txt'))
            if not caption_files:
                logger.warning(f"⚠️  {char_id}: No caption files found in {sd15_dir}")
                continue

            # SDXL output directory
            sdxl_dir = self.base_data_dir / film / 'lora_data' / 'training_data_sdxl' / training_path

            # Check if already expanded
            already_expanded = False
            if sdxl_dir.exists():
                sdxl_files = list(sdxl_dir.rglob('*.txt'))
                if len(sdxl_files) == len(caption_files):
                    metadata_file = sdxl_dir / 'sdxl_expansion_metadata.json'
                    if metadata_file.exists():
                        already_expanded = True
                        logger.info(f"✓ {char_id}: Already expanded ({len(sdxl_files)} files)")

            job = {
                'character_id': char_id,
                'character_name': config['character_name'],
                'film': film,
                'style': config['style'],
                'sd15_dir': sd15_dir,
                'sdxl_dir': sdxl_dir,
                'caption_count': len(caption_files),
                'already_expanded': already_expanded
            }

            jobs.append(job)

        return jobs

    def expand_all(
        self,
        characters: Optional[List[str]] = None,
        skip_existing: bool = True,
        max_files_per_char: Optional[int] = None
    ) -> Dict:
        """
        Expand captions for all characters.

        Args:
            characters: List of character IDs to process (None = all)
            skip_existing: Skip already-expanded characters
            max_files_per_char: Limit files per character (for testing)

        Returns:
            Summary results
        """
        # Scan characters
        jobs = self.scan_characters(characters)

        if not jobs:
            logger.error("No characters found to process")
            return {'error': 'No characters found'}

        # Filter out already-expanded if requested
        if skip_existing:
            jobs = [j for j in jobs if not j['already_expanded']]

        if not jobs:
            logger.info("All characters already expanded!")
            return {'message': 'All characters already expanded'}

        logger.info(f"\n{'='*70}")
        logger.info(f"SDXL CAPTION EXPANSION - BATCH PROCESSING")
        logger.info(f"{'='*70}")
        logger.info(f"Characters to process: {len(jobs)}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(f"Max files per character: {max_files_per_char or 'unlimited'}")
        logger.info(f"{'='*70}\n")

        # Display summary
        for i, job in enumerate(jobs, 1):
            logger.info(f"{i}. {job['character_id']} ({job['film']}, {job['style']})")
            logger.info(f"   SD1.5: {job['sd15_dir']}")
            logger.info(f"   SDXL:  {job['sdxl_dir']}")
            logger.info(f"   Captions: {job['caption_count']}")

        if self.dry_run:
            logger.info("\n✓ Dry run complete. Use --execute to run expansion.")
            return {'dry_run': True, 'jobs': jobs}

        # Confirm before proceeding (skip if --yes flag)
        if not getattr(self, 'auto_confirm', False):
            print(f"\n{'='*70}")
            print(f"Ready to expand captions for {len(jobs)} characters")
            print(f"Estimated cost: ~${len(jobs) * 1.0:.2f} (approximate)")
            print(f"{'='*70}")
            confirm = input("Proceed? (y/n): ")

            if confirm.lower() != 'y':
                logger.info("Cancelled by user")
                return {'cancelled': True}
        else:
            logger.info(f"Auto-confirmed: Proceeding with {len(jobs)} characters")

        # Create expander
        expander = SDXLCaptionExpander()

        # Process each character
        for i, job in enumerate(jobs, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"[{i}/{len(jobs)}] Processing: {job['character_id']}")
            logger.info(f"{'='*70}")

            try:
                result = expander.expand_directory(
                    input_dir=job['sd15_dir'],
                    output_dir=job['sdxl_dir'],
                    character_name=job['character_name'],
                    style=job['style'],
                    max_files=max_files_per_char
                )

                self.results.append({
                    'character_id': job['character_id'],
                    'status': 'success',
                    'result': result
                })

            except Exception as e:
                logger.error(f"Failed to process {job['character_id']}: {e}")
                self.results.append({
                    'character_id': job['character_id'],
                    'status': 'failed',
                    'error': str(e)
                })

        # Generate summary
        summary = self._generate_summary(jobs)
        return summary

    def _generate_summary(self, jobs: List[Dict]) -> Dict:
        """Generate batch processing summary."""
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']

        total_captions = sum(
            r['result']['statistics']['processed']
            for r in successful
        )

        total_cost = sum(
            r['result']['statistics']['estimated_cost_usd']
            for r in successful
        )

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_characters': len(jobs),
            'successful': len(successful),
            'failed': len(failed),
            'total_captions_expanded': total_captions,
            'total_estimated_cost_usd': total_cost,
            'results': self.results
        }

        # Save summary
        summary_file = Path('logs') / f'sdxl_expansion_batch_{datetime.now():%Y%m%d_%H%M%S}.json'
        summary_file.parent.mkdir(exist_ok=True)

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total characters: {summary['total_characters']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Total captions expanded: {summary['total_captions_expanded']}")
        logger.info(f"Total estimated cost: ${summary['total_estimated_cost_usd']:.2f}")
        logger.info(f"Summary saved: {summary_file}")
        logger.info(f"{'='*70}\n")

        if failed:
            logger.warning(f"Failed characters:")
            for r in failed:
                logger.warning(f"  - {r['character_id']}: {r['error']}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch expand SD1.5 captions to SDXL format for all characters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Scan all characters (dry run)
  python expand_all_sdxl_captions.py --dry-run

  # Expand all characters
  python expand_all_sdxl_captions.py --execute

  # Expand specific characters only
  python expand_all_sdxl_captions.py --execute --characters alberto giulia orion

  # Test with limited files
  python expand_all_sdxl_captions.py --execute --max-files 10

  # Force re-expansion (overwrite existing)
  python expand_all_sdxl_captions.py --execute --no-skip-existing

Characters supported:
  Luca: alberto, giulia
  Onward: ian, barley
  Turning Red: tyler
  Up: russell
  Orion: orion
  Elio: elio, bryce, caleb, glordon
  Coco: miguel
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Scan and report only, do not expand'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute caption expansion'
    )
    parser.add_argument(
        '--characters',
        nargs='+',
        help='Specific character IDs to process (default: all)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-expand already-expanded characters'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        help='Limit files per character (for testing)'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path('/mnt/data/ai_data/datasets/3d-anime'),
        help='Base directory for film datasets'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt (auto-confirm)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dry_run and not args.execute:
        parser.error("Must specify --dry-run or --execute")

    # Create batch expander
    batch_expander = BatchSDXLExpander(
        base_data_dir=args.base_dir,
        dry_run=args.dry_run
    )
    batch_expander.auto_confirm = args.yes

    # Run expansion
    try:
        results = batch_expander.expand_all(
            characters=args.characters,
            skip_existing=not args.no_skip_existing,
            max_files_per_char=args.max_files
        )

        if results.get('failed', 0) > 0:
            return 1

        return 0

    except Exception as e:
        logger.error(f"Batch expansion failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
