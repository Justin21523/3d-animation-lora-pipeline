"""
Quick validation script for CharacterLoRAPreparer.

Tests the full pipeline with a minimal configuration to verify:
1. Component initialization
2. Image loading
3. Quality filtering
4. Feature extraction
5. Clustering
6. Caption generation
7. Dataset assembly
8. Metadata saving
"""

import sys
from pathlib import Path
import logging
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from preparers.character_lora_preparer import CharacterLoRAPreparer


def test_character_preparer(
    input_dir: Path,
    output_dir: Path,
    character_name: str = "test_character"
):
    """
    Test CharacterLoRAPreparer with minimal configuration.

    Args:
        input_dir: Directory containing test images
        output_dir: Output directory for test results
        character_name: Character name for testing
    """

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger("CharacterPreparerTest")

    logger.info("=" * 60)
    logger.info("CharacterLoRAPreparer Validation Test")
    logger.info("=" * 60)

    # Minimal configuration using fast components
    config = {
        'device': 'cuda',
        'batch_size': 8,
        'repeats': 10,

        # Use CLIP for feature extraction (fast and reliable)
        'feature_extractor': {
            'type': 'clip',
            'model_name': 'openai/clip-vit-base-patch32',  # Fastest variant
        },

        # Use KMeans for clustering (fixed k=2 for small test data)
        'clusterer': {
            'type': 'kmeans',
            'n_clusters': 2  # Split images into 2 clusters for test
        },

        # Use template engine for captions (no VLM inference needed)
        'caption_engine': {
            'type': 'template'
        },

        # Minimal quality filters (very lenient for test)
        'quality_filters': [
            {'type': 'size', 'min_width': 128, 'min_height': 128},
            # Skip blur filter for quick test - segmented images may have artifacts
        ]
    }

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Character name: {character_name}")
    logger.info("")

    # Check if input directory exists and has images
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return False

    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in image_extensions]

    if not image_files:
        logger.error(f"No images found in {input_dir}")
        logger.info(f"Expected extensions: {image_extensions}")
        return False

    logger.info(f"Found {len(image_files)} images in input directory")
    logger.info("")

    try:
        # Initialize preparer
        logger.info("Step 1: Initializing CharacterLoRAPreparer...")
        preparer = CharacterLoRAPreparer(
            input_dir=input_dir,
            output_dir=output_dir,
            character_name=character_name,
            config=config
        )
        logger.info("✓ Preparer initialized successfully")
        logger.info("")

        # Run preparation pipeline
        logger.info("Step 2: Running preparation pipeline...")
        metadata = preparer.prepare()
        logger.info("✓ Pipeline completed successfully")
        logger.info("")

        # Verify outputs
        logger.info("Step 3: Verifying outputs...")

        # Check metadata
        metadata_path = output_dir / 'preparation_metadata.json'
        if not metadata_path.exists():
            logger.error("Metadata file not found!")
            return False

        logger.info(f"✓ Metadata saved: {metadata_path}")

        # Check dataset directory
        dataset_dir = Path(metadata['dataset_info']['dataset_dir'])
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return False

        logger.info(f"✓ Dataset directory created: {dataset_dir}")

        # Check images and captions
        dataset_images = list(dataset_dir.glob('*.png')) + list(dataset_dir.glob('*.jpg')) + list(dataset_dir.glob('*.jpeg'))
        dataset_captions = list(dataset_dir.glob('*.txt'))

        logger.info(f"✓ Dataset contains {len(dataset_images)} images")
        logger.info(f"✓ Dataset contains {len(dataset_captions)} caption files")

        if len(dataset_images) != len(dataset_captions):
            logger.warning(f"Image/caption count mismatch: {len(dataset_images)} images vs {len(dataset_captions)} captions")

        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Validation Summary")
        logger.info("=" * 60)
        logger.info(f"Character: {metadata['character_name']}")
        logger.info(f"Total images processed: {metadata['dataset_info']['num_images']}")
        logger.info(f"Clusters found: {metadata['dataset_info']['num_clusters']}")
        logger.info(f"Cluster sizes: {metadata['dataset_info']['cluster_sizes']}")
        logger.info(f"Repeats: {metadata['dataset_info']['repeats']}")
        logger.info(f"Time elapsed: {metadata['elapsed_seconds']:.1f}s")
        logger.info("")
        logger.info(f"Components used:")
        logger.info(f"  - Feature extractor: {metadata['components']['feature_extractor']}")
        logger.info(f"  - Clusterer: {metadata['components']['clusterer']}")
        logger.info(f"  - Caption engine: {metadata['components']['caption_engine']}")
        logger.info("=" * 60)
        logger.info("✓ All validation checks passed!")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Validation failed with error: {e}", exc_info=True)
        return False


def main():
    """CLI entry point for validation test."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate CharacterLoRAPreparer with test data"
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Input directory containing test images'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for test results'
    )

    parser.add_argument(
        '--character-name',
        type=str,
        default='test_character',
        help='Character name for testing (default: test_character)'
    )

    args = parser.parse_args()

    success = test_character_preparer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        character_name=args.character_name
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
