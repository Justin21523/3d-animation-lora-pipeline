"""
Dataset Organization Module for LoRA Training

Organizes filtered images and prompts into Kohya_ss training format.

Part of Module 5: Dataset Organization Tools
Converts raw generation outputs → structured training datasets

Format:
  output_dir/
    {repeat_count}_{concept_name}/
      image_001.png
      image_001.txt  (caption)
      image_002.png
      image_002.txt
      ...
    dataset_metadata.json

Author: LLMProvider Tooling
Date: 2025-11-30
"""

import argparse
import json
import logging
import sys
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from PIL import Image
from tqdm import tqdm


@dataclass
class DatasetConfig:
    """Configuration for dataset organization"""
    repeat_count: int = 12  # How many times to repeat images during training
    concept_name: str = "concept"  # Concept/character name for folder
    min_resolution: int = 512  # Minimum image resolution
    max_resolution: int = 2048  # Maximum image resolution
    target_resolution: int = 1024  # Target resolution for training
    resize_if_needed: bool = True  # Resize images if out of range
    copy_images: bool = True  # Copy vs move images
    generate_captions_if_missing: bool = False  # Generate simple captions for missing ones
    caption_template: Optional[str] = None  # Template for auto-generated captions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CaptionGenerator:
    """Simple caption generator for images without captions"""

    def __init__(self, template: Optional[str] = None):
        self.template = template or "{concept_name}, high quality, detailed"

    def generate(self, image_path: Path, concept_name: str, prompt_metadata: Optional[str] = None) -> str:
        """
        Generate caption from template or prompt metadata

        Args:
            image_path: Path to image
            concept_name: Concept name
            prompt_metadata: Optional prompt used for generation

        Returns:
            Generated caption string
        """
        if prompt_metadata:
            return prompt_metadata

        return self.template.format(concept_name=concept_name)


class ImageProcessor:
    """Handles image validation and processing"""

    @staticmethod
    def validate_image(image_path: Path, config: DatasetConfig) -> bool:
        """
        Validate image meets requirements

        Args:
            image_path: Path to image
            config: Dataset configuration

        Returns:
            True if valid
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # Check minimum resolution
                if width < config.min_resolution or height < config.min_resolution:
                    logging.warning(f"Image too small: {image_path} ({width}x{height})")
                    return False

                # Check maximum resolution
                if width > config.max_resolution or height > config.max_resolution:
                    logging.warning(f"Image too large: {image_path} ({width}x{height})")
                    return False if not config.resize_if_needed else True

                return True

        except Exception as e:
            logging.error(f"Failed to validate image {image_path}: {e}")
            return False

    @staticmethod
    def process_image(
        image_path: Path,
        output_path: Path,
        config: DatasetConfig
    ) -> bool:
        """
        Process and save image

        Args:
            image_path: Source image path
            output_path: Destination path
            config: Dataset configuration

        Returns:
            True if successful
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # Resize if needed
                if config.resize_if_needed:
                    if width > config.max_resolution or height > config.max_resolution:
                        # Calculate new size maintaining aspect ratio
                        ratio = min(config.target_resolution / width, config.target_resolution / height)
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)

                        img = img.resize((new_width, new_height), Image.LANCZOS)
                        logging.debug(f"Resized {image_path.name}: {width}x{height} → {new_width}x{new_height}")

                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save
                img.save(output_path, quality=95, optimize=True)
                return True

        except Exception as e:
            logging.error(f"Failed to process image {image_path}: {e}")
            return False


class DatasetOrganizer:
    """
    Main dataset organizer class

    Organizes filtered images into Kohya_ss format:
    - Creates {repeat}_{concept}/ directory structure
    - Copies/moves images with sequential naming
    - Creates matching .txt caption files
    - Generates metadata.json
    """

    def __init__(
        self,
        output_dir: Path,
        config: DatasetConfig
    ):
        """
        Initialize dataset organizer

        Args:
            output_dir: Root output directory
            config: Dataset configuration
        """
        self.output_dir = Path(output_dir)
        self.config = config

        # Create concept directory
        self.concept_dir = self.output_dir / f"{config.repeat_count}_{config.concept_name}"
        self.concept_dir.mkdir(parents=True, exist_ok=True)

        # Initialize caption generator
        self.caption_gen = CaptionGenerator(config.caption_template)

        # Statistics
        self.stats = {
            "total_images_processed": 0,
            "images_copied": 0,
            "images_skipped": 0,
            "captions_created": 0,
            "captions_from_source": 0
        }

    def organize_from_directory(
        self,
        source_dir: Path,
        prompts_file: Optional[Path] = None,
        metadata_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Organize images from a source directory

        Args:
            source_dir: Directory containing generated images
            prompts_file: Optional JSON file with prompts (maps image → prompt)
            metadata_file: Optional metadata from generation

        Returns:
            Organization results and statistics
        """
        source_dir = Path(source_dir)

        # Load prompts if provided
        prompts_map = {}
        if prompts_file and prompts_file.exists():
            with open(prompts_file, 'r') as f:
                prompts_data = json.load(f)
                # Support different formats
                if isinstance(prompts_data, dict):
                    if "generated_images" in prompts_data:
                        # From batch_image_generator format
                        for item in prompts_data["generated_images"]:
                            prompts_map[item["filename"]] = item["prompt"]
                    else:
                        # Direct filename → prompt mapping
                        prompts_map = prompts_data
                elif isinstance(prompts_data, list):
                    # List of prompts (match by index)
                    prompts_map = {f"image_{str(i).zfill(6)}.png": prompt
                                 for i, prompt in enumerate(prompts_data)}

        # Get all images from source directory
        image_files = sorted(source_dir.glob("*.png"))
        logging.info(f"Found {len(image_files)} images in {source_dir}")

        # Process each image
        for idx, source_image_path in enumerate(tqdm(image_files, desc="Organizing dataset")):
            self.stats["total_images_processed"] += 1

            # Validate image
            if not ImageProcessor.validate_image(source_image_path, self.config):
                self.stats["images_skipped"] += 1
                continue

            # Create output filename
            output_basename = f"image_{str(idx).zfill(6)}"
            output_image_path = self.concept_dir / f"{output_basename}.png"
            output_caption_path = self.concept_dir / f"{output_basename}.txt"

            # Process and save image
            if self.config.copy_images:
                success = ImageProcessor.process_image(source_image_path, output_image_path, self.config)
            else:
                # Move instead of copy
                try:
                    shutil.move(str(source_image_path), str(output_image_path))
                    success = True
                except Exception as e:
                    logging.error(f"Failed to move {source_image_path}: {e}")
                    success = False

            if not success:
                self.stats["images_skipped"] += 1
                continue

            self.stats["images_copied"] += 1

            # Create caption file
            caption = self._get_caption(source_image_path, prompts_map)
            with open(output_caption_path, 'w') as f:
                f.write(caption)

            self.stats["captions_created"] += 1

        # Create dataset metadata
        metadata = self._create_metadata(source_dir, prompts_file, metadata_file)
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Dataset organization complete: {self.stats['images_copied']} images")

        return {
            "concept_dir": str(self.concept_dir),
            "metadata_file": str(metadata_path),
            "stats": self.stats
        }

    def _get_caption(self, image_path: Path, prompts_map: Dict[str, str]) -> str:
        """
        Get or generate caption for image

        Args:
            image_path: Path to image
            prompts_map: Mapping of filename → prompt

        Returns:
            Caption string
        """
        # Check for existing caption file
        caption_path = image_path.with_suffix('.txt')
        if caption_path.exists():
            with open(caption_path, 'r') as f:
                self.stats["captions_from_source"] += 1
                return f.read().strip()

        # Check prompts map
        if image_path.name in prompts_map:
            self.stats["captions_from_source"] += 1
            return prompts_map[image_path.name]

        # Generate caption
        if self.config.generate_captions_if_missing:
            return self.caption_gen.generate(
                image_path,
                self.config.concept_name,
                prompt_metadata=None
            )

        # Default: empty caption
        return ""

    def _create_metadata(
        self,
        source_dir: Path,
        prompts_file: Optional[Path],
        metadata_file: Optional[Path]
    ) -> Dict[str, Any]:
        """Create dataset metadata"""
        metadata = {
            "dataset_version": "1.0",
            "concept_name": self.config.concept_name,
            "repeat_count": self.config.repeat_count,
            "source_directory": str(source_dir),
            "concept_directory": str(self.concept_dir),
            "config": self.config.to_dict(),
            "stats": self.stats,
            "prompts_file": str(prompts_file) if prompts_file else None,
            "generation_metadata": str(metadata_file) if metadata_file else None
        }

        return metadata


def main():
    """CLI interface for dataset organization"""
    parser = argparse.ArgumentParser(description="Organize images into Kohya_ss training dataset")

    # Required arguments
    parser.add_argument("--source-dir", type=str, required=True,
                       help="Source directory containing generated images")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for organized dataset")
    parser.add_argument("--concept-name", type=str, required=True,
                       help="Concept/character name")

    # Optional arguments
    parser.add_argument("--prompts-file", type=str,
                       help="JSON file with prompts (generation_metadata.json or prompts.json)")
    parser.add_argument("--metadata-file", type=str,
                       help="Generation metadata file")
    parser.add_argument("--repeat-count", type=int, default=12,
                       help="Repeat count for training (default: 12)")
    parser.add_argument("--min-resolution", type=int, default=512,
                       help="Minimum image resolution")
    parser.add_argument("--max-resolution", type=int, default=2048,
                       help="Maximum image resolution")
    parser.add_argument("--target-resolution", type=int, default=1024,
                       help="Target resolution for resizing")
    parser.add_argument("--no-resize", action="store_true",
                       help="Don't resize images")
    parser.add_argument("--move-images", action="store_true",
                       help="Move images instead of copying")
    parser.add_argument("--generate-captions", action="store_true",
                       help="Generate simple captions if missing")
    parser.add_argument("--caption-template", type=str,
                       help="Template for auto-generated captions (use {concept_name})")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(args.output_dir) / "organization.log")
        ]
    )

    # Create configuration
    config = DatasetConfig(
        repeat_count=args.repeat_count,
        concept_name=args.concept_name,
        min_resolution=args.min_resolution,
        max_resolution=args.max_resolution,
        target_resolution=args.target_resolution,
        resize_if_needed=not args.no_resize,
        copy_images=not args.move_images,
        generate_captions_if_missing=args.generate_captions,
        caption_template=args.caption_template
    )

    # Initialize organizer
    organizer = DatasetOrganizer(
        output_dir=Path(args.output_dir),
        config=config
    )

    # Run organization
    try:
        results = organizer.organize_from_directory(
            source_dir=Path(args.source_dir),
            prompts_file=Path(args.prompts_file) if args.prompts_file else None,
            metadata_file=Path(args.metadata_file) if args.metadata_file else None
        )

        logging.info("Organization completed successfully")
        logging.info(f"Concept directory: {results['concept_dir']}")
        logging.info(f"Images organized: {results['stats']['images_copied']}")
        logging.info(f"Captions created: {results['stats']['captions_created']}")

    except Exception as e:
        logging.error(f"Organization failed: {e}")
        raise


if __name__ == "__main__":
    main()
