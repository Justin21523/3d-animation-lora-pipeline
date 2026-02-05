"""
Abstract base class for high-level processors.

Processors orchestrate the full pipeline: feature extraction, clustering,
quality filtering, and caption generation for specific LoRA data types.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import json


class BaseProcessor(ABC):
    """
    Abstract base class for all processor implementations.

    Processors coordinate multiple components (feature extractors, clusterers,
    caption engines, quality filters) to prepare training data for a specific
    LoRA type (character, pose, expression, background, style).

    Subclasses must implement:
    - process(): Execute the full processing pipeline
    - get_output_structure(): Define the expected output directory structure

    Optional methods to override:
    - configure(): Customize configuration loading
    - validate_config(): Validate configuration parameters
    - save_metadata(): Save processing metadata
    - load_metadata(): Load previously saved metadata
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the processor.

        Args:
            config: Configuration dictionary with processor-specific parameters
            input_dir: Input directory containing source images
            output_dir: Output directory for processed results
        """
        self.config = config or {}
        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.logger = logging.getLogger(self.__class__.__name__)

        # Processing state
        self.is_processed = False
        self.metadata = {}

        # Validate configuration
        self.validate_config()

        # Allow subclasses to perform custom configuration
        self.configure()

        # Create output directory if specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the full processing pipeline.

        This is the main entry point for processing. Subclasses must implement
        the specific logic for their LoRA data type.

        Args:
            **kwargs: Additional processing parameters

        Returns:
            Dictionary with processing results:
            - success: Boolean indicating if processing succeeded
            - output_dir: Path to output directory
            - num_clusters: Number of clusters created
            - num_images: Total number of images processed
            - metadata: Additional processing metadata
        """
        pass

    @abstractmethod
    def get_output_structure(self) -> Dict[str, str]:
        """
        Define the expected output directory structure.

        Returns:
            Dictionary mapping logical names to relative paths:
            Example: {
                "clusters": "clusters/",
                "metadata": "metadata.json",
                "visualizations": "visualizations/",
            }
        """
        pass

    def save_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save processing metadata to a JSON file.

        Args:
            metadata: Optional metadata dict to save (uses self.metadata if None)

        Returns:
            Path to saved metadata file
        """
        if metadata is None:
            metadata = self.metadata

        if not self.output_dir:
            raise ValueError("output_dir must be set before saving metadata")

        metadata_file = self.output_dir / "processing_metadata.json"

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved metadata to {metadata_file}")
        return metadata_file

    def load_metadata(self, metadata_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load previously saved processing metadata.

        Args:
            metadata_file: Optional path to metadata file
                         (uses default location if None)

        Returns:
            Metadata dictionary
        """
        if metadata_file is None:
            if not self.output_dir:
                raise ValueError("output_dir must be set to load metadata")
            metadata_file = self.output_dir / "processing_metadata.json"

        metadata_file = Path(metadata_file)

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.logger.info(f"Loaded metadata from {metadata_file}")
        return self.metadata

    def validate_inputs(self):
        """
        Validate that input directory exists and contains data.

        Raises:
            ValueError: If inputs are invalid
        """
        if not self.input_dir:
            raise ValueError("input_dir must be set")

        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")

        # Check for image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        image_files = [
            f for f in self.input_dir.glob('*')
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            raise ValueError(f"No image files found in {self.input_dir}")

        self.logger.info(f"Found {len(image_files)} image files in {self.input_dir}")

    def configure(self):
        """
        Perform custom configuration setup.

        Subclasses can override this method to initialize components,
        set default parameters, or perform other setup tasks.
        """
        pass

    def validate_config(self):
        """
        Validate configuration parameters.

        Subclasses can override this method to check for required config keys
        and valid parameter ranges.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def get_progress_file(self) -> Path:
        """
        Get path to progress tracking file.

        Returns:
            Path to progress file
        """
        if not self.output_dir:
            raise ValueError("output_dir must be set")
        return self.output_dir / ".processing_progress"

    def save_progress(self, step: str, data: Dict[str, Any]):
        """
        Save processing progress to enable resumption.

        Args:
            step: Current processing step identifier
            data: Progress data for this step
        """
        progress_file = self.get_progress_file()
        progress_data = {
            "step": step,
            "data": data,
            "processor": self.__class__.__name__
        }

        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2)

        self.logger.debug(f"Saved progress at step: {step}")

    def load_progress(self) -> Optional[Dict[str, Any]]:
        """
        Load previously saved processing progress.

        Returns:
            Progress data dict, or None if no progress file exists
        """
        progress_file = self.get_progress_file()

        if not progress_file.exists():
            return None

        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)

        self.logger.info(f"Loaded progress from step: {progress_data.get('step')}")
        return progress_data

    def clear_progress(self):
        """Remove progress tracking file."""
        progress_file = self.get_progress_file()
        if progress_file.exists():
            progress_file.unlink()
            self.logger.debug("Cleared progress file")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dir={self.input_dir}, "
            f"output_dir={self.output_dir}, "
            f"processed={self.is_processed})"
        )
