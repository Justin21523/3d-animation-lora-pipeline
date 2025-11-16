"""
Base Stage Class for Pipeline Stages

Provides common infrastructure for all pipeline stages including:
- Progress tracking
- Error handling
- Logging
- Result validation
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


class BaseStage(ABC):
    """Abstract base class for pipeline stages"""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize stage

        Args:
            config: Stage configuration dictionary
            logger: Optional logger instance (creates one if not provided)
        """
        self.config = config
        self.logger = logger or self._create_logger()
        self.start_time = None
        self.end_time = None
        self.metadata = {}

    def _create_logger(self) -> logging.Logger:
        """Create default logger for this stage"""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the stage

        Args:
            input_data: Input data from previous stage or pipeline

        Returns:
            Dictionary containing stage results and metadata
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate stage configuration

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run stage with error handling and timing

        Args:
            input_data: Input data from previous stage

        Returns:
            Stage execution results
        """
        self.logger.info(f"Starting stage: {self.__class__.__name__}")
        self.start_time = datetime.now()

        try:
            # Validate configuration
            self.validate_config()

            # Execute stage
            result = self.execute(input_data)

            # Add timing metadata
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()

            result['metadata'] = result.get('metadata', {})
            result['metadata'].update({
                'stage_name': self.__class__.__name__,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'duration_seconds': duration
            })

            self.logger.info(
                f"Stage {self.__class__.__name__} completed in {duration:.2f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Stage {self.__class__.__name__} failed: {str(e)}")
            raise

    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """
        Save stage results to JSON file

        Args:
            results: Results dictionary to save
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Path objects to strings for JSON serialization
        serializable_results = self._make_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """
        Recursively convert non-serializable objects to serializable format

        Args:
            obj: Object to convert

        Returns:
            Serializable version of object
        """
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    def load_results(self, input_path: Path) -> Dict[str, Any]:
        """
        Load stage results from JSON file

        Args:
            input_path: Path to input JSON file

        Returns:
            Loaded results dictionary
        """
        with open(input_path, 'r') as f:
            results = json.load(f)

        self.logger.info(f"Results loaded from {input_path}")
        return results
