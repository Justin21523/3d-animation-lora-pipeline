"""
Batch configuration management.

Handles loading/saving batch processing configurations that define
multiple characters/scenes to process in one run.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class BatchConfig:
    """
    Batch processing configuration.

    Defines multiple jobs to execute with shared base configuration
    and per-job overrides.
    """

    def __init__(
        self,
        jobs: List[Dict[str, Any]],
        base_config: Optional[Dict[str, Any]] = None,
        batch_settings: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize batch configuration.

        Args:
            jobs: List of job definitions
            base_config: Base configuration applied to all jobs
            batch_settings: Batch execution settings (parallel, gpu_ids, etc.)
        """
        self.jobs = jobs
        self.base_config = base_config or {}
        self.batch_settings = batch_settings or {}

    def __len__(self) -> int:
        """Get number of jobs."""
        return len(self.jobs)

    def __iter__(self):
        """Iterate over jobs."""
        return iter(self.jobs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'jobs': self.jobs,
            'base_config': self.base_config,
            'batch_settings': self.batch_settings
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchConfig':
        """Create from dictionary."""
        return cls(
            jobs=data['jobs'],
            base_config=data.get('base_config', {}),
            batch_settings=data.get('batch_settings', {})
        )


def load_batch_config(config_path: Path) -> BatchConfig:
    """
    Load batch configuration from JSON or YAML file.

    Args:
        config_path: Path to batch config file

    Returns:
        BatchConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Batch config file not found: {config_path}")

    suffix = config_path.suffix.lower()

    if suffix == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif suffix in ['.yaml', '.yml']:
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. Install: pip install pyyaml"
            )
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported config format '{suffix}'. Use .json or .yaml/.yml"
        )

    return BatchConfig.from_dict(data)


def save_batch_config(config: BatchConfig, output_path: Path, format: str = 'json'):
    """
    Save batch configuration to file.

    Args:
        config: BatchConfig object
        output_path: Output file path
        format: Format ('json' or 'yaml')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif format == 'yaml':
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. Install: pip install pyyaml"
            )
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    else:
        raise ValueError(f"Unsupported format '{format}'. Use 'json' or 'yaml'")

    logger.info(f"Batch config saved to: {output_path}")


def create_batch_config_from_directory(
    input_root: Path,
    output_root: Path,
    preparer_type: str = 'character',
    base_config: Optional[Dict[str, Any]] = None,
    name_pattern: str = '*'
) -> BatchConfig:
    """
    Create batch configuration by scanning directory structure.

    Useful for quickly creating batch configs from organized directories:
    input_root/
        character1/
        character2/
        character3/

    Args:
        input_root: Root directory containing character subdirectories
        output_root: Root output directory
        preparer_type: Type of preparer to use
        base_config: Base configuration for all jobs
        name_pattern: Glob pattern for subdirectory names

    Returns:
        BatchConfig object
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    # Find all matching subdirectories
    subdirs = sorted([d for d in input_root.glob(name_pattern) if d.is_dir()])

    if not subdirs:
        raise ValueError(f"No subdirectories found in {input_root} matching '{name_pattern}'")

    # Create job for each subdirectory
    jobs = []
    for subdir in subdirs:
        character_name = subdir.name
        jobs.append({
            'job_id': f"{preparer_type}_{character_name}",
            'preparer_type': preparer_type,
            'input_dir': str(subdir),
            'output_dir': str(output_root / f"{character_name}_lora"),
            'name': character_name,
            'config': {}  # Will be merged with base_config
        })

    logger.info(f"Created batch config with {len(jobs)} jobs from {input_root}")

    return BatchConfig(
        jobs=jobs,
        base_config=base_config or {},
        batch_settings={}
    )
