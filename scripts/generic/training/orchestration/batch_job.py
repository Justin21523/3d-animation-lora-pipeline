"""
Batch job data structures and status tracking.
"""

from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchJob:
    """
    Single job in a batch processing pipeline.

    Represents one character/scene to be processed with its configuration.
    """

    # Job identification
    job_id: str
    preparer_type: str  # 'character', 'pose', 'expression', 'background', 'style'

    # Paths
    input_dir: Path
    output_dir: Path

    # Job-specific configuration
    name: str  # character_name or scene_name
    config: Dict[str, Any] = field(default_factory=dict)

    # Status tracking
    status: JobStatus = JobStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

    # Results
    result: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)

    def mark_running(self):
        """Mark job as running."""
        self.status = JobStatus.RUNNING
        self.start_time = datetime.now()

    def mark_completed(self, result: Dict[str, Any]):
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        self.end_time = datetime.now()
        self.result = result

    def mark_failed(self, error: str):
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.end_time = datetime.now()
        self.error_message = error

    def mark_skipped(self, reason: str):
        """Mark job as skipped."""
        self.status = JobStatus.SKIPPED
        self.error_message = reason

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'preparer_type': self.preparer_type,
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'name': self.name,
            'config': self.config,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'error_message': self.error_message,
            'result': self.result
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchJob':
        """Create job from dictionary."""
        # Convert datetime strings back to datetime objects
        start_time = datetime.fromisoformat(data['start_time']) if data.get('start_time') else None
        end_time = datetime.fromisoformat(data['end_time']) if data.get('end_time') else None

        return cls(
            job_id=data['job_id'],
            preparer_type=data['preparer_type'],
            input_dir=Path(data['input_dir']),
            output_dir=Path(data['output_dir']),
            name=data['name'],
            config=data.get('config', {}),
            status=JobStatus(data.get('status', 'pending')),
            start_time=start_time,
            end_time=end_time,
            error_message=data.get('error_message'),
            result=data.get('result')
        )
