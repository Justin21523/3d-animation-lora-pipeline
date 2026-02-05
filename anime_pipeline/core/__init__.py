"""
Core infrastructure modules for 2D animation pipeline.

Includes:
- Resource monitoring (GPU/CPU)
- Stage management
- Pipeline orchestration
- Stub framework (Phase 4)
- Metadata I/O utilities (Phase 4)
"""

from .resource_monitor import ResourceMonitor, ResourceStats
from .stage_manager import StageManager, PipelineStage, StageStatus
from .stub_framework import StubMode, StubConfig, StubRegistry
from .metadata_io import MetadataIO

__all__ = [
    "ResourceMonitor",
    "ResourceStats",
    "StageManager",
    "PipelineStage",
    "StageStatus",
    "StubMode",
    "StubConfig",
    "StubRegistry",
    "MetadataIO",
]
