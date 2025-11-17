"""
Pipeline Core Module

Provides orchestration and management for the end-to-end processing pipeline.

Components:
- orchestrator: Main pipeline coordinator
- stage_manager: Individual stage execution
- resource_monitor: GPU/CPU resource tracking
"""

from .orchestrator import PipelineOrchestrator
from .stage_manager import StageManager, PipelineStage
from .resource_monitor import ResourceMonitor

__all__ = [
    'PipelineOrchestrator',
    'StageManager',
    'PipelineStage',
    'ResourceMonitor',
]
