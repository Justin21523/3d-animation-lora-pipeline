"""
Base classes and abstract interfaces for the modular training pipeline.

This module provides abstract base classes that define the contracts for all
pluggable components in the training data preparation system.
"""

from .feature_extractor import BaseFeatureExtractor
from .clusterer import BaseClusterer
from .caption_engine import BaseCaptionEngine
from .quality_filter import BaseQualityFilter, CompositeQualityFilter
from .processor import BaseProcessor
from .checkpoint_manager import IndexCheckpointManager, StateCheckpointManager, save_progress, load_progress

__all__ = [
    'BaseFeatureExtractor',
    'BaseClusterer',
    'BaseCaptionEngine',
    'BaseQualityFilter',
    'CompositeQualityFilter',
    'BaseProcessor',
    'IndexCheckpointManager',
    'StateCheckpointManager',
    'save_progress',
    'load_progress',
]
