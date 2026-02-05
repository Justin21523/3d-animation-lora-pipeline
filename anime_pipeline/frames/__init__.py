"""
Frame extraction and preprocessing utilities.
"""

from .sampling import ExtractFramesConfig, extract_frames
from .dedupe import DedupeConfig, dedupe_frames

__all__ = ["ExtractFramesConfig", "extract_frames", "DedupeConfig", "dedupe_frames"]
