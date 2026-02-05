"""
Checkpoint Manager for Resumable Batch Processing

Provides checkpoint/resume functionality for long-running batch jobs.
Supports both index-based (for list processing) and custom state checkpointing.

Part of the base utilities for the training pipeline.

Author: LLMProvider Tooling
Date: 2025-11-30
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class IndexCheckpointManager:
    """
    Manages checkpoints for index-based processing (iterating through lists)

    Tracks:
    - Last processed index
    - Cumulative statistics
    - Timestamp information
    - Custom metadata

    Features:
    - Atomic writes to prevent corruption
    - Automatic backup of previous checkpoint
    - Validation on load
    """

    def __init__(self, checkpoint_dir: Path, filename: str = "checkpoint.json"):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to store checkpoint files
            filename: Name of checkpoint file
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.checkpoint_dir / filename
        self.backup_file = self.checkpoint_dir / f"{filename}.backup"

    def save(self, data: Dict[str, Any]):
        """
        Save checkpoint data atomically

        Args:
            data: Dictionary containing checkpoint state
        """
        # Add metadata
        checkpoint_data = {
            "checkpoint_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            **data
        }

        try:
            # Backup existing checkpoint if it exists
            if self.checkpoint_file.exists():
                import shutil
                shutil.copy2(self.checkpoint_file, self.backup_file)

            # Write to temp file first (atomic)
            temp_file = self.checkpoint_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            # Rename temp to actual (atomic operation)
            temp_file.replace(self.checkpoint_file)

            logging.debug(f"Checkpoint saved: {self.checkpoint_file}")

        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            # Try to restore from backup if write failed
            if self.backup_file.exists():
                import shutil
                shutil.copy2(self.backup_file, self.checkpoint_file)
                logging.info("Restored checkpoint from backup")

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data with validation

        Returns:
            Dict with checkpoint data, or None if no valid checkpoint exists
        """
        if not self.checkpoint_file.exists():
            logging.debug("No checkpoint file found")
            return None

        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)

            # Validate checkpoint version
            if "checkpoint_version" not in data:
                logging.warning("Checkpoint missing version info")

            logging.info(f"Loaded checkpoint from {data.get('timestamp', 'unknown time')}")
            return data

        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Failed to load checkpoint: {e}")

            # Try backup
            if self.backup_file.exists():
                try:
                    with open(self.backup_file, 'r') as f:
                        data = json.load(f)
                    logging.info("Loaded checkpoint from backup")
                    return data
                except Exception as backup_e:
                    logging.error(f"Backup also failed: {backup_e}")

            return None

    def clear(self):
        """Remove checkpoint files"""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.backup_file.exists():
                self.backup_file.unlink()
            logging.info("Checkpoint cleared")
        except Exception as e:
            logging.error(f"Failed to clear checkpoint: {e}")

    def exists(self) -> bool:
        """Check if a checkpoint exists"""
        return self.checkpoint_file.exists()


class StateCheckpointManager:
    """
    Manages checkpoints for arbitrary state (not index-based)

    Useful for complex workflows with multiple stages and custom state.
    """

    def __init__(self, checkpoint_dir: Path, filename: str = "state_checkpoint.json"):
        """
        Initialize state checkpoint manager

        Args:
            checkpoint_dir: Directory to store checkpoint files
            filename: Name of checkpoint file
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.checkpoint_dir / filename

    def save_state(self, stage: str, state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """
        Save workflow state for a specific stage

        Args:
            stage: Name of the current stage
            state: State dictionary for this stage
            metadata: Optional additional metadata
        """
        checkpoint_data = {
            "checkpoint_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "current_stage": stage,
            "state": state,
            "metadata": metadata or {}
        }

        try:
            # Atomic write
            temp_file = self.checkpoint_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            temp_file.replace(self.checkpoint_file)

            logging.debug(f"State checkpoint saved for stage: {stage}")

        except Exception as e:
            logging.error(f"Failed to save state checkpoint: {e}")

    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent state checkpoint

        Returns:
            Dict with stage, state, and metadata, or None if not found
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)

            logging.info(f"Loaded state checkpoint for stage: {data.get('current_stage')}")
            return data

        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Failed to load state checkpoint: {e}")
            return None

    def clear(self):
        """Remove state checkpoint"""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            logging.info("State checkpoint cleared")
        except Exception as e:
            logging.error(f"Failed to clear state checkpoint: {e}")

    def exists(self) -> bool:
        """Check if a state checkpoint exists"""
        return self.checkpoint_file.exists()


# Convenience functions for quick checkpoint operations

def save_progress(checkpoint_dir: Path, index: int, total: int, **kwargs):
    """
    Quick helper to save processing progress

    Args:
        checkpoint_dir: Where to save checkpoint
        index: Current index being processed
        total: Total number of items
        **kwargs: Additional data to save
    """
    manager = IndexCheckpointManager(checkpoint_dir)
    manager.save({
        "last_processed_index": index,
        "total_items": total,
        "progress_percent": (index + 1) / total * 100,
        **kwargs
    })


def load_progress(checkpoint_dir: Path) -> Optional[int]:
    """
    Quick helper to load last processed index

    Args:
        checkpoint_dir: Where checkpoint is stored

    Returns:
        Last processed index, or None if no checkpoint
    """
    manager = IndexCheckpointManager(checkpoint_dir)
    data = manager.load()
    if data:
        return data.get("last_processed_index")
    return None
