#!/usr/bin/env python3
"""
Checkpoint Manager for Resume Capability

Provides checkpoint/resume functionality for long-running scripts.
Allows scripts to save progress and resume from last checkpoint after interruption.

Features:
- Save processed items to checkpoint file
- Load checkpoint and skip already processed items
- Automatic checkpoint cleanup on completion
- Thread-safe operations
- Configurable checkpoint intervals

Usage:
    from scripts.core.utils.checkpoint_manager import CheckpointManager

    # Initialize
    checkpoint_mgr = CheckpointManager(
        checkpoint_path="output_dir/checkpoint.json",
        save_interval=100
    )

    # Check if resuming
    if checkpoint_mgr.exists():
        checkpoint_mgr.load()
        logger.info(f"Resuming from checkpoint: {len(checkpoint_mgr)} items processed")

    # Process items
    for item in items:
        if checkpoint_mgr.is_processed(item):
            continue  # Skip already processed

        # Process item...

        # Mark as processed
        checkpoint_mgr.mark_processed(item)

    # Clean up checkpoint on completion
    checkpoint_mgr.cleanup()

Author: AI Pipeline
Date: 2025-01-17
"""

import json
import logging
from pathlib import Path
from typing import Set, List, Union, Optional, Any, Dict
from datetime import datetime
import threading


class CheckpointManager:
    """
    Manages checkpoint/resume functionality for long-running operations.

    Thread-safe implementation with automatic saving at intervals.
    """

    def __init__(self,
                 checkpoint_path: Union[str, Path],
                 save_interval: int = 100,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint file
            save_interval: Save checkpoint every N items (0 = manual save only)
            logger: Logger instance
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.save_interval = save_interval
        self.logger = logger or logging.getLogger(__name__)

        # Thread-safe set of processed items
        self._processed: Set[str] = set()
        self._lock = threading.Lock()

        # Counters
        self._item_count = 0
        self._last_save_count = 0

        # Metadata
        self._metadata: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'last_updated': None,
            'total_items': 0
        }

    def exists(self) -> bool:
        """Check if checkpoint file exists."""
        return self.checkpoint_path.exists()

    def load(self) -> bool:
        """
        Load checkpoint from file.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.checkpoint_path.exists():
            self.logger.info("No checkpoint found, starting fresh")
            return False

        try:
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)

            with self._lock:
                self._processed = set(data.get('processed', []))
                self._metadata = data.get('metadata', {})
                self._item_count = len(self._processed)
                self._last_save_count = self._item_count

            self.logger.info(f"Loaded checkpoint: {len(self._processed)} items already processed")
            self.logger.info(f"  Created: {self._metadata.get('created_at', 'unknown')}")
            self.logger.info(f"  Last updated: {self._metadata.get('last_updated', 'unknown')}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

    def save(self, force: bool = False):
        """
        Save checkpoint to file.

        Args:
            force: Force save regardless of interval
        """
        # Check if save needed
        if not force and self.save_interval > 0:
            items_since_save = self._item_count - self._last_save_count
            if items_since_save < self.save_interval:
                return

        try:
            with self._lock:
                self._metadata['last_updated'] = datetime.now().isoformat()
                self._metadata['total_items'] = self._item_count

                data = {
                    'processed': sorted(list(self._processed)),
                    'metadata': self._metadata
                }

            # Ensure parent directory exists
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first, then rename (atomic operation)
            temp_path = self.checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)

            temp_path.rename(self.checkpoint_path)

            self._last_save_count = self._item_count

            self.logger.debug(f"Checkpoint saved: {self._item_count} items")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def is_processed(self, item: Union[str, Path]) -> bool:
        """
        Check if item has been processed.

        Args:
            item: Item identifier (file path, key, etc.)

        Returns:
            True if item was already processed
        """
        item_str = str(item)
        with self._lock:
            return item_str in self._processed

    def mark_processed(self, item: Union[str, Path]):
        """
        Mark item as processed and auto-save if interval reached.

        Args:
            item: Item identifier (file path, key, etc.)
        """
        item_str = str(item)

        with self._lock:
            if item_str not in self._processed:
                self._processed.add(item_str)
                self._item_count += 1

        # Auto-save at intervals
        if self.save_interval > 0:
            self.save()

    def mark_batch_processed(self, items: List[Union[str, Path]]):
        """
        Mark multiple items as processed (batch operation).

        Args:
            items: List of item identifiers
        """
        with self._lock:
            for item in items:
                item_str = str(item)
                if item_str not in self._processed:
                    self._processed.add(item_str)
                    self._item_count += 1

        # Auto-save after batch
        if self.save_interval > 0:
            self.save()

    def get_processed_items(self) -> List[str]:
        """Get list of all processed items."""
        with self._lock:
            return sorted(list(self._processed))

    def get_unprocessed_items(self, all_items: List[Union[str, Path]]) -> List[Path]:
        """
        Filter list to get only unprocessed items.

        Args:
            all_items: Complete list of items to process

        Returns:
            List of items not yet processed
        """
        unprocessed = []

        for item in all_items:
            if not self.is_processed(item):
                unprocessed.append(Path(item) if isinstance(item, str) else item)

        return unprocessed

    def cleanup(self):
        """
        Remove checkpoint file (call after successful completion).
        """
        try:
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
                self.logger.info("Checkpoint file removed (processing complete)")
        except Exception as e:
            self.logger.warning(f"Failed to remove checkpoint file: {e}")

    def update_metadata(self, key: str, value: Any):
        """
        Update checkpoint metadata.

        Args:
            key: Metadata key
            value: Metadata value (must be JSON-serializable)
        """
        with self._lock:
            self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get checkpoint metadata value.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        with self._lock:
            return self._metadata.get(key, default)

    def __len__(self) -> int:
        """Get number of processed items."""
        with self._lock:
            return len(self._processed)

    def __contains__(self, item: Union[str, Path]) -> bool:
        """Check if item is processed (supports 'in' operator)."""
        return self.is_processed(item)

    def __repr__(self) -> str:
        """String representation."""
        return f"CheckpointManager(path={self.checkpoint_path}, processed={len(self)})"
