#!/usr/bin/env python3
"""
Unified Metadata I/O for 2D Animation Pipeline

Eliminates duplicated metadata reading/writing code across modules.
Provides consistent Parquet/CSV handling with automatic fallback.

Problem Solved:
- Each module had its own _load_detections() / _write_records() functions
- Inconsistent error handling and format support
- ~100 lines of duplicated code per module

Solution:
- Single source of truth for metadata I/O
- Automatic Parquet/CSV fallback
- Consistent error handling and logging
- Type hints and validation

Supported Formats:
- Parquet (preferred for performance)
- CSV (fallback for compatibility)

Usage Example:
    from anime_pipeline.core.metadata_io import MetadataIO

    # Load records
    records = MetadataIO.load_records(
        path="detections.parquet",
        logger=logger
    )

    # Save records
    MetadataIO.save_records(
        records=detections,
        path="detections.parquet",
        logger=logger,
        prefer_parquet=True
    )

Author: Created for Phase 4 code deduplication
Date: 2025-01-XX
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import csv


class MetadataIO:
    """
    Unified metadata reading/writing with Parquet/CSV fallback.

    Provides consistent interface for all pipeline metadata operations.
    """

    @staticmethod
    def load_records(
        path: Path | str,
        logger: Optional[logging.Logger] = None,
        required_columns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load metadata records from Parquet or CSV.

        Automatic format detection and fallback:
        1. Try loading specified format
        2. If fails, try alternate format (.parquet <-> .csv)
        3. If both fail, return empty list

        Args:
            path: Path to metadata file (.parquet or .csv)
            logger: Optional logger for warnings
            required_columns: Optional list of required columns to validate

        Returns:
            List of record dictionaries

        Example:
            >>> records = MetadataIO.load_records("detections.parquet", logger)
            >>> print(f"Loaded {len(records)} records")
            >>> for rec in records:
            ...     print(rec['det_id'], rec['bbox_x1'])
        """
        path = Path(path)

        if logger:
            logger.debug(f"Loading metadata from {path}")

        # Check if file exists
        if not path.exists():
            # Try alternate extension
            if path.suffix == ".parquet":
                alt_path = path.with_suffix(".csv")
            elif path.suffix == ".csv":
                alt_path = path.with_suffix(".parquet")
            else:
                alt_path = None

            if alt_path and alt_path.exists():
                if logger:
                    logger.info(f"Primary file not found, using alternate: {alt_path}")
                path = alt_path
            else:
                if logger:
                    logger.warning(f"Metadata not found: {path}")
                return []

        # Load based on extension
        try:
            if path.suffix == ".parquet":
                records = MetadataIO._load_parquet(path, logger)
            elif path.suffix == ".csv":
                records = MetadataIO._load_csv(path, logger)
            else:
                # Unknown extension, try both
                if logger:
                    logger.warning(f"Unknown extension {path.suffix}, trying Parquet first")
                try:
                    records = MetadataIO._load_parquet(path, logger)
                except Exception:
                    records = MetadataIO._load_csv(path, logger)

            if logger:
                logger.info(f"Loaded {len(records)} records from {path}")

            # Validate required columns
            if required_columns and records:
                MetadataIO._validate_columns(records[0], required_columns, logger)

            return records

        except Exception as e:
            if logger:
                logger.error(f"Failed to load metadata from {path}: {e}")
            return []

    @staticmethod
    def _load_parquet(path: Path, logger: Optional[logging.Logger]) -> List[Dict[str, Any]]:
        """Load from Parquet format."""
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return df.to_dict(orient="records")
        except ImportError:
            if logger:
                logger.warning("pandas not available, cannot load Parquet")
            raise
        except Exception as e:
            if logger:
                logger.debug(f"Parquet load failed: {e}")
            raise

    @staticmethod
    def _load_csv(path: Path, logger: Optional[logging.Logger]) -> List[Dict[str, Any]]:
        """Load from CSV format."""
        try:
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_dict(orient="records")
        except ImportError:
            # Fallback to Python csv module
            if logger:
                logger.debug("pandas not available, using csv module")

            records = []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(dict(row))
            return records
        except Exception as e:
            if logger:
                logger.debug(f"CSV load failed: {e}")
            raise

    @staticmethod
    def _validate_columns(
        sample_record: Dict[str, Any],
        required_columns: List[str],
        logger: Optional[logging.Logger]
    ):
        """Validate that required columns are present."""
        missing = [col for col in required_columns if col not in sample_record]

        if missing:
            error_msg = f"Missing required columns: {missing}"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def save_records(
        records: List[Dict[str, Any]],
        path: Path | str,
        logger: Optional[logging.Logger] = None,
        prefer_parquet: bool = True,
        create_dirs: bool = True
    ) -> Path:
        """
        Save metadata records to Parquet (with CSV fallback).

        Automatic fallback strategy:
        1. Try preferred format (Parquet if prefer_parquet=True)
        2. If fails, try alternate format
        3. Return actual path used

        Args:
            records: List of record dictionaries
            path: Output path (.parquet or .csv)
            logger: Optional logger for info messages
            prefer_parquet: Prefer Parquet format (default: True)
            create_dirs: Create parent directories if needed

        Returns:
            Actual path used for saving (may differ from input if fallback occurred)

        Example:
            >>> detections = [
            ...     {"det_id": "d1", "bbox_x1": 10, "bbox_y1": 20},
            ...     {"det_id": "d2", "bbox_x1": 30, "bbox_y1": 40}
            ... ]
            >>> actual_path = MetadataIO.save_records(
            ...     records=detections,
            ...     path="detections.parquet",
            ...     logger=logger
            ... )
        """
        path = Path(path)

        # Create parent directories
        if create_dirs and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        # Handle empty records
        if not records:
            if logger:
                logger.warning(f"No records to save, creating empty file: {path}")
            path.touch()
            return path

        if logger:
            logger.debug(f"Saving {len(records)} records to {path}")

        # Determine format
        if prefer_parquet and path.suffix != ".parquet":
            # Change to .parquet
            path = path.with_suffix(".parquet")
        elif not prefer_parquet and path.suffix != ".csv":
            # Change to .csv
            path = path.with_suffix(".csv")

        # Try saving in preferred format
        try:
            if path.suffix == ".parquet":
                MetadataIO._save_parquet(records, path, logger)
            else:
                MetadataIO._save_csv(records, path, logger)

            if logger:
                logger.info(f"Saved {len(records)} records to {path}")

            return path

        except Exception as e:
            # Fallback to alternate format
            if logger:
                logger.warning(
                    f"Failed to save as {path.suffix}: {e}, "
                    f"trying alternate format"
                )

            if path.suffix == ".parquet":
                alt_path = path.with_suffix(".csv")
                MetadataIO._save_csv(records, alt_path, logger)
            else:
                alt_path = path.with_suffix(".parquet")
                try:
                    MetadataIO._save_parquet(records, alt_path, logger)
                except Exception:
                    # Parquet failed, stick with CSV
                    MetadataIO._save_csv(path, records, logger)
                    alt_path = path

            if logger:
                logger.info(f"Saved {len(records)} records to {alt_path} (fallback)")

            return alt_path

    @staticmethod
    def _save_parquet(
        records: List[Dict[str, Any]],
        path: Path,
        logger: Optional[logging.Logger]
    ):
        """Save to Parquet format."""
        try:
            import pandas as pd
            df = pd.DataFrame(records)
            df.to_parquet(path, index=False)
        except ImportError:
            if logger:
                logger.warning("pandas not available, cannot save Parquet")
            raise
        except Exception as e:
            if logger:
                logger.debug(f"Parquet save failed: {e}")
            raise

    @staticmethod
    def _save_csv(
        records: List[Dict[str, Any]],
        path: Path,
        logger: Optional[logging.Logger]
    ):
        """Save to CSV format."""
        try:
            import pandas as pd
            df = pd.DataFrame(records)
            df.to_csv(path, index=False)
        except ImportError:
            # Fallback to Python csv module
            if logger:
                logger.debug("pandas not available, using csv module")

            fieldnames = list(records[0].keys())
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
        except Exception as e:
            if logger:
                logger.debug(f"CSV save failed: {e}")
            raise

    @staticmethod
    def merge_records(
        records_list: List[List[Dict[str, Any]]],
        deduplicate_by: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Merge multiple record lists.

        Args:
            records_list: List of record lists to merge
            deduplicate_by: Optional key to deduplicate by (e.g., 'det_id')

        Returns:
            Merged record list

        Example:
            >>> batch1 = [{"id": "a", "val": 1}]
            >>> batch2 = [{"id": "b", "val": 2}]
            >>> merged = MetadataIO.merge_records([batch1, batch2])
            >>> len(merged)
            2
        """
        merged = []
        seen = set()

        for records in records_list:
            for record in records:
                if deduplicate_by:
                    key = record.get(deduplicate_by)
                    if key in seen:
                        continue
                    seen.add(key)

                merged.append(record)

        return merged

    @staticmethod
    def filter_records(
        records: List[Dict[str, Any]],
        filter_fn: callable
    ) -> List[Dict[str, Any]]:
        """
        Filter records by predicate function.

        Args:
            records: List of records
            filter_fn: Function (record) -> bool

        Returns:
            Filtered record list

        Example:
            >>> records = [
            ...     {"score": 0.9, "class": "person"},
            ...     {"score": 0.3, "class": "person"}
            ... ]
            >>> high_conf = MetadataIO.filter_records(
            ...     records,
            ...     lambda r: r["score"] > 0.5
            ... )
            >>> len(high_conf)
            1
        """
        return [r for r in records if filter_fn(r)]

    @staticmethod
    def update_records(
        records: List[Dict[str, Any]],
        update_fn: callable
    ) -> List[Dict[str, Any]]:
        """
        Update records in-place using update function.

        Args:
            records: List of records
            update_fn: Function (record) -> updated_record

        Returns:
            Updated record list (same list, modified in place)

        Example:
            >>> records = [{"id": "a", "bbox_x1": 10}]
            >>> MetadataIO.update_records(
            ...     records,
            ...     lambda r: {**r, "normalized": True}
            ... )
        """
        for i, record in enumerate(records):
            records[i] = update_fn(record)

        return records

    @staticmethod
    def get_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute basic statistics on record list.

        Args:
            records: List of records

        Returns:
            Statistics dictionary

        Example:
            >>> records = [{"score": 0.9}, {"score": 0.8}, {"score": 0.7}]
            >>> stats = MetadataIO.get_statistics(records)
            >>> print(stats['count'])
            3
        """
        if not records:
            return {
                'count': 0,
                'columns': [],
                'sample': None
            }

        stats = {
            'count': len(records),
            'columns': list(records[0].keys()),
            'sample': records[0] if records else None
        }

        # Compute numeric statistics if possible
        try:
            import pandas as pd
            df = pd.DataFrame(records)
            numeric_stats = df.describe().to_dict()
            stats['numeric_stats'] = numeric_stats
        except Exception:
            pass

        return stats
