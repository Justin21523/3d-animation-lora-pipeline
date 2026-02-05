"""
Frame deduplication utilities with CPU-only hashing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from anime_pipeline.utils.logging_utils import setup_logging
from anime_pipeline.utils.path_utils import ensure_dir


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class DedupeConfig:
    frames_dir: str = "data_frames/frames"
    output_metadata_path: str = "metadata/frames_dedupe.parquet"
    hash_method: str = "sha1"
    log_dir: Optional[str] = "logs"


def dedupe_frames(config: DedupeConfig, logger=None) -> List[Dict]:
    """
    Deduplicate frames by hashing file content. Keeps first occurrence.
    """
    logger = logger or setup_logging("dedupe_frames", config.log_dir)
    frames_dir = Path(config.frames_dir)
    if not frames_dir.exists():
        logger.warning("Frames directory %s does not exist.", frames_dir)
        return []

    files = sorted([p for p in frames_dir.glob("**/*") if p.suffix.lower() in IMAGE_SUFFIXES])
    seen_hashes: Dict[str, str] = {}
    records: List[Dict] = []

    for idx, path in enumerate(files):
        file_hash = _hash_file(path, method=config.hash_method)
        video_id = path.parent.name
        frame_id = f"{video_id}_{path.stem}"
        is_duplicate = file_hash in seen_hashes
        if not is_duplicate:
            seen_hashes[file_hash] = frame_id
        duplicate_of = seen_hashes[file_hash] if is_duplicate else None

        records.append(
            {
                "frame_id": frame_id,
                "video_id": video_id,
                "image_path": str(path),
                "hash": file_hash,
                "hash_method": config.hash_method,
                "is_kept_dedupe": not is_duplicate,
                "duplicate_of": duplicate_of,
                "order": idx,
            }
        )

    metadata_path = _write_metadata(records, config.output_metadata_path, logger)
    logger.info("Deduplicated %d frames, kept %d, metadata at %s", len(records), sum(r["is_kept_dedupe"] for r in records), metadata_path)
    return records


def _hash_file(path: Path, method: str = "sha1") -> str:
    hasher = hashlib.new(method)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_metadata(records: List[Dict], metadata_path: str | Path, logger) -> Path:
    target_path = Path(metadata_path)
    ensure_dir(target_path.parent)
    if not records:
        target_path.touch()
        return target_path

    try:
        import pandas as pd

        df = pd.DataFrame(records)
        if target_path.suffix == ".parquet":
            df.to_parquet(target_path, index=False)
        else:
            df.to_csv(target_path, index=False)
        return target_path
    except Exception as exc:  # pragma: no cover
        logger.warning("Falling back to CSV metadata due to %s", exc)
        csv_path = target_path.with_suffix(".csv")
        import csv

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        return csv_path
