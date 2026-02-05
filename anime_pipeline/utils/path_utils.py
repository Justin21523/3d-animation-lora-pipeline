"""
Path helpers used across the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def get_project_root() -> Path:
    """
    Infer project root by going two levels up from this file.
    """
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist and return the Path.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def list_media_files(directory: str | Path, suffixes: Iterable[str]) -> List[Path]:
    """
    List files in a directory matching given suffixes (case-insensitive).
    """
    directory = Path(directory)
    normalized = {s.lower() for s in suffixes}
    return sorted([p for p in directory.glob("**/*") if p.suffix.lower() in normalized])

