"""
Lightweight YAML-driven config loader.
"""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is optional but recommended
    yaml = None

T = TypeVar("T")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required to parse YAML configs. Install with `pip install pyyaml`.")  # pragma: no cover
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping, got {type(data)}.")
    return data


def load_config(path: str | Path, config_cls: Type[T], overrides: Optional[Dict[str, Any]] = None) -> T:
    """
    Load YAML into a dataclass instance with optional overrides.
    """
    if not is_dataclass(config_cls):
        raise TypeError("config_cls must be a dataclass type.")
    path = Path(path)
    data = _load_yaml(path)
    if overrides:
        data.update(overrides)

    kwargs: Dict[str, Any] = {}
    for field in fields(config_cls):
        if field.name in data:
            kwargs[field.name] = data[field.name]
        elif field.default is not MISSING:
            kwargs[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            kwargs[field.name] = field.default_factory()  # type: ignore[misc]
        else:
            raise ValueError(f"Missing required config field: {field.name}")
    return config_cls(**kwargs)  # type: ignore[arg-type]

