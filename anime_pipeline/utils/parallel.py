"""
Minimal concurrency utilities.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def concurrent_map(fn: Callable[[T], R], items: Sequence[T], max_workers: int = 4) -> List[R]:
    """
    Apply a function to a list of items with a thread pool.
    """
    results: List[R] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(fn, item): item for item in items}
        for future in as_completed(future_map):
            results.append(future.result())
    return results

